// Originally written in 2023 by Arman Uguray <arman.uguray@gmail.com>
// SPDX-License-Identifier: CC-BY-4.0

use bytemuck::{Pod, Zeroable};
use std::{
    path::Path,
    sync::{mpsc, Arc},
};

use crate::camera::{Camera, CameraUniforms};

pub struct PathTracer {
    device: wgpu::Device,
    queue: wgpu::Queue,

    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    post_uniforms: PostUniforms,
    post_uniform_buffer: wgpu::Buffer,

    window_width: u32,
    window_height: u32,
    surface_format: wgpu::TextureFormat,
    render_width: u32,
    render_height: u32,
    render_scale: f32,
    radiance_samples: [wgpu::Texture; 2],
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,

    pipeline: wgpu::RenderPipeline,
    render_group_layout: wgpu::BindGroupLayout,
    render_bind_groups: [wgpu::BindGroup; 2],
    scene_group_layout: wgpu::BindGroupLayout,
    upscale_pipeline: wgpu::RenderPipeline,
    upscale_bind_group_layout: wgpu::BindGroupLayout,
    upscale_bind_group: wgpu::BindGroup,
    upscale_sampler: wgpu::Sampler,
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct Uniforms {
    camera: CameraUniforms,
    width: u32,
    height: u32,
    frame_count: u32,
    samples_per_frame: u32,
    audio_energy: f32,
    reactive_lights: f32,
    _pad0: u32,
    _pad1: u32,
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct PostUniforms {
    width: u32,
    height: u32,
    strength: f32,
    radius: f32,
    edge_threshold: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

impl PathTracer {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> PathTracer {
        device.on_uncaptured_error(Arc::new(|error| {
            panic!("Aborting due to an error: {}", error);
        }));

        let shader_module = compile_shader_module(&device);
        let upscale_shader_module = compile_upscale_shader_module(&device);
        let (pipeline, render_group_layout, scene_group_layout) =
            create_pipeline(&device, &shader_module, surface_format);
        let (upscale_pipeline, upscale_bind_group_layout) =
            create_upscale_pipeline(&device, &upscale_shader_module, surface_format);

        // Initialize the uniform buffer.
        let uniforms = Uniforms {
            camera: CameraUniforms::zeroed(),
            width,
            height,
            frame_count: 0,
            samples_per_frame: 1,
            audio_energy: 0.0,
            reactive_lights: 0.0,
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let post_uniforms = PostUniforms {
            width,
            height,
            strength: 0.0,
            radius: 1.0,
            edge_threshold: 0.16,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let post_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post uniforms"),
            size: std::mem::size_of::<PostUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let radiance_samples = create_sample_textures(&device, width, height);
        let (output_texture, output_view) =
            create_output_texture(&device, width, height, surface_format);
        let upscale_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("upscale sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let upscale_bind_group = create_upscale_bind_group(
            &device,
            &upscale_bind_group_layout,
            &output_view,
            &upscale_sampler,
            &post_uniform_buffer,
        );
        let render_bind_groups = create_render_bind_groups(
            &device,
            &render_group_layout,
            &radiance_samples,
            &uniform_buffer,
        );

        PathTracer {
            device,
            queue,
            uniforms,
            uniform_buffer,
            post_uniforms,
            post_uniform_buffer,
            window_width: width,
            window_height: height,
            surface_format,
            render_width: width,
            render_height: height,
            render_scale: 1.0,
            radiance_samples,
            output_texture,
            output_view,
            pipeline,
            render_group_layout,
            render_bind_groups,
            scene_group_layout,
            upscale_pipeline,
            upscale_bind_group_layout,
            upscale_bind_group,
            upscale_sampler,
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn scene_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.scene_group_layout
    }

    pub fn reset_samples(&mut self) {
        self.uniforms.frame_count = 0;
    }

    pub fn set_samples_per_frame(&mut self, samples: u32) {
        self.uniforms.samples_per_frame = samples.clamp(1, 8);
    }

    pub fn set_post_filter(&mut self, strength: f32, radius: f32, edge_threshold: f32) {
        self.post_uniforms.strength = strength.clamp(0.0, 1.0);
        self.post_uniforms.radius = radius.clamp(0.5, 2.0);
        self.post_uniforms.edge_threshold = edge_threshold.clamp(0.02, 1.0);
    }

    pub fn set_audio_reactivity(&mut self, energy: f32, reactive_lights: f32) {
        self.uniforms.audio_energy = energy.clamp(0.0, 1.0);
        self.uniforms.reactive_lights = reactive_lights.clamp(0.0, 1.0);
    }

    pub fn set_render_scale(&mut self, scale: f32) {
        let scale = scale.clamp(0.1, 1.0);
        self.render_scale = scale;
        self.rebuild_render_textures();
    }

    /// Match the renderer to a new window/surface size. The internal render
    /// targets are sized from this times the current render scale.
    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        if width == self.window_width && height == self.window_height {
            return;
        }
        self.window_width = width;
        self.window_height = height;
        self.rebuild_render_textures();
    }

    /// Recreate the radiance/output textures (and their bind groups) for the
    /// current window size and render scale. A no-op if the derived dimensions
    /// are unchanged.
    fn rebuild_render_textures(&mut self) {
        let render_width =
            ((self.window_width as f32 * self.render_scale).round() as u32).max(1);
        let render_height =
            ((self.window_height as f32 * self.render_scale).round() as u32).max(1);
        if render_width == self.render_width && render_height == self.render_height {
            return;
        }

        self.render_width = render_width;
        self.render_height = render_height;
        self.uniforms.width = render_width;
        self.uniforms.height = render_height;
        self.post_uniforms.width = render_width;
        self.post_uniforms.height = render_height;
        self.reset_samples();

        self.radiance_samples =
            create_sample_textures(&self.device, render_width, render_height);
        self.render_bind_groups = create_render_bind_groups(
            &self.device,
            &self.render_group_layout,
            &self.radiance_samples,
            &self.uniform_buffer,
        );
        let (output_texture, output_view) = create_output_texture(
            &self.device,
            render_width,
            render_height,
            self.surface_format,
        );
        self.output_texture = output_texture;
        self.output_view = output_view;
        self.upscale_bind_group = create_upscale_bind_group(
            &self.device,
            &self.upscale_bind_group_layout,
            &self.output_view,
            &self.upscale_sampler,
            &self.post_uniform_buffer,
        );
    }

    pub fn save_screenshot(&self, path: &Path) -> anyhow::Result<()> {
        let width = self.render_width;
        let height = self.render_height;
        let bytes_per_pixel = 4u32;
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let padded_bytes_per_row =
            align_to(unpadded_bytes_per_row, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let output_buffer_size = padded_bytes_per_row as u64 * height as u64;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot readback"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("screenshot copy"),
                });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = output_buffer.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;

        let data = slice.get_mapped_range();
        let mut pixels = vec![0u8; (width * height * bytes_per_pixel) as usize];
        for y in 0..height as usize {
            let src_start = y * padded_bytes_per_row as usize;
            let src_end = src_start + unpadded_bytes_per_row as usize;
            let dst_start = y * unpadded_bytes_per_row as usize;
            pixels[dst_start..dst_start + unpadded_bytes_per_row as usize]
                .copy_from_slice(&data[src_start..src_end]);
        }
        drop(data);
        output_buffer.unmap();

        if self.surface_format == wgpu::TextureFormat::Bgra8Unorm {
            for px in pixels.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }

        image::save_buffer_with_format(
            path,
            &pixels,
            width,
            height,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )?;
        Ok(())
    }

    pub fn encode_frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        camera: &Camera,
        scene_resources: &wgpu::BindGroup,
        target: &wgpu::TextureView,
    ) {
        self.uniforms.camera = *camera.uniforms();
        self.uniforms.frame_count += 1;
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
        self.queue.write_buffer(
            &self.post_uniform_buffer,
            0,
            bytemuck::bytes_of(&self.post_uniforms),
        );

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("path tracer render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.output_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(
            0,
            &self.render_bind_groups[(self.uniforms.frame_count % 2) as usize],
            &[],
        );
        render_pass.set_bind_group(1, scene_resources, &[]);
        render_pass.draw(0..6, 0..1);
        drop(render_pass);

        let mut upscale_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("upscale render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        upscale_pass.set_pipeline(&self.upscale_pipeline);
        upscale_pass.set_bind_group(0, &self.upscale_bind_group, &[]);
        upscale_pass.draw(0..6, 0..1);
    }
}

fn compile_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    use std::borrow::Cow;

    let code = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders.wgsl"));
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(code)),
    })
}

fn compile_upscale_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    use std::borrow::Cow;

    let code = r#"
@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

struct PostUniforms {
  width: u32,
  height: u32,
  strength: f32,
  radius: f32,
  edge_threshold: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(2) var<uniform> post: PostUniforms;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

alias TriangleVertices = array<vec2f, 6>;
var<private> vertices: TriangleVertices = TriangleVertices(
  vec2f(-1.0,  1.0),
  vec2f(-1.0, -1.0),
  vec2f( 1.0,  1.0),
  vec2f( 1.0,  1.0),
  vec2f(-1.0, -1.0),
  vec2f( 1.0, -1.0),
);

@vertex fn upscale_vs(@builtin(vertex_index) vid: u32) -> VertexOut {
  let pos = vertices[vid];
  var out: VertexOut;
  out.position = vec4f(pos, 0.0, 1.0);
  out.uv = pos * vec2f(0.5, -0.5) + vec2f(0.5);
  return out;
}

fn luminance(color: vec3f) -> f32 {
  return dot(color, vec3f(0.2126, 0.7152, 0.0722));
}

fn neighbor_weight(center: vec3f, sample_color: vec3f, base_weight: f32) -> f32 {
  let luma_delta = abs(luminance(center) - luminance(sample_color));
  let edge = exp(-luma_delta / max(post.edge_threshold, 0.001));
  return base_weight * edge;
}

fn denoise(uv: vec2f) -> vec3f {
  let center = textureSample(source_texture, source_sampler, uv).rgb;
  if post.strength <= 0.001 {
    return center;
  }

  let texel = vec2f(1.0 / f32(post.width), 1.0 / f32(post.height)) * post.radius;
  var sum = center;
  var total = 1.0;

  let c0 = textureSample(source_texture, source_sampler, uv + vec2f( texel.x, 0.0)).rgb;
  let c1 = textureSample(source_texture, source_sampler, uv + vec2f(-texel.x, 0.0)).rgb;
  let c2 = textureSample(source_texture, source_sampler, uv + vec2f(0.0,  texel.y)).rgb;
  let c3 = textureSample(source_texture, source_sampler, uv + vec2f(0.0, -texel.y)).rgb;
  let c4 = textureSample(source_texture, source_sampler, uv + vec2f( texel.x,  texel.y)).rgb;
  let c5 = textureSample(source_texture, source_sampler, uv + vec2f(-texel.x,  texel.y)).rgb;
  let c6 = textureSample(source_texture, source_sampler, uv + vec2f( texel.x, -texel.y)).rgb;
  let c7 = textureSample(source_texture, source_sampler, uv + vec2f(-texel.x, -texel.y)).rgb;

  let w0 = neighbor_weight(center, c0, 0.70);
  let w1 = neighbor_weight(center, c1, 0.70);
  let w2 = neighbor_weight(center, c2, 0.70);
  let w3 = neighbor_weight(center, c3, 0.70);
  let w4 = neighbor_weight(center, c4, 0.38);
  let w5 = neighbor_weight(center, c5, 0.38);
  let w6 = neighbor_weight(center, c6, 0.38);
  let w7 = neighbor_weight(center, c7, 0.38);

  sum += c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3;
  sum += c4 * w4 + c5 * w5 + c6 * w6 + c7 * w7;
  total += w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7;

  return mix(center, sum / total, post.strength);
}

@fragment fn upscale_fs(in: VertexOut) -> @location(0) vec4f {
  return vec4f(denoise(in.uv), 1.0);
}
"#;
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("upscale shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(code)),
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    shader_module: &wgpu::ShaderModule,
    surface_format: wgpu::TextureFormat,
) -> (
    wgpu::RenderPipeline,
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
) {
    let render_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
    // Scene resources: materials(0), spheres(1), and the uniform-grid
    // accelerator — header(2), cell ranges(3), sphere indices(4). All read-only
    // storage in the fragment shader.
    let storage_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let scene_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene resource layout"),
            entries: &[
                storage_entry(0),
                storage_entry(1),
                storage_entry(2),
                storage_entry(3),
                storage_entry(4),
            ],
        });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("path tracer"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[Some(&render_group_layout), Some(&scene_group_layout)],
                ..Default::default()
            }),
        ),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            polygon_mode: wgpu::PolygonMode::Fill,
            ..Default::default()
        },
        vertex: wgpu::VertexState {
            module: shader_module,
            entry_point: Some("path_tracer_vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader_module,
            entry_point: Some("path_tracer_fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, render_group_layout, scene_group_layout)
}

fn create_sample_textures(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> [wgpu::Texture; 2] {
    let desc = wgpu::TextureDescriptor {
        label: Some("radiance samples"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    };
    // Create two textures with the same parameters.
    [device.create_texture(&desc), device.create_texture(&desc)]
}

fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment) * alignment
}

fn create_output_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scaled path tracer output"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn create_upscale_pipeline(
    device: &wgpu::Device,
    shader_module: &wgpu::ShaderModule,
    surface_format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
    let bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("upscale bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("upscale pipeline"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[Some(&bind_group_layout)],
                ..Default::default()
            }),
        ),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        vertex: wgpu::VertexState {
            module: shader_module,
            entry_point: Some("upscale_vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader_module,
            entry_point: Some("upscale_fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    (pipeline, bind_group_layout)
}

fn create_upscale_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    post_uniform_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("upscale bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: post_uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    })
}

fn create_render_bind_groups(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    textures: &[wgpu::Texture; 2],
    uniform_buffer: &wgpu::Buffer,
) -> [wgpu::BindGroup; 2] {
    let views = [
        textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
        textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
    ];
    [
        // Bind group with view[0] assigned to binding 1 and view[1] assigned to binding 2.
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&views[1]),
                },
            ],
        }),
        // Bind group with view[1] assigned to binding 1 and view[0] assigned to binding 2.
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&views[0]),
                },
            ],
        }),
    ]
}
