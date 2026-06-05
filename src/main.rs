// Originally written in 2023 by Arman Uguray <arman.uguray@gmail.com>
// SPDX-License-Identifier: CC-BY-4.0

use {
    anyhow::{Context, Result},
    winit::{
        event::{DeviceEvent, ElementState, Event, MouseScrollDelta, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::{Window, WindowBuilder},
    },
};

mod algebra;
mod camera;
mod gallery;
mod render;
mod scene;

const WIDTH: u32 = 1600;
const HEIGHT: u32 = 1200;

#[pollster::main]
async fn main() -> Result<()> {
    let event_loop = EventLoop::new()?;
    let window_size = winit::dpi::PhysicalSize::new(WIDTH, HEIGHT);
    let window = WindowBuilder::new()
        .with_inner_size(window_size)
        .with_resizable(false)
        .with_title("GPU Path Tracer".to_string())
        .build(&event_loop)?;

    let (device, queue, surface, surface_format) = connect_to_gpu(&window).await?;
    let mut renderer = render::PathTracer::new(device, queue, WIDTH, HEIGHT, surface_format);
    let mut gallery =
        gallery::Gallery::new(renderer.device(), renderer.scene_group_layout());

    let mut left_mouse_button_pressed = false;
    let mut right_mouse_button_pressed = false;
    let mut key_w = false;
    let mut key_s = false;
    let mut key_a = false;
    let mut key_d = false;
    let mut path_playback: Option<std::time::Instant> = None;

    event_loop.run(|event, control_handle| {
        control_handle.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => control_handle.exit(),
                WindowEvent::KeyboardInput {
                    device_id: _,
                    event,
                    ..
                } => {
                    use winit::keyboard::{KeyCode, PhysicalKey};

                    let pressed = event.state == ElementState::Pressed;
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::KeyW) => key_w = pressed,
                        PhysicalKey::Code(KeyCode::KeyS) => key_s = pressed,
                        PhysicalKey::Code(KeyCode::KeyA) => key_a = pressed,
                        PhysicalKey::Code(KeyCode::KeyD) => key_d = pressed,
                        _ => {
                            if !pressed {
                                return;
                            }
                            match event.physical_key {
                                PhysicalKey::Code(KeyCode::ArrowUp) => {
                                    gallery
                                        .current_camera_mut()
                                        .adjust_fov(1_f32.to_radians());
                                    renderer.reset_samples();
                                }
                                PhysicalKey::Code(KeyCode::ArrowDown) => {
                                    gallery
                                        .current_camera_mut()
                                        .adjust_fov(-1_f32.to_radians());
                                    renderer.reset_samples();
                                }
                                PhysicalKey::Code(KeyCode::ArrowLeft) => {
                                    gallery.select_previous();
                                    path_playback = None;
                                    renderer.reset_samples();
                                }
                                PhysicalKey::Code(KeyCode::ArrowRight) => {
                                    gallery.select_next();
                                    path_playback = None;
                                    renderer.reset_samples();
                                }
                                PhysicalKey::Code(KeyCode::KeyP) => {
                                    if path_playback.is_some() {
                                        path_playback = None;
                                    } else if gallery.current_path().is_some() {
                                        path_playback = Some(std::time::Instant::now());
                                    }
                                }
                                _ => (),
                            }
                        }
                    }
                }
                WindowEvent::MouseInput {
                    device_id: _,
                    state,
                    button,
                } => {
                    use winit::event::MouseButton;

                    let pressed = state == ElementState::Pressed;
                    match button {
                        MouseButton::Left => left_mouse_button_pressed = pressed,
                        MouseButton::Right => right_mouse_button_pressed = pressed,
                        _ => (),
                    }
                }
                WindowEvent::RedrawRequested => {
                    if let Some(start) = path_playback {
                        let elapsed = start.elapsed().as_secs_f32();
                        if let Some(cam) =
                            gallery.current_path().map(|p| p.camera_at(elapsed))
                        {
                            *gallery.current_camera_mut() = cam;
                            renderer.reset_samples();
                        }
                    }

                    const MOVE_SPEED: f32 = 0.05;
                    const STRAFE_SPEED: f32 = 0.05;
                    let camera = gallery.current_camera_mut();
                    let mut moved = false;
                    if key_w {
                        camera.fly(-MOVE_SPEED);
                        moved = true;
                    }
                    if key_s {
                        camera.fly(MOVE_SPEED);
                        moved = true;
                    }
                    if key_a {
                        camera.pan(-STRAFE_SPEED, 0.0);
                        moved = true;
                    }
                    if key_d {
                        camera.pan(STRAFE_SPEED, 0.0);
                        moved = true;
                    }
                    if moved {
                        renderer.reset_samples();
                    }

                    let frame: wgpu::SurfaceTexture = surface
                        .get_current_texture()
                        .expect("failed to get current texture");

                    let render_target = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    let scene = gallery.current_scene();
                    renderer.render_frame(&scene.camera, &scene.resources, &render_target);

                    frame.present();
                    window.request_redraw();
                }
                _ => (),
            },
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseWheel { delta } => {
                    let delta = match delta {
                        MouseScrollDelta::PixelDelta(delta) => 0.001 * delta.y as f32,
                        MouseScrollDelta::LineDelta(_, y) => y * 0.1,
                    };
                    gallery.current_camera_mut().zoom(delta);
                    renderer.reset_samples();
                }
                DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                    let dx = dx as f32 * 0.01;
                    let dy = dy as f32 * -0.01;
                    if left_mouse_button_pressed {
                        gallery.current_camera_mut().orbit(dx, dy);
                        renderer.reset_samples();
                    }
                    if right_mouse_button_pressed {
                        gallery.current_camera_mut().pan(dx, dy);
                        renderer.reset_samples();
                    }
                }
                _ => (),
            },
            _ => (),
        }
    })?;
    Ok(())
}

async fn connect_to_gpu(
    window: &Window,
) -> Result<(
    wgpu::Device,
    wgpu::Queue,
    wgpu::Surface<'_>,
    wgpu::TextureFormat,
)> {
    use wgpu::TextureFormat::{Bgra8Unorm, Rgba8Unorm};

    // Create an "instance" of wgpu. This is the entry-point to the API.
    let instance = wgpu::Instance::default();

    // Create a drawable "surface" that is associated with the window.
    let surface = instance.create_surface(window)?;

    // Request a GPU that is compatible with the surface. If the system has multiple GPUs then
    // pick the high performance one.
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .context("failed to find a compatible adapter")?;

    // Connect to the GPU. "device" represents the connection to the GPU and allows us to create
    // resources like buffers, textures, and pipelines. "queue" represents the command queue that
    // we use to submit commands to the GPU.
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .context("failed to connect to the GPU")?;

    // Configure the texture memory backing the surface. Our renderer will draw to a surface
    // texture every frame.
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .into_iter()
        .find(|it| matches!(it, Rgba8Unorm | Bgra8Unorm))
        .context("could not find preferred texture format (Rgba8Unorm or Bgra8Unorm)")?;
    let size = window.inner_size();
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 3,
    };
    surface.configure(&device, &config);

    Ok((device, queue, surface, format))
}
