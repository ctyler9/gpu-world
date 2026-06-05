// Originally written in 2025 by Arman Uguray <arman.uguray@gmail.com>
// SPDX-License-Identifier: CC-BY-4.0

use bytemuck::{Pod, Zeroable};

use crate::algebra::Vec3;

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct Material {
    color: Vec3,
    metallic_or_ior: f32,
    emission: Vec3,
    _pad: f32,
}

impl Material {
    pub fn lambertian(color: Vec3) -> Self {
        Self {
            color,
            metallic_or_ior: 0.,
            emission: Vec3::zero(),
            _pad: 0.,
        }
    }

    pub fn opaque(color: Vec3, metallic: f32) -> Self {
        Self {
            color,
            metallic_or_ior: metallic.max(0.00001),
            emission: Vec3::zero(),
            _pad: 0.,
        }
    }

    pub fn transparent_dielectric(color: Vec3, ior: f32) -> Self {
        Self {
            color,
            metallic_or_ior: -ior.abs(),
            emission: Vec3::zero(),
            _pad: 0.,
        }
    }

    pub fn metal(color: Vec3) -> Self {
        Self::opaque(color, 1.)
    }

    /// A surface that emits light of `color * strength`. Its `color` is black so
    /// it absorbs incoming light (acting as a clean area light), letting the
    /// `emission` term be the only thing it contributes.
    pub fn emissive(color: Vec3, strength: f32) -> Self {
        Self {
            color: Vec3::zero(),
            metallic_or_ior: 0.,
            emission: color * strength,
            _pad: 0.,
        }
    }
}

// Must match the `Material` struct stride in shaders.wgsl (vec3f aligns to 16).
const _: () = assert!(std::mem::size_of::<Material>() == 32);

pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct SphereBufferEntry {
    center: Vec3,
    radius: f32,
    material_index: u32,
    _pad: [u32; 3],
}

#[derive(Copy, Clone)]
pub struct MaterialId(u32);

impl MaterialId {
    /// Construct a handle for a material at a known buffer index. Used by the
    /// streaming layer, which pushes a fixed palette and references it by index.
    pub fn from_index(index: u32) -> Self {
        MaterialId(index)
    }
}

#[derive(Default)]
pub struct SceneBuilder {
    materials: Vec<Material>,
    spheres: Vec<SphereBufferEntry>,
}

impl SceneBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_material(&mut self, material: Material) -> MaterialId {
        self.materials.push(material);
        MaterialId((self.materials.len() - 1).try_into().unwrap())
    }

    /// Append a whole palette at once. The materials land at indices
    /// `[0, palette.len())` when this is the first thing added, so they line up
    /// with [`MaterialId::from_index`] used by procedural chunks.
    pub fn add_palette(&mut self, palette: &[Material]) {
        self.materials.extend_from_slice(palette);
    }

    pub fn add_sphere(&mut self, sphere: Sphere, material: MaterialId) {
        let entry = SphereBufferEntry {
            center: sphere.center,
            radius: sphere.radius,
            material_index: material.0,
            _pad: [0; 3],
        };
        self.spheres.push(entry)
    }

    /// Fluent, chainable sphere placement. Accepts a tuple/array position:
    /// `s.sphere((0., 0.5, 0.), 0.5, mat)`.
    pub fn sphere(
        &mut self,
        center: impl Into<Vec3>,
        radius: f32,
        material: MaterialId,
    ) -> &mut Self {
        self.add_sphere(
            Sphere {
                center: center.into(),
                radius,
            },
            material,
        );
        self
    }

    /// A large sphere standing in for an infinite ground plane at y = 0.
    pub fn ground(&mut self, material: MaterialId) -> &mut Self {
        self.sphere((0., -200.001, 0.), 200., material)
    }

    /// Place objects on a `cols` x `rows` grid in the XZ plane, centered on
    /// `origin`. The closure receives the builder, the `(col, row)` index, and
    /// the world-space position of that cell, so it can place one or many
    /// spheres (e.g. a cluster) and vary materials by index.
    pub fn grid(
        &mut self,
        origin: impl Into<Vec3>,
        cols: usize,
        rows: usize,
        spacing: f32,
        mut place: impl FnMut(&mut Self, (usize, usize), Vec3),
    ) -> &mut Self {
        let origin = origin.into();
        for j in 0..rows {
            for i in 0..cols {
                let x = (i as f32 - (cols.saturating_sub(1)) as f32 * 0.5) * spacing;
                let z = (j as f32 - (rows.saturating_sub(1)) as f32 * 0.5) * spacing;
                let pos = origin + Vec3::new(x, 0., z);
                place(self, (i, j), pos);
            }
        }
        self
    }

    /// Place `count` objects evenly around a circle in the XZ plane. The closure
    /// receives the builder, the index, and the world-space position.
    pub fn ring(
        &mut self,
        center: impl Into<Vec3>,
        radius: f32,
        count: usize,
        mut place: impl FnMut(&mut Self, usize, Vec3),
    ) -> &mut Self {
        let center = center.into();
        for i in 0..count {
            let angle = i as f32 / count as f32 * std::f32::consts::TAU;
            let pos = center + Vec3::new(angle.cos() * radius, 0., angle.sin() * radius);
            place(self, i, pos);
        }
        self
    }

    /// Build the scene bind group with a degenerate single-cell acceleration
    /// grid: every sphere lives in one cell, so the shader's grid traversal
    /// reduces to a brute-force loop. Used by static scenes, whose giant
    /// ground/void spheres would otherwise span a huge grid.
    pub fn build(
        self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        let n = self.spheres.len() as u32;
        let header = GridHeader::single_cell();
        let cell_ranges = [0u32, n]; // one cell: start 0, count n
        let indices: Vec<u32> = (0..n).collect();
        self.build_internal(device, layout, header, &cell_ranges, &indices)
    }

    /// Build the scene bind group with a precomputed uniform grid (see
    /// `world.rs`). `cell_ranges` is `(start, count)` pairs per cell and
    /// `sphere_indices` lists sphere indices grouped by cell — both referencing
    /// the spheres in the order they were added to this builder.
    pub fn build_with_grid(
        self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        header: GridHeader,
        cell_ranges: &[u32],
        sphere_indices: &[u32],
    ) -> wgpu::BindGroup {
        self.build_internal(device, layout, header, cell_ranges, sphere_indices)
    }

    fn build_internal(
        self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        header: GridHeader,
        cell_ranges: &[u32],
        sphere_indices: &[u32],
    ) -> wgpu::BindGroup {
        let material_buffer =
            create_storage_buffer_with_data(device, &self.materials, Some("materials"));
        let spheres_buffer =
            create_storage_buffer_with_data(device, &self.spheres, Some("spheres"));
        let header_buffer = create_storage_buffer_with_data(
            device,
            std::slice::from_ref(&header),
            Some("grid"),
        );
        let ranges_buffer =
            create_storage_buffer_with_data(device, cell_ranges, Some("cell ranges"));
        let indices_buffer =
            create_storage_buffer_with_data(device, sphere_indices, Some("sphere indices"));

        let entry = |binding, buffer| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: None,
            }),
        };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene resources"),
            layout,
            entries: &[
                entry(0, &material_buffer),
                entry(1, &spheres_buffer),
                entry(2, &header_buffer),
                entry(3, &ranges_buffer),
                entry(4, &indices_buffer),
            ],
        })
    }
}

/// Header for the uniform-grid acceleration structure plus per-scene background,
/// mirrored in `shaders.wgsl`. A single-cell grid (`nx == nz == 1`) means brute
/// force. `sky_amount` blends the procedural sky (1.0) with the solid `bg`
/// color (0.0) for rays that escape the scene.
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct GridHeader {
    min_x: f32,
    min_z: f32,
    inv_cell: f32,
    nx: u32,
    nz: u32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
    sky_amount: f32,
    _pad: [u32; 3],
}

// Must match the `GridHeader` struct in shaders.wgsl.
const _: () = assert!(std::mem::size_of::<GridHeader>() == 48);

impl GridHeader {
    pub fn new(min_x: f32, min_z: f32, inv_cell: f32, nx: u32, nz: u32) -> Self {
        Self {
            min_x,
            min_z,
            inv_cell,
            nx,
            nz,
            bg_r: 0.0,
            bg_g: 0.0,
            bg_b: 0.0,
            sky_amount: 1.0, // default: full procedural sky (existing behavior)
            _pad: [0; 3],
        }
    }

    /// Override the background: `sky_amount` of 1.0 keeps the procedural sky,
    /// 0.0 uses the solid `color` (e.g. black for a night world).
    pub fn with_background(mut self, color: Vec3, sky_amount: f32) -> Self {
        self.bg_r = color.x();
        self.bg_g = color.y();
        self.bg_b = color.z();
        self.sky_amount = sky_amount;
        self
    }

    fn single_cell() -> Self {
        Self::new(0.0, 0.0, 0.0, 1, 1)
    }
}

fn create_storage_buffer_with_data<T: Pod>(
    device: &wgpu::Device,
    data: &[T],
    label: Option<&str>,
) -> wgpu::Buffer {
    // wgpu rejects zero-sized buffers, so always allocate room for at least one
    // element even when the data is empty.
    let elem = std::mem::size_of::<T>();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label,
        size: (elem * data.len().max(1)) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: true,
    });
    if !data.is_empty() {
        let mut view = buffer.slice(..).get_mapped_range_mut();
        view.copy_from_slice(&bytemuck::cast_slice(data)[..std::mem::size_of_val(data)]);
    }
    buffer.unmap();
    buffer
}
