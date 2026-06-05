// Originally written in 2026 by Clay Tyler
// SPDX-License-Identifier: CC-BY-4.0

//! Endless procedural world: streams chunks of geometry in and out around the
//! camera. Geometry comes from [`crate::procgen`]; materials come from a single
//! shared per-world palette (referenced by index), so combining many chunks
//! never grows the material set.
//!
//! The GPU scene buffers are rebuilt only when the camera crosses a chunk
//! boundary (see [`World::update`]); while the camera sits inside one chunk the
//! buffers are untouched, letting the path tracer keep accumulating samples.

use std::collections::HashMap;

use crate::{
    algebra::Vec3,
    camera::Camera,
    procgen::{self, ChunkSphere, Style},
    scene::{GridHeader, Material, MaterialId, SceneBuilder, Sphere},
};

type ChunkCoord = (i32, i32);

/// How finely each chunk is subdivided for the acceleration grid (cells per
/// chunk edge). Aim for a few spheres per occupied cell.
const CELLS_PER_CHUNK: u32 = 2;

/// Radius of the optional ground sphere. Large so its top reads as a flat plane
/// at y = 0 across the visible region.
const GROUND_RADIUS: f32 = 4000.0;

/// Extra cached chunks beyond the rendered `view_radius`, kept warm so boundary
/// crossings don't trigger a synchronous generation spike.
const PREFETCH_MARGIN: i32 = 2;

/// Max chunks generated per `update` call. Spreads generation across frames;
/// must exceed the rate at which the camera reaches new chunks.
const PREFETCH_BUDGET: usize = 2;

pub struct WorldConfig {
    pub world_seed: u64,
    pub chunk_size: f32,
    /// Chebyshev radius, in chunks, of the loaded region around the camera.
    pub view_radius: i32,
    pub style: Style,
    /// Shared material palette; chunk `palette_index` values index into this.
    pub palette: Vec<Material>,
    /// Safety cap on total spheres assembled (brute-force budget guard).
    pub max_spheres: usize,
    /// Optional ground: a huge flat sphere using this palette index, if set.
    pub ground: Option<u32>,
    /// Solid background color used for escaped rays when `sky_amount` < 1.
    pub background: Vec3,
    /// Blends the procedural sky (1.0) with `background` (0.0). 1.0 by default.
    pub sky_amount: f32,
}

pub struct World {
    cfg: WorldConfig,
    camera: Camera,
    loaded: HashMap<ChunkCoord, Vec<ChunkSphere>>,
    center: Option<ChunkCoord>,
    resources: wgpu::BindGroup,
}

impl World {
    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        cfg: WorldConfig,
        camera: Camera,
    ) -> Self {
        let center = chunk_coord(camera.position(), cfg.chunk_size);
        let mut loaded = HashMap::new();
        // Ensure the rendered set exists so the first frame is complete; the
        // outer prefetch margin then warms up incrementally over later frames.
        ensure_chunks(&cfg, &mut loaded, center, cfg.view_radius);
        let resources = assemble(&cfg, &loaded, center, device, layout);
        World {
            cfg,
            camera,
            loaded,
            center: Some(center),
            resources,
        }
    }

    /// Called every frame. Incrementally prefetches chunks ahead of the camera
    /// (a small budget per call) so crossing a chunk boundary rarely triggers a
    /// synchronous generation spike. Returns `true` (and rebuilds the GPU
    /// buffers) only when the camera has crossed into a new chunk; the caller
    /// resets sample accumulation iff this returns `true`.
    pub fn update(&mut self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> bool {
        let center = chunk_coord(self.camera.position(), self.cfg.chunk_size);

        // Spread chunk generation across frames: top up the cache (keep radius)
        // a few chunks at a time, closest-first, and drop far ones. Slow camera
        // motion means chunks ahead are usually ready before we reach them.
        prefetch(&self.cfg, &mut self.loaded, center, PREFETCH_BUDGET);

        if self.center == Some(center) {
            return false;
        }

        // Crossed a boundary: guarantee the rendered set is complete (cheap if
        // prefetch kept up — typically nothing to generate), then rebuild.
        ensure_chunks(&self.cfg, &mut self.loaded, center, self.cfg.view_radius);
        self.resources = assemble(&self.cfg, &self.loaded, center, device, layout);
        self.center = Some(center);
        true
    }

    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    pub fn resources(&self) -> &wgpu::BindGroup {
        &self.resources
    }
}

/// World position -> chunk coordinate (floor division on the XZ plane).
fn chunk_coord(pos: Vec3, chunk_size: f32) -> ChunkCoord {
    (
        (pos.x() / chunk_size).floor() as i32,
        (pos.z() / chunk_size).floor() as i32,
    )
}

/// Radius (in chunks) of the cache, larger than what is rendered so chunks just
/// outside view are generated before the camera reaches them.
fn keep_radius(cfg: &WorldConfig) -> i32 {
    cfg.view_radius + PREFETCH_MARGIN
}

/// Generate every chunk within `radius` of `center` that isn't already cached.
fn ensure_chunks(
    cfg: &WorldConfig,
    loaded: &mut HashMap<ChunkCoord, Vec<ChunkSphere>>,
    center: ChunkCoord,
    radius: i32,
) {
    for dz in -radius..=radius {
        for dx in -radius..=radius {
            let key = (center.0 + dx, center.1 + dz);
            loaded.entry(key).or_insert_with(|| {
                procgen::generate_chunk(
                    cfg.world_seed,
                    key.0,
                    key.1,
                    cfg.chunk_size,
                    &cfg.style,
                )
            });
        }
    }
}

/// Evict cached chunks beyond the keep radius, then generate up to `budget`
/// missing chunks within it, nearest first. Returns the number generated.
fn prefetch(
    cfg: &WorldConfig,
    loaded: &mut HashMap<ChunkCoord, Vec<ChunkSphere>>,
    center: ChunkCoord,
    budget: usize,
) -> usize {
    let r = keep_radius(cfg);
    loaded.retain(|&(cx, cz), _| (cx - center.0).abs() <= r && (cz - center.1).abs() <= r);

    // Collect missing chunks within the keep radius, nearest-ring first so the
    // soon-to-be-rendered chunks are generated before the outer margin.
    let mut missing: Vec<ChunkCoord> = Vec::new();
    for dz in -r..=r {
        for dx in -r..=r {
            let key = (center.0 + dx, center.1 + dz);
            if !loaded.contains_key(&key) {
                missing.push(key);
            }
        }
    }
    missing.sort_by_key(|&(cx, cz)| (cx - center.0).abs().max((cz - center.1).abs()));

    let mut generated = 0;
    for key in missing.into_iter().take(budget) {
        loaded.insert(
            key,
            procgen::generate_chunk(
                cfg.world_seed,
                key.0,
                key.1,
                cfg.chunk_size,
                &cfg.style,
            ),
        );
        generated += 1;
    }
    generated
}

/// Flatten the rendered chunks (within `view_radius` of `center`) into a single
/// sphere list, capped at `max_spheres`. Iterates a deterministic window so the
/// cache's outer prefetch margin is excluded from what is rendered.
fn collect_spheres(
    cfg: &WorldConfig,
    loaded: &HashMap<ChunkCoord, Vec<ChunkSphere>>,
    center: ChunkCoord,
) -> Vec<ChunkSphere> {
    let r = cfg.view_radius;
    let mut out = Vec::new();
    for dz in -r..=r {
        for dx in -r..=r {
            let key = (center.0 + dx, center.1 + dz);
            if let Some(spheres) = loaded.get(&key) {
                for s in spheres {
                    if out.len() >= cfg.max_spheres {
                        return out;
                    }
                    out.push(*s);
                }
            }
        }
    }
    out
}

/// The uniform-grid acceleration structure over a set of spheres: cells bucket
/// the spheres so a ray only tests nearby ones. Indices reference the spheres in
/// the order they were assembled into the scene.
struct Grid {
    min_x: f32,
    min_z: f32,
    cell_size: f32,
    nx: u32,
    nz: u32,
    /// `(start, count)` pairs per cell, indexing `sphere_indices`.
    cell_ranges: Vec<u32>,
    /// Sphere indices grouped by cell, concatenated in cell order.
    sphere_indices: Vec<u32>,
}

impl Grid {
    fn header(&self) -> GridHeader {
        GridHeader::new(
            self.min_x,
            self.min_z,
            1.0 / self.cell_size,
            self.nx,
            self.nz,
        )
    }
}

/// Bucket `spheres` into a uniform grid covering the loaded region. Each sphere
/// is inserted into every cell its XZ footprint (`±radius`) overlaps — required
/// so a ray passing through any of those cells will test it.
fn build_grid(spheres: &[ChunkSphere], cfg: &WorldConfig, center: ChunkCoord) -> Grid {
    let r = cfg.view_radius;
    let cell_size = cfg.chunk_size / CELLS_PER_CHUNK as f32;
    let inv_cell = 1.0 / cell_size;
    let span = (2 * r + 1).max(1) as u32; // chunks per side
    let nx = span * CELLS_PER_CHUNK;
    let nz = span * CELLS_PER_CHUNK;
    let min_x = (center.0 - r) as f32 * cfg.chunk_size;
    let min_z = (center.1 - r) as f32 * cfg.chunk_size;
    let n_cells = (nx * nz) as usize;

    // Inclusive (clamped) cell footprint of a sphere on the XZ plane.
    let footprint = |s: &ChunkSphere| -> (i32, i32, i32, i32) {
        let x0 = (((s.center.x() - s.radius) - min_x) * inv_cell).floor() as i32;
        let x1 = (((s.center.x() + s.radius) - min_x) * inv_cell).floor() as i32;
        let z0 = (((s.center.z() - s.radius) - min_z) * inv_cell).floor() as i32;
        let z1 = (((s.center.z() + s.radius) - min_z) * inv_cell).floor() as i32;
        (
            x0.clamp(0, nx as i32 - 1),
            x1.clamp(0, nx as i32 - 1),
            z0.clamp(0, nz as i32 - 1),
            z1.clamp(0, nz as i32 - 1),
        )
    };

    // Counting sort: count per cell, prefix-sum to starts, then scatter.
    let mut counts = vec![0u32; n_cells];
    for s in spheres {
        let (x0, x1, z0, z1) = footprint(s);
        for iz in z0..=z1 {
            for ix in x0..=x1 {
                counts[(iz as u32 * nx + ix as u32) as usize] += 1;
            }
        }
    }
    let mut starts = vec![0u32; n_cells];
    let mut acc = 0u32;
    for c in 0..n_cells {
        starts[c] = acc;
        acc += counts[c];
    }
    let mut cursor = starts.clone();
    let mut sphere_indices = vec![0u32; acc as usize];
    for (si, s) in spheres.iter().enumerate() {
        let (x0, x1, z0, z1) = footprint(s);
        for iz in z0..=z1 {
            for ix in x0..=x1 {
                let c = (iz as u32 * nx + ix as u32) as usize;
                sphere_indices[cursor[c] as usize] = si as u32;
                cursor[c] += 1;
            }
        }
    }
    let mut cell_ranges = vec![0u32; n_cells * 2];
    for c in 0..n_cells {
        cell_ranges[c * 2] = starts[c];
        cell_ranges[c * 2 + 1] = counts[c];
    }

    Grid {
        min_x,
        min_z,
        cell_size,
        nx,
        nz,
        cell_ranges,
        sphere_indices,
    }
}

/// Build the GPU scene bind group: palette first (so `palette_index` lines up
/// with the material buffer index), then every loaded sphere, plus the uniform
/// grid built over that same sphere list (matching index order).
fn assemble(
    cfg: &WorldConfig,
    loaded: &HashMap<ChunkCoord, Vec<ChunkSphere>>,
    center: ChunkCoord,
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    let mut spheres = collect_spheres(cfg, loaded, center);

    // Optional ground: one huge sphere centered under the loaded region, its
    // top at y = 0. It spans every grid cell (cheaply) so all rays test it.
    if let Some(ground_index) = cfg.ground {
        let gx = (center.0 as f32 + 0.5) * cfg.chunk_size;
        let gz = (center.1 as f32 + 0.5) * cfg.chunk_size;
        spheres.push(ChunkSphere {
            center: Vec3::new(gx, -GROUND_RADIUS, gz),
            radius: GROUND_RADIUS,
            palette_index: ground_index,
        });
    }

    let mut builder = SceneBuilder::new();
    builder.add_palette(&cfg.palette);
    for s in &spheres {
        builder.add_sphere(
            Sphere {
                center: s.center,
                radius: s.radius,
            },
            MaterialId::from_index(s.palette_index),
        );
    }
    let grid = build_grid(&spheres, cfg, center);
    let header = grid
        .header()
        .with_background(cfg.background, cfg.sky_amount);
    builder.build_with_grid(
        device,
        layout,
        header,
        &grid.cell_ranges,
        &grid.sphere_indices,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::Material;

    fn test_cfg(view_radius: i32, max_spheres: usize) -> WorldConfig {
        WorldConfig {
            world_seed: 1,
            chunk_size: 8.0,
            view_radius,
            style: Style::Scatter {
                density: 5,
                min_r: 0.2,
                max_r: 0.5,
                y_jitter: 0.1,
                mats: vec![0, 1],
            },
            palette: vec![
                Material::lambertian(Vec3::new(0.8, 0.2, 0.2)),
                Material::metal(Vec3::new(0.8, 0.8, 0.9)),
            ],
            max_spheres,
            ground: None,
            background: Vec3::zero(),
            sky_amount: 1.0,
        }
    }

    #[test]
    fn chunk_coord_floors_on_xz() {
        let cs = 4.0;
        assert_eq!(chunk_coord(Vec3::new(0.0, 99.0, 0.0), cs), (0, 0));
        assert_eq!(chunk_coord(Vec3::new(3.9, 0.0, -0.1), cs), (0, -1));
        assert_eq!(chunk_coord(Vec3::new(-0.1, 0.0, 4.0), cs), (-1, 1));
        assert_eq!(chunk_coord(Vec3::new(8.0, 0.0, 8.0), cs), (2, 2));
    }

    #[test]
    fn ensure_chunks_fills_the_window() {
        let cfg = test_cfg(1, usize::MAX);
        let mut loaded = HashMap::new();
        ensure_chunks(&cfg, &mut loaded, (0, 0), cfg.view_radius);
        assert_eq!(loaded.len(), 9, "3x3 window for radius 1");
        for cx in -1..=1 {
            for cz in -1..=1 {
                assert!(loaded.contains_key(&(cx, cz)));
            }
        }
    }

    #[test]
    fn prefetch_respects_budget_and_fills_keep_radius() {
        let cfg = test_cfg(1, usize::MAX); // keep_radius = 1 + PREFETCH_MARGIN
        let keep = keep_radius(&cfg);
        let total = ((2 * keep + 1) * (2 * keep + 1)) as usize;

        // Each call generates at most the budget until the keep window is full.
        let mut loaded = HashMap::new();
        let mut calls = 0;
        loop {
            let made = prefetch(&cfg, &mut loaded, (0, 0), PREFETCH_BUDGET);
            assert!(made <= PREFETCH_BUDGET, "budget honored");
            calls += 1;
            if loaded.len() == total {
                break;
            }
            assert!(calls < 10_000, "should converge");
        }
        // Once full, further prefetch generates nothing.
        assert_eq!(prefetch(&cfg, &mut loaded, (0, 0), PREFETCH_BUDGET), 0);

        // Re-centering evicts everything now beyond the keep radius of the new
        // center and keeps the window size constant once refilled.
        while prefetch(&cfg, &mut loaded, (100, 100), PREFETCH_BUDGET) > 0 {}
        assert_eq!(loaded.len(), total, "window size constant after a big jump");
        for (&(cx, cz), _) in loaded.iter() {
            assert!((cx - 100).abs() <= keep && (cz - 100).abs() <= keep);
        }
    }

    #[test]
    fn collect_renders_only_view_radius_not_prefetch_margin() {
        let cfg = test_cfg(1, usize::MAX);
        let mut loaded = HashMap::new();
        // Fill the whole keep radius (view + margin).
        ensure_chunks(&cfg, &mut loaded, (0, 0), keep_radius(&cfg));
        // But collect only the 3x3 view window: 9 chunks * density 5 = 45.
        assert_eq!(collect_spheres(&cfg, &loaded, (0, 0)).len(), 45);
    }

    #[test]
    fn collect_respects_cap_and_counts() {
        let cfg = test_cfg(1, usize::MAX);
        let mut loaded = HashMap::new();
        ensure_chunks(&cfg, &mut loaded, (0, 0), cfg.view_radius);
        assert_eq!(collect_spheres(&cfg, &loaded, (0, 0)).len(), 45);

        let capped = test_cfg(1, 20);
        assert_eq!(
            collect_spheres(&capped, &loaded, (0, 0)).len(),
            20,
            "cap honored"
        );
    }

    #[test]
    fn all_palette_indices_are_valid() {
        let cfg = test_cfg(2, usize::MAX);
        let mut loaded = HashMap::new();
        ensure_chunks(&cfg, &mut loaded, (3, -4), cfg.view_radius);
        let palette_len = cfg.palette.len() as u32;
        for s in collect_spheres(&cfg, &loaded, (3, -4)) {
            assert!(s.palette_index < palette_len);
        }
    }

    // --- Uniform-grid acceleration tests ---

    #[test]
    fn grid_membership_matches_footprint() {
        let cfg = test_cfg(1, usize::MAX);
        let mut loaded = HashMap::new();
        ensure_chunks(&cfg, &mut loaded, (0, 0), cfg.view_radius);
        let spheres = collect_spheres(&cfg, &loaded, (0, 0));
        let grid = build_grid(&spheres, &cfg, (0, 0));

        // Internal consistency: indices length == sum of cell counts.
        let total: u32 = (0..grid.nx * grid.nz)
            .map(|c| grid.cell_ranges[c as usize * 2 + 1])
            .sum();
        assert_eq!(total as usize, grid.sphere_indices.len());

        // Reconstruct membership from the grid and compare to an independent
        // footprint computation for every sphere.
        let inv_cell = 1.0 / grid.cell_size;
        for (si, s) in spheres.iter().enumerate() {
            let x0 = ((((s.center.x() - s.radius) - grid.min_x) * inv_cell).floor() as i32)
                .clamp(0, grid.nx as i32 - 1);
            let x1 = ((((s.center.x() + s.radius) - grid.min_x) * inv_cell).floor() as i32)
                .clamp(0, grid.nx as i32 - 1);
            let z0 = ((((s.center.z() - s.radius) - grid.min_z) * inv_cell).floor() as i32)
                .clamp(0, grid.nz as i32 - 1);
            let z1 = ((((s.center.z() + s.radius) - grid.min_z) * inv_cell).floor() as i32)
                .clamp(0, grid.nz as i32 - 1);
            for iz in z0..=z1 {
                for ix in x0..=x1 {
                    let c = (iz as u32 * grid.nx + ix as u32) as usize;
                    let start = grid.cell_ranges[c * 2] as usize;
                    let count = grid.cell_ranges[c * 2 + 1] as usize;
                    let members = &grid.sphere_indices[start..start + count];
                    assert!(
                        members.contains(&(si as u32)),
                        "sphere {si} missing from cell ({ix},{iz}) it overlaps"
                    );
                }
            }
        }
    }

    // --- A Rust port of the shader's intersect + DDA, to validate that the
    // grid traversal never misses the true closest hit (the early-out bug
    // class) without needing a GPU. The brute-force and grid closest-hit must
    // agree for every ray.

    const EPSILON: f32 = 1e-3;

    fn dot(a: Vec3, b: Vec3) -> f32 {
        a.x() * b.x() + a.y() * b.y() + a.z() * b.z()
    }

    fn hit_sphere(origin: Vec3, dir: Vec3, s: &ChunkSphere) -> f32 {
        let v = origin - s.center;
        let a = dot(dir, dir);
        let b = dot(v, dir);
        let c = dot(v, v) - s.radius * s.radius;
        let disc = b * b - a * c;
        if disc < 0.0 {
            return -1.0;
        }
        let sq = disc.sqrt();
        let recip_a = 1.0 / a;
        let t1 = (-b - sq) * recip_a;
        let t2 = (-b + sq) * recip_a;
        let t = if t1 >= EPSILON { t1 } else { t2 };
        if t < EPSILON {
            -1.0
        } else {
            t
        }
    }

    fn brute_closest(spheres: &[ChunkSphere], origin: Vec3, dir: Vec3) -> f32 {
        let mut best = f32::MAX;
        for s in spheres {
            let t = hit_sphere(origin, dir, s);
            if t > 0.0 && t < best {
                best = t;
            }
        }
        best
    }

    /// Direct Rust mirror of `intersect_scene` in shaders.wgsl.
    fn grid_closest(grid: &Grid, spheres: &[ChunkSphere], origin: Vec3, dir: Vec3) -> f32 {
        let nx = grid.nx as i32;
        let nz = grid.nz as i32;
        let test_cell = |cell: usize, best: &mut f32| {
            let start = grid.cell_ranges[cell * 2] as usize;
            let count = grid.cell_ranges[cell * 2 + 1] as usize;
            for k in 0..count {
                let si = grid.sphere_indices[start + k] as usize;
                let t = hit_sphere(origin, dir, &spheres[si]);
                if t > 0.0 && t < *best {
                    *best = t;
                }
            }
        };

        let mut best = f32::MAX;
        if nx == 1 && nz == 1 {
            test_cell(0, &mut best);
            return best;
        }

        let cell = grid.cell_size;
        let inv_cell = 1.0 / cell;
        let (min_x, min_z) = (grid.min_x, grid.min_z);
        let max_x = min_x + nx as f32 * cell;
        let max_z = min_z + nz as f32 * cell;
        let (ox, oz) = (origin.x(), origin.z());
        let (dx, dz) = (dir.x(), dir.z());

        let mut t_enter = 0.0f32;
        let mut t_exit = f32::MAX;
        if dx.abs() < 1e-8 {
            if ox < min_x || ox > max_x {
                return f32::MAX;
            }
        } else {
            let inv = 1.0 / dx;
            let (mut t0, mut t1) = ((min_x - ox) * inv, (max_x - ox) * inv);
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_enter = t_enter.max(t0);
            t_exit = t_exit.min(t1);
        }
        if dz.abs() < 1e-8 {
            if oz < min_z || oz > max_z {
                return f32::MAX;
            }
        } else {
            let inv = 1.0 / dz;
            let (mut t0, mut t1) = ((min_z - oz) * inv, (max_z - oz) * inv);
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_enter = t_enter.max(t0);
            t_exit = t_exit.min(t1);
        }
        if t_enter > t_exit {
            return f32::MAX;
        }
        t_enter = t_enter.max(0.0);

        let ex = ox + dx * t_enter;
        let ez = oz + dz * t_enter;
        let mut ix = (((ex - min_x) * inv_cell).floor() as i32).clamp(0, nx - 1);
        let mut iz = (((ez - min_z) * inv_cell).floor() as i32).clamp(0, nz - 1);

        let step_x = if dx >= 0.0 { 1 } else { -1 };
        let step_z = if dz >= 0.0 { 1 } else { -1 };
        let bx = min_x + (ix + if dx >= 0.0 { 1 } else { 0 }) as f32 * cell;
        let bz = min_z + (iz + if dz >= 0.0 { 1 } else { 0 }) as f32 * cell;
        let mut t_max_x = if dx.abs() >= 1e-8 {
            (bx - ox) / dx
        } else {
            f32::MAX
        };
        let mut t_max_z = if dz.abs() >= 1e-8 {
            (bz - oz) / dz
        } else {
            f32::MAX
        };
        let t_delta_x = if dx.abs() >= 1e-8 {
            cell / dx.abs()
        } else {
            f32::MAX
        };
        let t_delta_z = if dz.abs() >= 1e-8 {
            cell / dz.abs()
        } else {
            f32::MAX
        };

        loop {
            test_cell((iz as u32 * grid.nx + ix as u32) as usize, &mut best);
            let t_cell_exit = t_max_x.min(t_max_z);
            if best <= t_cell_exit {
                return best;
            }
            if t_cell_exit > t_exit {
                break;
            }
            if t_max_x < t_max_z {
                ix += step_x;
                t_max_x += t_delta_x;
                if ix < 0 || ix >= nx {
                    break;
                }
            } else {
                iz += step_z;
                t_max_z += t_delta_z;
                if iz < 0 || iz >= nz {
                    break;
                }
            }
        }
        best
    }

    #[test]
    fn grid_traversal_matches_brute_force() {
        let cfg = test_cfg(2, usize::MAX);
        let mut loaded = HashMap::new();
        ensure_chunks(&cfg, &mut loaded, (0, 0), cfg.view_radius);
        let spheres = collect_spheres(&cfg, &loaded, (0, 0));
        let grid = build_grid(&spheres, &cfg, (0, 0));

        // Sweep many rays: random origins inside the region, directions that
        // include oblique, axis-aligned, and near-vertical cases.
        let mut rng = crate::procgen::Rng::seeded(0xBADC0DE);
        let span = (2 * cfg.view_radius + 1) as f32 * cfg.chunk_size;
        let mut checked = 0;
        for _ in 0..20_000 {
            let ox = grid.min_x + rng.next_f32() * span;
            let oz = grid.min_z + rng.next_f32() * span;
            let oy = rng.range_f32(-1.0, 3.0);
            let origin = Vec3::new(ox, oy, oz);
            // Direction: mostly oblique, occasionally axis-aligned / vertical.
            let dir = match rng.next_u32() % 5 {
                0 => Vec3::new(0.0, 1.0, 0.0),  // straight up
                1 => Vec3::new(1.0, 0.0, 0.0),  // +x
                2 => Vec3::new(0.0, 0.0, -1.0), // -z
                _ => Vec3::new(
                    rng.range_f32(-1.0, 1.0),
                    rng.range_f32(-0.5, 0.5),
                    rng.range_f32(-1.0, 1.0),
                ),
            };
            let g = grid_closest(&grid, &spheres, origin, dir);
            let b = brute_closest(&spheres, origin, dir);
            assert!(
                (g - b).abs() <= 1e-3 || (g == f32::MAX && b == f32::MAX),
                "grid/brute mismatch: grid={g} brute={b} origin={origin:?} dir={dir:?}"
            );
            checked += 1;
        }
        assert_eq!(checked, 20_000);
    }
}
