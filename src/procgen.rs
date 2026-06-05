// Originally written in 2026 by Clay Tyler
// SPDX-License-Identifier: CC-BY-4.0

//! CPU-side procedural generation: a deterministic RNG, continuous value noise,
//! and per-chunk geometry generators for a few content styles.
//!
//! Everything here is GPU-free and deterministic: given the same
//! `(world_seed, chunk_x, chunk_z)` and [`Style`], [`generate_chunk`] always
//! produces byte-identical output. Noise is sampled in *continuous world
//! coordinates* so terrain is seamless across chunk borders.

use crate::algebra::Vec3;

// --- Deterministic RNG (splitmix64) -----------------------------------------

/// A tiny deterministic PRNG (splitmix64). Cheap to seed per chunk.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn seeded(seed: u64) -> Self {
        Rng { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        finalize(self.state)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// A uniform float in `[0, 1)`, using the top 24 bits for the mantissa.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

/// The splitmix64 finalizing mix — also used as a standalone integer hash.
fn finalize(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Combine a world seed and integer chunk coordinates into one stable seed.
/// All three inputs are mixed so adjacent chunks are uncorrelated yet each is
/// reproducible.
pub fn chunk_seed(world_seed: u64, cx: i32, cz: i32) -> u64 {
    let a = finalize(world_seed ^ (cx as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    finalize(a ^ (cz as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F))
}

// --- Value noise (continuous world space) -----------------------------------

/// Hash an integer lattice point to a float in `[0, 1)`.
fn lattice_value(world_seed: u64, ix: i32, iz: i32) -> f32 {
    let h = chunk_seed(world_seed, ix, iz);
    (h >> 40) as f32 / (1u64 << 24) as f32
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// 2D value noise sampled at continuous world coordinates, returning `[0, 1]`.
/// It is a pure function of `(world_seed, x, z)`, so the same world point always
/// yields the same value regardless of which chunk asks — the seam guarantee.
pub fn value_noise_2d(world_seed: u64, x: f32, z: f32) -> f32 {
    let x0 = x.floor();
    let z0 = z.floor();
    let (ix, iz) = (x0 as i32, z0 as i32);
    let fx = x - x0;
    let fz = z - z0;
    let ux = smoothstep(fx);
    let uz = smoothstep(fz);

    let n00 = lattice_value(world_seed, ix, iz);
    let n10 = lattice_value(world_seed, ix + 1, iz);
    let n01 = lattice_value(world_seed, ix, iz + 1);
    let n11 = lattice_value(world_seed, ix + 1, iz + 1);

    lerp(lerp(n00, n10, ux), lerp(n01, n11, ux), uz)
}

/// Fractal Brownian motion: summed octaves of [`value_noise_2d`], returning
/// `[0, 1]`. Each octave perturbs the seed so the layers decorrelate.
pub fn fbm_2d(world_seed: u64, x: f32, z: f32, octaves: u32) -> f32 {
    let mut sum = 0.0;
    let mut norm = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    for o in 0..octaves.max(1) {
        let seed = world_seed ^ (o as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        sum += amp * value_noise_2d(seed, x * freq, z * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

// --- Chunk generation -------------------------------------------------------

/// One sphere emitted by a generator. `palette_index` references the world's
/// shared material palette (see the streaming layer).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkSphere {
    pub center: Vec3,
    pub radius: f32,
    pub palette_index: u32,
}

/// A content style with its parameters. Material choices are palette indices
/// (resolved from names by the data-driven layer), keeping this module
/// self-contained and testable.
#[derive(Debug, Clone)]
pub enum Style {
    /// Randomly scattered spheres resting on the ground.
    Scatter {
        density: u32,
        min_r: f32,
        max_r: f32,
        y_jitter: f32,
        mats: Vec<u32>,
    },
    /// A landscape: a sub-grid of spheres whose height follows world-space fbm.
    Terrain {
        /// World units per terrain sub-cell.
        cell: f32,
        /// World units per noise feature (larger = smoother hills).
        feature_size: f32,
        height_scale: f32,
        base_r: f32,
        octaves: u32,
        /// Palette indices banded low→high by height.
        bands: Vec<u32>,
    },
    /// A regular grid of vertical columns (stacked spheres).
    Architecture {
        spacing: f32,
        col_radius: f32,
        /// Spheres stacked per column.
        height: u32,
        mats: Vec<u32>,
    },
    /// Emissive spheres floating in a height band (a glowing swarm).
    Swarm {
        count: u32,
        y_lo: f32,
        y_hi: f32,
        r: f32,
        mats: Vec<u32>,
    },
    /// Endless interior corridor/room segments. Geometry is assembled directly
    /// by the streaming world layer so it can use native boxes/cylinders.
    Interior {
        wall_thickness: f32,
        height: f32,
        floor: u32,
        wall: u32,
        ceiling: u32,
        light: u32,
        column: u32,
    },
}

fn pick(rng: &mut Rng, mats: &[u32]) -> u32 {
    if mats.is_empty() {
        0
    } else {
        mats[(rng.next_u32() as usize) % mats.len()]
    }
}

/// Generate the spheres for one chunk. Deterministic in `(world_seed, cx, cz)`.
/// The chunk occupies world XZ `[cx, cx+1) * chunk_size × [cz, cz+1) * chunk_size`.
pub fn generate_chunk(
    world_seed: u64,
    cx: i32,
    cz: i32,
    chunk_size: f32,
    style: &Style,
) -> Vec<ChunkSphere> {
    let base_x = cx as f32 * chunk_size;
    let base_z = cz as f32 * chunk_size;
    let mut rng = Rng::seeded(chunk_seed(world_seed, cx, cz));
    let mut out = Vec::new();

    match style {
        Style::Scatter {
            density,
            min_r,
            max_r,
            y_jitter,
            mats,
        } => {
            for _ in 0..*density {
                let x = base_x + rng.next_f32() * chunk_size;
                let z = base_z + rng.next_f32() * chunk_size;
                let radius = rng.range_f32(*min_r, *max_r);
                let y = radius + rng.next_f32() * *y_jitter;
                let palette_index = pick(&mut rng, mats);
                out.push(ChunkSphere {
                    center: Vec3::new(x, y, z),
                    radius,
                    palette_index,
                });
            }
        }
        Style::Terrain {
            cell,
            feature_size,
            height_scale,
            base_r,
            octaves,
            bands,
        } => {
            let n = (chunk_size / cell).round().max(1.0) as i32;
            let fs = feature_size.max(1e-3);
            for j in 0..n {
                for i in 0..n {
                    let x = base_x + (i as f32 + 0.5) * cell;
                    let z = base_z + (j as f32 + 0.5) * cell;
                    let h = fbm_2d(world_seed, x / fs, z / fs, *octaves);
                    let y = h * height_scale;
                    let palette_index = if bands.is_empty() {
                        0
                    } else {
                        let band = ((h * bands.len() as f32) as usize).min(bands.len() - 1);
                        bands[band]
                    };
                    out.push(ChunkSphere {
                        center: Vec3::new(x, y, z),
                        radius: *base_r,
                        palette_index,
                    });
                }
            }
        }
        Style::Architecture {
            spacing,
            col_radius,
            height,
            mats,
        } => {
            let n = (chunk_size / spacing).floor().max(1.0) as i32;
            for j in 0..n {
                for i in 0..n {
                    // Skip a quarter of the columns (hashed) for ruined variety.
                    let skip = (rng.next_u32() & 3) == 0;
                    let palette_index = pick(&mut rng, mats);
                    if skip {
                        continue;
                    }
                    let x = base_x + (i as f32 + 0.5) * spacing;
                    let z = base_z + (j as f32 + 0.5) * spacing;
                    for k in 0..*height {
                        let y = col_radius + k as f32 * col_radius * 2.0;
                        out.push(ChunkSphere {
                            center: Vec3::new(x, y, z),
                            radius: *col_radius,
                            palette_index,
                        });
                    }
                }
            }
        }
        Style::Swarm {
            count,
            y_lo,
            y_hi,
            r,
            mats,
        } => {
            for _ in 0..*count {
                let x = base_x + rng.next_f32() * chunk_size;
                let z = base_z + rng.next_f32() * chunk_size;
                let y = rng.range_f32(*y_lo, *y_hi);
                let palette_index = pick(&mut rng, mats);
                out.push(ChunkSphere {
                    center: Vec3::new(x, y, z),
                    radius: *r,
                    palette_index,
                });
            }
        }
        Style::Interior { .. } => {}
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_varies_by_seed() {
        let mut a = Rng::seeded(42);
        let mut b = Rng::seeded(42);
        let mut c = Rng::seeded(43);
        let sa: Vec<u32> = (0..8).map(|_| a.next_u32()).collect();
        let sb: Vec<u32> = (0..8).map(|_| b.next_u32()).collect();
        let sc: Vec<u32> = (0..8).map(|_| c.next_u32()).collect();
        assert_eq!(sa, sb, "same seed must produce same sequence");
        assert_ne!(sa, sc, "different seed must diverge");
    }

    #[test]
    fn next_f32_in_unit_interval() {
        let mut rng = Rng::seeded(7);
        for _ in 0..10_000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn chunk_seed_combines_all_inputs() {
        assert_ne!(chunk_seed(1, 0, 0), chunk_seed(2, 0, 0));
        assert_ne!(chunk_seed(1, 0, 0), chunk_seed(1, 1, 0));
        assert_ne!(chunk_seed(1, 0, 0), chunk_seed(1, 0, 1));
        assert_ne!(chunk_seed(1, 5, 3), chunk_seed(1, 3, 5)); // order matters
        assert_eq!(chunk_seed(9, -2, 4), chunk_seed(9, -2, 4)); // reproducible
    }

    #[test]
    fn value_noise_is_continuous_and_bounded() {
        let seed = 1234;
        for i in 0..2000 {
            let x = i as f32 * 0.013 - 13.0;
            let z = i as f32 * -0.017 + 5.0;
            let a = value_noise_2d(seed, x, z);
            let b = value_noise_2d(seed, x + 0.001, z + 0.001);
            assert!((0.0..=1.0).contains(&a), "noise out of range: {a}");
            // Lipschitz-ish: tiny step => tiny change (guards seam continuity).
            assert!(
                (a - b).abs() < 0.02,
                "noise discontinuity at {x},{z}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn value_noise_same_point_same_value() {
        // The seam guarantee: a shared world point read from "either chunk".
        let p = value_noise_2d(77, 32.0, -16.0);
        let q = value_noise_2d(77, 32.0, -16.0);
        assert_eq!(p, q);
    }

    fn scatter_style() -> Style {
        Style::Scatter {
            density: 20,
            min_r: 0.2,
            max_r: 0.6,
            y_jitter: 0.1,
            mats: vec![1, 3, 5],
        }
    }

    #[test]
    fn generate_chunk_is_reproducible_and_location_dependent() {
        let s = scatter_style();
        let a = generate_chunk(100, 2, -3, 8.0, &s);
        let b = generate_chunk(100, 2, -3, 8.0, &s);
        let c = generate_chunk(100, 2, -2, 8.0, &s);
        assert_eq!(a, b, "same chunk coords must reproduce exactly");
        assert_ne!(a, c, "different chunk coords must differ");
        assert_eq!(a.len(), 20, "scatter density honored");
    }

    #[test]
    fn scatter_spheres_stay_in_chunk_xz_and_use_palette() {
        let s = scatter_style();
        let chunk_size = 8.0;
        let (cx, cz) = (4, -1);
        let spheres = generate_chunk(100, cx, cz, chunk_size, &s);
        let (bx, bz) = (cx as f32 * chunk_size, cz as f32 * chunk_size);
        for sp in &spheres {
            assert!(sp.center.x() >= bx && sp.center.x() <= bx + chunk_size);
            assert!(sp.center.z() >= bz && sp.center.z() <= bz + chunk_size);
            assert!([1, 3, 5].contains(&sp.palette_index));
        }
    }

    #[test]
    fn terrain_counts_and_bands() {
        let s = Style::Terrain {
            cell: 2.0,
            feature_size: 40.0,
            height_scale: 5.0,
            base_r: 1.0,
            octaves: 4,
            bands: vec![0, 1, 2],
        };
        let chunk_size = 8.0;
        let spheres = generate_chunk(5, 0, 0, chunk_size, &s);
        assert_eq!(spheres.len(), 16, "4x4 sub-grid for cell=2, chunk=8");
        for sp in &spheres {
            assert!(sp.palette_index < 3);
        }
    }

    #[test]
    fn architecture_columns_stack_and_stay_in_chunk() {
        let s = Style::Architecture {
            spacing: 4.0,
            col_radius: 0.5,
            height: 3,
            mats: vec![2, 4],
        };
        let chunk_size = 8.0;
        let (cx, cz) = (1, 1);
        let spheres = generate_chunk(9, cx, cz, chunk_size, &s);
        // 2x2 columns, some skipped; every column contributes `height` spheres,
        // so the count is a multiple of 3 and at most 4*3.
        assert!(!spheres.is_empty());
        assert_eq!(spheres.len() % 3, 0, "each column stacks `height` spheres");
        assert!(spheres.len() <= 12);
        let (bx, bz) = (cx as f32 * chunk_size, cz as f32 * chunk_size);
        for sp in &spheres {
            assert!(sp.center.x() >= bx && sp.center.x() <= bx + chunk_size);
            assert!(sp.center.z() >= bz && sp.center.z() <= bz + chunk_size);
            assert!([2, 4].contains(&sp.palette_index));
            assert!(sp.center.y() >= 0.5, "columns rest above ground");
        }
    }

    #[test]
    fn swarm_count_and_height_band() {
        let s = Style::Swarm {
            count: 30,
            y_lo: 4.0,
            y_hi: 9.0,
            r: 0.3,
            mats: vec![7],
        };
        let spheres = generate_chunk(5, -2, 2, 8.0, &s);
        assert_eq!(spheres.len(), 30);
        for sp in &spheres {
            assert!((4.0..=9.0).contains(&sp.center.y()));
            assert_eq!(sp.palette_index, 7);
        }
    }
}
