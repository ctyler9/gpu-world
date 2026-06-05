// Originally written in 2025 by Clay Tyler
// SPDX-License-Identifier: CC-BY-4.0

//! Data-driven scene definitions.
//!
//! A scene lives in a `.ron` file under `scenes/` and is deserialized into the
//! types below, then lowered onto the procedural [`SceneBuilder`] helpers. The
//! schema mirrors those helpers: alongside a plain `Sphere` you can drop a
//! `Grid` or `Ring` generator so procedural layouts stay compact in data.

use std::collections::{BTreeMap, HashMap};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use crate::{
    algebra::Vec3,
    audio::AudioMode,
    camera::{Camera, CameraKeyframe, CameraPath},
    procgen::Style,
    scene::{Material, SceneBuilder},
    world::WorldConfig,
};

/// A position written in RON as a 3-tuple, e.g. `(0.0, 0.5, 1.6)`.
type Vec3Def = [f32; 3];

fn white() -> Vec3Def {
    [1.0, 1.0, 1.0]
}

fn one() -> f32 {
    1.0
}

fn default_fov() -> f32 {
    30.0
}

fn yes() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
pub enum CameraMode {
    Manual,
    Path,
    Drift,
}

impl CameraMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Manual => "manual",
            Self::Path => "path",
            Self::Drift => "drift",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SceneMetadata {
    pub title: Option<String>,
    pub description: String,
    pub music: Option<AudioMode>,
    pub camera_mode: Option<CameraMode>,
    pub reactive_lights: f32,
}

#[derive(Debug, Deserialize)]
pub struct SceneDef {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    description: String,
    #[serde(default)]
    music: Option<AudioMode>,
    #[serde(default)]
    camera_mode: Option<CameraMode>,
    #[serde(default)]
    reactive_lights: f32,
    /// Named materials, referenced by the objects below.
    materials: HashMap<String, MaterialDef>,
    /// The things in the scene, placed in order.
    objects: Vec<ObjectDef>,
    camera: CameraDef,
    #[serde(default)]
    path: Option<PathDef>,
}

#[derive(Debug, Deserialize)]
enum MaterialDef {
    Lambertian(Vec3Def),
    /// `Opaque(color, metallic)` where metallic is in `[0, 1]`.
    Opaque(Vec3Def, f32),
    Metal(Vec3Def),
    /// Transparent dielectric; color defaults to white.
    Dielectric {
        ior: f32,
        #[serde(default = "white")]
        color: Vec3Def,
    },
    /// A light-emitting surface: `Emissive(color, strength)`.
    Emissive(Vec3Def, f32),
}

impl MaterialDef {
    fn lower(&self) -> Material {
        match *self {
            MaterialDef::Lambertian(c) => Material::lambertian(c.into()),
            MaterialDef::Opaque(c, m) => Material::opaque(c.into(), m),
            MaterialDef::Metal(c) => Material::metal(c.into()),
            MaterialDef::Dielectric { ior, color } => {
                Material::transparent_dielectric(color.into(), ior)
            }
            MaterialDef::Emissive(color, strength) => {
                Material::emissive(color.into(), strength)
            }
        }
    }
}

#[derive(Debug, Deserialize)]
enum ObjectDef {
    Sphere {
        at: Vec3Def,
        r: f32,
        mat: String,
    },
    Box {
        min: Vec3Def,
        max: Vec3Def,
        mat: String,
    },
    Cylinder {
        center: Vec3Def,
        radius: f32,
        y_min: f32,
        y_max: f32,
        mat: String,
    },
    WallPanel {
        min: Vec3Def,
        max: Vec3Def,
        mat: String,
    },
    Column {
        center: Vec3Def,
        radius: f32,
        y_min: f32,
        y_max: f32,
        mat: String,
    },
    Arch {
        center: Vec3Def,
        width: f32,
        height: f32,
        depth: f32,
        thickness: f32,
        mat: String,
    },
    /// A large sphere standing in for the ground plane at y = 0.
    Ground(String),
    /// A `cols` x `rows` grid of spheres on the XZ plane, centered on `origin`.
    Grid {
        #[serde(default)]
        origin: Vec3Def,
        cols: usize,
        rows: usize,
        spacing: f32,
        /// Height of each sphere's center.
        y: f32,
        r: f32,
        /// Materials cycled across cells (checkerboard if `checker`).
        materials: Vec<String>,
        #[serde(default)]
        checker: bool,
    },
    /// `count` stacks evenly spaced around a circle on the XZ plane.
    Ring {
        #[serde(default)]
        center: Vec3Def,
        radius: f32,
        count: usize,
        /// One or more spheres placed at every ring position.
        stack: Vec<StackItem>,
    },
    /// An axis-aligned box room with interior corners `min`..`max`. Each face is
    /// a huge sphere (radius `wall`) whose near surface reads as a flat wall;
    /// omit a face to leave it open. Faces: `floor`/`ceiling` (±y),
    /// `left`/`right` (∓x), `back`/`front` (∓z).
    Room {
        min: Vec3Def,
        max: Vec3Def,
        /// Radius of the wall spheres; bigger reads flatter. Defaults to 1000.
        #[serde(default = "wall_radius")]
        wall: f32,
        #[serde(default)]
        floor: Option<String>,
        #[serde(default)]
        ceiling: Option<String>,
        #[serde(default)]
        left: Option<String>,
        #[serde(default)]
        right: Option<String>,
        #[serde(default)]
        back: Option<String>,
        #[serde(default)]
        front: Option<String>,
    },
}

fn wall_radius() -> f32 {
    1000.0
}

/// One sphere within a [`ObjectDef::Ring`] stack.
#[derive(Debug, Deserialize)]
struct StackItem {
    /// Center height of this sphere.
    y: f32,
    r: f32,
    /// Either one material name (`mat: "blue"`) or a list cycled across the ring
    /// positions (`mat: ["red", "green", "blue"]`).
    mat: MatRef,
    /// Horizontal scale toward the ring center: `1.0` sits on the ring, smaller
    /// values pull the sphere inward (e.g. `0.3125` for an inner ring).
    #[serde(default = "one")]
    inset: f32,
}

/// A material reference that is either a single name or a list cycled by index.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MatRef {
    One(String),
    Many(Vec<String>),
}

impl MatRef {
    /// The material names this reference cycles through, in order.
    fn names(&self) -> &[String] {
        match self {
            MatRef::One(name) => std::slice::from_ref(name),
            MatRef::Many(names) => names,
        }
    }
}

#[derive(Debug, Deserialize)]
enum CameraDef {
    LookAt {
        eye: Vec3Def,
        center: Vec3Def,
        up: Vec3Def,
        /// Vertical field of view, in degrees.
        #[serde(default = "default_fov")]
        fov: f32,
    },
}

#[derive(Debug, Deserialize)]
enum PathDef {
    /// A looping circular orbit around `center`.
    Orbit {
        #[serde(default)]
        center: Vec3Def,
        radius: f32,
        height: f32,
        #[serde(default = "default_fov")]
        fov: f32,
        secs: f32,
        steps: usize,
    },
    /// An explicit list of keyframes.
    Keyframes {
        frames: Vec<KeyframeDef>,
        secs_per_segment: f32,
        #[serde(default = "yes")]
        looping: bool,
    },
}

#[derive(Debug, Deserialize)]
struct KeyframeDef {
    eye: Vec3Def,
    center: Vec3Def,
    #[serde(default = "up_y")]
    up: Vec3Def,
    #[serde(default = "default_fov")]
    fov: f32,
}

fn up_y() -> Vec3Def {
    [0.0, 1.0, 0.0]
}

impl SceneDef {
    /// Parse a scene from RON source text. The `implicit_some` extension lets
    /// optional fields like `path` be written without a `Some(..)` wrapper.
    pub fn from_ron(text: &str) -> Result<Self> {
        ron::Options::default()
            .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME)
            .from_str(text)
            .context("failed to parse scene RON")
    }

    pub fn metadata(&self) -> SceneMetadata {
        SceneMetadata {
            title: self.title.clone(),
            description: self.description.clone(),
            music: self.music,
            camera_mode: self.camera_mode,
            reactive_lights: self.reactive_lights,
        }
    }

    /// Lower this definition into the renderer-facing pieces, reusing the
    /// procedural [`SceneBuilder`] helpers for grids and rings.
    pub fn build(&self) -> Result<(Camera, Option<CameraPath>, SceneBuilder)> {
        let mut builder = SceneBuilder::new();

        // Register materials, remembering their ids by name.
        let mut ids = HashMap::with_capacity(self.materials.len());
        for (name, def) in &self.materials {
            ids.insert(name.clone(), builder.add_material(def.lower()));
        }
        let lookup = |name: &str| {
            ids.get(name)
                .copied()
                .ok_or_else(|| anyhow!("unknown material `{name}`"))
        };

        for object in &self.objects {
            match object {
                ObjectDef::Sphere { at, r, mat } => {
                    builder.sphere(*at, *r, lookup(mat)?);
                }
                ObjectDef::Box { min, max, mat }
                | ObjectDef::WallPanel { min, max, mat } => {
                    builder.cuboid(*min, *max, lookup(mat)?);
                }
                ObjectDef::Cylinder {
                    center,
                    radius,
                    y_min,
                    y_max,
                    mat,
                }
                | ObjectDef::Column {
                    center,
                    radius,
                    y_min,
                    y_max,
                    mat,
                } => {
                    builder.cylinder(*center, *radius, *y_min, *y_max, lookup(mat)?);
                }
                ObjectDef::Arch {
                    center,
                    width,
                    height,
                    depth,
                    thickness,
                    mat,
                } => {
                    let c: Vec3 = (*center).into();
                    let id = lookup(mat)?;
                    let half_w = width * 0.5;
                    let half_d = depth * 0.5;
                    let leg_r = thickness * 0.5;
                    let y0 = c.y();
                    let y1 = c.y() + height - thickness;
                    builder.cylinder(
                        (c.x() - half_w + leg_r, 0.0, c.z()),
                        leg_r,
                        y0,
                        y1,
                        id,
                    );
                    builder.cylinder(
                        (c.x() + half_w - leg_r, 0.0, c.z()),
                        leg_r,
                        y0,
                        y1,
                        id,
                    );
                    builder.cuboid(
                        (c.x() - half_w, y1, c.z() - half_d),
                        (c.x() + half_w, c.y() + height, c.z() + half_d),
                        id,
                    );
                }
                ObjectDef::Ground(mat) => {
                    builder.ground(lookup(mat)?);
                }
                ObjectDef::Grid {
                    origin,
                    cols,
                    rows,
                    spacing,
                    y,
                    r,
                    materials,
                    checker,
                } => {
                    if materials.is_empty() {
                        return Err(anyhow!("Grid needs at least one material"));
                    }
                    let mats: Vec<_> =
                        materials.iter().map(|m| lookup(m)).collect::<Result<_>>()?;
                    let (y, r, checker) = (*y, *r, *checker);
                    builder.grid(*origin, *cols, *rows, *spacing, |s, (i, j), pos| {
                        let idx = if checker { i + j } else { i + j * *cols };
                        let mat = mats[idx % mats.len()];
                        s.sphere((pos.x(), y, pos.z()), r, mat);
                    });
                }
                ObjectDef::Ring {
                    center,
                    radius,
                    count,
                    stack,
                } => {
                    // Resolve each stack item to (y, r, inset, materials-by-position).
                    let stack: Vec<(f32, f32, f32, Vec<_>)> = stack
                        .iter()
                        .map(|it| {
                            let mats = it
                                .mat
                                .names()
                                .iter()
                                .map(|n| lookup(n))
                                .collect::<Result<Vec<_>>>()?;
                            Ok((it.y, it.r, it.inset, mats))
                        })
                        .collect::<Result<_>>()?;
                    let center: Vec3 = (*center).into();
                    builder.ring(center, *radius, *count, |s, i, pos| {
                        for (y, r, inset, mats) in &stack {
                            let offset = (pos - center) * *inset;
                            let p = center + offset;
                            let mat = mats[i % mats.len()];
                            s.sphere((p.x(), *y, p.z()), *r, mat);
                        }
                    });
                }
                ObjectDef::Room {
                    min,
                    max,
                    wall,
                    floor,
                    ceiling,
                    left,
                    right,
                    back,
                    front,
                } => {
                    let lo: Vec3 = (*min).into();
                    let hi: Vec3 = (*max).into();
                    let (mx, my, mz) = (
                        (lo.x() + hi.x()) * 0.5,
                        (lo.y() + hi.y()) * 0.5,
                        (lo.z() + hi.z()) * 0.5,
                    );
                    let r = *wall;
                    // Each face: its material and the center of the wall sphere
                    // sitting just outside that face.
                    let faces = [
                        (floor, (mx, lo.y() - r, mz)),
                        (ceiling, (mx, hi.y() + r, mz)),
                        (left, (lo.x() - r, my, mz)),
                        (right, (hi.x() + r, my, mz)),
                        (back, (mx, my, lo.z() - r)),
                        (front, (mx, my, hi.z() + r)),
                    ];
                    for (mat, center) in faces {
                        if let Some(name) = mat {
                            builder.sphere(center, r, lookup(name)?);
                        }
                    }
                }
            }
        }

        let camera = self.camera.lower();
        let path = self.path.as_ref().map(PathDef::lower);
        Ok((camera, path, builder))
    }
}

impl CameraDef {
    fn lower(&self) -> Camera {
        match *self {
            CameraDef::LookAt {
                eye,
                center,
                up,
                fov,
            } => Camera::look_at(eye.into(), center.into(), up.into())
                .with_fov(fov.to_radians()),
        }
    }
}

// --- Procedural world definitions -------------------------------------------

/// Detect whether a `.ron` file describes a procedural world (`World((...))`)
/// rather than a static scene, by inspecting the first meaningful token. Lets
/// world files and static scene files share the `scenes/` directory.
pub fn is_world_source(text: &str) -> bool {
    for line in text.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") || t.starts_with("#!") {
            continue;
        }
        return t.starts_with("World");
    }
    false
}

/// World files are wrapped in `World(( .. ))` to distinguish them from the bare
/// `( .. )` tuple of a static scene.
#[derive(Debug, Deserialize)]
enum WorldWrapper {
    World(WorldDef),
}

#[derive(Debug, Deserialize)]
pub struct WorldDef {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    description: String,
    #[serde(default)]
    music: Option<AudioMode>,
    #[serde(default)]
    camera_mode: Option<CameraMode>,
    #[serde(default)]
    reactive_lights: f32,
    world_seed: u64,
    chunk_size: f32,
    /// Chebyshev radius, in chunks, of the loaded region.
    view_radius: i32,
    #[serde(default = "default_max_spheres")]
    max_spheres: usize,
    /// Shared, bounded material palette (deterministic order: sorted by name).
    palette: BTreeMap<String, MaterialDef>,
    biome: BiomeDef,
    camera: CameraDef,
    /// Optional ground plane material (a palette name); omit for no floor.
    #[serde(default)]
    ground: Option<String>,
    /// Background blend for escaped rays: 1.0 = procedural sky (default),
    /// 0.0 = the solid `background` color (e.g. a dark night world).
    #[serde(default = "default_sky")]
    sky: f32,
    /// Solid background color, used when `sky` < 1.0.
    #[serde(default)]
    background: Vec3Def,
}

fn default_max_spheres() -> usize {
    4096
}

fn default_sky() -> f32 {
    1.0
}

fn four() -> u32 {
    4
}

#[derive(Debug, Deserialize)]
enum BiomeDef {
    Scatter {
        density: u32,
        min_r: f32,
        max_r: f32,
        #[serde(default)]
        y_jitter: f32,
        mats: Vec<String>,
    },
    Terrain {
        cell: f32,
        feature_size: f32,
        height_scale: f32,
        base_r: f32,
        #[serde(default = "four")]
        octaves: u32,
        bands: Vec<String>,
    },
    Architecture {
        spacing: f32,
        col_radius: f32,
        height: u32,
        mats: Vec<String>,
    },
    Swarm {
        count: u32,
        y_lo: f32,
        y_hi: f32,
        r: f32,
        mats: Vec<String>,
    },
    Interior {
        wall_thickness: f32,
        height: f32,
        floor: String,
        wall: String,
        ceiling: String,
        light: String,
        column: String,
    },
}

impl WorldDef {
    pub fn from_ron(text: &str) -> Result<Self> {
        let WorldWrapper::World(def) = ron::Options::default()
            .with_default_extension(ron::extensions::Extensions::IMPLICIT_SOME)
            .from_str(text)
            .context("failed to parse world RON")?;
        Ok(def)
    }

    pub fn metadata(&self) -> SceneMetadata {
        SceneMetadata {
            title: self.title.clone(),
            description: self.description.clone(),
            music: self.music,
            camera_mode: self.camera_mode,
            reactive_lights: self.reactive_lights,
        }
    }

    /// Device-free lowering into a renderer-ready [`WorldConfig`] plus the spawn
    /// camera. The palette order is deterministic (BTreeMap is sorted by key),
    /// which matters because chunk geometry bakes palette indices.
    pub fn build(&self) -> Result<(WorldConfig, Camera)> {
        let mut palette = Vec::with_capacity(self.palette.len());
        let mut index_of = HashMap::new();
        for (name, def) in &self.palette {
            index_of.insert(name.clone(), palette.len() as u32);
            palette.push(def.lower());
        }
        let resolve = |names: &[String]| -> Result<Vec<u32>> {
            names
                .iter()
                .map(|n| {
                    index_of
                        .get(n)
                        .copied()
                        .ok_or_else(|| anyhow!("unknown palette material `{n}`"))
                })
                .collect()
        };

        let style = match &self.biome {
            BiomeDef::Scatter {
                density,
                min_r,
                max_r,
                y_jitter,
                mats,
            } => Style::Scatter {
                density: *density,
                min_r: *min_r,
                max_r: *max_r,
                y_jitter: *y_jitter,
                mats: resolve(mats)?,
            },
            BiomeDef::Terrain {
                cell,
                feature_size,
                height_scale,
                base_r,
                octaves,
                bands,
            } => Style::Terrain {
                cell: *cell,
                feature_size: *feature_size,
                height_scale: *height_scale,
                base_r: *base_r,
                octaves: *octaves,
                bands: resolve(bands)?,
            },
            BiomeDef::Architecture {
                spacing,
                col_radius,
                height,
                mats,
            } => Style::Architecture {
                spacing: *spacing,
                col_radius: *col_radius,
                height: *height,
                mats: resolve(mats)?,
            },
            BiomeDef::Swarm {
                count,
                y_lo,
                y_hi,
                r,
                mats,
            } => Style::Swarm {
                count: *count,
                y_lo: *y_lo,
                y_hi: *y_hi,
                r: *r,
                mats: resolve(mats)?,
            },
            BiomeDef::Interior {
                wall_thickness,
                height,
                floor,
                wall,
                ceiling,
                light,
                column,
            } => Style::Interior {
                wall_thickness: *wall_thickness,
                height: *height,
                floor: *index_of
                    .get(floor)
                    .ok_or_else(|| anyhow!("unknown interior floor material `{floor}`"))?,
                wall: *index_of
                    .get(wall)
                    .ok_or_else(|| anyhow!("unknown interior wall material `{wall}`"))?,
                ceiling: *index_of.get(ceiling).ok_or_else(|| {
                    anyhow!("unknown interior ceiling material `{ceiling}`")
                })?,
                light: *index_of
                    .get(light)
                    .ok_or_else(|| anyhow!("unknown interior light material `{light}`"))?,
                column: *index_of
                    .get(column)
                    .ok_or_else(|| anyhow!("unknown interior column material `{column}`"))?,
            },
        };

        let ground = match &self.ground {
            Some(name) => Some(
                index_of
                    .get(name)
                    .copied()
                    .ok_or_else(|| anyhow!("unknown ground material `{name}`"))?,
            ),
            None => None,
        };

        let cfg = WorldConfig {
            world_seed: self.world_seed,
            chunk_size: self.chunk_size,
            view_radius: self.view_radius,
            style,
            palette,
            max_spheres: self.max_spheres,
            ground,
            background: self.background.into(),
            sky_amount: self.sky,
        };
        Ok((cfg, self.camera.lower()))
    }
}

#[cfg(test)]
mod tests {
    use super::{is_world_source, SceneDef, WorldDef};

    /// Every shipped scene/world file must parse and lower without error.
    #[test]
    fn shipped_scenes_load() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/scenes");
        let mut count = 0;
        for entry in std::fs::read_dir(dir).expect("scenes dir") {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "ron") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            if is_world_source(&text) {
                let def = WorldDef::from_ron(&text)
                    .unwrap_or_else(|e| panic!("parsing world {}: {e:#}", path.display()));
                def.build()
                    .unwrap_or_else(|e| panic!("building world {}: {e:#}", path.display()));
            } else {
                let def = SceneDef::from_ron(&text)
                    .unwrap_or_else(|e| panic!("parsing {}: {e:#}", path.display()));
                def.build()
                    .unwrap_or_else(|e| panic!("building {}: {e:#}", path.display()));
            }
            count += 1;
        }
        assert!(count > 0, "no scene files found in {dir}");
    }

    /// Lowering is deterministic: same world text -> identical palette order and
    /// resolved indices (guards against HashMap nondeterminism in the palette).
    #[test]
    fn world_lowering_is_deterministic() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/scenes");
        for entry in std::fs::read_dir(dir).expect("scenes dir") {
            let path = entry.unwrap().path();
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            if !is_world_source(&text) {
                continue;
            }
            let a = WorldDef::from_ron(&text).unwrap().build().unwrap().0;
            let b = WorldDef::from_ron(&text).unwrap().build().unwrap().0;
            assert_eq!(a.palette.len(), b.palette.len());
            // Style mats/bands are baked from palette order; compare via Debug.
            assert_eq!(format!("{:?}", a.style), format!("{:?}", b.style));
        }
    }
}

impl PathDef {
    fn lower(&self) -> CameraPath {
        match self {
            PathDef::Orbit {
                center,
                radius,
                height,
                fov,
                secs,
                steps,
            } => CameraPath::orbit(
                (*center).into(),
                *radius,
                *height,
                fov.to_radians(),
                *secs,
                *steps,
            ),
            PathDef::Keyframes {
                frames,
                secs_per_segment,
                looping,
            } => CameraPath {
                keyframes: frames
                    .iter()
                    .map(|k| CameraKeyframe {
                        origin: k.eye.into(),
                        center: k.center.into(),
                        up: k.up.into(),
                        fov_y: k.fov.to_radians(),
                    })
                    .collect(),
                secs_per_segment: *secs_per_segment,
                looping: *looping,
            },
        }
    }
}
