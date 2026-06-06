// Originally written in 2025 by Arman Uguray <arman.uguray@gmail.com>
// SPDX-License-Identifier: CC-BY-4.0

use anyhow::Result;
#[cfg(not(target_arch = "wasm32"))]
use anyhow::Context;

use crate::{
    camera::{Camera, CameraPath},
    embedded_scenes::SCENES,
    scene_def::{self, SceneMetadata, WorldDef},
    world::World,
};

/// Directory the `.ron` files live in, used only by native hot-reload (R) to
/// re-read an edited scene. The web build has no filesystem and relies entirely
/// on the copies embedded by `build.rs`.
#[cfg(not(target_arch = "wasm32"))]
const SCENES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/scenes");

pub struct Gallery {
    current_index: usize,
    entries: Vec<Entry>,
}

/// One slot in the gallery: either a static authored scene or a procedural
/// streaming world. `name` is the scene's file stem (e.g. `"04_cornell_box"`),
/// used as a title fallback and to locate the file for native hot-reload.
struct Entry {
    source: SceneSource,
    // Read by native hot-reload; the web build only uses it at construction.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    name: String,
    metadata: SceneMetadata,
    initial_camera: Camera,
}

enum SceneSource {
    Static(StaticScene),
    World(World),
}

struct StaticScene {
    camera: Camera,
    resources: wgpu::BindGroup,
    camera_path: Option<CameraPath>,
}

#[derive(Clone)]
pub struct SceneOption {
    pub index: usize,
    pub title: String,
}

impl Gallery {
    /// Build every scene embedded by `build.rs` — static scenes and procedural
    /// worlds alike. Broken files are reported and skipped.
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let mut entries = Vec::new();
        for (name, text) in SCENES {
            match load_entry(name, text, device, layout) {
                Ok(entry) => entries.push(entry),
                Err(e) => eprintln!("warning: skipping {name}: {e:#}"),
            }
        }

        if entries.is_empty() {
            panic!("no scenes embedded; add a .ron file under scenes/");
        }

        Self {
            current_index: 0,
            entries,
        }
    }

    /// If the current slot is a procedural world, reconcile its loaded chunks
    /// with the camera position. Returns `true` when the world rebuilt (the
    /// caller should then reset sample accumulation).
    pub fn update_world(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
    ) -> bool {
        match &mut self.entries[self.current_index].source {
            SceneSource::World(w) => w.update(device, layout),
            SceneSource::Static(_) => false,
        }
    }

    /// Re-read the current entry's file from disk and rebuild it (static scene
    /// or world). Native-only: the web build has no filesystem and uses the
    /// embedded scenes. On error the previously loaded entry is kept.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn reload_current(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
    ) -> Result<()> {
        let entry = &mut self.entries[self.current_index];
        let path = std::path::Path::new(SCENES_DIR).join(format!("{}.ron", entry.name));
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let new_entry = load_entry(&entry.name.clone(), &text, device, layout)
            .with_context(|| format!("reloading {}", path.display()))?;
        *entry = new_entry;
        Ok(())
    }

    pub fn current_index(&self) -> usize {
        self.current_index
    }

    pub fn scene_options(&self) -> Vec<SceneOption> {
        self.entries
            .iter()
            .enumerate()
            .map(|(index, entry)| SceneOption {
                index,
                title: entry
                    .metadata
                    .title
                    .clone()
                    .unwrap_or_else(|| "Scene".to_string()),
            })
            .collect()
    }

    pub fn current_metadata(&self) -> &SceneMetadata {
        &self.entries[self.current_index].metadata
    }

    pub fn current_path(&self) -> Option<&CameraPath> {
        match &self.entries[self.current_index].source {
            SceneSource::Static(s) => s.camera_path.as_ref(),
            SceneSource::World(_) => None,
        }
    }

    pub fn current_camera(&self) -> &Camera {
        match &self.entries[self.current_index].source {
            SceneSource::Static(s) => &s.camera,
            SceneSource::World(w) => w.camera(),
        }
    }

    pub fn current_resources(&self) -> &wgpu::BindGroup {
        match &self.entries[self.current_index].source {
            SceneSource::Static(s) => &s.resources,
            SceneSource::World(w) => w.resources(),
        }
    }

    pub fn current_camera_mut(&mut self) -> &mut Camera {
        match &mut self.entries[self.current_index].source {
            SceneSource::Static(s) => &mut s.camera,
            SceneSource::World(w) => w.camera_mut(),
        }
    }

    pub fn reset_current_camera(&mut self) {
        let camera = self.entries[self.current_index].initial_camera.clone();
        *self.current_camera_mut() = camera;
    }

    pub fn select_index(&mut self, index: usize) {
        if index < self.entries.len() {
            self.current_index = index;
        }
    }

    pub fn select_next(&mut self) {
        self.current_index += 1;
        self.current_index %= self.entries.len();
    }

    pub fn select_previous(&mut self) {
        if self.current_index == 0 {
            self.current_index = self.entries.len() - 1;
        } else {
            self.current_index -= 1;
        }
    }
}

/// Build a scene from its name + `.ron` text, as either a static scene or a
/// procedural world, detected from the contents (`World(( .. ))` vs a bare
/// scene tuple).
fn load_entry(
    name: &str,
    text: &str,
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> Result<Entry> {
    let fallback_title = title_from_stem(name);
    if scene_def::is_world_source(text) {
        let def = WorldDef::from_ron(text)?;
        let mut metadata = def.metadata();
        metadata.title.get_or_insert(fallback_title);
        let (cfg, camera) = def.build()?;
        Ok(Entry {
            initial_camera: camera.clone(),
            source: SceneSource::World(World::new(device, layout, cfg, camera)),
            name: name.to_string(),
            metadata,
        })
    } else {
        let def = scene_def::SceneDef::from_ron(text)?;
        let mut metadata = def.metadata();
        metadata.title.get_or_insert(fallback_title);
        let (camera, camera_path, builder) = def.build()?;
        Ok(Entry {
            initial_camera: camera.clone(),
            source: SceneSource::Static(StaticScene {
                camera,
                resources: builder.build(device, layout),
                camera_path,
            }),
            name: name.to_string(),
            metadata,
        })
    }
}

/// Derive a display title from a file stem, e.g. `"04_cornell_box"` -> `"Cornell Box"`.
fn title_from_stem(stem: &str) -> String {
    let name = stem
        .trim_start_matches(|c: char| c.is_ascii_digit())
        .trim_start_matches('_');
    name.split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
