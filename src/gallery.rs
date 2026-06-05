// Originally written in 2025 by Arman Uguray <arman.uguray@gmail.com>
// SPDX-License-Identifier: CC-BY-4.0

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::{
    camera::{Camera, CameraPath},
    scene_def::{self, SceneMetadata, WorldDef},
    world::World,
};

/// Directory that scene `.ron` files are loaded from, relative to the crate.
const SCENES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/scenes");

pub struct Gallery {
    current_index: usize,
    entries: Vec<Entry>,
}

/// One slot in the gallery: either a static authored scene or a procedural
/// streaming world. `path` is set only for file-backed static scenes (so reload
/// knows what to re-read).
struct Entry {
    source: SceneSource,
    path: Option<PathBuf>,
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
    /// Load every `scenes/*.ron` file — static scenes and procedural worlds
    /// alike. Broken files are reported and skipped.
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let scene_paths = discover_scene_files(SCENES_DIR.as_ref()).unwrap_or_else(|e| {
            eprintln!("warning: could not read scenes from {SCENES_DIR}: {e:#}");
            Vec::new()
        });

        let mut entries = Vec::new();
        for path in scene_paths {
            match load_entry(&path, device, layout) {
                Ok(entry) => entries.push(entry),
                Err(e) => eprintln!("warning: skipping {}: {e:#}", path.display()),
            }
        }

        if entries.is_empty() {
            panic!("no scenes loaded from {SCENES_DIR}; add a .ron file there");
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

    /// Re-read the current entry's file and rebuild it (static scene or world).
    /// Reloading a world resets its view to the spawn camera. On error the
    /// previously loaded entry is kept.
    pub fn reload_current(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
    ) -> Result<()> {
        let entry = &mut self.entries[self.current_index];
        let Some(path) = entry.path.clone() else {
            return Ok(());
        };
        let new_entry = load_entry(&path, device, layout)
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

/// Collect `*.ron` files from `dir`, sorted by filename for a stable order.
fn discover_scene_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "ron"))
        .collect();
    paths.sort();
    Ok(paths)
}

/// Load a `.ron` file as either a static scene or a procedural world, detected
/// from its contents (`World(( .. ))` vs a bare scene tuple).
fn load_entry(
    path: &Path,
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> Result<Entry> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let fallback_title = title_from_path(path);
    if scene_def::is_world_source(&text) {
        let def = WorldDef::from_ron(&text)?;
        let mut metadata = def.metadata();
        metadata.title.get_or_insert(fallback_title);
        let (cfg, camera) = def.build()?;
        Ok(Entry {
            initial_camera: camera.clone(),
            source: SceneSource::World(World::new(device, layout, cfg, camera)),
            path: Some(path.to_path_buf()),
            metadata,
        })
    } else {
        let def = scene_def::SceneDef::from_ron(&text)?;
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
            path: Some(path.to_path_buf()),
            metadata,
        })
    }
}

fn title_from_path(path: &Path) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("scene");
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
