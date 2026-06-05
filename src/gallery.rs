// Originally written in 2025 by Arman Uguray <arman.uguray@gmail.com>
// SPDX-License-Identifier: CC-BY-4.0

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::{
    camera::{Camera, CameraPath},
    scene_def::{self, SceneDef, WorldDef},
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
            match load_source(&path, device, layout) {
                Ok(source) => entries.push(Entry {
                    source,
                    path: Some(path),
                }),
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
        let source = load_source(&path, device, layout)
            .with_context(|| format!("reloading {}", path.display()))?;
        entry.source = source;
        Ok(())
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
fn load_source(
    path: &Path,
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> Result<SceneSource> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    if scene_def::is_world_source(&text) {
        let (cfg, camera) = WorldDef::from_ron(&text)?.build()?;
        Ok(SceneSource::World(World::new(device, layout, cfg, camera)))
    } else {
        let (camera, camera_path, builder) = SceneDef::from_ron(&text)?.build()?;
        Ok(SceneSource::Static(StaticScene {
            camera,
            resources: builder.build(device, layout),
            camera_path,
        }))
    }
}
