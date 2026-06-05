# AGENTS.md

## Commands

```bash
cargo run            # debug build (fast compile, slow render)
cargo run --release  # optimized build (recommended for rendering)
cargo build          # compile without running
cargo fmt            # format (max line width: 93, see rustfmt.toml)
cargo clippy         # lint
```

No tests exist in this project.

## Architecture

A WebGPU path tracer written in Rust using **wgpu** + **winit**. The CPU side manages the window, camera, and GPU resources; all ray tracing happens in a single WGSL compute/fragment shader pass.

### Data flow per frame

1. `main.rs` handles input events and calls `renderer.render_frame(&scene.camera, &scene.resources, &render_target)`
2. `render.rs / PathTracer` uploads updated `Uniforms` (camera + frame count) to the GPU, runs a render pass using the path tracer pipeline, and submits the command buffer
3. The shader (`shaders.wgsl`) reads from two bind groups:
   - **Group 0** (`render_bind_group`): uniforms buffer + two ping-pong radiance sample textures
   - **Group 1** (`scene_bind_group`): materials + spheres storage buffers, plus the uniform-grid accelerator (header + cell ranges + sphere indices) — see "Procedural streaming worlds" below
4. The shader accumulates samples across frames using `frame_count` — resetting samples (via `renderer.reset_samples()`) zeroes this counter and restarts convergence

### Key design decisions

**Ping-pong accumulation**: Two `Rgba32Float` textures alternate as current/previous sample buffers. The shader blends the new sample with the running average using `frame_count`.

**Material encoding**: `material.metallic_or_ior` encodes three material types in one f32 — `0.0` = pure Lambertian, `> 0` = opaque with metallic factor, `< 0` = dielectric with IOR stored as negative value. See `scene.rs` and the shader's material handling.

**Camera model**: Orbit camera (`camera.rs`) with spherical coordinates (azimuth, altitude, distance, center). `fly()` translates both origin and center along the look direction for walk-through movement. `pan()` translates laterally. `orbit()` rotates in place.

**Camera paths**: `CameraPath` in `camera.rs` holds a list of `CameraKeyframe`s and linearly interpolates between them. Paths are defined per-scene in `gallery.rs`. Press **P** to toggle playback.

### Controls

| Input      | Action                                     |
| ---------- | ------------------------------------------ |
| W / S      | Fly forward / backward                     |
| A / D      | Strafe left / right                        |
| Left-drag  | Orbit around center                        |
| Right-drag | Pan                                        |
| Scroll     | Zoom                                       |
| ↑ / ↓      | Adjust FOV                                 |
| ← / →      | Switch scenes / worlds                     |
| P          | Toggle camera path playback                |
| R          | Hot-reload the current scene/world `.ron`  |
| F          | Toggle auto-drift (hands-off forward walk) |

### Adding a new scene

Scenes are data-driven `.ron` files in `scenes/`, loaded by `gallery.rs` and parsed in `scene_def.rs`. A static scene is a bare `( materials: {...}, objects: [...], camera: ..., path: ... )` tuple; objects include `Sphere`, `Ground`, `Grid`, `Ring`, and `Room` generators. Edit a file and press **R** to hot-reload.

### Procedural streaming worlds

A `.ron` file wrapped in `World(( .. ))` defines an endless procedural world (`scene_def::WorldDef` → `world::WorldConfig`). `world.rs` streams chunks around the camera (regenerating GPU buffers only on chunk-boundary crossings), with geometry from `procgen.rs` (deterministic per-chunk RNG + value noise; `Scatter`/`Terrain`/`Architecture`/`Swarm` biomes) and a shared material palette referenced by index. Fly with WASD or toggle **F** to drift.

**Acceleration**: intersection uses a 2D uniform grid over XZ (`GridHeader` + `cell_ranges` + `sphere_indices`), traversed by DDA in `shaders.wgsl`. Static scenes use a degenerate single-cell grid (`nx == nz == 1`), which is exactly brute force. The scene bind group (**Group 1**) is now five storage buffers: materials(0), spheres(1), grid header(2), cell ranges(3), sphere indices(4).
