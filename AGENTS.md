# AGENTS.md

## Workflow

After every change: run `cargo fmt`, then `cargo clippy`, then commit.

## Commands

```bash
cargo run            # debug build (fast compile, slow render)
cargo run --release  # optimized build (recommended for rendering)
cargo build          # compile without running
cargo fmt            # format (max line width: 93, see rustfmt.toml)
```

No tests exist in this project.

## Architecture

A WebGPU path tracer written in Rust using **wgpu** + **winit**. The CPU side manages the window, camera, and GPU resources; all ray tracing happens in a single WGSL compute/fragment shader pass.

### Data flow per frame

1. `main.rs` handles input events and calls `renderer.render_frame(&scene.camera, &scene.resources, &render_target)`
2. `render.rs / PathTracer` uploads updated `Uniforms` (camera + frame count) to the GPU, runs a render pass using the path tracer pipeline, and submits the command buffer
3. The shader (`shaders.wgsl`) reads from two bind groups:
   - **Group 0** (`render_bind_group`): uniforms buffer + two ping-pong radiance sample textures
   - **Group 1** (`scene_bind_group`): materials storage buffer + spheres storage buffer
4. The shader accumulates samples across frames using `frame_count` — resetting samples (via `renderer.reset_samples()`) zeroes this counter and restarts convergence

### Key design decisions

**Ping-pong accumulation**: Two `Rgba32Float` textures alternate as current/previous sample buffers. The shader blends the new sample with the running average using `frame_count`.

**Material encoding**: `material.metallic_or_ior` encodes three material types in one f32 — `0.0` = pure Lambertian, `> 0` = opaque with metallic factor, `< 0` = dielectric with IOR stored as negative value. See `scene.rs` and the shader's material handling.

**Camera model**: Orbit camera (`camera.rs`) with spherical coordinates (azimuth, altitude, distance, center). `fly()` translates both origin and center along the look direction for walk-through movement. `pan()` translates laterally. `orbit()` rotates in place.

**Camera paths**: `CameraPath` in `camera.rs` holds a list of `CameraKeyframe`s and linearly interpolates between them. Paths are defined per-scene in `gallery.rs`. Press **P** to toggle playback.

### Controls

| Input      | Action                      |
| ---------- | --------------------------- |
| W / S      | Fly forward / backward      |
| A / D      | Strafe left / right         |
| Left-drag  | Orbit around center         |
| Right-drag | Pan                         |
| Scroll     | Zoom                        |
| ↑ / ↓      | Adjust FOV                  |
| ← / →      | Switch scenes               |
| P          | Toggle camera path playback |

### Adding a new scene

Add a function returning `(Camera, Option<CameraPath>, SceneBuilder)` in `gallery.rs` and register it in `Gallery::new`. Use `SceneBuilder::add_material` / `add_sphere` to populate geometry. The scene's GPU resources are a `wgpu::BindGroup` built from `SceneBuilder::build`.
