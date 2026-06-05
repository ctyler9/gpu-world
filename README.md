# GPU Path Tracer

A real-time path tracer written in Rust with **wgpu** (GPU) and **winit**
(windowing), featuring data-driven scenes, endless procedurally-generated
streaming worlds, and a generative ambient soundtrack (**FunDSP** + **cpal**).

## Prerequisites

- **Rust** (stable) — install via [rustup](https://rustup.rs).
- **A GPU with Vulkan/Metal/DX12 support** and up-to-date drivers (wgpu).
- The system libraries below.

### System libraries

| Library | Why | Needed at |
|---|---|---|
| **ALSA** dev headers | `cpal` audio output (links `alsa-sys`) | build time (Linux) |
| **Vulkan** loader + GPU driver | `wgpu` rendering backend | run time (Linux) |
| Wayland or X11 session | `winit` window | run time (Linux) |

> On **macOS/Windows** no extra system libraries are required — wgpu uses
> Metal/DX12 and cpal uses CoreAudio/WASAPI out of the box.

#### Install (Linux)

```bash
# Fedora / RHEL
sudo dnf install alsa-lib-devel vulkan-loader mesa-vulkan-drivers

# Debian / Ubuntu
sudo apt install libasound2-dev libvulkan1 mesa-vulkan-drivers

# Arch
sudo pacman -S alsa-lib vulkan-icd-loader mesa
```

The **ALSA development headers** (`alsa-lib-devel` / `libasound2-dev` /
`alsa-lib`) are the one strict *build-time* requirement added by the audio
feature — without them the `alsa-sys` build script fails. If you have no audio
device, the app still runs and just prints `audio disabled: …`.

## Building & running

```bash
cargo run --release   # optimized; recommended (path tracing is heavy)
cargo run             # debug build (slow render)
cargo test            # run the unit tests
```

## Controls

| Input | Action |
|---|---|
| ← / → | Switch between scenes / worlds |
| W / S | Fly forward / backward |
| A / D | Strafe left / right |
| Left-drag | Look around |
| Right-drag | Pan |
| Scroll | Zoom |
| ↑ / ↓ | Adjust field of view |
| F | Toggle auto-drift (hands-off forward walk) |
| P | Toggle camera path playback (static scenes) |
| R | Hot-reload the current scene/world `.ron` |

Static scenes live in `scenes/*.ron`; the procedural worlds are the
`World(( .. ))` files (`scenes/10_world_*` … `13_world_*`). See `AGENTS.md` for
architecture details.
