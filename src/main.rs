// Originally written in 2023 by Arman Uguray <arman.uguray@gmail.com>
// SPDX-License-Identifier: CC-BY-4.0

use {
    anyhow::{Context, Result},
    std::time::Instant,
    winit::{
        event::{DeviceEvent, ElementState, Event, MouseScrollDelta, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::Window,
    },
};

mod algebra;
mod audio;
mod camera;
mod gallery;
mod procgen;
mod render;
mod scene;
mod scene_def;
mod ui;
mod world;

const WIDTH: u32 = 1600;
const HEIGHT: u32 = 1200;

#[pollster::main]
#[allow(deprecated)]
async fn main() -> Result<()> {
    let audio_mode = parse_audio_mode()?;
    let event_loop = EventLoop::new()?;
    let window_size = winit::dpi::PhysicalSize::new(WIDTH, HEIGHT);
    let window = event_loop.create_window(
        Window::default_attributes()
            .with_inner_size(window_size)
            .with_resizable(false)
            .with_title("GPU Path Tracer".to_string()),
    )?;

    let (device, queue, surface, surface_format) = connect_to_gpu(&window).await?;
    let mut renderer = render::PathTracer::new(device, queue, WIDTH, HEIGHT, surface_format);
    let mut gallery =
        gallery::Gallery::new(renderer.device(), renderer.scene_group_layout());

    let audio_controls = audio::AudioControls::new(audio_mode);
    audio_controls.set_scene_mode(gallery.current_metadata().music);
    let mut music_ui = ui::MusicSettingsUi::new(
        &window,
        renderer.device(),
        surface_format,
        audio_controls.clone(),
    );

    // Start the ambient soundtrack. Keep the stream alive for the whole run
    // (dropping it stops playback); a missing audio device is non-fatal.
    let _audio_stream = match audio::start(audio_controls.clone()) {
        Ok(stream) => Some(stream),
        Err(e) => {
            eprintln!("audio disabled: {e:#}");
            None
        }
    };

    let mut left_mouse_button_pressed = false;
    let mut right_mouse_button_pressed = false;
    let mut key_w = false;
    let mut key_s = false;
    let mut key_a = false;
    let mut key_d = false;
    let mut path_playback: Option<std::time::Instant> = None;
    let mut auto_drift = false;
    apply_camera_recommendation(
        &gallery,
        &mut path_playback,
        &mut auto_drift,
        Instant::now(),
    );
    let mut last_frame = std::time::Instant::now();
    let mut last_interaction = std::time::Instant::now();
    let mut mouse_sensitivity = 1.0_f32;

    event_loop.run(|event, control_handle| {
        control_handle.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { event, .. } => {
                let egui_response = music_ui.on_window_event(&window, &event);
                match event {
                    WindowEvent::CloseRequested => control_handle.exit(),
                    WindowEvent::KeyboardInput {
                        device_id: _,
                        event,
                        ..
                    } => {
                        if egui_response.consumed {
                            return;
                        }
                        use winit::keyboard::{KeyCode, PhysicalKey};

                        let pressed = event.state == ElementState::Pressed;
                        match event.physical_key {
                            PhysicalKey::Code(KeyCode::KeyW) => key_w = pressed,
                            PhysicalKey::Code(KeyCode::KeyS) => key_s = pressed,
                            PhysicalKey::Code(KeyCode::KeyA) => key_a = pressed,
                            PhysicalKey::Code(KeyCode::KeyD) => key_d = pressed,
                            _ => {
                                if !pressed {
                                    return;
                                }
                                match event.physical_key {
                                    PhysicalKey::Code(KeyCode::ArrowUp) => {
                                        gallery
                                            .current_camera_mut()
                                            .adjust_fov(1_f32.to_radians());
                                        renderer.reset_samples();
                                        last_interaction = std::time::Instant::now();
                                    }
                                    PhysicalKey::Code(KeyCode::ArrowDown) => {
                                        gallery
                                            .current_camera_mut()
                                            .adjust_fov(-1_f32.to_radians());
                                        renderer.reset_samples();
                                        last_interaction = std::time::Instant::now();
                                    }
                                    PhysicalKey::Code(KeyCode::ArrowLeft) => {
                                        gallery.select_previous();
                                        audio_controls.set_scene_mode(
                                            gallery.current_metadata().music,
                                        );
                                        apply_camera_recommendation(
                                            &gallery,
                                            &mut path_playback,
                                            &mut auto_drift,
                                            std::time::Instant::now(),
                                        );
                                        renderer.reset_samples();
                                        last_interaction = std::time::Instant::now();
                                    }
                                    PhysicalKey::Code(KeyCode::ArrowRight) => {
                                        gallery.select_next();
                                        audio_controls.set_scene_mode(
                                            gallery.current_metadata().music,
                                        );
                                        apply_camera_recommendation(
                                            &gallery,
                                            &mut path_playback,
                                            &mut auto_drift,
                                            std::time::Instant::now(),
                                        );
                                        renderer.reset_samples();
                                        last_interaction = std::time::Instant::now();
                                    }
                                    PhysicalKey::Code(KeyCode::KeyP) => {
                                        if path_playback.is_some() {
                                            path_playback = None;
                                        } else if gallery.current_path().is_some() {
                                            path_playback = Some(std::time::Instant::now());
                                        }
                                        last_interaction = std::time::Instant::now();
                                    }
                                    PhysicalKey::Code(KeyCode::KeyR) => {
                                        match gallery.reload_current(
                                            renderer.device(),
                                            renderer.scene_group_layout(),
                                        ) {
                                            Ok(()) => {
                                                audio_controls.set_scene_mode(
                                                    gallery.current_metadata().music,
                                                );
                                                apply_camera_recommendation(
                                                    &gallery,
                                                    &mut path_playback,
                                                    &mut auto_drift,
                                                    std::time::Instant::now(),
                                                );
                                                renderer.reset_samples();
                                                last_interaction = std::time::Instant::now();
                                            }
                                            Err(e) => eprintln!("reload failed: {e:#}"),
                                        }
                                    }
                                    PhysicalKey::Code(KeyCode::KeyF) => {
                                        // Toggle slow automatic forward drift, for
                                        // hands-off streaming walkthroughs.
                                        auto_drift = !auto_drift;
                                        last_interaction = std::time::Instant::now();
                                    }
                                    _ => (),
                                }
                            }
                        }
                    }
                    WindowEvent::MouseInput {
                        device_id: _,
                        state,
                        button,
                    } => {
                        if egui_response.consumed {
                            return;
                        }
                        use winit::event::MouseButton;

                        let pressed = state == ElementState::Pressed;
                        match button {
                            MouseButton::Left => left_mouse_button_pressed = pressed,
                            MouseButton::Right => right_mouse_button_pressed = pressed,
                            _ => (),
                        }
                        if pressed {
                            last_interaction = std::time::Instant::now();
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        let now = std::time::Instant::now();
                        if let Some(start) = path_playback {
                            let elapsed = start.elapsed().as_secs_f32();
                            if let Some(cam) =
                                gallery.current_path().map(|p| p.camera_at(elapsed))
                            {
                                *gallery.current_camera_mut() = cam;
                                renderer.reset_samples();
                                last_interaction = now;
                            }
                        }

                        // Frame-rate-independent movement: scale speeds by the time
                        // since the last frame (world units per second).
                        let dt = now.duration_since(last_frame).as_secs_f32();
                        last_frame = now;
                        const MOVE_SPEED: f32 = 3.0;
                        const STRAFE_SPEED: f32 = 3.0;
                        const DRIFT_SPEED: f32 = 2.0;
                        let camera = gallery.current_camera_mut();
                        let mut moved = false;
                        if key_w {
                            camera.fly(-MOVE_SPEED * dt);
                            moved = true;
                        }
                        if key_s {
                            camera.fly(MOVE_SPEED * dt);
                            moved = true;
                        }
                        if key_a {
                            camera.pan(-STRAFE_SPEED * dt, 0.0);
                            moved = true;
                        }
                        if key_d {
                            camera.pan(STRAFE_SPEED * dt, 0.0);
                            moved = true;
                        }
                        if auto_drift {
                            camera.fly(-DRIFT_SPEED * dt);
                            moved = true;
                        }
                        if moved {
                            renderer.reset_samples();
                            last_interaction = now;
                        }

                        // Stream procedural chunks around the (possibly moved) camera.
                        // Rebuilds the scene buffers only on chunk-boundary crossings;
                        // reset accumulation when that happens.
                        if gallery
                            .update_world(renderer.device(), renderer.scene_group_layout())
                        {
                            renderer.reset_samples();
                            last_interaction = now;
                        }

                        let performance = music_ui.performance_settings();
                        let ui_active = music_ui.panel_open() || music_ui.captures_pointer();
                        let waiting_for_full_quality =
                            now.duration_since(last_interaction).as_secs_f32()
                                < performance.full_quality_delay;
                        let interactive_preview = ui_active || waiting_for_full_quality;
                        let render_scale =
                            if performance.dynamic_resolution && interactive_preview {
                                performance.interactive_scale
                            } else {
                                1.0
                            };
                        renderer.set_render_scale(render_scale);

                        let (
                            interactive_samples,
                            post_strength,
                            post_radius,
                            edge_threshold,
                        ) = match performance.motion_quality {
                            ui::MotionQuality::Fast => (2, 0.18, 1.0, 0.20),
                            ui::MotionQuality::Balanced => (4, 0.34, 1.0, 0.16),
                            ui::MotionQuality::Pretty => (6, 0.48, 1.25, 0.13),
                        };
                        let samples_per_frame = if interactive_preview {
                            interactive_samples
                        } else {
                            1
                        };
                        renderer.set_samples_per_frame(samples_per_frame);
                        renderer.set_post_filter(
                            if interactive_preview {
                                post_strength
                            } else {
                                0.0
                            },
                            post_radius,
                            edge_threshold,
                        );

                        let frame = match surface.get_current_texture() {
                            wgpu::CurrentSurfaceTexture::Success(frame)
                            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
                            wgpu::CurrentSurfaceTexture::Timeout
                            | wgpu::CurrentSurfaceTexture::Occluded => {
                                window.request_redraw();
                                return;
                            }
                            wgpu::CurrentSurfaceTexture::Outdated
                            | wgpu::CurrentSurfaceTexture::Lost => {
                                eprintln!("surface changed; restart the app to recreate it");
                                control_handle.exit();
                                return;
                            }
                            wgpu::CurrentSurfaceTexture::Validation => {
                                panic!("surface validation error");
                            }
                        };

                        let render_target = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());

                        let mut encoder = renderer.device().create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("render frame"),
                            },
                        );

                        renderer.encode_frame(
                            &mut encoder,
                            gallery.current_camera(),
                            gallery.current_resources(),
                            &render_target,
                        );

                        let mut command_buffers = music_ui.render(
                            &window,
                            renderer.device(),
                            renderer.queue(),
                            &mut encoder,
                            &render_target,
                            &ui::AppUiState {
                                scenes: gallery.scene_options(),
                                current_scene: gallery.current_index(),
                                scene_description: gallery
                                    .current_metadata()
                                    .description
                                    .clone(),
                                scene_music: gallery.current_metadata().music,
                                scene_camera_mode: gallery.current_metadata().camera_mode,
                                path_available: gallery.current_path().is_some(),
                                path_playing: path_playback.is_some(),
                                auto_drift,
                                fov_degrees: gallery.current_camera().fov_y().to_degrees(),
                                mouse_sensitivity,
                            },
                        );
                        if music_ui.take_interacted() {
                            last_interaction = std::time::Instant::now();
                        }
                        let ui_actions = music_ui.take_actions();
                        command_buffers.push(encoder.finish());
                        renderer.queue().submit(command_buffers);

                        frame.present();
                        if apply_ui_actions(
                            ui_actions,
                            &mut gallery,
                            &audio_controls,
                            &mut renderer,
                            &mut path_playback,
                            &mut auto_drift,
                            &mut mouse_sensitivity,
                            now,
                        ) {
                            last_interaction = std::time::Instant::now();
                        }
                        window.request_redraw();
                    }
                    _ => (),
                }
            }
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseWheel { delta } => {
                    if music_ui.captures_pointer() {
                        return;
                    }
                    let delta = match delta {
                        MouseScrollDelta::PixelDelta(delta) => 0.001 * delta.y as f32,
                        MouseScrollDelta::LineDelta(_, y) => y * 0.1,
                    };
                    gallery.current_camera_mut().zoom(delta);
                    renderer.reset_samples();
                    last_interaction = std::time::Instant::now();
                }
                DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                    if music_ui.captures_pointer() {
                        return;
                    }
                    let dx = dx as f32 * 0.01 * mouse_sensitivity;
                    let dy = dy as f32 * -0.01 * mouse_sensitivity;
                    if left_mouse_button_pressed {
                        gallery.current_camera_mut().orbit(dx, dy);
                        renderer.reset_samples();
                        last_interaction = std::time::Instant::now();
                    }
                    if right_mouse_button_pressed {
                        gallery.current_camera_mut().pan(dx, dy);
                        renderer.reset_samples();
                        last_interaction = std::time::Instant::now();
                    }
                }
                _ => (),
            },
            _ => (),
        }
    })?;
    Ok(())
}

fn apply_camera_recommendation(
    gallery: &gallery::Gallery,
    path_playback: &mut Option<Instant>,
    auto_drift: &mut bool,
    now: Instant,
) {
    match gallery
        .current_metadata()
        .camera_mode
        .unwrap_or(scene_def::CameraMode::Manual)
    {
        scene_def::CameraMode::Manual => {
            *path_playback = None;
            *auto_drift = false;
        }
        scene_def::CameraMode::Path => {
            *path_playback = gallery.current_path().map(|_| now);
            *auto_drift = false;
        }
        scene_def::CameraMode::Drift => {
            *path_playback = None;
            *auto_drift = true;
        }
    }
}

fn apply_ui_actions(
    actions: ui::UiActions,
    gallery: &mut gallery::Gallery,
    audio_controls: &audio::AudioControls,
    renderer: &mut render::PathTracer,
    path_playback: &mut Option<Instant>,
    auto_drift: &mut bool,
    mouse_sensitivity: &mut f32,
    now: Instant,
) -> bool {
    let mut interacted = false;

    if let Some(index) = actions.select_scene {
        if index != gallery.current_index() {
            gallery.select_index(index);
            audio_controls.set_scene_mode(gallery.current_metadata().music);
            apply_camera_recommendation(gallery, path_playback, auto_drift, now);
            renderer.reset_samples();
            interacted = true;
        }
    }

    if actions.toggle_path {
        if path_playback.is_some() {
            *path_playback = None;
        } else if gallery.current_path().is_some() {
            *path_playback = Some(now);
            *auto_drift = false;
        }
        renderer.reset_samples();
        interacted = true;
    }

    if actions.reset_camera {
        gallery.reset_current_camera();
        if path_playback.is_some() && gallery.current_path().is_some() {
            *path_playback = Some(now);
        }
        renderer.reset_samples();
        interacted = true;
    }

    if let Some(enabled) = actions.set_auto_drift {
        *auto_drift = enabled;
        if enabled {
            *path_playback = None;
        }
        renderer.reset_samples();
        interacted = true;
    }

    if let Some(fov) = actions.set_fov_degrees {
        gallery.current_camera_mut().set_fov(fov.to_radians());
        renderer.reset_samples();
        interacted = true;
    }

    if let Some(value) = actions.set_mouse_sensitivity {
        *mouse_sensitivity = value.clamp(0.25, 3.0);
        interacted = true;
    }

    interacted
}

fn parse_audio_mode() -> Result<audio::AudioMode> {
    let mut args = std::env::args().skip(1);
    let mut audio_mode = audio::AudioMode::Auto;

    while let Some(arg) = args.next() {
        let value = if let Some(value) = arg.strip_prefix("--audio-mode=") {
            Some(value.to_string())
        } else if let Some(value) = arg.strip_prefix("--mode=") {
            Some(value.to_string())
        } else if arg == "--audio-mode" || arg == "--mode" {
            Some(
                args.next()
                    .context("--audio-mode requires one of: auto, dorian, lydian, aeolian, mixolydian, ionian")?,
            )
        } else {
            anyhow::bail!(
                "unknown argument {arg:?}; use --audio-mode <{}>",
                audio::AudioMode::names()
            );
        };

        if let Some(value) = value {
            audio_mode = audio::AudioMode::parse(&value).with_context(|| {
                format!(
                    "unknown audio mode {value:?}; use one of: {}",
                    audio::AudioMode::names()
                )
            })?;
        }
    }

    Ok(audio_mode)
}

async fn connect_to_gpu(
    window: &Window,
) -> Result<(
    wgpu::Device,
    wgpu::Queue,
    wgpu::Surface<'_>,
    wgpu::TextureFormat,
)> {
    use wgpu::TextureFormat::{Bgra8Unorm, Rgba8Unorm};

    // Create an "instance" of wgpu. This is the entry-point to the API.
    let instance = wgpu::Instance::default();

    // Create a drawable "surface" that is associated with the window.
    let surface = instance.create_surface(window)?;

    // Request a GPU that is compatible with the surface. If the system has multiple GPUs then
    // pick the high performance one.
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .context("failed to find a compatible adapter")?;

    // Connect to the GPU. "device" represents the connection to the GPU and allows us to create
    // resources like buffers, textures, and pipelines. "queue" represents the command queue that
    // we use to submit commands to the GPU.
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .context("failed to connect to the GPU")?;

    // Configure the texture memory backing the surface. Our renderer will draw to a surface
    // texture every frame.
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .into_iter()
        .find(|it| matches!(it, Rgba8Unorm | Bgra8Unorm))
        .context("could not find preferred texture format (Rgba8Unorm or Bgra8Unorm)")?;
    let size = window.inner_size();
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 3,
    };
    surface.configure(&device, &config);

    Ok((device, queue, surface, format))
}
