use crate::{
    audio::{AudioControls, AudioMode},
    gallery::SceneOption,
    scene_def::CameraMode,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MotionQuality {
    Fast,
    Balanced,
    Pretty,
}

impl MotionQuality {
    pub fn all() -> &'static [Self] {
        &[Self::Fast, Self::Balanced, Self::Pretty]
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Fast => "Fast",
            Self::Balanced => "Balanced",
            Self::Pretty => "Pretty",
        }
    }
}

#[derive(Clone, Copy)]
pub struct PerformanceSettings {
    pub dynamic_resolution: bool,
    pub interactive_scale: f32,
    pub motion_quality: MotionQuality,
    pub full_quality_delay: f32,
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            dynamic_resolution: false,
            interactive_scale: 0.5,
            motion_quality: MotionQuality::Balanced,
            full_quality_delay: 0.5,
        }
    }
}

pub struct MusicSettingsUi {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    controls: AudioControls,
    performance: PerformanceSettings,
    panel_open: bool,
    interacted: bool,
    actions: UiActions,
}

#[derive(Clone)]
pub struct AppUiState {
    pub scenes: Vec<SceneOption>,
    pub current_scene: usize,
    pub scene_description: String,
    pub scene_music: Option<AudioMode>,
    pub scene_camera_mode: Option<CameraMode>,
    pub path_available: bool,
    pub path_playing: bool,
    pub auto_drift: bool,
    pub fov_degrees: f32,
    pub mouse_sensitivity: f32,
}

#[derive(Default)]
pub struct UiActions {
    pub select_scene: Option<usize>,
    pub toggle_path: bool,
    pub reset_camera: bool,
    pub screenshot: bool,
    pub set_auto_drift: Option<bool>,
    pub set_fov_degrees: Option<f32>,
    pub set_mouse_sensitivity: Option<f32>,
}

impl MusicSettingsUi {
    pub fn new(
        window: &winit::window::Window,
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        controls: AudioControls,
    ) -> Self {
        let ctx = egui::Context::default();
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        );
        let renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );

        Self {
            ctx,
            state,
            renderer,
            controls,
            performance: PerformanceSettings::default(),
            panel_open: false,
            interacted: false,
            actions: UiActions::default(),
        }
    }

    pub fn on_window_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> egui_winit::EventResponse {
        self.state.on_window_event(window, event)
    }

    pub fn captures_pointer(&self) -> bool {
        self.ctx.egui_wants_pointer_input() || self.ctx.is_pointer_over_egui()
    }

    pub fn panel_open(&self) -> bool {
        self.panel_open
    }

    pub fn performance_settings(&self) -> PerformanceSettings {
        self.performance
    }

    pub fn take_interacted(&mut self) -> bool {
        std::mem::take(&mut self.interacted)
    }

    pub fn take_actions(&mut self) -> UiActions {
        std::mem::take(&mut self.actions)
    }

    pub fn render(
        &mut self,
        window: &winit::window::Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        app: &AppUiState,
    ) -> Vec<wgpu::CommandBuffer> {
        let raw_input = self.state.take_egui_input(window);
        let ctx = self.ctx.clone();
        let output = ctx.run_ui(raw_input, |ui| self.show(ui.ctx(), app));
        self.state
            .handle_platform_output(window, output.platform_output);

        for (id, image_delta) in &output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, *id, image_delta);
        }

        let pixels_per_point = self.ctx.pixels_per_point();
        let paint_jobs = self.ctx.tessellate(output.shapes, pixels_per_point);
        let size = window.inner_size();
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point,
        };

        let extra_command_buffers = self.renderer.update_buffers(
            device,
            queue,
            encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            self.renderer.render(
                &mut render_pass.forget_lifetime(),
                &paint_jobs,
                &screen_descriptor,
            );
        }

        for id in &output.textures_delta.free {
            self.renderer.free_texture(id);
        }

        extra_command_buffers
    }

    fn show(&mut self, ctx: &egui::Context, app: &AppUiState) {
        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.panel_open = false;
        }

        egui::Area::new("music_settings_gear".into())
            .anchor(egui::Align2::RIGHT_TOP, [-16.0, 16.0])
            .show(ctx, |ui| {
                if ui.button("⚙").on_hover_text("Settings").clicked() {
                    self.panel_open = !self.panel_open;
                    self.interacted = true;
                }
            });

        if self.panel_open {
            let mut open = true;
            egui::Window::new("Settings")
                .anchor(egui::Align2::RIGHT_TOP, [-16.0, 56.0])
                .collapsible(false)
                .resizable(false)
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.set_min_width(260.0);

                    ui.label("Scene");
                    let mut scene_index = app.current_scene;
                    let selected_title = app
                        .scenes
                        .iter()
                        .find(|scene| scene.index == app.current_scene)
                        .map(|scene| scene.title.as_str())
                        .unwrap_or("Scene");
                    egui::ComboBox::from_label("Scene")
                        .selected_text(selected_title)
                        .show_ui(ui, |ui| {
                            for scene in &app.scenes {
                                ui.selectable_value(
                                    &mut scene_index,
                                    scene.index,
                                    scene.title.as_str(),
                                );
                            }
                        });
                    if scene_index != app.current_scene {
                        self.actions.select_scene = Some(scene_index);
                        self.interacted = true;
                    }
                    if !app.scene_description.is_empty() {
                        ui.label(app.scene_description.as_str());
                    }
                    let music_label =
                        app.scene_music.map(AudioMode::label).unwrap_or("auto");
                    let camera_label = app
                        .scene_camera_mode
                        .map(CameraMode::label)
                        .unwrap_or("manual");
                    ui.label(format!("Recommended: {music_label} / {camera_label}"));

                    ui.horizontal(|ui| {
                        let path_label = if app.path_playing {
                            "Pause path"
                        } else {
                            "Play path"
                        };
                        if ui
                            .add_enabled(app.path_available, egui::Button::new(path_label))
                            .clicked()
                        {
                            self.actions.toggle_path = true;
                            self.interacted = true;
                        }
                        if ui.button("Reset camera").clicked() {
                            self.actions.reset_camera = true;
                            self.interacted = true;
                        }
                        if ui.button("Screenshot").clicked() {
                            self.actions.screenshot = true;
                            self.interacted = true;
                        }
                    });
                    let mut auto_drift = app.auto_drift;
                    if ui.checkbox(&mut auto_drift, "Auto drift").changed() {
                        self.actions.set_auto_drift = Some(auto_drift);
                        self.interacted = true;
                    }

                    let mut fov = app.fov_degrees;
                    if ui
                        .add(egui::Slider::new(&mut fov, 10.0..=120.0).text("FOV"))
                        .changed()
                    {
                        self.actions.set_fov_degrees = Some(fov);
                        self.interacted = true;
                    }

                    let mut mouse_sensitivity = app.mouse_sensitivity;
                    if ui
                        .add(
                            egui::Slider::new(&mut mouse_sensitivity, 0.25..=3.0)
                                .text("Mouse sensitivity"),
                        )
                        .changed()
                    {
                        self.actions.set_mouse_sensitivity = Some(mouse_sensitivity);
                        self.interacted = true;
                    }

                    ui.separator();
                    ui.label("Music");

                    let mut mode = self.controls.mode();
                    egui::ComboBox::from_label("Mode")
                        .selected_text(mode.label())
                        .show_ui(ui, |ui| {
                            for &candidate in AudioMode::all() {
                                ui.selectable_value(&mut mode, candidate, candidate.label());
                            }
                        });
                    if mode != self.controls.mode() {
                        self.controls.set_mode(mode);
                        self.interacted = true;
                    }

                    let mut volume = self.controls.volume();
                    if ui
                        .add(egui::Slider::new(&mut volume, 0.0..=2.0).text("Volume"))
                        .changed()
                    {
                        self.controls.set_volume(volume);
                        self.interacted = true;
                    }

                    let mut arpeggio = self.controls.arpeggio();
                    if ui
                        .add(egui::Slider::new(&mut arpeggio, 0.0..=2.0).text("Arpeggio"))
                        .changed()
                    {
                        self.controls.set_arpeggio(arpeggio);
                        self.interacted = true;
                    }

                    let mut warmth = self.controls.warmth();
                    if ui
                        .add(egui::Slider::new(&mut warmth, 0.0..=1.0).text("Warmth"))
                        .changed()
                    {
                        self.controls.set_warmth(warmth);
                        self.interacted = true;
                    }

                    let mut bass = self.controls.bass();
                    if ui
                        .add(egui::Slider::new(&mut bass, 0.0..=1.0).text("Bass"))
                        .changed()
                    {
                        self.controls.set_bass(bass);
                        self.interacted = true;
                    }

                    let mut shimmer = self.controls.shimmer();
                    if ui
                        .add(egui::Slider::new(&mut shimmer, 0.0..=1.0).text("Shimmer"))
                        .changed()
                    {
                        self.controls.set_shimmer(shimmer);
                        self.interacted = true;
                    }

                    ui.separator();
                    ui.label("Performance");

                    let mut motion_quality = self.performance.motion_quality;
                    egui::ComboBox::from_label("Motion quality")
                        .selected_text(motion_quality.label())
                        .show_ui(ui, |ui| {
                            for &candidate in MotionQuality::all() {
                                ui.selectable_value(
                                    &mut motion_quality,
                                    candidate,
                                    candidate.label(),
                                );
                            }
                        });
                    if motion_quality != self.performance.motion_quality {
                        self.performance.motion_quality = motion_quality;
                        self.interacted = true;
                    }

                    if ui
                        .checkbox(
                            &mut self.performance.dynamic_resolution,
                            "Dynamic resolution",
                        )
                        .changed()
                    {
                        self.interacted = true;
                    }

                    if self.performance.dynamic_resolution
                        && ui
                            .add(
                                egui::Slider::new(
                                    &mut self.performance.interactive_scale,
                                    0.35..=1.0,
                                )
                                .text("Interactive scale"),
                            )
                            .changed()
                    {
                        self.interacted = true;
                    }

                    if ui
                        .add(
                            egui::Slider::new(
                                &mut self.performance.full_quality_delay,
                                0.0..=2.0,
                            )
                            .text("Settle delay"),
                        )
                        .changed()
                    {
                        self.interacted = true;
                    }
                });

            if !open {
                self.panel_open = false;
                self.interacted = true;
            }
        }
    }
}
