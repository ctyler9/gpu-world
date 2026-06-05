use crate::audio::{AudioControls, AudioMode};

pub struct MusicSettingsUi {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    controls: AudioControls,
    panel_open: bool,
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
            panel_open: false,
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

    pub fn render(
        &mut self,
        window: &winit::window::Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) -> Vec<wgpu::CommandBuffer> {
        let raw_input = self.state.take_egui_input(window);
        let ctx = self.ctx.clone();
        let output = ctx.run_ui(raw_input, |ui| self.show(ui.ctx()));
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

    fn show(&mut self, ctx: &egui::Context) {
        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.panel_open = false;
        }

        egui::Area::new("music_settings_gear".into())
            .anchor(egui::Align2::RIGHT_TOP, [-16.0, 16.0])
            .show(ctx, |ui| {
                if ui.button("⚙").on_hover_text("Music settings").clicked() {
                    self.panel_open = !self.panel_open;
                }
            });

        if self.panel_open {
            let mut open = true;
            egui::Window::new("Music")
                .anchor(egui::Align2::RIGHT_TOP, [-16.0, 56.0])
                .collapsible(false)
                .resizable(false)
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.set_min_width(260.0);

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
                    }

                    let mut volume = self.controls.volume();
                    if ui
                        .add(egui::Slider::new(&mut volume, 0.0..=2.0).text("Volume"))
                        .changed()
                    {
                        self.controls.set_volume(volume);
                    }

                    let mut arpeggio = self.controls.arpeggio();
                    if ui
                        .add(egui::Slider::new(&mut arpeggio, 0.0..=2.0).text("Arpeggio"))
                        .changed()
                    {
                        self.controls.set_arpeggio(arpeggio);
                    }

                    let mut warmth = self.controls.warmth();
                    if ui
                        .add(egui::Slider::new(&mut warmth, 0.0..=1.0).text("Warmth"))
                        .changed()
                    {
                        self.controls.set_warmth(warmth);
                    }
                });

            if !open {
                self.panel_open = false;
            }
        }
    }
}
