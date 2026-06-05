// Originally written in 2024 by Arman Uguray <arman.uguray@gmail.com>
// SPDX-License-Identifier: CC-BY-4.0

use {
    bytemuck::{Pod, Zeroable},
    std::f32::consts::{FRAC_PI_2, PI},
};

use crate::algebra::Vec3;

const DEFAULT_FOV_Y: f32 = 45_f32.to_radians();
const MIN_FOV_Y: f32 = 10_f32.to_radians();
const MAX_FOV_Y: f32 = 170_f32.to_radians();

#[derive(Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct CameraUniforms {
    origin: Vec3,
    fov_y: f32,
    u: Vec3,
    _pad1: u32,
    v: Vec3,
    _pad2: u32,
    w: Vec3,
    _pad3: u32,
}

pub struct CameraKeyframe {
    pub origin: Vec3,
    pub center: Vec3,
    pub up: Vec3,
    pub fov_y: f32,
}

pub struct CameraPath {
    pub keyframes: Vec<CameraKeyframe>,
    /// Seconds to travel between each adjacent keyframe pair.
    pub secs_per_segment: f32,
    pub looping: bool,
}

impl CameraPath {
    /// A looping circular orbit around `center` at the given `radius` and
    /// `height`, sampled into `steps` segments. Saves hand-writing keyframes for
    /// the common "fly around the subject" shot.
    pub fn orbit(
        center: Vec3,
        radius: f32,
        height: f32,
        fov_y: f32,
        secs_per_segment: f32,
        steps: usize,
    ) -> Self {
        let up = Vec3::new(0., 1., 0.);
        let steps = steps.max(1);
        let keyframes = (0..=steps)
            .map(|i| {
                let angle = i as f32 / steps as f32 * std::f32::consts::TAU;
                CameraKeyframe {
                    origin: center
                        + Vec3::new(angle.cos() * radius, height, angle.sin() * radius),
                    center,
                    up,
                    fov_y,
                }
            })
            .collect();
        Self {
            keyframes,
            secs_per_segment,
            looping: true,
        }
    }

    pub fn camera_at(&self, elapsed_secs: f32) -> Camera {
        let n = self.keyframes.len();
        assert!(n >= 1);
        if n == 1 {
            let k = &self.keyframes[0];
            return Camera::look_at(k.origin, k.center, k.up).with_fov(k.fov_y);
        }
        let segments = (n - 1) as f32;
        let t = if self.looping {
            (elapsed_secs / self.secs_per_segment) % segments
        } else {
            (elapsed_secs / self.secs_per_segment).min(segments)
        };
        let seg = (t as usize).min(n - 2);
        let frac = t - seg as f32;
        let a = &self.keyframes[seg];
        let b = &self.keyframes[seg + 1];
        Camera::look_at(
            a.origin + (b.origin - a.origin) * frac,
            a.center + (b.center - a.center) * frac,
            a.up,
        )
        .with_fov(a.fov_y + (b.fov_y - a.fov_y) * frac)
    }
}

#[derive(Clone)]
pub struct Camera {
    uniforms: CameraUniforms,
    center: Vec3,
    up: Vec3,
    distance: f32,
    azimuth: f32,
    altitude: f32,
}

impl Camera {
    pub fn look_at(origin: Vec3, center: Vec3, up: Vec3) -> Camera {
        let center_to_origin = origin - center;
        let distance = center_to_origin.length().max(0.01); // Prevent distance of 0
        let neg_w = center_to_origin.normalized();
        let azimuth = neg_w.x().atan2(neg_w.z());
        let altitude = neg_w.y().asin();
        Self::from_spherical_coords(center, up, distance, azimuth, altitude)
    }

    pub fn from_spherical_coords(
        center: Vec3,
        up: Vec3,
        distance: f32,
        azimuth: f32,
        altitude: f32,
    ) -> Camera {
        let mut camera = Camera {
            uniforms: CameraUniforms::zeroed(),
            center,
            up,
            distance,
            azimuth,
            altitude,
        };
        camera.uniforms.fov_y = DEFAULT_FOV_Y;
        camera.calculate_uniforms();
        camera
    }

    pub fn with_fov(mut self, fov_y: f32) -> Self {
        self.uniforms.fov_y = fov_y;
        self
    }

    pub fn uniforms(&self) -> &CameraUniforms {
        &self.uniforms
    }

    /// The camera's world-space position. Drives procedural chunk streaming.
    pub fn position(&self) -> Vec3 {
        self.uniforms.origin
    }

    pub fn zoom(&mut self, displacement: f32) {
        self.distance = (self.distance - displacement).max(0.0); // Prevent negative distance
        self.uniforms.origin = self.center - self.distance * self.uniforms.w;
    }

    pub fn pan(&mut self, du: f32, dv: f32) {
        let pan = du * self.uniforms.u + dv * self.uniforms.v;
        self.center += pan;
        self.uniforms.origin += pan;
    }

    pub fn fly(&mut self, displacement: f32) {
        let forward = -self.uniforms.w * displacement;
        self.center += forward;
        self.uniforms.origin += forward;
    }

    pub fn orbit(&mut self, du: f32, dv: f32) {
        const MAX_ALT: f32 = FRAC_PI_2 - 1e-6;
        self.altitude = (self.altitude + dv).clamp(-MAX_ALT, MAX_ALT);
        self.azimuth += du;
        self.azimuth %= 2. * PI;
        self.calculate_uniforms();
    }

    pub fn adjust_fov(&mut self, delta: f32) {
        let fov_y = self.uniforms.fov_y;
        self.uniforms.fov_y = (fov_y + delta).clamp(MIN_FOV_Y, MAX_FOV_Y);
    }

    fn calculate_uniforms(&mut self) {
        let w = {
            let (y, xz_scale) = self.altitude.sin_cos();
            let (x, z) = self.azimuth.sin_cos();
            -Vec3::new(x * xz_scale, y, z * xz_scale)
        };
        let origin = self.center - self.distance * w;
        let u = w.cross(&self.up).normalized();
        let v = u.cross(&w);
        self.uniforms.origin = origin;
        self.uniforms.u = u;
        self.uniforms.v = v;
        self.uniforms.w = w;
    }
}
