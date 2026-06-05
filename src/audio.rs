// Originally written in 2026 by Clay Tyler
// SPDX-License-Identifier: CC-BY-4.0

//! A lightweight ambient soundtrack played through cpal on its own audio
//! thread. The callback avoids expensive effects so it stays stable while the
//! renderer is busy.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use std::sync::{
    atomic::{AtomicU32, AtomicU8, Ordering},
    Arc,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioMode {
    Auto,
    Dorian,
    Lydian,
    Aeolian,
    Mixolydian,
    Ionian,
}

impl AudioMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "dorian" => Some(Self::Dorian),
            "lydian" => Some(Self::Lydian),
            "aeolian" | "minor" => Some(Self::Aeolian),
            "mixolydian" => Some(Self::Mixolydian),
            "ionian" | "major" => Some(Self::Ionian),
            _ => None,
        }
    }

    pub fn names() -> &'static str {
        "auto, dorian, lydian, aeolian, mixolydian, ionian"
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Dorian => "dorian",
            Self::Lydian => "lydian",
            Self::Aeolian => "aeolian",
            Self::Mixolydian => "mixolydian",
            Self::Ionian => "ionian",
        }
    }

    pub fn all() -> &'static [Self] {
        &[
            Self::Auto,
            Self::Dorian,
            Self::Lydian,
            Self::Aeolian,
            Self::Mixolydian,
            Self::Ionian,
        ]
    }

    fn from_index(index: u8) -> Self {
        match index {
            1 => Self::Dorian,
            2 => Self::Lydian,
            3 => Self::Aeolian,
            4 => Self::Mixolydian,
            5 => Self::Ionian,
            _ => Self::Auto,
        }
    }

    fn index(self) -> u8 {
        match self {
            Self::Auto => 0,
            Self::Dorian => 1,
            Self::Lydian => 2,
            Self::Aeolian => 3,
            Self::Mixolydian => 4,
            Self::Ionian => 5,
        }
    }
}

#[derive(Clone)]
pub struct AudioControls {
    inner: Arc<AudioControlsInner>,
}

struct AudioControlsInner {
    mode: AtomicU8,
    volume: AtomicU32,
    arpeggio: AtomicU32,
    warmth: AtomicU32,
}

impl AudioControls {
    pub fn new(mode: AudioMode) -> Self {
        Self {
            inner: Arc::new(AudioControlsInner {
                mode: AtomicU8::new(mode.index()),
                volume: AtomicU32::new(1.0_f32.to_bits()),
                arpeggio: AtomicU32::new(1.0_f32.to_bits()),
                warmth: AtomicU32::new(0.65_f32.to_bits()),
            }),
        }
    }

    pub fn mode(&self) -> AudioMode {
        AudioMode::from_index(self.inner.mode.load(Ordering::Relaxed))
    }

    pub fn set_mode(&self, mode: AudioMode) {
        self.inner.mode.store(mode.index(), Ordering::Relaxed);
    }

    pub fn volume(&self) -> f32 {
        f32::from_bits(self.inner.volume.load(Ordering::Relaxed))
    }

    pub fn set_volume(&self, value: f32) {
        self.inner
            .volume
            .store(value.clamp(0.0, 2.0).to_bits(), Ordering::Relaxed);
    }

    pub fn arpeggio(&self) -> f32 {
        f32::from_bits(self.inner.arpeggio.load(Ordering::Relaxed))
    }

    pub fn set_arpeggio(&self, value: f32) {
        self.inner
            .arpeggio
            .store(value.clamp(0.0, 2.0).to_bits(), Ordering::Relaxed);
    }

    pub fn warmth(&self) -> f32 {
        f32::from_bits(self.inner.warmth.load(Ordering::Relaxed))
    }

    pub fn set_warmth(&self, value: f32) {
        self.inner
            .warmth
            .store(value.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }
}

/// Start the ambient soundtrack on the default output device. The returned
/// [`cpal::Stream`] must be kept alive for audio to keep playing — drop it to
/// stop.
pub fn start(controls: AudioControls) -> Result<cpal::Stream> {
    let host = cpal::default_host();
    let device = output_device(&host)?;
    let device_name = device.name().unwrap_or_else(|_| "unknown".to_string());
    let config = device.default_output_config()?;
    eprintln!(
        "audio started: {device_name}, {:?}, {} Hz, {} channels",
        config.sample_format(),
        config.sample_rate().0,
        config.channels()
    );
    match config.sample_format() {
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into(), controls),
        cpal::SampleFormat::I16 => run::<i16>(&device, &config.into(), controls),
        cpal::SampleFormat::U16 => run::<u16>(&device, &config.into(), controls),
        other => anyhow::bail!("unsupported sample format: {other:?}"),
    }
}

fn output_device(host: &cpal::Host) -> Result<cpal::Device> {
    if let Ok(requested) = std::env::var("GPU_WORLD_AUDIO_DEVICE") {
        let requested = requested.to_lowercase();
        let mut names = Vec::new();
        for device in host.output_devices()? {
            let name = device.name().unwrap_or_else(|_| "unknown".to_string());
            if name.to_lowercase().contains(&requested) {
                return Ok(device);
            }
            names.push(name);
        }

        anyhow::bail!(
            "no audio output device matched GPU_WORLD_AUDIO_DEVICE={requested:?}; available: {}",
            names.join(", ")
        );
    }

    host.default_output_device()
        .context("no audio output device")
}

/// Build the synth and wire it to a cpal output stream of sample type `T`.
fn run<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    controls: AudioControls,
) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let mut synth = AmbientSynth::new(config.sample_rate.0 as f32, controls);

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            // Pull one stereo pair per output frame, fanning out to however many
            // channels the device has.
            for frame in data.chunks_mut(channels) {
                let (left, right) = synth.next_stereo();
                for (ch, slot) in frame.iter_mut().enumerate() {
                    let value = if ch % 2 == 0 { left } else { right };
                    *slot = T::from_sample(value);
                }
            }
        },
        |err| eprintln!("audio stream error: {err}"),
        None,
    )?;
    stream.play()?;
    Ok(stream)
}

struct AmbientSynth {
    controls: AudioControls,
    active_mode: AudioMode,
    sample_rate: f32,
    phases: [f32; 6],
    detune_phases: [f32; 6],
    frequencies: [f32; 6],
    previous_phases: [f32; 6],
    previous_detune_phases: [f32; 6],
    previous_frequencies: [f32; 6],
    target_frequencies: [f32; 6],
    lfo_phase: f32,
    pan_phase: f32,
    arp_phase: f32,
    arp_frequency: f32,
    arp_envelope: f32,
    chord_envelope: f32,
    previous_chord_envelope: f32,
    lowpass_left: f32,
    lowpass_right: f32,
    current_chord: u64,
    current_arp_step: u64,
    sample_index: u64,
}

impl AmbientSynth {
    fn new(sample_rate: f32, controls: AudioControls) -> Self {
        let active_mode = controls.mode();
        let mut synth = Self {
            controls,
            active_mode,
            sample_rate,
            phases: [0.0; 6],
            detune_phases: [0.0; 6],
            frequencies: [0.0; 6],
            previous_phases: [0.0; 6],
            previous_detune_phases: [0.0; 6],
            previous_frequencies: [0.0; 6],
            target_frequencies: [0.0; 6],
            lfo_phase: 0.0,
            pan_phase: 0.0,
            arp_phase: 0.0,
            arp_frequency: 0.0,
            arp_envelope: 0.0,
            chord_envelope: 0.0,
            previous_chord_envelope: 0.0,
            lowpass_left: 0.0,
            lowpass_right: 0.0,
            current_chord: u64::MAX,
            current_arp_step: u64::MAX,
            sample_index: 0,
        };
        synth.update_harmony();
        synth.frequencies = synth.target_frequencies;
        synth
    }

    fn next_stereo(&mut self) -> (f32, f32) {
        const TAU: f32 = std::f32::consts::TAU;
        const GAINS: [f32; 6] = [0.105, 0.082, 0.058, 0.035, 0.017, 0.010];
        let warmth = self.controls.warmth();
        let detune_depth = 0.0012 + warmth * 0.0026;
        let detune_blend = 0.16 + warmth * 0.18;

        self.update_harmony();

        let mut current_mono = 0.0;
        let mut previous_mono = 0.0;
        for (index, gain) in GAINS.iter().enumerate() {
            self.phases[index] = wrap_phase(
                self.phases[index] + TAU * self.frequencies[index] / self.sample_rate,
            );
            let detune = if index % 2 == 0 {
                1.0 - detune_depth
            } else {
                1.0 + detune_depth
            };
            self.detune_phases[index] = wrap_phase(
                self.detune_phases[index]
                    + TAU * self.frequencies[index] * detune / self.sample_rate,
            );

            let voice_lfo = (self.lfo_phase + index as f32 * 0.73).sin() * 0.08 + 0.92;
            let voice = self.phases[index].sin() * (1.0 - detune_blend)
                + self.detune_phases[index].sin() * detune_blend;
            current_mono += voice * gain * voice_lfo;

            if self.previous_chord_envelope > 0.0 {
                self.previous_phases[index] = wrap_phase(
                    self.previous_phases[index]
                        + TAU * self.previous_frequencies[index] / self.sample_rate,
                );
                self.previous_detune_phases[index] = wrap_phase(
                    self.previous_detune_phases[index]
                        + TAU * self.previous_frequencies[index] * detune / self.sample_rate,
                );
                let previous_voice = self.previous_phases[index].sin()
                    * (1.0 - detune_blend)
                    + self.previous_detune_phases[index].sin() * detune_blend;
                previous_mono += previous_voice * gain * voice_lfo;
            }
        }

        self.lfo_phase = wrap_phase(self.lfo_phase + TAU * 0.024 / self.sample_rate);
        self.pan_phase = wrap_phase(self.pan_phase + TAU * 0.007 / self.sample_rate);
        self.arp_phase =
            wrap_phase(self.arp_phase + TAU * self.arp_frequency / self.sample_rate);
        self.arp_envelope *= 0.99993;
        self.chord_envelope =
            (self.chord_envelope + 1.0 / (self.sample_rate * 5.0)).min(1.0);
        self.previous_chord_envelope =
            (self.previous_chord_envelope - 1.0 / (self.sample_rate * 5.0)).max(0.0);

        let breathe = 0.58 + 0.09 * self.lfo_phase.sin();
        let attack_samples = self.sample_rate * 4.0;
        let attack = (self.sample_index as f32 / attack_samples).min(1.0);
        self.sample_index += 1;

        let current_pad = current_mono * smoothstep(self.chord_envelope);
        let previous_pad = previous_mono * smoothstep(self.previous_chord_envelope);
        let pad = ((current_pad + previous_pad) * breathe * attack).tanh() * 0.34;
        let arp =
            self.arp_phase.sin() * self.arp_envelope * 0.014 * self.controls.arpeggio();
        let pan = self.pan_phase.sin() * 0.07;
        let arp_pan = -pan * 0.30;

        let left = pad * (0.80 - pan) + arp * (0.68 - arp_pan);
        let right = pad * (0.80 + pan) + arp * (0.68 + arp_pan);

        let smooth = 0.095 - warmth * 0.05;
        self.lowpass_left += (left - self.lowpass_left) * smooth;
        self.lowpass_right += (right - self.lowpass_right) * smooth;

        let volume = self.controls.volume();
        (self.lowpass_left * volume, self.lowpass_right * volume)
    }

    fn update_harmony(&mut self) {
        const CHORD_SECONDS: f32 = 12.0;
        const ARP_SECONDS: f32 = 2.25;

        let chord_samples = (self.sample_rate * CHORD_SECONDS) as u64;
        let arp_samples = (self.sample_rate * ARP_SECONDS) as u64;
        let chord = self.sample_index / chord_samples;
        let arp_step = self.sample_index / arp_samples;
        let selected_mode = self.controls.mode();

        if chord != self.current_chord || selected_mode != self.active_mode {
            let had_previous_chord = self.current_chord != u64::MAX;
            self.current_chord = chord;
            self.active_mode = selected_mode;
            let mode = current_mode(chord, selected_mode);
            let progression_index = (chord as usize) % mode.progression.len();
            let degree = mode.progression[progression_index];
            let root = scale_midi(mode.root_midi, mode.scale, degree, -2);
            let third = scale_midi(mode.root_midi, mode.scale, degree + 2, -1);
            let fifth = scale_midi(mode.root_midi, mode.scale, degree + 4, -1);
            let seventh = scale_midi(mode.root_midi, mode.scale, degree + 6, 0);
            let ninth = scale_midi(mode.root_midi, mode.scale, degree + 8, 0);

            self.target_frequencies = [
                midi_to_hz(root),
                midi_to_hz(fifth),
                midi_to_hz(root + 12),
                midi_to_hz(third + 12),
                midi_to_hz(seventh),
                midi_to_hz(ninth),
            ];
            if had_previous_chord {
                self.previous_phases = self.phases;
                self.previous_detune_phases = self.detune_phases;
                self.previous_frequencies = self.frequencies;
                self.previous_chord_envelope = 1.0;
            }
            self.frequencies = self.target_frequencies;
            self.chord_envelope = 0.0;
            self.arp_envelope = 0.0;
        }

        if arp_step != self.current_arp_step {
            self.current_arp_step = arp_step;
            let mode = current_mode(chord, selected_mode);
            let progression_index = (chord as usize) % mode.progression.len();
            let degree = mode.progression[progression_index];
            let pattern = [0, 2, 4, 2, 6, 4, 2, 0];
            let note = scale_midi(
                mode.root_midi,
                mode.scale,
                degree + pattern[(arp_step as usize) % pattern.len()],
                0,
            );
            self.arp_frequency = midi_to_hz(note);
            self.arp_envelope = 0.48;
        }
    }
}

struct ModeProgression {
    root_midi: i32,
    scale: [i32; 7],
    progression: [usize; 8],
}

fn current_mode(chord: u64, mode: AudioMode) -> &'static ModeProgression {
    const IONIAN: [i32; 7] = [0, 2, 4, 5, 7, 9, 11];
    const DORIAN: [i32; 7] = [0, 2, 3, 5, 7, 9, 10];
    const LYDIAN: [i32; 7] = [0, 2, 4, 6, 7, 9, 11];
    const MIXOLYDIAN: [i32; 7] = [0, 2, 4, 5, 7, 9, 10];
    const AEOLIAN: [i32; 7] = [0, 2, 3, 5, 7, 8, 10];

    const MODES: [ModeProgression; 5] = [
        ModeProgression {
            root_midi: 50,
            scale: DORIAN,
            progression: [0, 3, 5, 1, 4, 2, 3, 0],
        },
        ModeProgression {
            root_midi: 53,
            scale: LYDIAN,
            progression: [0, 1, 4, 2, 3, 1, 5, 4],
        },
        ModeProgression {
            root_midi: 48,
            scale: AEOLIAN,
            progression: [0, 5, 2, 6, 3, 5, 4, 0],
        },
        ModeProgression {
            root_midi: 55,
            scale: MIXOLYDIAN,
            progression: [0, 4, 5, 3, 0, 6, 5, 4],
        },
        ModeProgression {
            root_midi: 57,
            scale: IONIAN,
            progression: [0, 3, 4, 0, 5, 3, 1, 4],
        },
    ];

    match mode {
        AudioMode::Auto => &MODES[((chord / 8) as usize) % MODES.len()],
        AudioMode::Dorian => &MODES[0],
        AudioMode::Lydian => &MODES[1],
        AudioMode::Aeolian => &MODES[2],
        AudioMode::Mixolydian => &MODES[3],
        AudioMode::Ionian => &MODES[4],
    }
}

fn scale_midi(root_midi: i32, scale: [i32; 7], degree: usize, octave_offset: i32) -> i32 {
    let octave = (degree / 7) as i32 + octave_offset;
    root_midi + scale[degree % 7] + octave * 12
}

fn midi_to_hz(note: i32) -> f32 {
    440.0 * 2.0_f32.powf((note as f32 - 69.0) / 12.0)
}

fn smoothstep(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

fn wrap_phase(phase: f32) -> f32 {
    if phase >= std::f32::consts::TAU {
        phase - std::f32::consts::TAU
    } else {
        phase
    }
}
