// Originally written in 2026 by Clay Tyler
// SPDX-License-Identifier: CC-BY-4.0

//! A lightweight ambient soundtrack played through cpal on its own audio
//! thread. The callback avoids expensive effects so it stays stable while the
//! renderer is busy.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

/// Start the ambient soundtrack on the default output device. The returned
/// [`cpal::Stream`] must be kept alive for audio to keep playing — drop it to
/// stop.
pub fn start() -> Result<cpal::Stream> {
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
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into()),
        cpal::SampleFormat::I16 => run::<i16>(&device, &config.into()),
        cpal::SampleFormat::U16 => run::<u16>(&device, &config.into()),
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
fn run<T>(device: &cpal::Device, config: &cpal::StreamConfig) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let mut synth = AmbientSynth::new(config.sample_rate.0 as f32);

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
    sample_rate: f32,
    phases: [f32; 6],
    detune_phases: [f32; 6],
    frequencies: [f32; 6],
    target_frequencies: [f32; 6],
    lfo_phase: f32,
    pan_phase: f32,
    arp_phase: f32,
    arp_frequency: f32,
    arp_envelope: f32,
    lowpass_left: f32,
    lowpass_right: f32,
    current_chord: u64,
    current_arp_step: u64,
    sample_index: u64,
}

impl AmbientSynth {
    fn new(sample_rate: f32) -> Self {
        let mut synth = Self {
            sample_rate,
            phases: [0.0; 6],
            detune_phases: [0.0; 6],
            frequencies: [0.0; 6],
            target_frequencies: [0.0; 6],
            lfo_phase: 0.0,
            pan_phase: 0.0,
            arp_phase: 0.0,
            arp_frequency: 0.0,
            arp_envelope: 0.0,
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

        self.update_harmony();

        let mut mono = 0.0;
        for (index, gain) in GAINS.iter().enumerate() {
            self.frequencies[index] +=
                (self.target_frequencies[index] - self.frequencies[index]) * 0.000014;
            self.phases[index] = wrap_phase(
                self.phases[index] + TAU * self.frequencies[index] / self.sample_rate,
            );
            let detune = if index % 2 == 0 { 0.9975 } else { 1.0025 };
            self.detune_phases[index] = wrap_phase(
                self.detune_phases[index]
                    + TAU * self.frequencies[index] * detune / self.sample_rate,
            );

            let voice_lfo = (self.lfo_phase + index as f32 * 0.73).sin() * 0.08 + 0.92;
            let voice =
                self.phases[index].sin() * 0.74 + self.detune_phases[index].sin() * 0.26;
            mono += voice * gain * voice_lfo;
        }

        self.lfo_phase = wrap_phase(self.lfo_phase + TAU * 0.024 / self.sample_rate);
        self.pan_phase = wrap_phase(self.pan_phase + TAU * 0.007 / self.sample_rate);
        self.arp_phase =
            wrap_phase(self.arp_phase + TAU * self.arp_frequency / self.sample_rate);
        self.arp_envelope *= 0.99993;

        let breathe = 0.58 + 0.09 * self.lfo_phase.sin();
        let attack_samples = self.sample_rate * 4.0;
        let attack = (self.sample_index as f32 / attack_samples).min(1.0);
        self.sample_index += 1;

        let pad = (mono * breathe * attack).tanh() * 0.36;
        let arp = self.arp_phase.sin() * self.arp_envelope * 0.007;
        let pan = self.pan_phase.sin() * 0.07;
        let arp_pan = -pan * 0.30;

        let left = pad * (0.80 - pan) + arp * (0.68 - arp_pan);
        let right = pad * (0.80 + pan) + arp * (0.68 + arp_pan);

        self.lowpass_left += (left - self.lowpass_left) * 0.065;
        self.lowpass_right += (right - self.lowpass_right) * 0.065;

        (self.lowpass_left, self.lowpass_right)
    }

    fn update_harmony(&mut self) {
        const CHORD_SECONDS: f32 = 12.0;
        const ARP_SECONDS: f32 = 3.0;

        let chord_samples = (self.sample_rate * CHORD_SECONDS) as u64;
        let arp_samples = (self.sample_rate * ARP_SECONDS) as u64;
        let chord = self.sample_index / chord_samples;
        let arp_step = self.sample_index / arp_samples;

        if chord != self.current_chord {
            self.current_chord = chord;
            let mode = current_mode(chord);
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
        }

        if arp_step != self.current_arp_step {
            self.current_arp_step = arp_step;
            let mode = current_mode(chord);
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
            self.arp_envelope = 0.32;
        }
    }
}

struct ModeProgression {
    root_midi: i32,
    scale: [i32; 7],
    progression: [usize; 8],
}

fn current_mode(chord: u64) -> &'static ModeProgression {
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

    &MODES[((chord / 8) as usize) % MODES.len()]
}

fn scale_midi(root_midi: i32, scale: [i32; 7], degree: usize, octave_offset: i32) -> i32 {
    let octave = (degree / 7) as i32 + octave_offset;
    root_midi + scale[degree % 7] + octave * 12
}

fn midi_to_hz(note: i32) -> f32 {
    440.0 * 2.0_f32.powf((note as f32 - 69.0) / 12.0)
}

fn wrap_phase(phase: f32) -> f32 {
    if phase >= std::f32::consts::TAU {
        phase - std::f32::consts::TAU
    } else {
        phase
    }
}
