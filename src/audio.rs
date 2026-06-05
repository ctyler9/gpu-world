// Originally written in 2026 by Clay Tyler
// SPDX-License-Identifier: CC-BY-4.0

//! A Smash Hit-inspired ambient soundtrack, generated live with FunDSP and
//! played through cpal on its own audio thread.
//!
//! Signal chain: slow synth pad + sub-bass → lowpass filter whose cutoff is
//! swept by a slow LFO → stereo reverb. The whole thing is a 0-input, 2-output
//! generator, so the audio callback just pulls stereo sample pairs from it.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use fundsp::hacker32::*;

/// Start the ambient soundtrack on the default output device. The returned
/// [`cpal::Stream`] must be kept alive for audio to keep playing — drop it to
/// stop.
pub fn start() -> Result<cpal::Stream> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .context("no audio output device")?;
    let config = device.default_output_config()?;
    match config.sample_format() {
        cpal::SampleFormat::F32 => run::<f32>(&device, &config.into()),
        cpal::SampleFormat::I16 => run::<i16>(&device, &config.into()),
        cpal::SampleFormat::U16 => run::<u16>(&device, &config.into()),
        other => anyhow::bail!("unsupported sample format: {other:?}"),
    }
}

/// Build the synth graph and wire it to a cpal output stream of sample type `T`.
fn run<T>(device: &cpal::Device, config: &cpal::StreamConfig) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;

    let mut graph = build_graph();
    graph.set_sample_rate(config.sample_rate.0 as f64);
    graph.allocate();

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            // Pull one stereo pair per output frame, fanning out to however many
            // channels the device has.
            for frame in data.chunks_mut(channels) {
                let (left, right) = graph.get_stereo();
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

/// The ambient bed: pad voices + sub-bass → LFO-swept lowpass → stereo reverb.
/// A 0-input, 2-output generator.
fn build_graph() -> impl AudioUnit {
    // Slow synth pad: a low fundamental with two quieter harmonics.
    let pad = sine_hz(80.0) * 0.25 & triangle_hz(160.0) * 0.12 & triangle_hz(240.0) * 0.06;

    // Sub-bass — felt more than heard.
    let sub = sine_hz(40.0) * 0.10;

    // Slow LFO sweeping the filter cutoff between ~800 Hz and ~3200 Hz at
    // 0.07 Hz (one full sweep every ~14 s).
    let cutoff = lfo(|t: f32| {
        let phase = (t * 0.07 * std::f32::consts::TAU).sin() * 0.5 + 0.5;
        xerp(800.0, 3200.0, phase)
    });

    // Mix voices to mono, then feed [signal, cutoff, Q] into the lowpass.
    let voices = pad & sub;
    let filtered = voices >> (pass() | cutoff | constant(1.2)) >> lowpass();

    // Fan mono out to stereo, then blend dry with a wet stereo reverb tail
    // (room 20 m, decay 3.5 s, HF damping 0.5) at ~65% wet.
    let reverb = reverb_stereo(20.0, 3.5, 0.5);
    filtered >> split::<U2>() >> (multipass::<U2>() & 0.65 * reverb)
}
