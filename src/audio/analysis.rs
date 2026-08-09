use std::collections::VecDeque;
use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

pub const SAMPLE_RATE: u32 = 48_000;
pub const FFT_SIZE: usize = 2_048;
pub const HOP_SIZE: usize = 512;

const MIN_FREQUENCY_HZ: f32 = 80.0;
const MAX_FREQUENCY_HZ: f32 = 4_000.0;
const SILENCE_THRESHOLD_DBFS: f32 = -50.0;
const ONSET_HISTORY_SECONDS: f32 = 8.0;
const ONSET_THRESHOLD: f32 = 0.18;
const ONSET_REFRACTORY_SECONDS: f32 = 0.08;
const TEMPO_MIN_BPM: f32 = 55.0;
const TEMPO_MAX_BPM: f32 = 200.0;
const TEMPO_MIN_HISTORY_SECONDS: f32 = 4.0;
const TEMPO_UPDATE_SECONDS: f32 = 0.5;
const TEMPO_CONFIDENCE_THRESHOLD: f32 = 0.35;

fn analysis_frame_rate() -> f32 {
    SAMPLE_RATE as f32 / HOP_SIZE as f32
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BandEnergies {
    pub low: f32,
    pub mid: f32,
    pub high: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AudioFeatures {
    pub rms: f32,
    pub dominant_hz: Option<f32>,
    pub spectral_flux: f32,
    pub spectral_flatness: f32,
    pub bands: BandEnergies,
    pub onset_rate_hz: f32,
    pub rhythmic_irregularity: f32,
    pub chaos: f32,
    pub bpm: Option<f32>,
    pub tempo_confidence: f32,
    pub beat_count: u64,
    pub beat_strength: f32,
}

pub struct Analyzer {
    samples: Vec<f32>,
    sample_start: usize,
    fft: Arc<dyn Fft<f32>>,
    fft_buffer: Vec<Complex<f32>>,
    previous_spectrum: Vec<f32>,
    flux_history: VecDeque<f32>,
    onset_envelope: VecDeque<f32>,
    onset_times: VecDeque<f32>,
    previous_previous_onset_strength: f32,
    previous_onset_strength: f32,
    previous_onset_time: f32,
    last_onset_time: Option<f32>,
    processed_samples: u64,
    last_tempo_update_time: f32,
    bpm: Option<f32>,
    tempo_confidence: f32,
    next_beat_time: Option<f32>,
    beat_count: u64,
    last_beat_strength: f32,
    pending_beat_strength: Option<f32>,
    smoothed_flatness: f32,
    smoothed_bands: BandEnergies,
    smoothed_onset_rate: f32,
    smoothed_irregularity: f32,
    smoothed_chaos: f32,
}

impl Analyzer {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);

        Self {
            samples: Vec::with_capacity(FFT_SIZE * 4),
            sample_start: 0,
            fft,
            fft_buffer: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            previous_spectrum: vec![0.0; FFT_SIZE / 2],
            flux_history: VecDeque::with_capacity(analysis_frame_rate() as usize + 1),
            onset_envelope: VecDeque::with_capacity(
                (analysis_frame_rate() * ONSET_HISTORY_SECONDS) as usize + 1,
            ),
            onset_times: VecDeque::with_capacity(64),
            previous_previous_onset_strength: 0.0,
            previous_onset_strength: 0.0,
            previous_onset_time: 0.0,
            last_onset_time: None,
            processed_samples: 0,
            last_tempo_update_time: 0.0,
            bpm: None,
            tempo_confidence: 0.0,
            next_beat_time: None,
            beat_count: 0,
            last_beat_strength: 0.0,
            pending_beat_strength: None,
            smoothed_flatness: 0.0,
            smoothed_bands: BandEnergies::default(),
            smoothed_onset_rate: 0.0,
            smoothed_irregularity: 0.0,
            smoothed_chaos: 0.0,
        }
    }

    pub fn push(&mut self, samples: &[f32]) -> Option<AudioFeatures> {
        self.samples.extend_from_slice(samples);
        let mut latest = None;

        while self.samples.len() - self.sample_start >= FFT_SIZE {
            latest = Some(self.process_frame());
            self.sample_start += HOP_SIZE;
        }

        if self.sample_start >= FFT_SIZE * 4 {
            self.samples.drain(..self.sample_start);
            self.sample_start = 0;
        }

        latest
    }

    fn process_frame(&mut self) -> AudioFeatures {
        let frame_time = (self.processed_samples + FFT_SIZE as u64) as f32 / SAMPLE_RATE as f32;
        let frame = &self.samples[self.sample_start..self.sample_start + FFT_SIZE];
        let rms = calculate_rms(frame);
        let dbfs = 20.0 * rms.max(0.000_001).log10();
        let mean = frame.iter().sum::<f32>() / FFT_SIZE as f32;

        for (index, fft_sample) in self.fft_buffer.iter_mut().enumerate() {
            let hann =
                0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / (FFT_SIZE - 1) as f32).cos();
            *fft_sample = Complex::new((frame[index] - mean) * hann, 0.0);
        }

        self.fft.process(&mut self.fft_buffer);

        let magnitudes: Vec<f32> = self.fft_buffer[..FFT_SIZE / 2]
            .iter()
            .map(|sample| sample.norm())
            .collect();
        let powers: Vec<f32> = self.fft_buffer[..FFT_SIZE / 2]
            .iter()
            .map(Complex::norm_sqr)
            .collect();

        let raw_flux = calculate_spectral_flux(&magnitudes, &self.previous_spectrum);
        self.previous_spectrum.copy_from_slice(&magnitudes);
        let onset_strength = self.normalized_onset_strength(raw_flux);
        let spectral_flux = (raw_flux / (raw_flux + 0.08)).clamp(0.0, 1.0);

        self.onset_envelope.push_back(onset_strength);
        truncate_history(
            &mut self.onset_envelope,
            (analysis_frame_rate() * ONSET_HISTORY_SECONDS).ceil() as usize,
        );

        let detected_onset = self.detect_onset(onset_strength, frame_time);
        if let Some((onset_time, _strength)) = detected_onset {
            self.onset_times.push_back(onset_time);
        }
        while self
            .onset_times
            .front()
            .is_some_and(|time| frame_time - time > ONSET_HISTORY_SECONDS)
        {
            self.onset_times.pop_front();
        }

        self.update_tempo(frame_time);
        self.update_beat_clock(frame_time, detected_onset);

        let flatness = calculate_spectral_flatness(&powers);
        let bands = calculate_band_energies(&powers);
        let onset_rate = self
            .onset_times
            .iter()
            .filter(|time| frame_time - **time <= 2.0)
            .count() as f32
            / 2.0;
        let irregularity = calculate_rhythmic_irregularity(&self.onset_times);
        let chaos =
            (0.45 * flatness + 0.35 * irregularity + 0.20 * (onset_rate / 8.0).clamp(0.0, 1.0))
                .clamp(0.0, 1.0);

        self.smoothed_flatness = smooth_feature(self.smoothed_flatness, flatness);
        self.smoothed_bands.low = smooth_feature(self.smoothed_bands.low, bands.low);
        self.smoothed_bands.mid = smooth_feature(self.smoothed_bands.mid, bands.mid);
        self.smoothed_bands.high = smooth_feature(self.smoothed_bands.high, bands.high);
        let smoothed_band_total =
            self.smoothed_bands.low + self.smoothed_bands.mid + self.smoothed_bands.high;
        if bands != BandEnergies::default() && smoothed_band_total > f32::EPSILON {
            self.smoothed_bands.low /= smoothed_band_total;
            self.smoothed_bands.mid /= smoothed_band_total;
            self.smoothed_bands.high /= smoothed_band_total;
        }
        self.smoothed_onset_rate = smooth_feature(self.smoothed_onset_rate, onset_rate);
        self.smoothed_irregularity = smooth_feature(self.smoothed_irregularity, irregularity);
        self.smoothed_chaos = smooth_feature(self.smoothed_chaos, chaos);

        let dominant_hz = if dbfs < SILENCE_THRESHOLD_DBFS {
            None
        } else {
            dominant_frequency(&powers)
        };

        self.processed_samples += HOP_SIZE as u64;

        AudioFeatures {
            rms,
            dominant_hz,
            spectral_flux,
            spectral_flatness: self.smoothed_flatness,
            bands: self.smoothed_bands,
            onset_rate_hz: self.smoothed_onset_rate,
            rhythmic_irregularity: self.smoothed_irregularity,
            chaos: self.smoothed_chaos,
            bpm: self.bpm,
            tempo_confidence: self.tempo_confidence,
            beat_count: self.beat_count,
            beat_strength: self.last_beat_strength,
        }
    }

    fn normalized_onset_strength(&mut self, raw_flux: f32) -> f32 {
        let baseline = if self.flux_history.is_empty() {
            raw_flux
        } else {
            median(self.flux_history.iter().copied())
        };
        let excess = (raw_flux - baseline * 1.5).max(0.0);
        let strength = (excess / (baseline * 3.0 + 0.02)).clamp(0.0, 1.0);

        self.flux_history.push_back(raw_flux);
        truncate_history(
            &mut self.flux_history,
            analysis_frame_rate().ceil() as usize,
        );
        strength
    }

    fn detect_onset(&mut self, current_strength: f32, frame_time: f32) -> Option<(f32, f32)> {
        let candidate_time = self.previous_onset_time;
        let candidate_strength = self.previous_onset_strength;
        let is_local_peak = candidate_strength > self.previous_previous_onset_strength
            && candidate_strength >= current_strength;
        let outside_refractory = self
            .last_onset_time
            .is_none_or(|last| candidate_time - last >= ONSET_REFRACTORY_SECONDS);

        let detected =
            if candidate_strength >= ONSET_THRESHOLD && is_local_peak && outside_refractory {
                self.last_onset_time = Some(candidate_time);
                Some((candidate_time, candidate_strength))
            } else {
                None
            };

        self.previous_previous_onset_strength = self.previous_onset_strength;
        self.previous_onset_strength = current_strength;
        self.previous_onset_time = frame_time;
        detected
    }

    fn update_tempo(&mut self, frame_time: f32) {
        if frame_time - self.last_tempo_update_time < TEMPO_UPDATE_SECONDS
            || self.onset_envelope.len()
                < (analysis_frame_rate() * TEMPO_MIN_HISTORY_SECONDS) as usize
        {
            return;
        }

        self.last_tempo_update_time = frame_time;
        let (bpm, confidence) = estimate_tempo(&self.onset_envelope, self.bpm);

        if let Some(bpm) = bpm {
            self.bpm = Some(match self.bpm {
                Some(previous) => previous + (bpm - previous) * 0.2,
                None => bpm,
            });
            self.tempo_confidence =
                smooth_feature(self.tempo_confidence, confidence.clamp(0.0, 1.0));
        } else {
            self.tempo_confidence = smooth_feature(self.tempo_confidence, 0.0);
            if self.tempo_confidence < 0.05 {
                self.bpm = None;
                self.next_beat_time = None;
                self.pending_beat_strength = None;
            }
        }
    }

    fn update_beat_clock(&mut self, frame_time: f32, onset: Option<(f32, f32)>) {
        let confident_period = self
            .bpm
            .filter(|_| self.tempo_confidence >= TEMPO_CONFIDENCE_THRESHOLD)
            .map(|bpm| 60.0 / bpm);

        let Some(period) = confident_period else {
            self.next_beat_time = None;
            self.pending_beat_strength = None;
            if let Some((_, strength)) = onset {
                self.beat_count = self.beat_count.wrapping_add(1);
                self.last_beat_strength = strength.clamp(0.0, 1.0);
            }
            return;
        };

        if self.next_beat_time.is_none() {
            let anchor = onset
                .map(|(time, _)| time)
                .or(self.last_onset_time)
                .unwrap_or(frame_time);
            self.next_beat_time = Some(anchor + period);
        }

        if let (Some((onset_time, strength)), Some(next_beat)) = (onset, self.next_beat_time)
            && (onset_time - next_beat).abs() <= period * 0.2
        {
            self.next_beat_time = Some(next_beat + (onset_time - next_beat) * 0.35);
            self.pending_beat_strength = Some(
                self.pending_beat_strength
                    .unwrap_or(0.0)
                    .max(strength.clamp(0.0, 1.0)),
            );
        }

        while self.next_beat_time.is_some_and(|next| frame_time >= next) {
            self.beat_count = self.beat_count.wrapping_add(1);
            self.last_beat_strength = self
                .pending_beat_strength
                .take()
                .unwrap_or((0.55 + self.tempo_confidence.clamp(0.0, 1.0) * 0.45).clamp(0.0, 1.0));
            self.next_beat_time = self.next_beat_time.map(|next| next + period);
        }
    }
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

fn calculate_spectral_flux(current: &[f32], previous: &[f32]) -> f32 {
    let useful_bins = current.len().min(previous.len()).max(1);
    current
        .iter()
        .zip(previous)
        .skip(1)
        .map(|(current, previous)| ((1.0 + current).ln() - (1.0 + previous).ln()).max(0.0))
        .sum::<f32>()
        / useful_bins as f32
}

fn calculate_spectral_flatness(powers: &[f32]) -> f32 {
    if powers.len() <= 1 {
        return 0.0;
    }

    let bins = &powers[1..];
    let arithmetic_mean = bins.iter().sum::<f32>() / bins.len() as f32;
    if arithmetic_mean <= f32::EPSILON {
        return 0.0;
    }

    let geometric_mean =
        (bins.iter().map(|power| (power + 1.0e-12).ln()).sum::<f32>() / bins.len() as f32).exp();

    (geometric_mean / arithmetic_mean).clamp(0.0, 1.0)
}

fn calculate_band_energies(powers: &[f32]) -> BandEnergies {
    let mut bands = BandEnergies::default();
    let bin_width = SAMPLE_RATE as f32 / FFT_SIZE as f32;

    for (bin, power) in powers.iter().enumerate().skip(1) {
        let frequency = bin as f32 * bin_width;
        if frequency < 250.0 {
            bands.low += power;
        } else if frequency < 2_000.0 {
            bands.mid += power;
        } else {
            bands.high += power;
        }
    }

    let total = bands.low + bands.mid + bands.high;
    if total > f32::EPSILON {
        bands.low /= total;
        bands.mid /= total;
        bands.high /= total;
    }
    bands
}

fn dominant_frequency(powers: &[f32]) -> Option<f32> {
    let bin_width_hz = SAMPLE_RATE as f32 / FFT_SIZE as f32;
    let first_bin = (MIN_FREQUENCY_HZ / bin_width_hz).ceil() as usize;
    let last_bin = (MAX_FREQUENCY_HZ / bin_width_hz).floor() as usize;
    let dominant_bin = (first_bin..=last_bin).max_by(|left, right| {
        powers[*left]
            .partial_cmp(&powers[*right])
            .unwrap_or(std::cmp::Ordering::Equal)
    })?;

    Some(dominant_bin as f32 * bin_width_hz)
}

fn calculate_rhythmic_irregularity(onset_times: &VecDeque<f32>) -> f32 {
    if onset_times.len() < 5 {
        return 0.0;
    }

    let intervals: Vec<f32> = onset_times
        .iter()
        .zip(onset_times.iter().skip(1))
        .map(|(left, right)| right - left)
        .collect();
    let mean = intervals.iter().sum::<f32>() / intervals.len() as f32;
    if mean <= f32::EPSILON {
        return 0.0;
    }

    let variance = intervals
        .iter()
        .map(|interval| (interval - mean).powi(2))
        .sum::<f32>()
        / intervals.len() as f32;
    (variance.sqrt() / mean / 0.75).clamp(0.0, 1.0)
}

fn estimate_tempo(envelope: &VecDeque<f32>, previous_bpm: Option<f32>) -> (Option<f32>, f32) {
    let values: Vec<f32> = envelope.iter().copied().collect();
    let frame_rate = analysis_frame_rate();
    let minimum_lag = (frame_rate * 60.0 / TEMPO_MAX_BPM).floor() as usize;
    let maximum_lag = (frame_rate * 60.0 / TEMPO_MIN_BPM).ceil() as usize;
    if values.len() <= maximum_lag {
        return (None, 0.0);
    }

    let mut best_lag = 0;
    let mut best_score = 0.0;
    let mut correlations = Vec::with_capacity(maximum_lag - minimum_lag + 1);

    for lag in minimum_lag..=maximum_lag {
        let mut cross = 0.0;
        let mut left_energy = 0.0;
        let mut right_energy = 0.0;

        for index in lag..values.len() {
            let left = values[index];
            let right = values[index - lag];
            cross += left * right;
            left_energy += left * left;
            right_energy += right * right;
        }

        let correlation = if left_energy > 0.0 && right_energy > 0.0 {
            cross / (left_energy * right_energy).sqrt()
        } else {
            0.0
        };
        let bpm = 60.0 * frame_rate / lag as f32;
        let continuity = previous_bpm
            .map(|previous| {
                let octave_distance = (bpm / previous).log2().abs();
                (-octave_distance.powi(2) / 0.5).exp()
            })
            .unwrap_or(0.0);
        let score = correlation * (0.9 + 0.1 * continuity);
        correlations.push(correlation);

        if score > best_score {
            best_score = score;
            best_lag = lag;
        }
    }

    if best_lag == 0 || best_score <= f32::EPSILON {
        return (None, 0.0);
    }

    let mean = correlations.iter().sum::<f32>() / correlations.len() as f32;
    let best_correlation = correlations[best_lag - minimum_lag];
    let confidence = ((best_correlation - mean) / (1.0 - mean).max(0.001)).clamp(0.0, 1.0);
    (Some(60.0 * frame_rate / best_lag as f32), confidence)
}

fn smooth_feature(current: f32, target: f32) -> f32 {
    let amount = if target > current { 0.35 } else { 0.08 };
    current + (target - current) * amount
}

fn truncate_history(history: &mut VecDeque<f32>, maximum_length: usize) {
    while history.len() > maximum_length {
        history.pop_front();
    }
}

fn median(values: impl Iterator<Item = f32>) -> f32 {
    let mut values: Vec<f32> = values.collect();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

pub fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    (samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32).sqrt()
}

#[cfg(test)]
#[path = "analysis/tests.rs"]
mod tests;
