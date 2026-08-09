use super::*;

fn sine_wave(frequency_hz: f32, amplitude: f32, sample_count: usize) -> Vec<f32> {
    (0..sample_count)
        .map(|sample_index| {
            let time = sample_index as f32 / SAMPLE_RATE as f32;
            amplitude * (std::f32::consts::TAU * frequency_hz * time).sin()
        })
        .collect()
}

fn analyze_sine(frequency_hz: f32, amplitude: f32) -> AudioFeatures {
    let mut analyzer = Analyzer::new();
    analyzer
        .push(&sine_wave(frequency_hz, amplitude, FFT_SIZE))
        .expect("a complete FFT window should produce audio features")
}

#[test]
fn empty_is_zero() {
    assert_eq!(calculate_rms(&[]), 0.0);
}

#[test]
fn zero_is_zero() {
    assert_eq!(calculate_rms(&[0.0, 0.0]), 0.0);
}

#[test]
fn positive_and_negative_one_have_rms_of_one() {
    let actual = calculate_rms(&[1.0, -1.0]);
    assert!((actual - 1.0).abs() < 0.001);
}

#[test]
fn half_amplitude_sine_has_expected_rms() {
    let input = sine_wave(440.0, 0.5, SAMPLE_RATE as usize);
    let actual = calculate_rms(&input);
    let expected = 0.5 / 2.0_f32.sqrt();
    assert!((actual - expected).abs() < 0.001);
}

#[test]
fn incomplete_window_produces_no_features() {
    let mut analyzer = Analyzer::new();
    assert!(analyzer.push(&vec![0.0; FFT_SIZE - 1]).is_none());
}

#[test]
fn overlapping_windows_produce_a_new_frame_after_each_hop() {
    let mut analyzer = Analyzer::new();
    assert!(analyzer.push(&vec![0.0; FFT_SIZE]).is_some());
    assert!(analyzer.push(&vec![0.0; HOP_SIZE - 1]).is_none());
    assert!(analyzer.push(&[0.0]).is_some());
}

#[test]
fn silence_has_no_dominant_frequency() {
    let mut analyzer = Analyzer::new();
    let features = analyzer.push(&vec![0.0; FFT_SIZE]).unwrap();
    assert_eq!(features.rms, 0.0);
    assert_eq!(features.dominant_hz, None);
}

#[test]
fn detects_440_hz_sine() {
    let actual = analyze_sine(440.0, 0.5).dominant_hz.unwrap();
    assert!((actual - 440.0).abs() <= SAMPLE_RATE as f32 / FFT_SIZE as f32);
}

#[test]
fn detects_1000_hz_sine() {
    let actual = analyze_sine(1_000.0, 0.5).dominant_hz.unwrap();
    assert!((actual - 1_000.0).abs() <= SAMPLE_RATE as f32 / FFT_SIZE as f32);
}

#[test]
fn amplitude_changes_rms_without_changing_frequency() {
    let quiet = analyze_sine(440.0, 0.1);
    let loud = analyze_sine(440.0, 0.8);
    let bin_width_hz = SAMPLE_RATE as f32 / FFT_SIZE as f32;
    assert!(loud.rms > quiet.rms);
    assert!((loud.dominant_hz.unwrap() - quiet.dominant_hz.unwrap()).abs() <= bin_width_hz);
}

#[test]
fn tones_are_assigned_to_the_expected_frequency_bands() {
    let low = analyze_sine(100.0, 0.5).bands;
    let mid = analyze_sine(1_000.0, 0.5).bands;
    let high = analyze_sine(8_000.0, 0.5).bands;
    assert!(low.low > low.mid && low.low > low.high);
    assert!(mid.mid > mid.low && mid.mid > mid.high);
    assert!(high.high > high.low && high.high > high.mid);
}

#[test]
fn noise_is_flatter_than_a_sine_wave() {
    let sine = analyze_sine(440.0, 0.5).spectral_flatness;
    let mut state = 0x1234_5678_u32;
    let noise: Vec<f32> = (0..FFT_SIZE)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    let mut analyzer = Analyzer::new();
    let noise = analyzer.push(&noise).unwrap().spectral_flatness;
    assert!(noise > sine);
}

#[test]
fn sudden_sound_produces_more_flux_than_a_steady_tone() {
    let mut analyzer = Analyzer::new();
    analyzer.push(&vec![0.0; FFT_SIZE + HOP_SIZE]).unwrap();
    let onset = analyzer
        .push(&sine_wave(440.0, 0.8, HOP_SIZE))
        .unwrap()
        .spectral_flux;
    let mut steady = onset;
    for _ in 0..8 {
        steady = analyzer
            .push(&sine_wave(440.0, 0.8, HOP_SIZE))
            .unwrap()
            .spectral_flux;
    }
    assert!(onset > steady);
}

#[test]
fn irregular_onsets_score_higher_than_regular_onsets() {
    let regular = VecDeque::from(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
    let irregular = VecDeque::from(vec![0.0, 0.2, 0.9, 1.1, 2.0, 2.2]);
    assert!(
        calculate_rhythmic_irregularity(&irregular) > calculate_rhythmic_irregularity(&regular)
    );
}

#[test]
fn autocorrelation_finds_120_bpm_pulse_train() {
    let frame_rate = analysis_frame_rate();
    let mut envelope = VecDeque::new();
    for frame in 0..(frame_rate * 8.0) as usize {
        let time = frame as f32 / frame_rate;
        let phase = (time % 0.5).min(0.5 - (time % 0.5));
        envelope.push_back(if phase < 1.0 / frame_rate { 1.0 } else { 0.0 });
    }

    let (bpm, confidence) = estimate_tempo(&envelope, None);
    assert!((bpm.unwrap() - 120.0).abs() <= 2.0);
    assert!(confidence > 0.5);
}

#[test]
fn beat_strength_stays_latched_until_the_next_beat() {
    let mut analyzer = Analyzer::new();
    analyzer.update_beat_clock(0.1, Some((0.1, 0.8)));
    assert_eq!(analyzer.beat_count, 1);
    assert_eq!(analyzer.last_beat_strength, 0.8);

    analyzer.update_beat_clock(0.2, None);
    assert_eq!(analyzer.beat_count, 1);
    assert_eq!(analyzer.last_beat_strength, 0.8);
}

#[test]
fn analyzer_tracks_a_120_bpm_audio_pulse_train() {
    let mut samples = vec![0.0; SAMPLE_RATE as usize * 10];
    let beat_interval = SAMPLE_RATE as usize / 2;
    for beat_start in (beat_interval..samples.len()).step_by(beat_interval) {
        for offset in 0..1_024 {
            if let Some(sample) = samples.get_mut(beat_start + offset) {
                let time = offset as f32 / SAMPLE_RATE as f32;
                *sample = 0.8 * (std::f32::consts::TAU * 1_000.0 * time).sin();
            }
        }
    }

    let mut analyzer = Analyzer::new();
    let mut latest = AudioFeatures::default();
    for chunk in samples.chunks(960) {
        if let Some(features) = analyzer.push(chunk) {
            latest = features;
        }
    }

    assert!((latest.bpm.unwrap() - 120.0).abs() <= 2.0);
    assert!(latest.tempo_confidence > 0.35);
    assert!(latest.beat_count > 0);
    assert!(latest.beat_strength > 0.0);
}
