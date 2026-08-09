
use super::*;

fn pixel_at(frame: &[u8], x: u32, y: u32) -> &[u8] {
    let start = ((y * WIDTH + x) * 4) as usize;
    &frame[start..start + 4]
}

#[test]
fn clear_frame_is_fully_transparent() {
    let mut frame = vec![255; (WIDTH * HEIGHT * 4) as usize];
    clear_frame(&mut frame);
    assert_eq!(pixel_at(&frame, WIDTH / 2, HEIGHT / 2), BACKGROUND);
}

#[test]
fn boid_draws_in_its_colour() {
    let mut frame = vec![0; (WIDTH * HEIGHT * 4) as usize];
    clear_frame(&mut frame);
    draw_boid(
        &mut frame,
        [WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0],
        [1.0, 0.0],
        1.0,
        0.0,
        [240, 80, 160, 255],
    );
    assert_eq!(pixel_at(&frame, WIDTH / 2, HEIGHT / 2), [240, 80, 160, 255]);
}

#[test]
fn fading_boid_remains_transparent() {
    let mut frame = vec![0; (WIDTH * HEIGHT * 4) as usize];
    clear_frame(&mut frame);
    fill_triangle(
        &mut frame,
        [[10.0, 10.0], [20.0, 10.0], [10.0, 20.0]],
        [240, 80, 160, 255],
        0.5,
    );

    assert_eq!(pixel_at(&frame, 11, 11), [240, 80, 160, 128]);
}

#[test]
fn invisible_and_offscreen_boids_are_safe() {
    let mut frame = vec![0; (WIDTH * HEIGHT * 4) as usize];
    clear_frame(&mut frame);
    draw_boid(&mut frame, [20.0, 20.0], [1.0, 0.0], 0.0, 0.0, [255; 4]);
    draw_boid(&mut frame, [-100.0, -100.0], [1.0, 0.0], 1.0, 0.0, [255; 4]);
    assert_eq!(pixel_at(&frame, 20, 20), BACKGROUND);
}

#[test]
fn ripple_pulse_makes_a_boid_larger() {
    let mut normal = vec![0; (WIDTH * HEIGHT * 4) as usize];
    let mut pulsing = normal.clone();
    let position = [WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0];
    draw_boid(&mut normal, position, [1.0, 0.0], 1.0, 0.0, [255; 4]);
    draw_boid(&mut pulsing, position, [1.0, 0.0], 1.0, 1.0, [255; 4]);

    let visible_pixels = |frame: &[u8]| frame.chunks_exact(4).filter(|pixel| pixel[3] > 0).count();
    assert!(visible_pixels(&pulsing) > visible_pixels(&normal));
}

#[test]
fn silence_uses_black() {
    assert_eq!(oklch_from_audio(0.0, None).to_rgba(), [0, 0, 0, 255]);
}

#[test]
fn colour_smoothing_moves_toward_the_target_over_time() {
    let features = AudioFeatures {
        rms: 0.2,
        dominant_hz: Some(1_000.0),
        ..AudioFeatures::default()
    };
    let target = oklch_from_audio(features.rms, features.dominant_hz);
    let mut smoother = ColourSmoother::new();

    smoother.update(1.0 / 60.0, &features);
    assert!(smoother.current.lightness > 0.0);
    assert!(smoother.current.lightness < target.lightness);
    assert!(smoother.current.chroma < target.chroma);

    for _ in 0..300 {
        smoother.update(1.0 / 60.0, &features);
    }
    assert!((smoother.current.lightness - target.lightness).abs() < 0.001);
    assert!((smoother.current.chroma - target.chroma).abs() < 0.001);
}

#[test]
fn beats_make_colour_transitions_more_responsive() {
    let ordinary_features = AudioFeatures {
        rms: 0.2,
        dominant_hz: Some(1_000.0),
        ..AudioFeatures::default()
    };
    let beat_features = AudioFeatures {
        beat_count: 1,
        beat_strength: 1.0,
        ..ordinary_features
    };
    let mut ordinary = ColourSmoother::new();
    let mut on_beat = ColourSmoother::new();

    ordinary.update(1.0 / 60.0, &ordinary_features);
    on_beat.update(1.0 / 60.0, &beat_features);

    assert!(on_beat.current.lightness > ordinary.current.lightness);
    assert!(on_beat.current.chroma > ordinary.current.chroma);
}

#[test]
fn hue_uses_the_shortest_path_around_the_colour_wheel() {
    assert_eq!(shortest_hue_delta(350.0, 10.0), 20.0);
    assert_eq!(shortest_hue_delta(10.0, 350.0), -20.0);
}

#[test]
fn low_and_high_frequencies_have_different_colours() {
    let low = oklch_from_audio(0.1, Some(80.0)).to_rgba();
    let high = oklch_from_audio(0.1, Some(4_000.0)).to_rgba();

    assert_ne!(low, high);
}

#[test]
fn higher_pitch_produces_higher_lightness() {
    let low = oklch_from_audio(0.0, Some(80.0)).to_rgba();
    let high = oklch_from_audio(0.0, Some(4_000.0)).to_rgba();
    let low_brightness: u16 = low[..3].iter().map(|channel| u16::from(*channel)).sum();
    let high_brightness: u16 = high[..3].iter().map(|channel| u16::from(*channel)).sum();

    assert!(high_brightness > low_brightness);
}

#[test]
fn louder_audio_produces_more_chroma() {
    let quiet = oklch_from_audio(0.0, Some(440.0)).to_rgba();
    let loud = oklch_from_audio(0.32, Some(440.0)).to_rgba();
    let channel_spread = |colour: [u8; 4]| {
        let channels = &colour[..3];
        channels.iter().max().unwrap() - channels.iter().min().unwrap()
    };

    assert!(channel_spread(loud) > channel_spread(quiet));
}

#[test]
fn invalid_frequency_uses_black() {
    assert_eq!(
        oklch_from_audio(0.1, Some(f32::NAN)).to_rgba(),
        [0, 0, 0, 255]
    );
    assert_eq!(
        oklch_from_audio(0.1, Some(f32::INFINITY)).to_rgba(),
        [0, 0, 0, 255]
    );
    assert_eq!(
        oklch_from_audio(0.1, Some(-440.0)).to_rgba(),
        [0, 0, 0, 255]
    );
}

#[test]
fn out_of_gamut_colour_reduces_only_its_chroma() {
    let requested_chroma = 0.5;
    let mapped_chroma = gamut_mapped_chroma(0.7, requested_chroma, 140.0);
    let mapped_rgb = oklch_to_linear_srgb(0.7, mapped_chroma, 140.0);

    assert!(mapped_chroma > 0.0);
    assert!(mapped_chroma < requested_chroma);
    assert!(is_in_srgb_gamut(mapped_rgb));
}

#[test]
fn loudness_position_is_bounded() {
    assert_eq!(loudness_position(0.0), 0.0);
    assert_eq!(loudness_position(f32::NAN), 0.0);
    assert_eq!(loudness_position(f32::INFINITY), 1.0);
    assert_eq!(loudness_position(10.0), 1.0);
}
