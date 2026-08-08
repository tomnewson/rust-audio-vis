pub const WIDTH: u32 = 640;
pub const HEIGHT: u32 = 480;

const BACKGROUND: [u8; 4] = [0, 0, 0, 0];
const QUIET_COLOUR: [u8; 4] = [96, 100, 112, 255];
const MIN_DBFS: f32 = -50.0;
const MAX_DBFS: f32 = -10.0;
const MIN_FREQUENCY_HZ: f32 = 80.0;
const MAX_FREQUENCY_HZ: f32 = 4_000.0;
const MIN_LIGHTNESS: f32 = 0.35;
const MAX_LIGHTNESS: f32 = 0.85;
const MAX_CHROMA: f32 = 0.24;
const MIN_HUE_DEGREES: f32 = 20.0;
const MAX_HUE_DEGREES: f32 = 320.0;

/// Maps loudness and dominant frequency to an OKLCH colour, then converts it
/// to the sRGB bytes used by the pixel buffer.
///
/// Loudness controls chroma. Pitch controls both lightness and hue, using a
/// logarithmic frequency scale so each musical octave has similar visual room.
pub fn colour_from_audio(rms: f32, dominant_hz: Option<f32>) -> [u8; 4] {
    let Some(frequency) = dominant_hz.filter(|frequency| frequency.is_finite() && *frequency > 0.0)
    else {
        return QUIET_COLOUR;
    };

    let pitch_position = pitch_position(Some(frequency)).unwrap_or(0.0);
    let lightness = MIN_LIGHTNESS + pitch_position * (MAX_LIGHTNESS - MIN_LIGHTNESS);
    let hue = MIN_HUE_DEGREES + pitch_position * (MAX_HUE_DEGREES - MIN_HUE_DEGREES);
    let chroma = loudness_position(rms) * MAX_CHROMA;
    let [red, green, blue] = oklch_to_srgb(lightness, chroma, hue);

    [red, green, blue, 255]
}

pub(crate) fn loudness_position(rms: f32) -> f32 {
    if rms.is_nan() || rms <= 0.0 {
        return 0.0;
    }

    if rms == f32::INFINITY {
        return 1.0;
    }

    let dbfs = 20.0 * rms.log10();
    ((dbfs - MIN_DBFS) / (MAX_DBFS - MIN_DBFS)).clamp(0.0, 1.0)
}

pub(crate) fn pitch_position(dominant_hz: Option<f32>) -> Option<f32> {
    dominant_hz
        .filter(|frequency| frequency.is_finite() && *frequency > 0.0)
        .map(|frequency| {
            let frequency = frequency.clamp(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ);
            ((frequency / MIN_FREQUENCY_HZ).ln() / (MAX_FREQUENCY_HZ / MIN_FREQUENCY_HZ).ln())
                .clamp(0.0, 1.0)
        })
}

fn oklch_to_srgb(lightness: f32, chroma: f32, hue_degrees: f32) -> [u8; 3] {
    let chroma = gamut_mapped_chroma(lightness, chroma, hue_degrees);
    let linear_rgb = oklch_to_linear_srgb(lightness, chroma, hue_degrees);

    linear_rgb.map(|channel| {
        let encoded = linear_to_srgb(channel.clamp(0.0, 1.0));
        (encoded * 255.0).round() as u8
    })
}

/// Finds the largest displayable chroma while leaving OKLCH lightness and hue
/// unchanged. This avoids the hue shifts caused by clamping RGB channels
/// independently.
fn gamut_mapped_chroma(lightness: f32, requested_chroma: f32, hue_degrees: f32) -> f32 {
    if is_in_srgb_gamut(oklch_to_linear_srgb(
        lightness,
        requested_chroma,
        hue_degrees,
    )) {
        return requested_chroma;
    }

    let mut lower = 0.0;
    let mut upper = requested_chroma;

    for _ in 0..16 {
        let candidate = (lower + upper) / 2.0;
        let rgb = oklch_to_linear_srgb(lightness, candidate, hue_degrees);

        if is_in_srgb_gamut(rgb) {
            lower = candidate;
        } else {
            upper = candidate;
        }
    }

    lower
}

fn oklch_to_linear_srgb(lightness: f32, chroma: f32, hue_degrees: f32) -> [f32; 3] {
    let hue_radians = hue_degrees.to_radians();
    let a = chroma * hue_radians.cos();
    let b = chroma * hue_radians.sin();

    let l_root = lightness + 0.396_337_78 * a + 0.215_803_76 * b;
    let m_root = lightness - 0.105_561_346 * a - 0.063_854_17 * b;
    let s_root = lightness - 0.089_484_18 * a - 1.291_485_5 * b;

    let l = l_root * l_root * l_root;
    let m = m_root * m_root * m_root;
    let s = s_root * s_root * s_root;

    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_4 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

fn is_in_srgb_gamut(rgb: [f32; 3]) -> bool {
    rgb.into_iter()
        .all(|channel| channel.is_finite() && (0.0..=1.0).contains(&channel))
}

fn linear_to_srgb(channel: f32) -> f32 {
    if channel <= 0.003_130_8 {
        12.92 * channel
    } else {
        1.055 * channel.powf(1.0 / 2.4) - 0.055
    }
}

pub fn clear_frame(frame: &mut [u8]) {
    for pixel in frame.chunks_exact_mut(4) {
        pixel.copy_from_slice(&BACKGROUND);
    }
}

pub fn draw_boid(
    frame: &mut [u8],
    position: [f32; 2],
    velocity: [f32; 2],
    visibility: f32,
    ripple_pulse: f32,
    colour: [u8; 4],
) {
    if !position.into_iter().all(f32::is_finite) || !velocity.into_iter().all(f32::is_finite) {
        return;
    }

    let visibility = visibility.clamp(0.0, 1.0);
    let eased_visibility = visibility * visibility * (3.0 - 2.0 * visibility);
    if eased_visibility <= 0.0 {
        return;
    }

    let velocity_length = velocity[0].hypot(velocity[1]);
    let direction = if velocity_length > 0.001 {
        [velocity[0] / velocity_length, velocity[1] / velocity_length]
    } else {
        [1.0, 0.0]
    };
    let perpendicular = [-direction[1], direction[0]];
    let pulse_scale = 1.0 + ripple_pulse.clamp(0.0, 1.0) * 0.6;
    let length = 9.0 * eased_visibility * pulse_scale;
    let half_width = 4.0 * eased_visibility * pulse_scale;
    let rear_x = position[0] - direction[0] * length * 0.55;
    let rear_y = position[1] - direction[1] * length * 0.55;
    let vertices = [
        [
            position[0] + direction[0] * length,
            position[1] + direction[1] * length,
        ],
        [
            rear_x + perpendicular[0] * half_width,
            rear_y + perpendicular[1] * half_width,
        ],
        [
            rear_x - perpendicular[0] * half_width,
            rear_y - perpendicular[1] * half_width,
        ],
    ];

    fill_triangle(frame, vertices, colour, eased_visibility);
}

fn fill_triangle(frame: &mut [u8], vertices: [[f32; 2]; 3], colour: [u8; 4], opacity: f32) {
    let min_x = vertices
        .iter()
        .map(|vertex| vertex[0])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, WIDTH.saturating_sub(1) as f32) as u32;
    let max_x = vertices
        .iter()
        .map(|vertex| vertex[0])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, WIDTH.saturating_sub(1) as f32) as u32;
    let min_y = vertices
        .iter()
        .map(|vertex| vertex[1])
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, HEIGHT.saturating_sub(1) as f32) as u32;
    let max_y = vertices
        .iter()
        .map(|vertex| vertex[1])
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, HEIGHT.saturating_sub(1) as f32) as u32;

    let alpha = opacity * (colour[3] as f32 / 255.0);
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let edges = [
                edge(vertices[0], vertices[1], point),
                edge(vertices[1], vertices[2], point),
                edge(vertices[2], vertices[0], point),
            ];
            let has_negative = edges.iter().any(|value| *value < 0.0);
            let has_positive = edges.iter().any(|value| *value > 0.0);

            if !(has_negative && has_positive) {
                let start = ((y * WIDTH + x) * 4) as usize;
                if let Some(pixel) = frame.get_mut(start..start + 4) {
                    let destination_alpha = pixel[3] as f32 / 255.0;
                    let output_alpha = alpha + destination_alpha * (1.0 - alpha);

                    for channel in 0..3 {
                        let source = colour[channel] as f32 / 255.0;
                        let destination = pixel[channel] as f32 / 255.0;
                        let output = if output_alpha > 0.0 {
                            (source * alpha + destination * destination_alpha * (1.0 - alpha))
                                / output_alpha
                        } else {
                            0.0
                        };
                        pixel[channel] = (output * 255.0).round() as u8;
                    }
                    pixel[3] = (output_alpha * 255.0).round() as u8;
                }
            }
        }
    }
}

fn edge(start: [f32; 2], end: [f32; 2], point: [f32; 2]) -> f32 {
    (point[0] - start[0]) * (end[1] - start[1]) - (point[1] - start[1]) * (end[0] - start[0])
}

#[cfg(test)]
mod tests {
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

        let visible_pixels =
            |frame: &[u8]| frame.chunks_exact(4).filter(|pixel| pixel[3] > 0).count();
        assert!(visible_pixels(&pulsing) > visible_pixels(&normal));
    }

    #[test]
    fn silence_has_a_muted_colour() {
        assert_eq!(colour_from_audio(0.0, None), QUIET_COLOUR);
    }

    #[test]
    fn low_and_high_frequencies_have_different_colours() {
        let low = colour_from_audio(0.1, Some(80.0));
        let high = colour_from_audio(0.1, Some(4_000.0));

        assert_ne!(low, high);
    }

    #[test]
    fn higher_pitch_produces_higher_lightness() {
        let low = colour_from_audio(0.0, Some(80.0));
        let high = colour_from_audio(0.0, Some(4_000.0));
        let low_brightness: u16 = low[..3].iter().map(|channel| u16::from(*channel)).sum();
        let high_brightness: u16 = high[..3].iter().map(|channel| u16::from(*channel)).sum();

        assert!(high_brightness > low_brightness);
    }

    #[test]
    fn louder_audio_produces_more_chroma() {
        let quiet = colour_from_audio(0.0, Some(440.0));
        let loud = colour_from_audio(0.32, Some(440.0));
        let channel_spread = |colour: [u8; 4]| {
            let channels = &colour[..3];
            channels.iter().max().unwrap() - channels.iter().min().unwrap()
        };

        assert!(channel_spread(loud) > channel_spread(quiet));
    }

    #[test]
    fn invalid_frequency_has_a_muted_colour() {
        assert_eq!(colour_from_audio(0.1, Some(f32::NAN)), QUIET_COLOUR);
        assert_eq!(colour_from_audio(0.1, Some(f32::INFINITY)), QUIET_COLOUR);
        assert_eq!(colour_from_audio(0.1, Some(-440.0)), QUIET_COLOUR);
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
}
