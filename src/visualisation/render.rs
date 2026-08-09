use crate::audio::AudioFeatures;

pub const WIDTH: u32 = 640;
pub const HEIGHT: u32 = 480;

const BLACK_BACKGROUND: [u8; 4] = [0, 0, 0, 255];
const WHITE_BACKGROUND: [u8; 4] = [255, 255, 255, 255];
const TRANSPARENT_BACKGROUND: [u8; 4] = [0, 0, 0, 0];
const MIN_DBFS: f32 = -50.0;
const MAX_DBFS: f32 = -10.0;
const MIN_FREQUENCY_HZ: f32 = 80.0;
const MAX_FREQUENCY_HZ: f32 = 4_000.0;
const MIN_LIGHTNESS: f32 = 0.35;
const MAX_LIGHTNESS: f32 = 0.85;
const MAX_CHROMA: f32 = 0.24;
const MIN_HUE_DEGREES: f32 = 20.0;
const MAX_HUE_DEGREES: f32 = 320.0;
const SLOW_COLOUR_TAU_SECONDS: f32 = 0.32;
const FAST_COLOUR_TAU_SECONDS: f32 = 0.055;
const BEAT_RESPONSE_SECONDS: f32 = 0.22;
pub(super) const COLOUR_VARIANT_COUNT: usize = 33;
const HUE_VARIATION_DEGREES: f32 = 45.0;
const LIGHTNESS_VARIATION: f32 = 0.035;
const PULSE_LIGHTNESS_BOOST: f32 = 0.12;
const PULSE_CHROMA_BOOST: f32 = 0.04;
const MAX_PALETTE_LIGHTNESS: f32 = 0.95;
const MAX_PALETTE_CHROMA: f32 = 0.30;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BackgroundMode {
    Black,
    White,
    #[default]
    Transparent,
    Boid,
}

impl BackgroundMode {
    pub fn next(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Transparent,
            Self::Transparent => Self::Boid,
            Self::Boid => Self::Black,
        }
    }

    fn colour(self, palette: &ColourPalette) -> [u8; 4] {
        match self {
            Self::Black => BLACK_BACKGROUND,
            Self::White => WHITE_BACKGROUND,
            Self::Transparent => TRANSPARENT_BACKGROUND,
            Self::Boid => palette.base_colour(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OklchColour {
    lightness: f32,
    chroma: f32,
    hue_degrees: f32,
}

impl OklchColour {
    const BLACK: Self = Self {
        lightness: 0.0,
        chroma: 0.0,
        hue_degrees: MIN_HUE_DEGREES,
    };

    fn to_rgba(self) -> [u8; 4] {
        let [red, green, blue] = oklch_to_srgb(self.lightness, self.chroma, self.hue_degrees);
        [red, green, blue, 255]
    }
}

#[derive(Debug, Clone)]
pub struct ColourPalette {
    normal: [[u8; 4]; COLOUR_VARIANT_COUNT],
    pulsing: [[u8; 4]; COLOUR_VARIANT_COUNT],
}

impl ColourPalette {
    fn from_base(base: OklchColour) -> Self {
        let normal = std::array::from_fn(|index| {
            palette_variant(base, palette_phase(index), false).to_rgba()
        });
        let pulsing = std::array::from_fn(|index| {
            palette_variant(base, palette_phase(index), true).to_rgba()
        });
        Self { normal, pulsing }
    }

    pub fn colour_for(&self, index: u8, ripple_pulse: f32) -> [u8; 4] {
        let index = usize::from(index).min(COLOUR_VARIANT_COUNT - 1);
        let normal = self.normal[index];
        let pulsing = self.pulsing[index];
        let amount = finite_unit(ripple_pulse);
        let mut colour = [0; 4];
        for channel in 0..3 {
            colour[channel] = (normal[channel] as f32
                + (pulsing[channel] as f32 - normal[channel] as f32) * amount)
                .round() as u8;
        }
        colour[3] = 255;
        colour
    }

    fn base_colour(&self) -> [u8; 4] {
        self.normal[COLOUR_VARIANT_COUNT / 2]
    }
}

fn palette_phase(index: usize) -> f32 {
    index as f32 / (COLOUR_VARIANT_COUNT - 1) as f32 * 2.0 - 1.0
}

fn palette_variant(base: OklchColour, phase: f32, pulsing: bool) -> OklchColour {
    if base.lightness <= 0.000_1 {
        return OklchColour::BLACK;
    }

    OklchColour {
        lightness: (base.lightness
            + phase * LIGHTNESS_VARIATION
            + if pulsing { PULSE_LIGHTNESS_BOOST } else { 0.0 })
        .clamp(0.0, MAX_PALETTE_LIGHTNESS),
        chroma: (base.chroma + if pulsing { PULSE_CHROMA_BOOST } else { 0.0 })
            .clamp(0.0, MAX_PALETTE_CHROMA),
        hue_degrees: (base.hue_degrees + phase * HUE_VARIATION_DEGREES).rem_euclid(360.0),
    }
}

pub struct ColourSmoother {
    current: OklchColour,
    last_beat_count: u64,
    beat_response: f32,
}

impl ColourSmoother {
    pub fn new() -> Self {
        Self {
            current: OklchColour::BLACK,
            last_beat_count: 0,
            beat_response: 0.0,
        }
    }

    pub fn update(&mut self, elapsed_seconds: f32, features: &AudioFeatures) -> ColourPalette {
        let elapsed_seconds = if elapsed_seconds.is_finite() {
            elapsed_seconds.clamp(0.0, 0.1)
        } else {
            0.0
        };

        if features.beat_count != self.last_beat_count {
            self.beat_response = self
                .beat_response
                .max(0.5 + finite_unit(features.beat_strength) * 0.5);
            self.last_beat_count = features.beat_count;
        }

        let responsiveness = finite_unit(features.spectral_flux).max(self.beat_response);
        let tau = SLOW_COLOUR_TAU_SECONDS
            + responsiveness * (FAST_COLOUR_TAU_SECONDS - SLOW_COLOUR_TAU_SECONDS);
        let interpolation = if elapsed_seconds > 0.0 {
            1.0 - (-elapsed_seconds / tau).exp()
        } else {
            0.0
        };
        let target = oklch_from_audio(features.rms, features.dominant_hz);

        if target.lightness > 0.000_1 {
            if target.chroma > 0.000_1
                && (self.current.chroma <= 0.000_1 || self.current.lightness <= 0.000_1)
            {
                self.current.hue_degrees = target.hue_degrees;
            } else if target.chroma > 0.000_1 {
                self.current.hue_degrees = (self.current.hue_degrees
                    + shortest_hue_delta(self.current.hue_degrees, target.hue_degrees)
                        * interpolation)
                    .rem_euclid(360.0);
            }

            self.current.lightness += (target.lightness - self.current.lightness) * interpolation;
            self.current.chroma += (target.chroma - self.current.chroma) * interpolation;
        }
        self.beat_response =
            (self.beat_response - elapsed_seconds / BEAT_RESPONSE_SECONDS).max(0.0);

        ColourPalette::from_base(self.current)
    }
}

impl Default for ColourSmoother {
    fn default() -> Self {
        Self::new()
    }
}

/// Maps loudness and dominant frequency to an OKLCH target. Loudness controls
/// chroma, while pitch controls lightness and hue on a logarithmic scale.
fn oklch_from_audio(rms: f32, dominant_hz: Option<f32>) -> OklchColour {
    let Some(frequency) = dominant_hz.filter(|frequency| frequency.is_finite() && *frequency > 0.0)
    else {
        return OklchColour::BLACK;
    };

    let pitch_position = pitch_position(Some(frequency)).unwrap_or(0.0);
    let lightness = MIN_LIGHTNESS + pitch_position * (MAX_LIGHTNESS - MIN_LIGHTNESS);
    let hue = MIN_HUE_DEGREES + pitch_position * (MAX_HUE_DEGREES - MIN_HUE_DEGREES);
    let chroma = loudness_position(rms) * MAX_CHROMA;
    OklchColour {
        lightness,
        chroma,
        hue_degrees: hue,
    }
}

fn shortest_hue_delta(current: f32, target: f32) -> f32 {
    (target - current + 180.0).rem_euclid(360.0) - 180.0
}

fn finite_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
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

pub fn clear_frame(frame: &mut [u8], background: BackgroundMode, palette: &ColourPalette) {
    let colour = background.colour(palette);
    for pixel in frame.chunks_exact_mut(4) {
        pixel.copy_from_slice(&colour);
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
                    if alpha >= 1.0 {
                        pixel.copy_from_slice(&colour);
                    } else {
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
}

fn edge(start: [f32; 2], end: [f32; 2], point: [f32; 2]) -> f32 {
    (point[0] - start[0]) * (end[1] - start[1]) - (point[1] - start[1]) * (end[0] - start[0])
}

#[cfg(test)]
#[path = "render/tests.rs"]
mod tests;
