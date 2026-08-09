use crate::audio::AudioFeatures;
use crate::visualisation::canvas::CanvasSize;
use bytemuck::{Pod, Zeroable};
use pixels::wgpu::{self, util::DeviceExt};

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
            Self::Transparent => Self::Boid,
            Self::Boid => Self::White,
            Self::White => Self::Black,
            Self::Black => Self::Transparent,
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

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BoidInstance {
    position: [f32; 2],
    velocity: [f32; 2],
    colour: [u8; 4],
    visibility: f32,
    ripple_pulse: f32,
}

impl BoidInstance {
    pub fn new(
        position: [f32; 2],
        velocity: [f32; 2],
        visibility: f32,
        ripple_pulse: f32,
        colour: [u8; 4],
    ) -> Self {
        Self {
            position,
            velocity,
            colour,
            visibility: finite_unit(visibility),
            ripple_pulse: finite_unit(ripple_pulse),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewportUniform {
    size: [f32; 2],
    _padding: [f32; 2],
}

pub struct GpuRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    instance_buffer: wgpu::Buffer,
    viewport_buffer: wgpu::Buffer,
    viewport_bind_group: wgpu::BindGroup,
    max_instances: usize,
}

impl GpuRenderer {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        max_instances: usize,
    ) -> Self {
        let vertices: [[f32; 2]; 3] = [[1.0, 0.0], [-0.55, 1.0], [-0.55, -1.0]];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boid vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("boid instances"),
            size: (max_instances * std::mem::size_of::<BoidInstance>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let viewport = ViewportUniform {
            size: CanvasSize::default().as_array(),
            _padding: [0.0; 2],
        };
        let viewport_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("boid viewport"),
            contents: bytemuck::bytes_of(&viewport),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let viewport_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("boid viewport layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<ViewportUniform>() as u64
                    ),
                },
                count: None,
            }],
        });
        let viewport_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("boid viewport bind group"),
            layout: &viewport_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: viewport_buffer.as_entire_binding(),
            }],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("boid shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("boids.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("boid pipeline layout"),
            bind_group_layouts: &[Some(&viewport_layout)],
            immediate_size: 0,
        });
        let vertex_attributes = wgpu::vertex_attr_array![0 => Float32x2];
        let instance_attributes = wgpu::vertex_attr_array![
            1 => Float32x2,
            2 => Float32x2,
            3 => Unorm8x4,
            4 => Float32x2
        ];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("boid pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attributes,
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<BoidInstance>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &instance_attributes,
                    },
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            vertex_buffer,
            instance_buffer,
            viewport_buffer,
            viewport_bind_group,
            max_instances,
        }
    }

    pub fn prepare(&self, queue: &wgpu::Queue, viewport: [f32; 2], instances: &[BoidInstance]) {
        assert!(instances.len() <= self.max_instances);
        let viewport = ViewportUniform {
            size: viewport,
            _padding: [0.0; 2],
        };
        queue.write_buffer(&self.viewport_buffer, 0, bytemuck::bytes_of(&viewport));
        if !instances.is_empty() {
            queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(instances));
        }
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        render_target: &wgpu::TextureView,
        background: BackgroundMode,
        palette: &ColourPalette,
        instance_count: usize,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("boid render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: render_target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear_colour(background.colour(palette))),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.viewport_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        pass.draw(0..3, 0..instance_count as u32);
    }
}

fn clear_colour(colour: [u8; 4]) -> wgpu::Color {
    wgpu::Color {
        r: srgb_to_linear(colour[0]) as f64,
        g: srgb_to_linear(colour[1]) as f64,
        b: srgb_to_linear(colour[2]) as f64,
        a: colour[3] as f64 / 255.0,
    }
}

fn srgb_to_linear(channel: u8) -> f32 {
    let encoded = channel as f32 / 255.0;
    if encoded <= 0.040_45 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
#[path = "render/tests.rs"]
mod tests;
