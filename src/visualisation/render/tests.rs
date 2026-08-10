use super::*;
use crate::visualisation::MAX_BOIDS;

#[test]
fn background_modes_resolve_to_the_expected_clear_colour() {
    let base = OklchColour {
        lightness: 0.6,
        chroma: 0.15,
        hue_degrees: 200.0,
    };
    let palette = ColourPalette::from_base(base);
    for (mode, expected) in [
        (BackgroundMode::Black, BLACK_BACKGROUND),
        (BackgroundMode::White, WHITE_BACKGROUND),
        (BackgroundMode::Transparent, TRANSPARENT_BACKGROUND),
        (BackgroundMode::Boid, base.to_rgba()),
    ] {
        assert_eq!(mode.colour(&palette), expected);
    }
}

#[test]
fn background_modes_cycle_in_display_order() {
    assert_eq!(BackgroundMode::Boid.next(), BackgroundMode::Transparent);
    assert_eq!(BackgroundMode::Transparent.next(), BackgroundMode::White);
    assert_eq!(BackgroundMode::White.next(), BackgroundMode::Black);
    assert_eq!(BackgroundMode::Black.next(), BackgroundMode::Boid);
}

#[test]
fn boid_instance_preserves_inputs_and_clamps_effects() {
    let instance = BoidInstance::new([320.0, 240.0], [1.0, 0.0], 2.0, -1.0, [240, 80, 160, 255]);

    assert_eq!(instance.position, [320.0, 240.0]);
    assert_eq!(instance.velocity, [1.0, 0.0]);
    assert_eq!(instance.colour, [240, 80, 160, 255]);
    assert_eq!(instance.visibility, 1.0);
    assert_eq!(instance.ripple_pulse, 0.0);
}

#[test]
fn gpu_instance_layout_matches_the_vertex_buffer_layout() {
    assert_eq!(std::mem::size_of::<BoidInstance>(), 28);
    assert_eq!(std::mem::align_of::<BoidInstance>(), 4);
    assert_eq!(std::mem::offset_of!(BoidInstance, position), 0);
    assert_eq!(std::mem::offset_of!(BoidInstance, velocity), 8);
    assert_eq!(std::mem::offset_of!(BoidInstance, colour), 16);
    assert_eq!(std::mem::offset_of!(BoidInstance, visibility), 20);
    assert_eq!(std::mem::offset_of!(BoidInstance, ripple_pulse), 24);
}

#[test]
#[ignore = "manual GPU performance benchmark"]
fn benchmark_gpu_renderer_at_4k() {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("a high-performance GPU adapter is required");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("a GPU device is required");
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("4k benchmark target"),
            size: wgpu::Extent3d {
                width: 3_840,
                height: 2_160,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let renderer = GpuRenderer::new(&device, format, MAX_BOIDS);
        let canvas = CanvasSize::from_surface(3_840, 2_160);
        let palette = ColourPalette::from_base(OklchColour {
            lightness: 0.65,
            chroma: 0.18,
            hue_degrees: 220.0,
        });
        let instances: Vec<BoidInstance> = (0..MAX_BOIDS)
            .map(|index| {
                let column = index % 40;
                let row = index / 40;
                BoidInstance::new(
                    [
                        (column as f32 + 0.5) * canvas.width / 40.0,
                        (row as f32 + 0.5) * canvas.height / 25.0,
                    ],
                    [80.0, 40.0],
                    1.0,
                    (index % 10) as f32 / 10.0,
                    palette.colour_for((index % COLOUR_VARIANT_COUNT) as u8, 0.5),
                )
            })
            .collect();
        renderer.prepare(&queue, canvas.as_array(), &instances);

        let render_frame = || {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("4k benchmark encoder"),
            });
            renderer.render(
                &mut encoder,
                &target_view,
                BackgroundMode::Boid,
                &palette,
                instances.len(),
            );
            queue.submit(Some(encoder.finish()));
        };

        for _ in 0..30 {
            render_frame();
        }
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU warmup should finish");

        const FRAMES: usize = 300;
        let started = std::time::Instant::now();
        for _ in 0..FRAMES {
            render_frame();
        }
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU benchmark should finish");
        let milliseconds = started.elapsed().as_secs_f64() * 1_000.0 / FRAMES as f64;
        eprintln!(
            "4k GPU render: {milliseconds:.3} ms/frame on {}",
            adapter.get_info().name
        );
    });
}

#[test]
fn clear_colour_converts_srgb_to_linear_and_preserves_alpha() {
    let clear = clear_colour([128, 64, 32, 128]);
    assert!((clear.r - srgb_to_linear(128) as f64).abs() < f64::EPSILON);
    assert!((clear.g - srgb_to_linear(64) as f64).abs() < f64::EPSILON);
    assert!((clear.b - srgb_to_linear(32) as f64).abs() < f64::EPSILON);
    assert!((clear.a - 128.0 / 255.0).abs() < f64::EPSILON);
}

#[test]
fn missing_pitch_target_is_black() {
    assert_eq!(oklch_from_audio(0.0, None).to_rgba(), [0, 0, 0, 255]);
}

#[test]
fn silence_holds_the_last_colour_while_boids_fade_out() {
    let audible = AudioFeatures {
        rms: 0.2,
        dominant_hz: Some(1_000.0),
        ..AudioFeatures::default()
    };
    let mut smoother = ColourSmoother::new();
    for _ in 0..300 {
        smoother.update(1.0 / 60.0, &audible);
    }
    let colour_before_silence = smoother.current;

    for _ in 0..300 {
        smoother.update(1.0 / 60.0, &AudioFeatures::default());
    }

    assert_eq!(smoother.current, colour_before_silence);
}

#[test]
fn silence_before_any_audio_stays_black() {
    let mut smoother = ColourSmoother::new();
    let palette = smoother.update(1.0, &AudioFeatures::default());

    assert_eq!(smoother.current, OklchColour::BLACK);
    assert_eq!(
        palette.colour_for((COLOUR_VARIANT_COUNT / 2) as u8, 0.0),
        [0, 0, 0, 255]
    );
}

#[test]
fn palette_centre_preserves_the_smoothed_audio_colour() {
    let base = OklchColour {
        lightness: 0.6,
        chroma: 0.15,
        hue_degrees: 200.0,
    };
    let palette = ColourPalette::from_base(base);

    assert_eq!(palette.normal[COLOUR_VARIANT_COUNT / 2], base.to_rgba());
}

#[test]
fn palette_spreads_boids_across_a_cohesive_hue_range() {
    let base = OklchColour {
        lightness: 0.6,
        chroma: 0.15,
        hue_degrees: 200.0,
    };
    let low = palette_variant(base, -1.0, false);
    let high = palette_variant(base, 1.0, false);
    let palette = ColourPalette::from_base(base);

    assert_eq!(low.hue_degrees, 155.0);
    assert_eq!(high.hue_degrees, 245.0);
    assert_ne!(palette.normal[0], palette.normal[COLOUR_VARIANT_COUNT - 1]);
}

#[test]
fn palette_hue_variation_wraps_around_the_colour_wheel() {
    let base = OklchColour {
        lightness: 0.6,
        chroma: 0.15,
        hue_degrees: 350.0,
    };

    let wrapped = palette_variant(base, 1.0, false);
    assert!((wrapped.hue_degrees - 35.0).abs() < f32::EPSILON);
    assert_eq!(wrapped.to_rgba()[3], 255);
}

#[test]
fn ripple_pulse_brightens_and_saturates_a_palette_variant() {
    let base = OklchColour {
        lightness: 0.55,
        chroma: 0.12,
        hue_degrees: 120.0,
    };
    let normal = palette_variant(base, 0.0, false);
    let pulsing = palette_variant(base, 0.0, true);
    let palette = ColourPalette::from_base(base);
    let index = (COLOUR_VARIANT_COUNT / 2) as u8;

    assert!(pulsing.lightness > normal.lightness);
    assert!(pulsing.chroma > normal.chroma);
    assert_eq!(palette.colour_for(index, 0.0), normal.to_rgba());
    assert_eq!(palette.colour_for(index, 1.0), pulsing.to_rgba());
}

#[test]
fn black_palette_remains_black_even_during_a_ripple() {
    let palette = ColourPalette::from_base(OklchColour::BLACK);

    for index in 0..COLOUR_VARIANT_COUNT {
        assert_eq!(palette.normal[index], [0, 0, 0, 255]);
        assert_eq!(palette.pulsing[index], [0, 0, 0, 255]);
        assert_eq!(palette.colour_for(index as u8, 1.0), [0, 0, 0, 255]);
    }
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
