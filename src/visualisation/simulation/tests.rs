use std::collections::HashSet;
use std::time::Instant;

use super::*;
use crate::audio::BandEnergies;

fn loud_features() -> AudioFeatures {
    AudioFeatures {
        rms: 1.0,
        dominant_hz: Some(440.0),
        bands: BandEnergies {
            low: 0.3,
            mid: 0.4,
            high: 0.3,
        },
        ..AudioFeatures::default()
    }
}

fn default_canvas() -> CanvasSize {
    CanvasSize::default()
}

fn calculate_acceleration_brute_force(
    index: usize,
    boids: &[Boid],
    inputs: SimulationInputs,
) -> Vec2 {
    let boid = &boids[index];
    let mut separation = Vec2::ZERO;
    let mut alignment = Vec2::ZERO;
    let mut cohesion = Vec2::ZERO;
    let mut neighbours = 0_u32;

    for (other_index, other) in boids.iter().enumerate() {
        if index == other_index || other.visibility <= 0.0 {
            continue;
        }

        let offset = toroidal_offset(boid.position, other.position, default_canvas());
        let distance_squared = offset.length_squared();
        if distance_squared > NEIGHBOUR_RADIUS * NEIGHBOUR_RADIUS {
            continue;
        }

        neighbours += 1;
        alignment += other.velocity;
        cohesion += offset;
        if distance_squared < SEPARATION_RADIUS * SEPARATION_RADIUS {
            separation += offset * (-1.0 / distance_squared.max(1.0));
        }
    }

    if neighbours == 0 {
        return Vec2::ZERO;
    }

    let count = neighbours as f32;
    let separation = steering_force(separation, boid.velocity, inputs);
    let alignment = steering_force(alignment / count, boid.velocity, inputs);
    let cohesion = steering_force(cohesion / count, boid.velocity, inputs);
    (separation * inputs.separation_weight
        + alignment * inputs.alignment_weight
        + cohesion * inputs.cohesion_weight)
        .limited(inputs.max_force)
}

#[test]
fn population_maps_from_zero_to_maximum() {
    assert_eq!(
        SimulationInputs::from_audio(&AudioFeatures::default()).target_population(),
        0
    );
    assert_eq!(
        SimulationInputs::from_audio(&loud_features()).target_population(),
        MAX_BOIDS
    );
}

#[test]
fn spatial_grid_matches_brute_force_flocking() {
    let mut rng = XorShift64::new(99);
    let inputs = SimulationInputs::from_audio(&loud_features());
    let mut boids: Vec<Boid> = (0..200)
        .map(|_| {
            let mut boid = random_boid(&mut rng, inputs.max_speed, default_canvas());
            boid.visibility = 1.0;
            boid
        })
        .collect();
    boids[0].position = Vec2::new(2.0, 2.0);
    boids[1].position = Vec2::new(default_canvas().width - 2.0, default_canvas().height - 2.0);

    let mut grid = SpatialGrid::new(default_canvas());
    grid.rebuild(&boids);
    for index in 0..boids.len() {
        let grid_result = calculate_acceleration(index, &boids, &grid, inputs);
        let brute_result = calculate_acceleration_brute_force(index, &boids, inputs);
        assert!((grid_result.x - brute_result.x).abs() < 0.01);
        assert!((grid_result.y - brute_result.y).abs() < 0.01);
    }
}

#[test]
fn spatial_grid_has_nine_unique_neighbouring_cells() {
    let grid = SpatialGrid::new(default_canvas());
    for neighbouring_cells in &grid.neighbouring_cells {
        let unique: HashSet<usize> = neighbouring_cells.iter().copied().collect();
        assert_eq!(unique.len(), 9);
    }
}

#[test]
fn narrow_canvas_grid_does_not_visit_the_same_cell_twice() {
    let grid = SpatialGrid::new(CanvasSize {
        width: 100.0,
        height: 480.0,
    });
    let neighbours = grid.neighbouring_cells_for(Vec2::new(50.0, 240.0));
    let unique: HashSet<usize> = neighbours.iter().copied().collect();

    assert_eq!(unique.len(), neighbours.len());
}

#[test]
fn catch_up_work_is_bounded() {
    let mut simulation = BoidSimulation::with_seed(100);
    let stats = simulation.update(MAX_FRAME_TIME, &AudioFeatures::default());

    assert_eq!(stats.fixed_steps, MAX_FIXED_STEPS_PER_FRAME);
    assert!(stats.dropped_seconds >= FIXED_TIME_STEP);
    assert!(simulation.accumulator < FIXED_TIME_STEP);
}

#[test]
#[ignore = "manual simulation performance benchmark"]
fn benchmark_population_sizes() {
    for population in [100, 250, 500, 1_000] {
        let mut simulation = BoidSimulation::with_seed(population as u64);
        let inputs = SimulationInputs::from_audio(&loud_features());
        simulation.reconcile_population(population, inputs.max_speed);
        for boid in &mut simulation.boids {
            boid.visibility = 1.0;
        }

        for _ in 0..30 {
            simulation.step(FIXED_TIME_STEP, inputs);
        }

        const ITERATIONS: usize = 100;
        let grid_started = Instant::now();
        for _ in 0..ITERATIONS {
            simulation.spatial_grid.rebuild(&simulation.boids);
            for index in 0..simulation.boids.len() {
                std::hint::black_box(calculate_acceleration(
                    index,
                    &simulation.boids,
                    &simulation.spatial_grid,
                    inputs,
                ));
            }
        }
        let grid_ms = grid_started.elapsed().as_secs_f64() * 1_000.0 / ITERATIONS as f64;

        let brute_started = Instant::now();
        for _ in 0..ITERATIONS {
            for index in 0..simulation.boids.len() {
                std::hint::black_box(calculate_acceleration_brute_force(
                    index,
                    &simulation.boids,
                    inputs,
                ));
            }
        }
        let brute_ms = brute_started.elapsed().as_secs_f64() * 1_000.0 / ITERATIONS as f64;
        eprintln!(
            "{population} boids: grid {grid_ms:.3} ms, brute {brute_ms:.3} ms, {:.1}x faster",
            brute_ms / grid_ms
        );
    }
}

#[test]
fn fast_paced_audio_has_a_much_higher_speed_limit() {
    let slow = AudioFeatures {
        dominant_hz: Some(440.0),
        onset_rate_hz: 0.5,
        bpm: Some(60.0),
        tempo_confidence: 1.0,
        ..AudioFeatures::default()
    };
    let fast = AudioFeatures {
        onset_rate_hz: 6.0,
        bpm: Some(190.0),
        ..slow
    };

    let slow_inputs = SimulationInputs::from_audio(&slow);
    let fast_inputs = SimulationInputs::from_audio(&fast);
    assert!(fast_inputs.max_speed > slow_inputs.max_speed * 2.5);
    assert!(fast_inputs.max_force > slow_inputs.max_force * 2.5);
}

#[test]
fn boids_fade_in_and_are_removed_after_fading_out() {
    let mut simulation = BoidSimulation::with_seed(1);
    let loud = loud_features();
    for _ in 0..60 {
        simulation.update(1.0 / 60.0, &loud);
    }
    assert!(!simulation.boids.is_empty());
    assert!(simulation.boids.iter().all(|boid| boid.visibility > 0.0));

    for _ in 0..300 {
        simulation.update(1.0 / 60.0, &AudioFeatures::default());
    }
    assert!(simulation.boids.is_empty());
}

#[test]
fn wrapping_keeps_positions_inside_the_canvas() {
    let mut simulation = BoidSimulation::with_seed(2);
    simulation.boids.push(Boid {
        position: Vec2::new(default_canvas().width - 0.1, default_canvas().height - 0.1),
        velocity: Vec2::new(100.0, 100.0),
        visibility: 1.0,
        target_visible: true,
        wander_angle: 0.0,
        ripple_pulse: 0.0,
        colour_index: 0,
    });
    let mut inputs = SimulationInputs::from_audio(&loud_features());
    inputs.wander_strength = 0.0;
    simulation.step(0.1, inputs);
    let position = simulation.boids[0].position;
    assert!((0.0..simulation.canvas.width).contains(&position.x));
    assert!((0.0..simulation.canvas.height).contains(&position.y));
}

#[test]
fn resizing_preserves_relative_positions_and_rebuilds_the_grid() {
    let mut simulation = BoidSimulation::with_seed(7);
    simulation.boids.push(Boid {
        position: Vec2::new(320.0, 240.0),
        velocity: Vec2::new(10.0, 5.0),
        visibility: 1.0,
        target_visible: true,
        wander_angle: 0.0,
        ripple_pulse: 0.0,
        colour_index: 0,
    });

    simulation.resize_surface(3_840, 2_160);

    let canvas = simulation.canvas();
    assert!((simulation.boids[0].position.x / canvas.width - 0.5).abs() < 0.001);
    assert!((simulation.boids[0].position.y / canvas.height - 0.5).abs() < 0.001);
    assert_eq!(simulation.spatial_grid.canvas, canvas);
}

#[test]
fn each_new_beat_spawns_a_primary_and_two_trailing_ripples() {
    let mut simulation = BoidSimulation::with_seed(3);
    let mut features = loud_features();
    features.beat_count = 1;
    features.beat_strength = 0.8;

    simulation.update(FIXED_TIME_STEP, &features);
    assert_eq!(simulation.ripples.len(), 3);
    assert_eq!(simulation.ripples[0].delay_seconds, 0.0);
    assert!((simulation.ripples[0].strength - 0.4).abs() < f32::EPSILON);
    assert_eq!(simulation.ripples[0].width, RIPPLE_WIDTH);
    assert!(simulation.ripples[1].delay_seconds > 0.0);
    assert!(simulation.ripples[1].strength < simulation.ripples[0].strength);
    assert!(simulation.ripples[1].width < simulation.ripples[0].width);
    assert_eq!(simulation.ripples[2].force_multiplier, 0.0);

    simulation.update(FIXED_TIME_STEP, &features);
    assert_eq!(simulation.ripples.len(), 3);

    features.beat_count = 2;
    simulation.update(FIXED_TIME_STEP, &features);
    assert_eq!(simulation.ripples.len(), 6);
}

#[test]
fn ripple_only_affects_boids_at_its_wavefront() {
    let ripple = BeatRipple {
        origin: Vec2::new(100.0, 100.0),
        radius: 50.0,
        strength: 1.0,
        width: RIPPLE_WIDTH,
        force_multiplier: 1.0,
        delay_seconds: 0.0,
    };

    let (acceleration, pulse) =
        ripple_effect_at(Vec2::new(150.0, 100.0), &[ripple], default_canvas());
    assert!(acceleration.x > 0.0);
    assert_eq!(pulse, 1.0);
    assert_eq!(
        ripple_effect_at(Vec2::new(110.0, 100.0), &[ripple], default_canvas()),
        (Vec2::ZERO, 0.0)
    );
    assert_eq!(
        ripple_effect_at(Vec2::new(180.0, 100.0), &[ripple], default_canvas()),
        (Vec2::ZERO, 0.0)
    );
}

#[test]
fn ripple_crosses_the_toroidal_edge() {
    let ripple = BeatRipple {
        origin: Vec2::new(default_canvas().width - 10.0, 100.0),
        radius: 20.0,
        strength: 1.0,
        width: RIPPLE_WIDTH,
        force_multiplier: 1.0,
        delay_seconds: 0.0,
    };

    let (acceleration, pulse) =
        ripple_effect_at(Vec2::new(10.0, 100.0), &[ripple], default_canvas());
    assert!(acceleration.x > 0.0);
    assert_eq!(pulse, 1.0);
}

#[test]
fn stronger_beats_create_stronger_ripple_forces() {
    let position = Vec2::new(150.0, 100.0);
    let weak = BeatRipple {
        origin: Vec2::new(100.0, 100.0),
        radius: 50.0,
        strength: 0.5,
        width: RIPPLE_WIDTH,
        force_multiplier: 1.0,
        delay_seconds: 0.0,
    };
    let strong = BeatRipple {
        strength: 1.0,
        ..weak
    };

    let (weak_force, weak_pulse) = ripple_effect_at(position, &[weak], default_canvas());
    let (strong_force, strong_pulse) = ripple_effect_at(position, &[strong], default_canvas());
    assert!(strong_force.length() > weak_force.length());
    assert!(strong_pulse > weak_pulse);
}

#[test]
fn ripple_is_removed_after_crossing_the_canvas() {
    let mut simulation = BoidSimulation::with_seed(5);
    simulation.ripples.push(BeatRipple {
        origin: Vec2::ZERO,
        radius: maximum_toroidal_distance(default_canvas()) + RIPPLE_WIDTH * 0.5,
        strength: 1.0,
        width: RIPPLE_WIDTH,
        force_multiplier: 1.0,
        delay_seconds: 0.0,
    });

    simulation.step(
        FIXED_TIME_STEP,
        SimulationInputs::from_audio(&AudioFeatures::default()),
    );
    assert!(simulation.ripples.is_empty());
}

#[test]
fn identical_seeds_produce_identical_flocks() {
    let mut first = BoidSimulation::with_seed(4);
    let mut second = BoidSimulation::with_seed(4);
    let features = loud_features();
    for _ in 0..30 {
        first.update(1.0 / 60.0, &features);
        second.update(1.0 / 60.0, &features);
    }
    assert_eq!(first.boids.len(), second.boids.len());
    for (left, right) in first.boids.iter().zip(&second.boids) {
        assert_eq!(left.position, right.position);
        assert_eq!(left.velocity, right.velocity);
        assert_eq!(left.colour_index, right.colour_index);
    }
}

#[test]
fn flock_uses_multiple_stable_colour_variants() {
    let mut simulation = BoidSimulation::with_seed(6);
    let inputs = SimulationInputs::from_audio(&loud_features());
    simulation.reconcile_population(100, inputs.max_speed);

    let original_indices: Vec<u8> = simulation
        .boids
        .iter()
        .map(|boid| boid.colour_index)
        .collect();
    let unique_indices: HashSet<u8> = original_indices.iter().copied().collect();
    assert!(unique_indices.len() > 1);
    assert!(
        original_indices
            .iter()
            .all(|index| usize::from(*index) < COLOUR_VARIANT_COUNT)
    );

    simulation.reconcile_population(50, inputs.max_speed);
    simulation.reconcile_population(100, inputs.max_speed);

    let reactivated_indices: Vec<u8> = simulation
        .boids
        .iter()
        .map(|boid| boid.colour_index)
        .collect();
    assert_eq!(reactivated_indices, original_indices);
}

#[test]
fn separation_pushes_close_boids_apart() {
    let boids = vec![
        Boid {
            position: Vec2::new(100.0, 100.0),
            velocity: Vec2::ZERO,
            visibility: 1.0,
            target_visible: true,
            wander_angle: 0.0,
            ripple_pulse: 0.0,
            colour_index: 0,
        },
        Boid {
            position: Vec2::new(105.0, 100.0),
            velocity: Vec2::ZERO,
            visibility: 1.0,
            target_visible: true,
            wander_angle: 0.0,
            ripple_pulse: 0.0,
            colour_index: 1,
        },
    ];
    let inputs = SimulationInputs {
        population_level: 0.0,
        max_speed: 100.0,
        max_force: 50.0,
        separation_weight: 1.0,
        alignment_weight: 0.0,
        cohesion_weight: 0.0,
        wander_strength: 0.0,
    };
    let mut grid = SpatialGrid::new(default_canvas());
    grid.rebuild(&boids);
    assert!(calculate_acceleration(0, &boids, &grid, inputs).x < 0.0);
    assert!(calculate_acceleration(1, &boids, &grid, inputs).x > 0.0);
}
