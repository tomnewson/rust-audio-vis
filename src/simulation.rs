use crate::analysis::AudioFeatures;
use crate::render::{HEIGHT, WIDTH, draw_boid, loudness_position, pitch_position};

const MAX_BOIDS: usize = 250;
const FIXED_TIME_STEP: f32 = 1.0 / 60.0;
const MAX_FRAME_TIME: f32 = 0.1;
const LIFECYCLE_SECONDS: f32 = 0.35;
const NEIGHBOUR_RADIUS: f32 = 60.0;
const SEPARATION_RADIUS: f32 = 18.0;
const RIPPLE_SPEED: f32 = 360.0;
const RIPPLE_WIDTH: f32 = 40.0;
const RIPPLE_FORCE: f32 = 300.0;
const RIPPLE_SPEED_HEADROOM: f32 = 0.5;
const MIN_RIPPLE_STRENGTH: f32 = 0.6;
const MAX_ACTIVE_RIPPLES: usize = 6;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Vec2 {
    x: f32,
    y: f32,
}

impl Vec2 {
    const ZERO: Self = Self { x: 0.0, y: 0.0 };

    fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    fn length(self) -> f32 {
        self.x.hypot(self.y)
    }

    fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    fn normalized_or_zero(self) -> Self {
        let length = self.length();
        if length > 0.000_1 {
            self / length
        } else {
            Self::ZERO
        }
    }

    fn limited(self, maximum: f32) -> Self {
        let length = self.length();
        if length > maximum && length > 0.0 {
            self * (maximum / length)
        } else {
            self
        }
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }
}

impl std::ops::AddAssign for Vec2 {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self {
        Self::new(self.x * scalar, self.y * scalar)
    }
}

impl std::ops::Div<f32> for Vec2 {
    type Output = Self;

    fn div(self, scalar: f32) -> Self {
        Self::new(self.x / scalar, self.y / scalar)
    }
}

#[derive(Debug, Clone)]
struct Boid {
    position: Vec2,
    velocity: Vec2,
    visibility: f32,
    target_visible: bool,
    wander_angle: f32,
    ripple_pulse: f32,
}

#[derive(Debug, Clone, Copy)]
struct BeatRipple {
    origin: Vec2,
    radius: f32,
    strength: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct SimulationInputs {
    population_level: f32,
    max_speed: f32,
    max_force: f32,
    separation_weight: f32,
    alignment_weight: f32,
    cohesion_weight: f32,
    wander_strength: f32,
}

impl SimulationInputs {
    pub fn from_audio(features: &AudioFeatures) -> Self {
        let pitch = pitch_position(features.dominant_hz).unwrap_or(0.0);
        let chaos = finite_unit(features.chaos);
        let spectral_flatness = finite_unit(features.spectral_flatness);
        let low = finite_unit(features.bands.low);
        let mid = finite_unit(features.bands.mid);
        let high = finite_unit(features.bands.high);
        let rhythmic_irregularity = finite_unit(features.rhythmic_irregularity);
        let tempo_position = features
            .bpm
            .filter(|bpm| bpm.is_finite())
            .map(|bpm| ((bpm - 55.0) / (200.0 - 55.0)).clamp(0.0, 1.0))
            .unwrap_or(0.0)
            * finite_unit(features.tempo_confidence);

        Self {
            population_level: loudness_position(features.rms),
            max_speed: 30.0 + pitch * 120.0,
            max_force: 35.0
                + (features.onset_rate_hz / 8.0).clamp(0.0, 1.0) * 45.0
                + tempo_position * 10.0,
            separation_weight: 1.6 * (1.0 + high * 0.8) * (1.0 + chaos * 0.8),
            alignment_weight: 1.0
                * (1.0 + mid * 0.8)
                * (1.0 - chaos * 0.7)
                * (1.0 - spectral_flatness * 0.15),
            cohesion_weight: 0.8 * (1.0 + low * 0.8),
            wander_strength: chaos * 45.0
                + rhythmic_irregularity * 10.0
                + spectral_flatness * 5.0
                + finite_unit(features.spectral_flux) * 30.0,
        }
    }

    fn target_population(self) -> usize {
        (self.population_level * MAX_BOIDS as f32).round() as usize
    }
}

pub struct BoidSimulation {
    boids: Vec<Boid>,
    rng: XorShift64,
    accumulator: f32,
    population_level: f32,
    smoothed_speed: f32,
    last_beat_count: u64,
    ripples: Vec<BeatRipple>,
}

impl BoidSimulation {
    pub fn new() -> Self {
        Self::with_seed(0x4d59_5df4_d0f3_3173)
    }

    fn with_seed(seed: u64) -> Self {
        Self {
            boids: Vec::with_capacity(MAX_BOIDS),
            rng: XorShift64::new(seed),
            accumulator: 0.0,
            population_level: 0.0,
            smoothed_speed: 30.0,
            last_beat_count: 0,
            ripples: Vec::with_capacity(MAX_ACTIVE_RIPPLES),
        }
    }

    pub fn update(&mut self, elapsed_seconds: f32, features: &AudioFeatures) {
        let elapsed_seconds = if elapsed_seconds.is_finite() {
            elapsed_seconds.clamp(0.0, MAX_FRAME_TIME)
        } else {
            0.0
        };
        let mut inputs = SimulationInputs::from_audio(features);
        let population_tau = if inputs.population_level > self.population_level {
            0.12
        } else {
            0.5
        };
        self.population_level = exponential_smooth(
            self.population_level,
            inputs.population_level,
            elapsed_seconds,
            population_tau,
        );
        self.smoothed_speed =
            exponential_smooth(self.smoothed_speed, inputs.max_speed, elapsed_seconds, 0.18);
        inputs.population_level = self.population_level;
        inputs.max_speed = self.smoothed_speed;

        self.reconcile_population(inputs.target_population(), inputs.max_speed);

        if features.beat_count != self.last_beat_count {
            self.spawn_ripple(features.beat_strength);
            self.last_beat_count = features.beat_count;
        }

        self.accumulator += elapsed_seconds;
        while self.accumulator >= FIXED_TIME_STEP {
            self.step(FIXED_TIME_STEP, inputs);
            self.accumulator -= FIXED_TIME_STEP;
        }
    }

    pub fn draw(&self, frame: &mut [u8], colour: [u8; 4]) {
        for boid in &self.boids {
            draw_boid(
                frame,
                [boid.position.x, boid.position.y],
                [boid.velocity.x, boid.velocity.y],
                boid.visibility,
                boid.ripple_pulse,
                colour,
            );
        }
    }

    fn reconcile_population(&mut self, target: usize, max_speed: f32) {
        let mut active = self.boids.iter().filter(|boid| boid.target_visible).count();

        if active < target {
            for boid in self.boids.iter_mut().filter(|boid| !boid.target_visible) {
                boid.target_visible = true;
                active += 1;
                if active == target {
                    return;
                }
            }

            while active < target {
                self.boids.push(random_boid(&mut self.rng, max_speed));
                active += 1;
            }
        } else if active > target {
            for boid in self
                .boids
                .iter_mut()
                .rev()
                .filter(|boid| boid.target_visible)
                .take(active - target)
            {
                boid.target_visible = false;
            }
        }
    }

    fn spawn_ripple(&mut self, strength: f32) {
        let visible_count = self.boids.iter().filter(|boid| boid.target_visible).count();
        if visible_count == 0 {
            return;
        }

        let selected = ((self.rng.next_f32() * visible_count as f32) as usize)
            .min(visible_count.saturating_sub(1));
        let origin = self
            .boids
            .iter()
            .filter(|boid| boid.target_visible)
            .nth(selected)
            .map(|boid| boid.position)
            .expect("the visible boid count was checked");

        if self.ripples.len() == MAX_ACTIVE_RIPPLES {
            self.ripples.remove(0);
        }
        self.ripples.push(BeatRipple {
            origin,
            radius: 0.0,
            strength: finite_unit(strength).max(MIN_RIPPLE_STRENGTH),
        });
    }

    fn step(&mut self, elapsed_seconds: f32, inputs: SimulationInputs) {
        let accelerations: Vec<Vec2> = (0..self.boids.len())
            .map(|index| calculate_acceleration(index, &self.boids, inputs))
            .collect();
        let ripple_effects: Vec<(Vec2, f32)> = self
            .boids
            .iter()
            .map(|boid| ripple_effect_at(boid.position, &self.ripples))
            .collect();

        for ((boid, acceleration), (ripple_acceleration, ripple_pulse)) in
            self.boids.iter_mut().zip(accelerations).zip(ripple_effects)
        {
            boid.wander_angle += self.rng.range(-1.0, 1.0) * (0.4 + inputs.wander_strength / 30.0);
            let wander = Vec2::new(boid.wander_angle.cos(), boid.wander_angle.sin())
                * inputs.wander_strength;
            boid.velocity += (acceleration + wander).limited(inputs.max_force) * elapsed_seconds;
            boid.velocity += ripple_acceleration * elapsed_seconds;
            let speed_limit =
                inputs.max_speed * (1.0 + ripple_pulse.clamp(0.0, 1.0) * RIPPLE_SPEED_HEADROOM);
            boid.velocity = boid.velocity.limited(speed_limit.max(1.0));
            boid.position += boid.velocity * elapsed_seconds;
            boid.position.x = boid.position.x.rem_euclid(WIDTH as f32);
            boid.position.y = boid.position.y.rem_euclid(HEIGHT as f32);
            boid.ripple_pulse = ripple_pulse.clamp(0.0, 1.0);

            let visibility_change = elapsed_seconds / LIFECYCLE_SECONDS;
            if boid.target_visible {
                boid.visibility = (boid.visibility + visibility_change).min(1.0);
            } else {
                boid.visibility = (boid.visibility - visibility_change).max(0.0);
            }
        }

        self.boids
            .retain(|boid| boid.target_visible || boid.visibility > 0.0);
        for ripple in &mut self.ripples {
            ripple.radius += RIPPLE_SPEED * elapsed_seconds;
        }
        let maximum_radius = maximum_toroidal_distance() + RIPPLE_WIDTH * 0.5;
        self.ripples
            .retain(|ripple| ripple.radius <= maximum_radius);
    }
}

impl Default for BoidSimulation {
    fn default() -> Self {
        Self::new()
    }
}

fn calculate_acceleration(index: usize, boids: &[Boid], inputs: SimulationInputs) -> Vec2 {
    let boid = &boids[index];
    let mut separation = Vec2::ZERO;
    let mut alignment = Vec2::ZERO;
    let mut cohesion = Vec2::ZERO;
    let mut neighbours = 0_u32;

    for (other_index, other) in boids.iter().enumerate() {
        if index == other_index || other.visibility <= 0.0 {
            continue;
        }

        let offset = toroidal_offset(boid.position, other.position);
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

fn ripple_effect_at(position: Vec2, ripples: &[BeatRipple]) -> (Vec2, f32) {
    let mut acceleration = Vec2::ZERO;
    let mut visual_pulse = 0.0_f32;
    let half_width = RIPPLE_WIDTH * 0.5;

    for ripple in ripples {
        let offset = toroidal_offset(ripple.origin, position);
        let distance = offset.length();
        let distance_from_front = (distance - ripple.radius).abs();
        if distance_from_front >= half_width {
            continue;
        }

        let phase = distance_from_front / half_width;
        let envelope = 0.5 + 0.5 * (std::f32::consts::PI * phase).cos();
        let influence = envelope * ripple.strength;
        acceleration += offset.normalized_or_zero() * (RIPPLE_FORCE * influence);
        visual_pulse = visual_pulse.max(influence);
    }

    (
        acceleration.limited(RIPPLE_FORCE),
        visual_pulse.clamp(0.0, 1.0),
    )
}

fn maximum_toroidal_distance() -> f32 {
    (WIDTH as f32 * 0.5).hypot(HEIGHT as f32 * 0.5)
}

fn steering_force(desired: Vec2, current_velocity: Vec2, inputs: SimulationInputs) -> Vec2 {
    if desired.length_squared() <= f32::EPSILON {
        Vec2::ZERO
    } else {
        (desired.normalized_or_zero() * inputs.max_speed - current_velocity)
            .limited(inputs.max_force)
    }
}

fn toroidal_offset(from: Vec2, to: Vec2) -> Vec2 {
    let mut offset = to - from;
    let half_width = WIDTH as f32 / 2.0;
    let half_height = HEIGHT as f32 / 2.0;
    if offset.x > half_width {
        offset.x -= WIDTH as f32;
    } else if offset.x < -half_width {
        offset.x += WIDTH as f32;
    }
    if offset.y > half_height {
        offset.y -= HEIGHT as f32;
    } else if offset.y < -half_height {
        offset.y += HEIGHT as f32;
    }
    offset
}

fn random_boid(rng: &mut XorShift64, max_speed: f32) -> Boid {
    let angle = rng.range(0.0, std::f32::consts::TAU);
    let speed = rng.range(max_speed * 0.6, max_speed.max(1.0));
    Boid {
        position: Vec2::new(rng.range(0.0, WIDTH as f32), rng.range(0.0, HEIGHT as f32)),
        velocity: Vec2::new(angle.cos(), angle.sin()) * speed,
        visibility: 0.0,
        target_visible: true,
        wander_angle: rng.range(0.0, std::f32::consts::TAU),
        ripple_pulse: 0.0,
    }
}

fn exponential_smooth(current: f32, target: f32, elapsed_seconds: f32, tau: f32) -> f32 {
    if elapsed_seconds <= 0.0 {
        return current;
    }
    let amount = 1.0 - (-elapsed_seconds / tau).exp();
    current + (target - current) * amount
}

fn finite_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[derive(Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as u32) as f32 / u32::MAX as f32
    }

    fn range(&mut self, minimum: f32, maximum: f32) -> f32 {
        minimum + self.next_f32() * (maximum - minimum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::BandEnergies;

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
            position: Vec2::new(WIDTH as f32 - 0.1, HEIGHT as f32 - 0.1),
            velocity: Vec2::new(100.0, 100.0),
            visibility: 1.0,
            target_visible: true,
            wander_angle: 0.0,
            ripple_pulse: 0.0,
        });
        let mut inputs = SimulationInputs::from_audio(&loud_features());
        inputs.wander_strength = 0.0;
        simulation.step(0.1, inputs);
        let position = simulation.boids[0].position;
        assert!((0.0..WIDTH as f32).contains(&position.x));
        assert!((0.0..HEIGHT as f32).contains(&position.y));
    }

    #[test]
    fn each_new_beat_spawns_one_ripple() {
        let mut simulation = BoidSimulation::with_seed(3);
        let mut features = loud_features();
        features.beat_count = 1;
        features.beat_strength = 0.8;

        simulation.update(FIXED_TIME_STEP, &features);
        assert_eq!(simulation.ripples.len(), 1);

        simulation.update(FIXED_TIME_STEP, &features);
        assert_eq!(simulation.ripples.len(), 1);

        features.beat_count = 2;
        simulation.update(FIXED_TIME_STEP, &features);
        assert_eq!(simulation.ripples.len(), 2);
    }

    #[test]
    fn ripple_only_affects_boids_at_its_wavefront() {
        let ripple = BeatRipple {
            origin: Vec2::new(100.0, 100.0),
            radius: 50.0,
            strength: 1.0,
        };

        let (acceleration, pulse) = ripple_effect_at(Vec2::new(150.0, 100.0), &[ripple]);
        assert!(acceleration.x > 0.0);
        assert_eq!(pulse, 1.0);
        assert_eq!(
            ripple_effect_at(Vec2::new(110.0, 100.0), &[ripple]),
            (Vec2::ZERO, 0.0)
        );
        assert_eq!(
            ripple_effect_at(Vec2::new(180.0, 100.0), &[ripple]),
            (Vec2::ZERO, 0.0)
        );
    }

    #[test]
    fn ripple_crosses_the_toroidal_edge() {
        let ripple = BeatRipple {
            origin: Vec2::new(WIDTH as f32 - 10.0, 100.0),
            radius: 20.0,
            strength: 1.0,
        };

        let (acceleration, pulse) = ripple_effect_at(Vec2::new(10.0, 100.0), &[ripple]);
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
        };
        let strong = BeatRipple {
            strength: 1.0,
            ..weak
        };

        let (weak_force, weak_pulse) = ripple_effect_at(position, &[weak]);
        let (strong_force, strong_pulse) = ripple_effect_at(position, &[strong]);
        assert!(strong_force.length() > weak_force.length());
        assert!(strong_pulse > weak_pulse);
    }

    #[test]
    fn ripple_is_removed_after_crossing_the_canvas() {
        let mut simulation = BoidSimulation::with_seed(5);
        simulation.ripples.push(BeatRipple {
            origin: Vec2::ZERO,
            radius: maximum_toroidal_distance() + RIPPLE_WIDTH * 0.5,
            strength: 1.0,
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
        }
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
            },
            Boid {
                position: Vec2::new(105.0, 100.0),
                velocity: Vec2::ZERO,
                visibility: 1.0,
                target_visible: true,
                wander_angle: 0.0,
                ripple_pulse: 0.0,
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
        assert!(calculate_acceleration(0, &boids, inputs).x < 0.0);
        assert!(calculate_acceleration(1, &boids, inputs).x > 0.0);
    }
}
