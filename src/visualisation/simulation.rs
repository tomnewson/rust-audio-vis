use crate::audio::AudioFeatures;
use crate::visualisation::render::{
    COLOUR_VARIANT_COUNT, ColourPalette, HEIGHT, WIDTH, draw_boid, loudness_position,
    pitch_position,
};

const MAX_BOIDS: usize = 500;
const FIXED_TIME_STEP: f32 = 1.0 / 60.0;
const MAX_FRAME_TIME: f32 = 0.1;
const MAX_FIXED_STEPS_PER_FRAME: usize = 4;
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
    colour_index: u8,
}

#[derive(Debug, Clone, Copy)]
struct BeatRipple {
    origin: Vec2,
    radius: f32,
    strength: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct StepEffect {
    flock_acceleration: Vec2,
    ripple_acceleration: Vec2,
    ripple_pulse: f32,
}

struct SpatialGrid {
    columns: usize,
    rows: usize,
    cell_width: f32,
    cell_height: f32,
    cells: Vec<Vec<usize>>,
    neighbouring_cells: Vec<[usize; 9]>,
}

impl SpatialGrid {
    fn new() -> Self {
        let columns = ((WIDTH as f32 / NEIGHBOUR_RADIUS).floor() as usize).max(1);
        let rows = ((HEIGHT as f32 / NEIGHBOUR_RADIUS).floor() as usize).max(1);
        let cell_width = WIDTH as f32 / columns as f32;
        let cell_height = HEIGHT as f32 / rows as f32;
        let cell_count = columns * rows;
        let expected_boids_per_cell = (MAX_BOIDS / cell_count).max(1);
        let cells = (0..cell_count)
            .map(|_| Vec::with_capacity(expected_boids_per_cell * 2))
            .collect();
        let mut neighbouring_cells = Vec::with_capacity(cell_count);

        for row in 0..rows {
            for column in 0..columns {
                let mut neighbours = [0; 9];
                let mut next = 0;
                for row_offset in -1_isize..=1 {
                    for column_offset in -1_isize..=1 {
                        let neighbour_column =
                            (column as isize + column_offset).rem_euclid(columns as isize) as usize;
                        let neighbour_row =
                            (row as isize + row_offset).rem_euclid(rows as isize) as usize;
                        neighbours[next] = neighbour_row * columns + neighbour_column;
                        next += 1;
                    }
                }
                neighbouring_cells.push(neighbours);
            }
        }

        Self {
            columns,
            rows,
            cell_width,
            cell_height,
            cells,
            neighbouring_cells,
        }
    }

    fn rebuild(&mut self, boids: &[Boid]) {
        for cell in &mut self.cells {
            cell.clear();
        }

        for (index, boid) in boids.iter().enumerate() {
            if boid.visibility > 0.0 {
                let cell = self.cell_for_position(boid.position);
                self.cells[cell].push(index);
            }
        }
    }

    fn cell_for_position(&self, position: Vec2) -> usize {
        let x = position.x.rem_euclid(WIDTH as f32);
        let y = position.y.rem_euclid(HEIGHT as f32);
        let column = ((x / self.cell_width) as usize).min(self.columns - 1);
        let row = ((y / self.cell_height) as usize).min(self.rows - 1);
        row * self.columns + column
    }

    fn neighbouring_cells_for(&self, position: Vec2) -> &[usize; 9] {
        &self.neighbouring_cells[self.cell_for_position(position)]
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SimulationUpdateStats {
    pub fixed_steps: usize,
    pub dropped_seconds: f32,
    pub boid_count: usize,
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
        let onset_rate_position = finite_unit(features.onset_rate_hz / 8.0);
        let tempo_position = features
            .bpm
            .filter(|bpm| bpm.is_finite())
            .map(|bpm| ((bpm - 55.0) / (200.0 - 55.0)).clamp(0.0, 1.0))
            .unwrap_or(0.0)
            * finite_unit(features.tempo_confidence);
        let onset_pace = finite_unit((features.onset_rate_hz - 1.0) / 5.0) * 0.9;
        let pace = tempo_position.max(onset_pace);

        Self {
            population_level: loudness_position(features.rms),
            max_speed: 30.0 + pitch * 120.0 + pace * 220.0,
            max_force: 35.0 + onset_rate_position * 45.0 + pace * 110.0,
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
    spatial_grid: SpatialGrid,
    step_effects: Vec<StepEffect>,
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
            spatial_grid: SpatialGrid::new(),
            step_effects: Vec::with_capacity(MAX_BOIDS),
        }
    }

    pub fn update(
        &mut self,
        elapsed_seconds: f32,
        features: &AudioFeatures,
    ) -> SimulationUpdateStats {
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
        let mut fixed_steps = 0;
        while self.accumulator >= FIXED_TIME_STEP && fixed_steps < MAX_FIXED_STEPS_PER_FRAME {
            self.step(FIXED_TIME_STEP, inputs);
            self.accumulator -= FIXED_TIME_STEP;
            fixed_steps += 1;
        }

        let dropped_seconds = if self.accumulator >= FIXED_TIME_STEP {
            let complete_steps = (self.accumulator / FIXED_TIME_STEP).floor();
            let dropped = complete_steps * FIXED_TIME_STEP;
            self.accumulator -= dropped;
            dropped
        } else {
            0.0
        };

        SimulationUpdateStats {
            fixed_steps,
            dropped_seconds,
            boid_count: self.boids.len(),
        }
    }

    pub fn draw(&self, frame: &mut [u8], palette: &ColourPalette) {
        for boid in &self.boids {
            let colour = palette.colour_for(boid.colour_index, boid.ripple_pulse);
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
        self.spatial_grid.rebuild(&self.boids);
        self.step_effects.clear();
        for (index, boid) in self.boids.iter().enumerate() {
            let (ripple_acceleration, ripple_pulse) =
                ripple_effect_at(boid.position, &self.ripples);
            self.step_effects.push(StepEffect {
                flock_acceleration: calculate_acceleration(
                    index,
                    &self.boids,
                    &self.spatial_grid,
                    inputs,
                ),
                ripple_acceleration,
                ripple_pulse,
            });
        }

        for (boid, effect) in self.boids.iter_mut().zip(self.step_effects.iter().copied()) {
            boid.wander_angle += self.rng.range(-1.0, 1.0) * (0.4 + inputs.wander_strength / 30.0);
            let wander = Vec2::new(boid.wander_angle.cos(), boid.wander_angle.sin())
                * inputs.wander_strength;
            boid.velocity +=
                (effect.flock_acceleration + wander).limited(inputs.max_force) * elapsed_seconds;
            boid.velocity += effect.ripple_acceleration * elapsed_seconds;
            let speed_limit = inputs.max_speed
                * (1.0 + effect.ripple_pulse.clamp(0.0, 1.0) * RIPPLE_SPEED_HEADROOM);
            boid.velocity = boid.velocity.limited(speed_limit.max(1.0));
            boid.position += boid.velocity * elapsed_seconds;
            boid.position.x = boid.position.x.rem_euclid(WIDTH as f32);
            boid.position.y = boid.position.y.rem_euclid(HEIGHT as f32);
            boid.ripple_pulse = effect.ripple_pulse.clamp(0.0, 1.0);

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

fn calculate_acceleration(
    index: usize,
    boids: &[Boid],
    grid: &SpatialGrid,
    inputs: SimulationInputs,
) -> Vec2 {
    let boid = &boids[index];
    let mut separation = Vec2::ZERO;
    let mut alignment = Vec2::ZERO;
    let mut cohesion = Vec2::ZERO;
    let mut neighbours = 0_u32;

    for cell_index in grid.neighbouring_cells_for(boid.position) {
        for &other_index in &grid.cells[*cell_index] {
            if index == other_index {
                continue;
            }
            let other = &boids[other_index];

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
    let colour_index = ((rng.next_f32() * COLOUR_VARIANT_COUNT as f32) as usize)
        .min(COLOUR_VARIANT_COUNT - 1) as u8;
    Boid {
        position: Vec2::new(rng.range(0.0, WIDTH as f32), rng.range(0.0, HEIGHT as f32)),
        velocity: Vec2::new(angle.cos(), angle.sin()) * speed,
        visibility: 0.0,
        target_visible: true,
        wander_angle: rng.range(0.0, std::f32::consts::TAU),
        ripple_pulse: 0.0,
        colour_index,
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
#[path = "simulation/tests.rs"]
mod tests;
