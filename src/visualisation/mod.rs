mod canvas;
mod render;
mod simulation;

pub use canvas::{INITIAL_WINDOW_HEIGHT, INITIAL_WINDOW_WIDTH};
pub use render::{BackgroundMode, BoidInstance, ColourPalette, ColourSmoother, GpuRenderer};
pub use simulation::{BoidSimulation, MAX_BOIDS, SimulationUpdateStats};
