mod render;
mod simulation;

pub use render::{ColourSmoother, HEIGHT, WIDTH, clear_frame};
pub use simulation::{BoidSimulation, SimulationUpdateStats};
