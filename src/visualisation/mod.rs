mod render;
mod simulation;

pub use render::{ColourPalette, ColourSmoother, HEIGHT, WIDTH, clear_frame};
pub use simulation::{BoidSimulation, SimulationUpdateStats};
