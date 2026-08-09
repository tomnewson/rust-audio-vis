mod render;
mod simulation;

pub use render::{BackgroundMode, ColourPalette, ColourSmoother, HEIGHT, WIDTH, clear_frame};
pub use simulation::{BoidSimulation, SimulationUpdateStats};
