pub const INITIAL_WINDOW_WIDTH: u32 = 640;
pub const INITIAL_WINDOW_HEIGHT: u32 = 480;
const REFERENCE_HEIGHT: f32 = 480.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CanvasSize {
    pub width: f32,
    pub height: f32,
}

impl CanvasSize {
    pub fn from_surface(width: u32, height: u32) -> Self {
        if width == 0 || height == 0 {
            return Self::default();
        }

        Self {
            width: REFERENCE_HEIGHT * width as f32 / height as f32,
            height: REFERENCE_HEIGHT,
        }
    }

    pub fn as_array(self) -> [f32; 2] {
        [self.width, self.height]
    }
}

impl Default for CanvasSize {
    fn default() -> Self {
        Self {
            width: INITIAL_WINDOW_WIDTH as f32,
            height: INITIAL_WINDOW_HEIGHT as f32,
        }
    }
}

#[cfg(test)]
#[path = "canvas/tests.rs"]
mod tests;
