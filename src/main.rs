mod analysis;
mod audio;
mod render;
mod simulation;

use std::error::Error;
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Instant;

use analysis::AudioFeatures;
use audio::{AudioMessage, AudioWorker, InputMode};
use pixels::wgpu::{Color, CompositeAlphaMode};
use pixels::{Pixels, PixelsBuilder, ScalingMode, SurfaceTexture};
use render::{ColourSmoother, HEIGHT, WIDTH, clear_frame};
use simulation::BoidSimulation;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

#[derive(Debug, PartialEq, Eq)]
struct LaunchOptions {
    demo_mode: bool,
    input_mode: InputMode,
}

struct App {
    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'static>>,
    features: AudioFeatures,
    colour_smoother: ColourSmoother,
    simulation: BoidSimulation,
    audio_worker: Option<AudioWorker>,
    audio_receiver: Option<Receiver<AudioMessage>>,
    audio_error: Option<String>,
    demo_mode: bool,
    started_at: Instant,
    last_frame_at: Instant,
}

impl App {
    fn new(demo_mode: bool, input_mode: InputMode) -> Self {
        let (audio_worker, audio_receiver) = if demo_mode {
            (None, None)
        } else {
            let (worker, receiver) = AudioWorker::spawn(input_mode);
            (Some(worker), Some(receiver))
        };

        Self {
            window: None,
            pixels: None,
            features: AudioFeatures::default(),
            colour_smoother: ColourSmoother::new(),
            simulation: BoidSimulation::new(),
            audio_worker,
            audio_receiver,
            audio_error: None,
            demo_mode,
            started_at: Instant::now(),
            last_frame_at: Instant::now(),
        }
    }

    fn update_demo(&mut self) {
        let time = self.started_at.elapsed().as_secs_f32();
        let volume = ((time * 2.0).sin() + 1.0) / 2.0;
        let frequency = 80.0 * 50.0_f32.powf(volume);
        let beat_phase = (time % 0.5) / 0.5;
        let onset_strength = (1.0 - beat_phase * 8.0).clamp(0.0, 1.0);
        let low = ((time * 0.7).sin() + 1.0) * 0.5;
        let mid = ((time * 0.9 + 2.0).sin() + 1.0) * 0.5;
        let high = ((time * 1.3 + 4.0).sin() + 1.0) * 0.5;
        let band_total = (low + mid + high).max(0.001);

        self.features = AudioFeatures {
            rms: volume,
            dominant_hz: Some(frequency),
            spectral_flux: onset_strength,
            spectral_flatness: ((time * 0.23).sin() + 1.0) * 0.25,
            bands: analysis::BandEnergies {
                low: low / band_total,
                mid: mid / band_total,
                high: high / band_total,
            },
            onset_rate_hz: 2.0 + ((time * 0.4).sin() + 1.0) * 2.0,
            rhythmic_irregularity: ((time * 0.31).sin() + 1.0) * 0.25,
            chaos: ((time * 0.27).sin() + 1.0) * 0.35,
            bpm: Some(120.0),
            tempo_confidence: 1.0,
            beat_count: (time * 2.0).floor() as u64,
            beat_strength: 1.0,
        };
    }

    fn receive_audio(&mut self) {
        let Some(receiver) = self.audio_receiver.as_ref() else {
            return;
        };

        let mut newest_features = None;
        let mut newest_error = None;
        let mut newest_switch_error = None;

        for message in receiver.try_iter() {
            match message {
                AudioMessage::Features(features) => newest_features = Some(features),
                AudioMessage::Failed(error) => newest_error = Some(error),
                AudioMessage::SwitchFailed(error) => newest_switch_error = Some(error),
            }
        }

        if let Some(error) = newest_switch_error {
            eprintln!("{error}");
        }

        if let Some(error) = newest_error {
            eprintln!("Audio capture stopped: {error}");
            self.features = AudioFeatures::default();
            self.audio_error = Some(error);

            if let Some(window) = &self.window {
                window.set_title("Rust Audio Visualiser — Audio unavailable");
            }
        } else if let Some(features) = newest_features {
            self.features = features;
        }
    }

    fn draw(&mut self) -> Result<(), pixels::Error> {
        let now = Instant::now();
        let elapsed_seconds = now.duration_since(self.last_frame_at).as_secs_f32();
        self.last_frame_at = now;

        if self.demo_mode {
            self.update_demo();
        } else {
            self.receive_audio();
        }

        let colour = self.colour_smoother.update(elapsed_seconds, &self.features);
        self.simulation.update(elapsed_seconds, &self.features);

        if let Some(pixels) = self.pixels.as_mut() {
            clear_frame(pixels.frame_mut());
            self.simulation.draw(pixels.frame_mut(), colour);
            pixels.render()?;
        }

        Ok(())
    }

    fn shutdown_audio(&mut self) {
        if let Some(worker) = self.audio_worker.as_mut() {
            worker.shutdown();
        }
    }

    fn toggle_audio_input(&self) {
        if let Some(worker) = &self.audio_worker
            && let Err(error) = worker.toggle_input()
        {
            eprintln!("Could not switch audio input: {error}");
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
            return;
        }

        let title = if self.demo_mode {
            "Rust Audio Visualiser — Demo"
        } else if self.audio_error.is_some() {
            "Rust Audio Visualiser — Audio unavailable"
        } else {
            "Rust Audio Visualiser"
        };

        let attributes = Window::default_attributes()
            .with_title(title)
            .with_transparent(true)
            .with_inner_size(LogicalSize::new(WIDTH as f64, HEIGHT as f64))
            .with_min_inner_size(LogicalSize::new(320.0, 240.0));

        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                eprintln!("Could not create the window: {error}");
                self.shutdown_audio();
                event_loop.exit();
                return;
            }
        };

        let size = window.inner_size();
        let surface = SurfaceTexture::new(size.width, size.height, Arc::clone(&window));
        let mut pixels = match PixelsBuilder::new(WIDTH, HEIGHT, surface)
            .alpha_mode(CompositeAlphaMode::PreMultiplied)
            .clear_color(Color::TRANSPARENT)
            .build()
        {
            Ok(pixels) => pixels,
            Err(error) => {
                eprintln!("Could not create the pixel surface: {error}");
                self.shutdown_audio();
                event_loop.exit();
                return;
            }
        };
        pixels.set_scaling_mode(ScalingMode::Fill);

        window.request_redraw();
        self.pixels = Some(pixels);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if self
            .window
            .as_ref()
            .is_none_or(|window| window.id() != window_id)
        {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                self.shutdown_audio();
                event_loop.exit();
            }
            WindowEvent::Resized(size) if size.width > 0 && size.height > 0 => {
                if let Some(pixels) = self.pixels.as_mut()
                    && let Err(error) = pixels.resize_surface(size.width, size.height)
                {
                    eprintln!("Could not resize the pixel surface: {error}");
                    self.shutdown_audio();
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && !event.repeat
                    && event.physical_key == PhysicalKey::Code(KeyCode::KeyI) =>
            {
                self.toggle_audio_input();
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.draw() {
                    eprintln!("Could not draw the next frame: {error}");
                    self.shutdown_audio();
                    event_loop.exit();
                    return;
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn parse_launch_options(
    arguments: impl IntoIterator<Item = String>,
) -> Result<LaunchOptions, String> {
    let mut options = LaunchOptions {
        demo_mode: false,
        input_mode: InputMode::Loopback,
    };
    let mut arguments = arguments.into_iter();

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--demo" => options.demo_mode = true,
            "--input" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| "--input requires either 'loopback' or 'mic'".to_owned())?;
                options.input_mode = parse_input_mode(&value)?;
            }
            _ => {
                if let Some(value) = argument.strip_prefix("--input=") {
                    options.input_mode = parse_input_mode(value)?;
                } else {
                    return Err(format!(
                        "unknown argument '{argument}'; use --input loopback, --input mic, or --demo"
                    ));
                }
            }
        }
    }

    Ok(options)
}

fn parse_input_mode(value: &str) -> Result<InputMode, String> {
    match value {
        "loopback" => Ok(InputMode::Loopback),
        "mic" => Ok(InputMode::Microphone),
        _ => Err(format!(
            "unknown input '{value}'; expected 'loopback' or 'mic'"
        )),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_launch_options(std::env::args().skip(1))?;
    let event_loop = EventLoop::new()?;
    let mut app = App::new(options.demo_mode, options.input_mode);
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn loopback_is_the_default_input() {
        assert_eq!(
            parse_launch_options(Vec::new()).unwrap(),
            LaunchOptions {
                demo_mode: false,
                input_mode: InputMode::Loopback,
            }
        );
    }

    #[test]
    fn microphone_can_be_selected_at_launch() {
        assert_eq!(
            parse_launch_options(arguments(&["--input", "mic"]))
                .unwrap()
                .input_mode,
            InputMode::Microphone
        );
        assert_eq!(
            parse_launch_options(arguments(&["--input=mic"]))
                .unwrap()
                .input_mode,
            InputMode::Microphone
        );
    }

    #[test]
    fn invalid_input_is_rejected() {
        assert!(parse_launch_options(arguments(&["--input", "file"])).is_err());
        assert!(parse_launch_options(arguments(&["--input"])).is_err());
    }
}
