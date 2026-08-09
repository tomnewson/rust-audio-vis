mod audio;
mod visualisation;

use std::error::Error;
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Instant;

use audio::{AudioFeatures, AudioMessage, AudioWorker, BandEnergies, InputMode};
use pixels::wgpu::{Color, CompositeAlphaMode};
use pixels::{Pixels, PixelsBuilder, ScalingMode, SurfaceTexture};
use visualisation::{
    BackgroundMode, BoidSimulation, ColourPalette, ColourSmoother, HEIGHT, WIDTH, clear_frame,
};
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
    background_mode: BackgroundMode,
    show_stats: bool,
}

struct PerformanceStats {
    interval_started: Instant,
    simulation_ms: Vec<f64>,
    rendering_ms: Vec<f64>,
    frame_ms: Vec<f64>,
    fixed_steps: usize,
    dropped_seconds: f32,
    boid_count: usize,
}

impl PerformanceStats {
    fn new() -> Self {
        Self {
            interval_started: Instant::now(),
            simulation_ms: Vec::with_capacity(120),
            rendering_ms: Vec::with_capacity(120),
            frame_ms: Vec::with_capacity(120),
            fixed_steps: 0,
            dropped_seconds: 0.0,
            boid_count: 0,
        }
    }

    fn record(
        &mut self,
        simulation_ms: f64,
        rendering_ms: f64,
        frame_ms: f64,
        simulation: visualisation::SimulationUpdateStats,
    ) {
        self.simulation_ms.push(simulation_ms);
        self.rendering_ms.push(rendering_ms);
        self.frame_ms.push(frame_ms);
        self.fixed_steps += simulation.fixed_steps;
        self.dropped_seconds += simulation.dropped_seconds;
        self.boid_count = simulation.boid_count;

        let interval_seconds = self.interval_started.elapsed().as_secs_f64();
        if interval_seconds < 1.0 {
            return;
        }

        let frame_count = self.frame_ms.len();
        let fps = frame_count as f64 / interval_seconds;
        let steps_per_frame = self.fixed_steps as f64 / frame_count.max(1) as f64;
        eprintln!(
            "perf boids={} fps={fps:.1} frame={:.2}/{:.2}ms sim={:.2}/{:.2}ms render={:.2}/{:.2}ms steps/frame={steps_per_frame:.2} dropped={:.2}ms",
            self.boid_count,
            average(&self.frame_ms),
            percentile(&mut self.frame_ms, 0.95),
            average(&self.simulation_ms),
            percentile(&mut self.simulation_ms, 0.95),
            average(&self.rendering_ms),
            percentile(&mut self.rendering_ms, 0.95),
            self.dropped_seconds as f64 * 1_000.0,
        );

        self.interval_started = Instant::now();
        self.simulation_ms.clear();
        self.rendering_ms.clear();
        self.frame_ms.clear();
        self.fixed_steps = 0;
        self.dropped_seconds = 0.0;
    }
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
    background_mode: BackgroundMode,
    performance_stats: Option<PerformanceStats>,
    started_at: Instant,
    last_frame_at: Instant,
}

impl App {
    fn new(
        demo_mode: bool,
        input_mode: InputMode,
        background_mode: BackgroundMode,
        show_stats: bool,
    ) -> Self {
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
            background_mode,
            performance_stats: show_stats.then(PerformanceStats::new),
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
            bands: BandEnergies {
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
                window.set_title("raesboida (audio unavailable)");
            }
        } else if let Some(features) = newest_features {
            self.features = features;
        }
    }

    fn draw(&mut self) -> Result<(), pixels::Error> {
        let frame_started = Instant::now();
        let now = Instant::now();
        let elapsed_seconds = now.duration_since(self.last_frame_at).as_secs_f32();
        self.last_frame_at = now;

        if self.demo_mode {
            self.update_demo();
        } else {
            self.receive_audio();
        }

        let palette: ColourPalette = self.colour_smoother.update(elapsed_seconds, &self.features);
        let simulation_started = Instant::now();
        let simulation_stats = self.simulation.update(elapsed_seconds, &self.features);
        let simulation_ms = simulation_started.elapsed().as_secs_f64() * 1_000.0;

        let rendering_started = Instant::now();
        if let Some(pixels) = self.pixels.as_mut() {
            clear_frame(pixels.frame_mut(), self.background_mode, &palette);
            self.simulation.draw(pixels.frame_mut(), &palette);
            pixels.render()?;
        }
        let rendering_ms = rendering_started.elapsed().as_secs_f64() * 1_000.0;

        if let Some(stats) = &mut self.performance_stats {
            stats.record(
                simulation_ms,
                rendering_ms,
                frame_started.elapsed().as_secs_f64() * 1_000.0,
                simulation_stats,
            );
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

    fn toggle_background(&mut self) {
        self.background_mode = self.background_mode.next();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
            return;
        }

        let title = if self.demo_mode {
            "raesboida (demo)"
        } else if self.audio_error.is_some() {
            "raesboida (audio unavailable)"
        } else {
            "raesboida"
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
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && !event.repeat
                    && event.physical_key == PhysicalKey::Code(KeyCode::KeyB) =>
            {
                self.toggle_background();
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
        background_mode: BackgroundMode::Transparent,
        show_stats: false,
    };
    let mut arguments = arguments.into_iter();

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--demo" => options.demo_mode = true,
            "--stats" => options.show_stats = true,
            "--input" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| "--input requires either 'loopback' or 'mic'".to_owned())?;
                options.input_mode = parse_input_mode(&value)?;
            }
            "--background" => {
                let value = arguments.next().ok_or_else(|| {
                    "--background requires 'black', 'white', 'transparent', or 'boid'".to_owned()
                })?;
                options.background_mode = parse_background_mode(&value)?;
            }
            _ => {
                if let Some(value) = argument.strip_prefix("--input=") {
                    options.input_mode = parse_input_mode(value)?;
                } else if let Some(value) = argument.strip_prefix("--background=") {
                    options.background_mode = parse_background_mode(value)?;
                } else {
                    return Err(format!(
                        "unknown argument '{argument}'; use --input loopback|mic, --background black|white|transparent|boid, --demo, or --stats"
                    ));
                }
            }
        }
    }

    Ok(options)
}

fn parse_background_mode(value: &str) -> Result<BackgroundMode, String> {
    match value {
        "black" => Ok(BackgroundMode::Black),
        "white" => Ok(BackgroundMode::White),
        "transparent" => Ok(BackgroundMode::Transparent),
        "boid" => Ok(BackgroundMode::Boid),
        _ => Err(format!(
            "unknown background '{value}'; expected 'black', 'white', 'transparent', or 'boid'"
        )),
    }
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

fn average(samples: &[f64]) -> f64 {
    samples.iter().sum::<f64>() / samples.len().max(1) as f64
}

fn percentile(samples: &mut [f64], percentile: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(f64::total_cmp);
    let index = ((samples.len() - 1) as f64 * percentile).round() as usize;
    samples[index]
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = parse_launch_options(std::env::args().skip(1))?;
    let event_loop = EventLoop::new()?;
    let mut app = App::new(
        options.demo_mode,
        options.input_mode,
        options.background_mode,
        options.show_stats,
    );
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
                background_mode: BackgroundMode::Transparent,
                show_stats: false,
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

    #[test]
    fn background_can_be_selected_at_launch() {
        assert_eq!(
            parse_launch_options(arguments(&["--background", "black"]))
                .unwrap()
                .background_mode,
            BackgroundMode::Black
        );
        assert_eq!(
            parse_launch_options(arguments(&["--background=white"]))
                .unwrap()
                .background_mode,
            BackgroundMode::White
        );
        assert_eq!(
            parse_launch_options(arguments(&["--background", "transparent"]))
                .unwrap()
                .background_mode,
            BackgroundMode::Transparent
        );
        assert_eq!(
            parse_launch_options(arguments(&["--background=boid"]))
                .unwrap()
                .background_mode,
            BackgroundMode::Boid
        );
    }

    #[test]
    fn invalid_background_is_rejected() {
        assert!(parse_launch_options(arguments(&["--background", "blue"])).is_err());
        assert!(parse_launch_options(arguments(&["--background"])).is_err());
    }

    #[test]
    fn performance_stats_can_be_enabled() {
        assert!(
            parse_launch_options(arguments(&["--stats"]))
                .unwrap()
                .show_stats
        );
    }
}
