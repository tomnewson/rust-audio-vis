use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TrySendError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use flexaudio::{Event, OutputFormat, SourceKind, Stream, StreamConfig, open};

use crate::analysis::{Analyzer, AudioFeatures, SAMPLE_RATE};

pub enum AudioMessage {
    Features(AudioFeatures),
    Failed(String),
    SwitchFailed(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Loopback,
    Microphone,
}

impl InputMode {
    fn toggled(self) -> Self {
        match self {
            Self::Loopback => Self::Microphone,
            Self::Microphone => Self::Loopback,
        }
    }

    fn source_kind(self) -> SourceKind {
        match self {
            Self::Loopback => SourceKind::SystemLoopback,
            Self::Microphone => SourceKind::Mic,
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Loopback => "system audio",
            Self::Microphone => "microphone audio",
        }
    }
}

pub struct AudioWorker {
    stop: Arc<AtomicBool>,
    command_sender: Sender<AudioCommand>,
    handle: Option<JoinHandle<()>>,
}

enum AudioCommand {
    ToggleInput,
}

impl AudioWorker {
    pub fn spawn(input_mode: InputMode) -> (Self, Receiver<AudioMessage>) {
        let (sender, receiver) = mpsc::sync_channel(2);
        let (command_sender, command_receiver) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);

        let handle = thread::spawn(move || {
            if let Err(error) = capture_audio(input_mode, &worker_stop, &sender, &command_receiver)
            {
                send_failure(&worker_stop, &sender, error);
            }
        });

        (
            Self {
                stop,
                command_sender,
                handle: Some(handle),
            },
            receiver,
        )
    }

    pub fn toggle_input(&self) -> Result<(), String> {
        self.command_sender
            .send(AudioCommand::ToggleInput)
            .map_err(|_| "the audio worker is no longer running".to_owned())
    }

    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::SeqCst);

        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn send_failure(stop: &AtomicBool, sender: &SyncSender<AudioMessage>, error: String) {
    let mut message = AudioMessage::Failed(error);

    while !stop.load(Ordering::SeqCst) {
        match sender.try_send(message) {
            Ok(()) | Err(TrySendError::Disconnected(_)) => return,
            Err(TrySendError::Full(returned_message)) => {
                message = returned_message;
                thread::sleep(Duration::from_millis(2));
            }
        }
    }
}

impl Drop for AudioWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn stream_config(input_mode: InputMode) -> StreamConfig {
    StreamConfig {
        kind: input_mode.source_kind(),
        output: OutputFormat {
            sample_rate: SAMPLE_RATE,
            channels: 1,
        },
        ..Default::default()
    }
}

fn capture_audio(
    input_mode: InputMode,
    stop: &AtomicBool,
    sender: &SyncSender<AudioMessage>,
    command_receiver: &Receiver<AudioCommand>,
) -> Result<(), String> {
    let config = stream_config(input_mode);

    let mut stream = open(config)
        .map_err(|error| format!("could not open {}: {error}", input_mode.description()))?;

    if let Err(error) = stream.start() {
        stream.stop();
        return Err(format!(
            "could not start {}: {error}",
            input_mode.description()
        ));
    }

    let result = capture_loop(&mut stream, input_mode, stop, sender, command_receiver);
    stream.stop();
    result
}

fn capture_loop(
    stream: &mut Stream,
    mut input_mode: InputMode,
    stop: &AtomicBool,
    sender: &SyncSender<AudioMessage>,
    command_receiver: &Receiver<AudioCommand>,
) -> Result<(), String> {
    let mut analyzer = Analyzer::new();

    'capture: while !stop.load(Ordering::SeqCst) {
        for command in command_receiver.try_iter() {
            match command {
                AudioCommand::ToggleInput => {
                    let new_mode = input_mode.toggled();
                    match stream.switch_source(stream_config(new_mode)) {
                        Ok(()) => {
                            input_mode = new_mode;
                            analyzer = Analyzer::new();
                        }
                        Err(error) => {
                            let _ = sender.try_send(AudioMessage::SwitchFailed(format!(
                                "could not switch to {}: {error}",
                                new_mode.description()
                            )));
                        }
                    }
                }
            }
        }

        while let Some(event) = stream.poll_event() {
            match event {
                Event::PermissionDenied => {
                    return Err(format!(
                        "permission to capture {} was denied",
                        input_mode.description()
                    ));
                }
                Event::DeviceLost => {
                    return Err(format!("the {} device was lost", input_mode.description()));
                }
                Event::Error(error) => {
                    return Err(format!(
                        "{} capture failed: {error}",
                        input_mode.description()
                    ));
                }
                _ => {}
            }
        }

        match stream.poll_chunk() {
            Some(chunk) => {
                if let Some(features) = analyzer.push(&chunk.data) {
                    match sender.try_send(AudioMessage::Features(features)) {
                        Ok(()) | Err(TrySendError::Full(_)) => {}
                        Err(TrySendError::Disconnected(_)) => break 'capture,
                    }
                }
            }
            None => thread::sleep(Duration::from_millis(2)),
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_mode_toggles_in_both_directions() {
        assert_eq!(InputMode::Loopback.toggled(), InputMode::Microphone);
        assert_eq!(InputMode::Microphone.toggled(), InputMode::Loopback);
    }

    #[test]
    fn input_modes_select_the_expected_capture_sources() {
        assert_eq!(
            stream_config(InputMode::Loopback).kind,
            SourceKind::SystemLoopback
        );
        assert_eq!(stream_config(InputMode::Microphone).kind, SourceKind::Mic);
    }
}
