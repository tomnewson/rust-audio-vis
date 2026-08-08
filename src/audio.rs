use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use flexaudio::{Event, OutputFormat, SourceKind, Stream, StreamConfig, open};

use crate::analysis::{Analyzer, AudioFeatures, SAMPLE_RATE};

pub enum AudioMessage {
    Features(AudioFeatures),
    Failed(String),
}

pub struct AudioWorker {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl AudioWorker {
    pub fn spawn() -> (Self, Receiver<AudioMessage>) {
        let (sender, receiver) = mpsc::sync_channel(2);
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);

        let handle = thread::spawn(move || {
            if let Err(error) = capture_audio(&worker_stop, &sender) {
                send_failure(&worker_stop, &sender, error);
            }
        });

        (
            Self {
                stop,
                handle: Some(handle),
            },
            receiver,
        )
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

fn capture_audio(stop: &AtomicBool, sender: &SyncSender<AudioMessage>) -> Result<(), String> {
    let config = StreamConfig {
        kind: SourceKind::SystemLoopback,
        output: OutputFormat {
            sample_rate: SAMPLE_RATE,
            channels: 1,
        },
        ..Default::default()
    };

    let mut stream =
        open(config).map_err(|error| format!("could not open system audio: {error}"))?;

    if let Err(error) = stream.start() {
        stream.stop();
        return Err(format!("could not start system audio: {error}"));
    }

    let result = capture_loop(&mut stream, stop, sender);
    stream.stop();
    result
}

fn capture_loop(
    stream: &mut Stream,
    stop: &AtomicBool,
    sender: &SyncSender<AudioMessage>,
) -> Result<(), String> {
    let mut analyzer = Analyzer::new();

    'capture: while !stop.load(Ordering::SeqCst) {
        while let Some(event) = stream.poll_event() {
            match event {
                Event::PermissionDenied => {
                    return Err("permission to capture system audio was denied".to_owned());
                }
                Event::DeviceLost => {
                    return Err("the system audio device was lost".to_owned());
                }
                Event::Error(error) => {
                    return Err(format!("system audio capture failed: {error}"));
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
