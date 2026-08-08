use std::{
    thread,
    time::{Duration, Instant},
};

use flexaudio::{OutputFormat, SourceKind, StreamConfig, open};

fn calculate_rms(samples: &[f32]) -> f32 {
    // RMS - Root Mean Square
    // Turn block of pos and neg audio samples
    // into one pos number approximating volume

    if samples.is_empty() {
        return 0.0;
    };
    // list of samples, square each, calculate mean, root mean
    ((samples.iter().map(|sample| sample * sample).sum::<f32>()) / samples.len() as f32).sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let devices = flexaudio::devices()?;
    println!("{devices:#?}");

    let config = StreamConfig {
        kind: SourceKind::SystemLoopback,
        output: OutputFormat {
            sample_rate: 48_000,
            channels: 1,
        },
        ..Default::default()
    };

    let mut stream = open(config)?;
    stream.start()?;

    // listen to system audio for ten seconds
    let finish_at = Instant::now() + Duration::from_secs(10);

    while Instant::now() < finish_at {
        match stream.poll_chunk() {
            Some(chunk) => {
                let rms = calculate_rms(&chunk.data);
                println!("RMS: {rms:.6}")
            }
            None => thread::sleep(Duration::from_millis(2)),
        }
    }

    Ok(())
}
