# Audio-to-visual mapping

This document describes how analyzed audio features currently affect the visualiser.

## Overview

| Audio feature | Main visual result |
| --- | --- |
| RMS loudness | Number of boids and colour chroma, including the optional boid-colour background |
| Dominant frequency (pitch) | Colour hue, colour lightness, boid speed, and the optional boid-colour background |
| Spectral flux | More wandering and faster colour transitions |
| Spectral flatness | Slightly less alignment, more wandering, and greater calculated chaos |
| Low-frequency energy | Stronger cohesion |
| Mid-frequency energy | Stronger alignment |
| High-frequency energy | Stronger separation |
| Onset rate | Faster motion, stronger steering, and greater calculated chaos |
| Rhythmic irregularity | More wandering and greater calculated chaos |
| Chaos | More separation and wandering, but less alignment |
| BPM and tempo confidence | Faster motion and stronger steering |
| Beat event and strength | An expanding ripple that pushes, enlarges, and brightens boids |

Several inputs deliberately overlap. For example, fast-paced audio can be recognized either from a confident BPM or from a high onset rate.

The flocking terms mean:

| Output | Meaning |
| --- | --- |
| Separation | Steer away from nearby boids to avoid crowding |
| Alignment | Steer toward nearby boids' average direction |
| Cohesion | Steer toward the local flock |
| Wander | Add continuously changing random steering |
| Maximum force | Limit how sharply a boid can change velocity |
| Maximum speed | Limit how fast a boid can travel |

## Shared normalized values

The simulation converts raw measurements into values between `0.0` and `1.0` before using them.

### Loudness

```text
loudness_db = 20 × log10(rms)
loudness = clamp((loudness_db - (-50)) / (-10 - (-50)), 0, 1)
```

- At or below `-50 dBFS`: `0.0`
- At or above `-10 dBFS`: `1.0`
- The logarithmic dB scale makes the response closer to perceived loudness than using RMS directly.

### Pitch

```text
frequency = clamp(dominant_frequency_hz, 80, 4000)
pitch = ln(frequency / 80) / ln(4000 / 80)
```

- `80 Hz` becomes `0.0`.
- `4000 Hz` becomes `1.0`.
- The logarithmic mapping better matches musical pitch, where equal frequency ratios represent equal pitch intervals.

### Pace

```text
tempo = clamp((bpm - 55) / (200 - 55), 0, 1) × tempo_confidence
onset_pace = 0.9 × clamp((onset_rate_hz - 1) / 5, 0, 1)
pace = max(tempo, onset_pace)
```

`pace` can therefore react quickly to frequent attacks even before the slower tempo detector becomes confident.

## Detailed feature mappings

### RMS loudness

RMS measures the average energy of the current audio window.

It controls:

- Target boid count: `round(loudness × 1,000)`.
- Base OKLCH chroma: `loudness × 0.24`.

Population changes are smoothed:

- Increasing population uses a `0.12 second` smoothing time constant.
- Decreasing population uses a `0.5 second` smoothing time constant.
- Individual boids fade in or out over `0.35 seconds`.

Loudness does not directly alter boid size or speed.

### Dominant frequency

Dominant frequency is the strongest FFT frequency between `80 Hz` and `4000 Hz`.

It controls:

- Base OKLCH lightness: `0.35 + pitch × 0.50`.
- Base OKLCH hue: `20° + pitch × 300°`.
- Speed contribution: `pitch × 120 pixels/second`.

Higher pitches are therefore lighter, farther around the hue wheel, and faster.

When no pitch is available:

- At startup, the palette remains black.
- After valid audio has been seen, the last colour is retained while the boids fade out.
- The movement calculation treats missing pitch as `0.0`.

### Spectral flux

Spectral flux measures how much the spectrum has increased since the previous analysis frame. Sudden attacks and rapid timbral changes produce higher values.

It controls:

- Wander contribution: `spectral_flux × 30`.
- Colour-transition responsiveness.

The colour smoother uses:

```text
responsiveness = max(spectral_flux, recent_beat_response)
colour_time_constant = 0.32 + responsiveness × (0.055 - 0.32) seconds
```

Higher flux therefore makes hue, lightness, and chroma follow the audio more quickly.

### Spectral flatness

Spectral flatness compares noise-like energy with tonal energy. A pure tone is low; broad noise is high.

It directly controls:

- Alignment multiplier: `1 - spectral_flatness × 0.15`.
- Wander contribution: `spectral_flatness × 5`.

It also contributes `45%` of the calculated chaos value, so it indirectly increases separation and wandering while reducing alignment.

### Frequency bands

FFT power is divided into normalized proportions:

| Band | Frequencies | Direct mapping |
| --- | --- | --- |
| Low | Below `250 Hz` | Cohesion multiplier: `1 + low × 0.8` |
| Mid | `250–2000 Hz` | Alignment multiplier: `1 + mid × 0.8` |
| High | `2000 Hz` and above | Separation multiplier: `1 + high × 0.8` |

The three values normally add up to `1.0`. They describe the distribution of energy, not total loudness.

### Onset rate

Onset rate is the number of detected attacks per second, measured over the most recent two seconds.

It controls:

- Steering-force contribution: `clamp(onset_rate_hz / 8, 0, 1) × 45`.
- The `onset_pace` component used by speed and steering force.
- `20%` of calculated chaos after normalizing by `8 onsets/second`.

### Rhythmic irregularity

Rhythmic irregularity measures variation between recent onset intervals. Even spacing approaches `0.0`; inconsistent spacing approaches `1.0`.

It controls:

- Wander contribution: `rhythmic_irregularity × 10`.
- `35%` of calculated chaos.

### Chaos

Chaos is a combined feature calculated during audio analysis:

```text
chaos = clamp(
    spectral_flatness × 0.45
    + rhythmic_irregularity × 0.35
    + clamp(onset_rate_hz / 8, 0, 1) × 0.20,
    0,
    1
)
```

It controls:

- Separation multiplier: `1 + chaos × 0.8`.
- Alignment multiplier: `1 - chaos × 0.7`.
- Wander contribution: `chaos × 45`.

High chaos makes the flock looser, less synchronized, and less predictable.

### BPM and tempo confidence

BPM is estimated between `55` and `200 BPM`. At least four seconds of history are needed, and the estimate is updated every `0.5 seconds`.

BPM is multiplied by tempo confidence before it affects the simulation. A high BPM with poor confidence therefore has little influence.

The resulting `tempo` value competes with `onset_pace`; whichever is higher becomes `pace`.

### Beat events and beat strength

A change in `beat_count` creates a three-ring ripple at a randomly selected visible boid. The main ring begins immediately; two trailing rings begin from the same point after `70 ms` and `140 ms`.

Beat strength is clamped to `0.0–1.0`, with a minimum raw main-ring strength of `0.6`. All ring intensity is then scaled to `50%`, so the main ring's final strength is `0.3–0.5`. Stronger beats have a stronger effect. The first trailing ring uses `60%` of the main-ring strength and the second uses `30%`.

All rings travel at `360 pixels/second`. Their other properties are:

- Main ring: `40 pixels` wide with up to `300 pixels/second²` outward acceleration.
- First trailing ring: `30 pixels` wide with `25%` of the main ring's acceleration.
- Second trailing ring: `22 pixels` wide and visual-only; it adds no outward acceleration.
- Maximum temporary speed-limit increase: `50%`.
- Maximum temporary boid-size increase: `60%`.
- Pulse lightness boost: `+0.12` OKLCH lightness.
- Pulse chroma boost: `+0.04` OKLCH chroma.

Each ring follows a smooth cosine envelope across its wavefront. A boid is affected only while a wavefront passes over it.

A beat also accelerates colour smoothing. Its initial response is:

```text
beat_response = 0.5 + beat_strength × 0.5
```

It decreases at `1 / 0.22` per second, so the strongest possible response lasts `0.22 seconds`.

## Visual-output formulas

This section shows the final combined values used by the boid simulation.

### Number and visibility

```text
target_boids = round(loudness × 1,000)
```

Boid opacity and geometric size both ease with lifecycle visibility. This makes boids smoothly appear and disappear instead of popping.

### Speed and steering force

```text
maximum_speed = 30 + pitch × 120 + pace × 220
maximum_force = 35 + clamp(onset_rate_hz / 8, 0, 1) × 45 + pace × 110
```

Maximum speed is smoothed with a `0.18 second` time constant. A ripple can temporarily raise an affected boid's speed limit by up to `50%`.

### Flocking behavior

```text
separation = 1.6 × (1 + high_band × 0.8) × (1 + chaos × 0.8)

alignment = 1.0
    × (1 + mid_band × 0.8)
    × (1 - chaos × 0.7)
    × (1 - spectral_flatness × 0.15)

cohesion = 0.8 × (1 + low_band × 0.8)

wander = chaos × 45
    + rhythmic_irregularity × 10
    + spectral_flatness × 5
    + spectral_flux × 30
```

### Colour

```text
base_lightness = 0.35 + pitch × 0.50
base_chroma = loudness × 0.24
base_hue = 20° + pitch × 300°
```

The base OKLCH colour is converted into a palette of `33` variants. Each boid receives a stable random palette index:

- Hue variation: `-45°` to `+45°` around the base hue.
- Lightness variation: `-0.035` to `+0.035` correlated with the hue offset.
- The centre variant exactly matches the smoothed base colour.
- Ripple pulses interpolate toward a lighter, more chromatic version of the same variant.
- Out-of-gamut colours reduce chroma while retaining OKLCH lightness and hue as closely as sRGB permits.

## Analysis timing and smoothing

- Input sample rate: `48,000 Hz`.
- FFT window: `2048 samples`, approximately `42.7 ms`.
- Analysis hop: `512 samples`, approximately `10.7 ms` between feature updates.
- Dominant pitch is disabled below `-50 dBFS`.
- Spectral flatness, band energies, onset rate, rhythmic irregularity, chaos, BPM, and tempo confidence are smoothed before reaching the visualisation.
- General analysis smoothing responds faster when a feature rises (`0.35` per update) than when it falls (`0.08` per update).

## Background

The black, white, and transparent modes are not audio-reactive. Boid-colour mode uses the exact smoothed base OKLCH colour for the current frame before the `±45°` per-boid hue variation is applied. It does not receive a ripple colour boost. Like the boid palette, it holds the last valid colour while the boids fade out during silence.
