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
