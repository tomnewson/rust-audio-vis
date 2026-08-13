#!/usr/bin/env bash
set -euo pipefail

# Wraps a binary/example in a minimal .app bundle with NSAudioCaptureUsageDescription
# and launches it via `open`, so macOS attributes the system-audio-capture TCC request
# to the bundle instead of to the terminal (which can fail to request the perms it needs and silently fail)
#
# Usage:
#   scripts/macos-bundle.sh <name> [--example] [-- <args...>]
#
# Examples:
#   scripts/macos-bundle.sh visualiser
#   scripts/macos-bundle.sh capture_probe --example

NAME="${1:?usage: macos-bundle.sh <name> [--example] [-- <args...>]}"
shift

IS_EXAMPLE=0
if [[ "${1:-}" == "--example" ]]; then
  IS_EXAMPLE=1
  shift
fi
if [[ "${1:-}" == "--" ]]; then
  shift
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$IS_EXAMPLE" == 1 ]]; then
  cargo build --example "$NAME"
  BIN_PATH="target/debug/examples/$NAME"
else
  cargo build --bin "$NAME"
  BIN_PATH="target/debug/$NAME"
fi

APP="target/macos-bundle/$NAME.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

ICONSET="assets/icons/AppIcon.iconset"
ICNS="target/macos-bundle/AppIcon.icns"
if [[ -d "$ICONSET" ]] && { [[ ! -f "$ICNS" ]] || [[ "$ICONSET" -nt "$ICNS" ]]; }; then
  iconutil -c icns "$ICONSET" -o "$ICNS"
fi
[[ -f "$ICNS" ]] && cp -f "$ICNS" "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$NAME</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>dev.local.rust-audio-vis</string>
    <key>CFBundleName</key>
    <string>$NAME</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.4</string>
    <key>NSAudioCaptureUsageDescription</key>
    <string>rust-audio-vis visualises system audio playback in real time.</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>rust-audio-vis visualises microphone input in real time.</string>
</dict>
</plist>
PLIST

cp -f "$BIN_PATH" "$APP/Contents/MacOS/$NAME"

codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true

LOG="$(mktemp -t "${NAME}-bundle").log"
echo "Bundled: $APP" >&2
echo "Output: $LOG" >&2
open -W -n --stdout "$LOG" --stderr "$LOG" "$APP" --args "$@"
cat "$LOG"
