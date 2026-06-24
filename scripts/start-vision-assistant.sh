#!/bin/sh
set -e

APP_DIR="/root/efficientdet-lite0"
BT_AUTOCONNECT="/usr/local/bin/bt-headset-autoconnect.sh"
BT_GUI="$APP_DIR/bluetooth_headset_gui.py"
BT_SETUP_MARKER="/tmp/vision-assistant-bt-setup-required"

cd "$APP_DIR"

if aplay -l | grep -qi "USB Audio"; then
    echo "[startup] USB headset detected, using ALSA default output"
    AUDIO_PLAYER='aplay -D default {file}'
else
    echo "[startup] USB headset not found, trying Bluetooth headset"
    if [ -e "$BT_SETUP_MARKER" ]; then
        echo "[startup] Bluetooth setup marker exists, opening headset GUI"
        DISPLAY="${DISPLAY:-:0}" XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/0}" python3 "$BT_GUI" --autoconnect-script "$BT_AUTOCONNECT"
        rm -f "$BT_SETUP_MARKER"
    elif ! "$BT_AUTOCONNECT"; then
        echo "[startup] Bluetooth auto connect failed, opening headset GUI"
        touch "$BT_SETUP_MARKER"
        DISPLAY="${DISPLAY:-:0}" XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/0}" python3 "$BT_GUI" --autoconnect-script "$BT_AUTOCONNECT"
        rm -f "$BT_SETUP_MARKER"
    fi
    export XDG_RUNTIME_DIR=/run/user/0
    export PULSE_RUNTIME_PATH=/run/user/0/pulse
    AUDIO_PLAYER='paplay {file}'
fi

python3 camera_detect.py \
  --ethosu on \
  --source /dev/video2 \
  --stereo-distance \
  --state-output vision_state.json \
  --no-display &

python3 voice_agent.py \
  --state-input vision_state.json \
  --input-mode vosk-wav \
  --vosk-model models/vosk-model-small-cn-0.22 \
  --arecord-device auto \
  --record-seconds 2 \
  --tts cached \
  --audio-cache-dir audio_cache \
  --audio-player "$AUDIO_PLAYER" \
  --trigger-mode gpio \
  --gpio-chip /dev/gpiochip0 \
  --gpio-line 2 \
  --gpio-edge falling \
  --gpio-bias pull-down \
  --gpio-debounce-ms 8 \
  --gpiomon-style long-option \
  --gpio-debug \
  --gpio-event-timeout 1 \
  --audio-timeout 5 \
  --record-timeout-extra 2
