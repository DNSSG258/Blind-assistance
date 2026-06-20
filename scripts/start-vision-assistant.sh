#!/bin/sh
set -e

APP_DIR="/root/Blind-assistance"

export XDG_RUNTIME_DIR=/run/user/0
export PULSE_RUNTIME_PATH=/run/user/0/pulse
unset PULSE_SERVER

cd "$APP_DIR"

if aplay -l | grep -qi "USB Audio"; then
    echo "[startup] USB headset detected, using ALSA default output"
    AUDIO_PLAYER='aplay -D default {file}'
else
    echo "[startup] USB headset not found, trying Bluetooth headset"
    /usr/local/bin/bt-headset-autoconnect.sh
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
