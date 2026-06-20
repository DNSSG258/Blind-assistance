# Blind-assistance / 视界无界

Blind-assistance is an embedded vision-and-voice assistive prototype for the NXP FRDM-IMX93 board. It combines real-time object detection, stereo distance estimation, GPIO-triggered offline speech recognition, and cached Chinese voice prompts. The project is maintained by the Suzhou University project team **视界无界**.

本项目是面向 NXP FRDM-IMX93 开发板的嵌入式视觉辅助原型系统，集成实时目标检测、双目测距、GPIO 按键触发的离线语音识别，以及中文缓存语音播报。项目归属：**苏州大学 视界无界**。

## Features / 功能

- EfficientDet Lite0 object detection accelerated by Ethos-U.
- Stereo camera distance estimation from a side-by-side camera stream.
- Traffic-light color classification.
- Offline Chinese speech recognition with Vosk.
- Cached Chinese audio prompts for low-latency playback.
- Audio output priority: USB headset first, Bluetooth headset fallback.
- GPIO push-to-talk trigger for board-side interaction.

- 基于 Ethos-U 加速的 EfficientDet Lite0 目标检测。
- 基于双目摄像头的前方障碍物距离估计。
- 红绿灯颜色识别。
- 基于 Vosk 的离线中文语音识别。
- 使用缓存 WAV 音频降低中文播报延迟。
- 音频输出优先使用 USB 耳机，无 USB 时自动连接蓝牙耳机。
- 使用 GPIO 按键触发语音交互。

## Repository Contents / 仓库内容

This repository contains source code, deployment templates, and documentation only. It does **not** include model files, generated audio cache files, or packaged binary tools.

本仓库只包含源码、部署模板和文档，**不包含**模型文件、生成的音频缓存文件和第三方二进制工具。

Included:

```text
camera_detect.py
voice_agent.py
coco-labels-2014_2017.txt
scripts/
systemd/
LICENSE
THIRD_PARTY_NOTICES.md
requirements.txt
```

Not included:

```text
models/
tools/piper/
audio_cache/
*.tflite
*.onnx
*.dll
*.exe
*.ort
vision_state.json
```

## Hardware / 硬件

- NXP FRDM-IMX93 board.
- Side-by-side stereo camera, expected at `/dev/video2`.
- Microphone input for `arecord`.
- USB headset or Bluetooth headset for audio output.
- GPIO button connected to `/dev/gpiochip0` line `2`.
- 12V to 5V Voltage Regulator Module.

- NXP FRDM-IMX93 开发板。
- 双目摄像头，默认使用 `/dev/video2`。
- 麦克风录音输入。
- USB 耳机或蓝牙耳机音频输出。
- GPIO 按键，默认使用 `/dev/gpiochip0` 的 line `2`。
- 12V降5V稳压模块。

## External Assets / 外部文件

Download or prepare the following files locally before running the project.

运行前需要自行下载或生成以下文件。

### EfficientDet Lite0 int8

Download:

```text
https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
```

Place as:

```text
efficientdet_lite0_int8.tflite
```

### EfficientDet Lite0 Vela

The Vela model is generated locally from the int8 TFLite model for Ethos-U deployment.

Vela 版模型由上面的 int8 TFLite 模型使用 Vela 本地编译生成。

Place as:

```text
efficientdet_lite0_int8_vela.tflite
```

### Vosk Chinese Model

Download from:

```text
https://alphacephei.com/vosk/models
```

Use `vosk-model-small-cn-0.22` and place as:

```text
models/vosk-model-small-cn-0.22/
```

### Piper Chinese Voice

Download `zh_CN-huayan-medium.onnx` and `zh_CN-huayan-medium.onnx.json` from:

```text
https://huggingface.co/rhasspy/piper-voices/tree/v1.0.0/zh/zh_CN/huayan/medium
```

Place as:

```text
models/zh_CN-huayan-medium.onnx
models/zh_CN-huayan-medium.onnx.json
```

### Piper Runtime

Piper binaries and runtime dependencies are not distributed in this repository. If cached audio needs to be regenerated, install Piper separately and refer to `voice_agent.py --build-audio-cache`.

本仓库不分发 Piper 可执行文件及其运行时依赖。如需重新生成缓存音频，请单独安装 Piper，并参考 `voice_agent.py --build-audio-cache`。

## Run on Board / 板端运行

Start the camera process:

```bash
python3 camera_detect.py \
  --ethosu on \
  --source /dev/video2 \
  --stereo-distance \
  --state-output vision_state.json \
  --no-display &
```

Start the voice agent:

```bash
python3 voice_agent.py \
  --state-input vision_state.json \
  --input-mode vosk-wav \
  --vosk-model models/vosk-model-small-cn-0.22 \
  --arecord-device auto \
  --record-seconds 2 \
  --tts cached \
  --audio-cache-dir audio_cache \
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
```

## Audio Startup / 音频启动逻辑

The provided startup script template uses this priority:

1. Use USB headset if detected.
2. If no USB headset is detected, run the Bluetooth headset setup script.
3. Start the camera process and voice agent after an audio output path is available.

提供的启动脚本模板使用以下优先级：

1. 检测到 USB 耳机时优先使用 USB 耳机。
2. 未检测到 USB 耳机时运行蓝牙耳机连接脚本。
3. 音频输出准备完成后启动视觉检测和语音助手。

Install the startup script:

```bash
cp scripts/start-vision-assistant.sh /usr/local/bin/start-vision-assistant.sh
nano /usr/local/bin/start-vision-assistant.sh
chmod +x /usr/local/bin/start-vision-assistant.sh
```

Update `APP_DIR` in the script to match the board-side project path.

请根据板端实际路径修改脚本中的 `APP_DIR`。

Install the Bluetooth example script:

```bash
cp scripts/bt-headset-autoconnect.example.sh /usr/local/bin/bt-headset-autoconnect.sh
nano /usr/local/bin/bt-headset-autoconnect.sh
chmod +x /usr/local/bin/bt-headset-autoconnect.sh
```

Replace:

```bash
HEADSET_MAC="AA:BB:CC:DD:EE:FF"
```

with the target Bluetooth headset MAC address.

## Systemd Autostart / 开机自启

Background service:

```bash
cp systemd/vision-assistant.service /etc/systemd/system/vision-assistant.service
systemctl daemon-reload
systemctl enable vision-assistant.service
systemctl start vision-assistant.service
```

View logs:

```bash
journalctl -u vision-assistant.service -f
```

Console-output service for direct terminal display on `tty1`:

```bash
cp systemd/vision-assistant-console.service /etc/systemd/system/vision-assistant-console.service
systemctl daemon-reload
systemctl enable vision-assistant-console.service
systemctl start vision-assistant-console.service
```

If the board uses a serial console instead of `/dev/tty1`, edit `TTYPath` in the service file.

如果开发板使用串口终端而不是 `/dev/tty1`，需要修改 service 文件中的 `TTYPath`。

## Project Team / 项目团队

- Project: Blind-assistance / 视界无界
- Organization: 苏州大学 视界无界
- Team members: fill in the final names before submission.

- 项目名称：Blind-assistance / 视界无界
- 项目归属：苏州大学 视界无界
- 团队成员：张睿、殷安豪、黄家豪

## License / 开源协议

This project is released under the MIT License. See `LICENSE`.

本项目采用 MIT 协议开源，详见 `LICENSE`。

Some code in `camera_detect.py` includes NXP MIT-licensed sample code and keeps the original copyright header. Third-party models and tools are listed in `THIRD_PARTY_NOTICES.md`.

`camera_detect.py` 中包含 NXP MIT 协议示例代码，并保留了原始版权声明。第三方模型和工具说明见 `THIRD_PARTY_NOTICES.md`。
