# Third-Party Notices

This project is released under the MIT License. It also uses or documents the following third-party components. Model files, generated audio cache files, and packaged binary tools are intentionally not included in this repository.

## Source Code and Runtime Dependencies

- NXP sample code portions in `camera_detect.py`
  - License: MIT
  - Notice: keep the existing `Copyright 2024 NXP` and `SPDX-License-Identifier: MIT` header.

- NumPy
  - License: BSD-3-Clause
  - Website: https://numpy.org/

- OpenCV
  - License: Apache-2.0
  - Website: https://opencv.org/

- Vosk Python API
  - License: Apache-2.0
  - Website: https://alphacephei.com/vosk/

## External Models and Tools Not Distributed in This Repository

- MediaPipe EfficientDet Lite0 int8 model
  - License: Apache-2.0
  - Download: https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite
  - Expected local path: `efficientdet_lite0_int8.tflite`

- EfficientDet Lite0 Vela model
  - Generated locally from the int8 TFLite model with Vela for Ethos-U deployment.
  - Expected local path: `efficientdet_lite0_int8_vela.tflite`

- Vosk Chinese small model `vosk-model-small-cn-0.22`
  - License: Apache-2.0
  - Download: https://alphacephei.com/vosk/models
  - Expected local path: `models/vosk-model-small-cn-0.22/`

- Piper voice `zh_CN-huayan-medium`
  - License: MIT for the Piper voices repository metadata/model release.
  - Download: https://huggingface.co/rhasspy/piper-voices/tree/v1.0.0/zh/zh_CN/huayan/medium
  - Expected local paths:
    - `models/zh_CN-huayan-medium.onnx`
    - `models/zh_CN-huayan-medium.onnx.json`

- Piper
  - License: MIT
  - Website: https://github.com/rhasspy/piper

- piper-phonemize
  - License: MIT
  - Website: https://github.com/rhasspy/piper-phonemize

- ONNX Runtime
  - License: MIT
  - Website: https://onnxruntime.ai/

- eSpeak NG
  - License: GPL-3.0-or-later
  - Website: https://github.com/espeak-ng/espeak-ng

## Generated Files

The `audio_cache/` directory contains generated WAV files for cached speech playback. It is not included in this repository and can be regenerated locally.
