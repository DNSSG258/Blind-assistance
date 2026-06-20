#!/usr/bin/env python3
# Copyright 2024 NXP
# SPDX-License-Identifier: MIT

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
CPU_MODEL = BASE_DIR / "efficientdet_lite0_int8.tflite"
VELA_MODEL = BASE_DIR / "efficientdet_lite0_int8_vela.tflite"
DEFAULT_LABELS = BASE_DIR / "coco-labels-2014_2017.txt"
INPUT_SIZE = 320
RAW_TOP_K = 1000
FAST_RAW_TOP_K = 200
STEREO_LEFT_WIDTH = 640
STEREO_HEIGHT = 480
STEREO_FRAME_WIDTH = STEREO_LEFT_WIDTH * 2
STEREO_BASELINE_M = 0.06
STEREO_FOCAL_PX = 416.0
STEREO_DISPARITY_OFFSET = 2.5
STEREO_DEPTH_SCALE = 0.5
STEREO_DEPTH_INTERVAL = 3
STEREO_NUM_DISPARITIES = 64
STEREO_BLOCK_SIZE = 7
STEREO_MIN_DISPARITY = 0
STEREO_UNIQUENESS = 5
DEFAULT_THRESHOLD = 0.6
DEFAULT_MAX_DETECTIONS = 10
TRAFFIC_LIGHT_MIN_PIXELS = 3
TRAFFIC_LIGHT_MIN_RATIO = 0.002
BOX_YX_SCALE = 10.0
BOX_HW_SCALE = 5.0
ETHOSU_DELEGATE_CANDIDATES = (
    BASE_DIR / "libethosu_delegate.so",
    Path("/usr/lib/libethosu_delegate.so"),
    Path("/usr/local/lib/libethosu_delegate.so"),
    Path("/usr/lib/aarch64-linux-gnu/libethosu_delegate.so"),
    Path("/usr/lib/tensorflow-lite/libethosu_delegate.so"),
    Path("/usr/lib/ethosu/libethosu_delegate.so"),
)


class UnsupportedModelOutputError(RuntimeError):
    pass


def import_tflite():
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate

        return Interpreter, load_delegate
    except ImportError:
        try:
            import tensorflow as tf

            return tf.lite.Interpreter, tf.lite.experimental.load_delegate
        except ImportError as exc:
            raise RuntimeError(
                "No TensorFlow Lite runtime found. Install python3-tflite-runtime "
                "on the board, or install TensorFlow on a development machine."
            ) from exc


def load_labels(path):
    with open(path, "r", encoding="utf-8") as labels_file:
        return [line.strip() for line in labels_file if line.strip()]


def find_ethosu_delegate():
    for path in ETHOSU_DELEGATE_CANDIDATES:
        if path.exists():
            return path
    return None


def resolve_runtime(args):
    delegate_library = None

    if args.delegate_library:
        delegate_library = Path(args.delegate_library)
    elif args.ethosu in ("auto", "on"):
        delegate_library = find_ethosu_delegate()
        if delegate_library is None and args.ethosu == "on":
            searched = "\n  ".join(str(path) for path in ETHOSU_DELEGATE_CANDIDATES)
            raise RuntimeError(f"Ethos-U delegate library was not found. Searched:\n  {searched}")

    if args.model:
        model_path = args.model
    elif delegate_library and VELA_MODEL.exists():
        model_path = VELA_MODEL
    else:
        model_path = CPU_MODEL

    if model_path.name.endswith("_vela.tflite") and delegate_library is None:
        print(
            "Warning: Vela model selected without an Ethos-U delegate. "
            "Use --ethosu on or --delegate-library, or select the CPU model.",
            file=sys.stderr,
        )

    return model_path, delegate_library


def make_colors(count):
    random.seed(42)
    return [
        (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255),
        )
        for _ in range(count)
    ]


def generate_efficientdet_anchors(image_size=INPUT_SIZE):
    anchors = []
    min_level = 3
    max_level = 7
    num_scales = 3
    anchor_scale = 4.0
    aspect_ratios = (1.0, 2.0, 0.5)

    for level in range(min_level, max_level + 1):
        stride = 2 ** level
        feature_size = int(np.ceil(image_size / stride))

        for y in range(feature_size):
            y_center = (y + 0.5) * stride / image_size
            for x in range(feature_size):
                x_center = (x + 0.5) * stride / image_size
                for scale_octave in range(num_scales):
                    scale = anchor_scale * (2 ** (scale_octave / num_scales))
                    for aspect_ratio in aspect_ratios:
                        aspect_x = np.sqrt(aspect_ratio)
                        aspect_y = 1.0 / aspect_x
                        height = scale * aspect_y * stride / image_size
                        width = scale * aspect_x * stride / image_size
                        anchors.append([y_center, x_center, height, width])

    return np.asarray(anchors, dtype=np.float32)


def build_interpreter(model_path, delegate_library=None, delegate_option=None, threads=2):
    Interpreter, load_delegate = import_tflite()
    delegates = []

    if delegate_library:
        options = {}
        for item in delegate_option or []:
            if "=" not in item:
                raise ValueError(f"Delegate option must be KEY=VALUE, got: {item}")
            key, value = item.split("=", 1)
            options[key] = value
        delegates.append(load_delegate(delegate_library, options))

    try:
        interpreter = Interpreter(
            model_path=str(model_path),
            experimental_delegates=delegates,
            num_threads=threads,
        )
    except TypeError:
        interpreter = Interpreter(
            model_path=str(model_path),
            experimental_delegates=delegates,
        )

    interpreter.allocate_tensors()
    return interpreter


def describe_tensor(detail):
    shape = detail.get("shape", [])
    shape_text = "x".join(str(int(value)) for value in shape)
    return f"name={detail.get('name')} index={detail.get('index')} shape={shape_text} dtype={detail.get('dtype')}"


def print_model_io(input_details, output_details):
    print("Model inputs:")
    for detail in input_details:
        print("  " + describe_tensor(detail))
    print("Model outputs:")
    for detail in output_details:
        print("  " + describe_tensor(detail))


def get_output_array(interpreter, detail):
    return interpreter.get_tensor(detail["index"])


def decode_nms_outputs(interpreter, output_details):
    tensors = [(detail, get_output_array(interpreter, detail)) for detail in output_details]
    boxes = scores = classes = None
    num = None

    for detail, tensor in tensors:
        shape = list(tensor.shape)
        name = detail.get("name", "").lower()

        if len(shape) == 3 and shape[-1] == 4:
            boxes = tensor
        elif len(shape) in (1, 2) and "score" in name:
            scores = tensor
        elif len(shape) in (1, 2) and ("class" in name or "category" in name):
            classes = tensor
        elif len(shape) in (1, 2) and ("count" in name or "num" in name):
            num = tensor

    vectors = []
    for detail, tensor in tensors:
        shape = list(tensor.shape)
        if len(shape) == 2 and shape[0] == 1 and shape[1] != 4:
            vectors.append((detail, tensor))

    if boxes is not None and (scores is None or classes is None) and len(vectors) >= 2:
        non_count_vectors = []
        for detail, tensor in vectors:
            name = detail.get("name", "").lower()
            if "count" not in name and "num" not in name:
                non_count_vectors.append((detail, tensor))

        if len(non_count_vectors) >= 2:
            first = non_count_vectors[0][1]
            second = non_count_vectors[1][1]
            first_is_integerish = np.allclose(first, np.round(first))
            second_is_integerish = np.allclose(second, np.round(second))

            if first_is_integerish and not second_is_integerish:
                classes, scores = first, second
            elif second_is_integerish and not first_is_integerish:
                scores, classes = first, second
            else:
                classes, scores = first, second

    if boxes is None or scores is None or classes is None:
        raise UnsupportedModelOutputError(
            "Unsupported model outputs. This script expects an EfficientDet model with "
            "postprocessed detection outputs: boxes, classes, scores, and optionally count."
        )

    if num is not None:
        try:
            count = int(np.squeeze(num))
            boxes = boxes[:, :count, :]
            scores = scores[:, :count]
            classes = classes[:, :count]
        except (TypeError, ValueError):
            pass

    return boxes, scores, classes


def decode_efficientdet_boxes(encoded_boxes, anchors):
    if encoded_boxes.shape[0] != anchors.shape[0]:
        raise UnsupportedModelOutputError(
            f"Raw box count {encoded_boxes.shape[0]} does not match EfficientDet anchor count {anchors.shape[0]}."
        )

    y_center = encoded_boxes[:, 0] / BOX_YX_SCALE * anchors[:, 2] + anchors[:, 0]
    x_center = encoded_boxes[:, 1] / BOX_YX_SCALE * anchors[:, 3] + anchors[:, 1]
    height = np.exp(np.clip(encoded_boxes[:, 2] / BOX_HW_SCALE, -10.0, 10.0)) * anchors[:, 2]
    width = np.exp(np.clip(encoded_boxes[:, 3] / BOX_HW_SCALE, -10.0, 10.0)) * anchors[:, 3]

    ymin = y_center - height / 2.0
    xmin = x_center - width / 2.0
    ymax = y_center + height / 2.0
    xmax = x_center + width / 2.0
    return np.stack([ymin, xmin, ymax, xmax], axis=1)


def normalize_scores(scores, force_sigmoid=False):
    if force_sigmoid or (scores.size and (np.nanmin(scores) < 0.0 or np.nanmax(scores) > 1.0)):
        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -80.0, 80.0)))
    return scores


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def select_classes(scores, labels, class_offset, ignore_na_labels):
    start = max(0, class_offset)
    if start >= scores.shape[1]:
        start = 0

    valid_class_ids = np.arange(start, scores.shape[1])
    if ignore_na_labels:
        valid_class_ids = np.asarray(
            [
                class_id
                for class_id in valid_class_ids
                if class_id >= len(labels) or labels[class_id].upper() != "N/A"
            ],
            dtype=np.int32,
        )

    if valid_class_ids.size == 0:
        valid_class_ids = np.arange(scores.shape[1])

    valid_scores = scores[:, valid_class_ids]
    best_indices = np.argmax(valid_scores, axis=1)
    class_ids = valid_class_ids[best_indices]
    class_scores = valid_scores[np.arange(scores.shape[0]), best_indices]
    return class_ids, class_scores


def valid_class_ids_for_scores(scores, labels, class_offset, ignore_na_labels):
    start = max(0, class_offset)
    if start >= scores.shape[1]:
        start = 0

    valid_class_ids = np.arange(start, scores.shape[1])
    if ignore_na_labels:
        valid_class_ids = np.asarray(
            [
                class_id
                for class_id in valid_class_ids
                if class_id >= len(labels) or labels[class_id].upper() != "N/A"
            ],
            dtype=np.int32,
        )

    if valid_class_ids.size == 0:
        valid_class_ids = np.arange(scores.shape[1])

    return valid_class_ids


def scores_are_logits(scores, mode):
    if mode == "logits":
        return True
    if mode == "prob":
        return False
    return bool(scores.size and (np.nanmin(scores) < 0.0 or np.nanmax(scores) > 1.0))


def select_top_candidates(scores, labels, class_offset, ignore_na_labels, threshold, top_k, score_mode, candidate_multiplier):
    start = max(0, class_offset)
    if start >= scores.shape[1]:
        start = 0

    score_view = scores[:, start:]
    use_logits = scores_are_logits(score_view, score_mode)
    flat_scores = score_view.reshape(-1)
    candidate_count = min(flat_scores.size, max(top_k * candidate_multiplier, top_k + 256))

    if candidate_count < flat_scores.size:
        flat_indices = np.argpartition(flat_scores, -candidate_count)[-candidate_count:]
    else:
        flat_indices = np.arange(flat_scores.size)

    raw_scores = flat_scores[flat_indices]
    order = np.argsort(raw_scores)[::-1]
    flat_indices = flat_indices[order]
    raw_scores = raw_scores[order]

    num_classes = score_view.shape[1]
    anchor_ids = flat_indices // num_classes
    class_ids = (flat_indices % num_classes) + start

    if ignore_na_labels and labels:
        valid_mask = np.asarray(
            [
                class_id >= len(labels) or labels[int(class_id)].upper() != "N/A"
                for class_id in class_ids
            ],
            dtype=bool,
        )
        anchor_ids = anchor_ids[valid_mask]
        class_ids = class_ids[valid_mask]
        raw_scores = raw_scores[valid_mask]

    if anchor_ids.size == 0:
        return anchor_ids.astype(np.int32), class_ids.astype(np.int32), raw_scores.astype(np.float32)

    class_scores = sigmoid(raw_scores) if use_logits else raw_scores
    score_mask = class_scores >= threshold
    anchor_ids = anchor_ids[score_mask]
    class_ids = class_ids[score_mask]
    class_scores = class_scores[score_mask]

    if anchor_ids.size == 0:
        return anchor_ids.astype(np.int32), class_ids.astype(np.int32), class_scores.astype(np.float32)

    _, first_indices = np.unique(anchor_ids, return_index=True)
    first_indices = np.sort(first_indices)
    anchor_ids = anchor_ids[first_indices]
    class_ids = class_ids[first_indices]
    class_scores = class_scores[first_indices]

    if anchor_ids.size > top_k:
        anchor_ids = anchor_ids[:top_k]
        class_ids = class_ids[:top_k]
        class_scores = class_scores[:top_k]

    return anchor_ids.astype(np.int32), class_ids.astype(np.int32), class_scores.astype(np.float32)


def run_nms(
    boxes,
    scores,
    threshold,
    iou_threshold,
    max_detections,
    anchors=None,
    labels=None,
    class_offset=1,
    ignore_na_labels=True,
):
    boxes = np.squeeze(boxes, axis=0)
    scores = np.squeeze(scores, axis=0)

    if boxes.ndim != 2 or boxes.shape[-1] != 4 or scores.ndim != 2:
        raise UnsupportedModelOutputError(
            f"Unsupported raw output shape: boxes={boxes.shape}, scores={scores.shape}"
        )

    scores = normalize_scores(scores)
    class_ids, class_scores = select_classes(
        scores,
        labels or [],
        class_offset,
        ignore_na_labels,
    )
    keep = np.where(class_scores >= threshold)[0]

    if keep.size == 0:
        return (
            np.zeros((1, 0, 4), dtype=np.float32),
            np.zeros((1, 0), dtype=np.float32),
            np.zeros((1, 0), dtype=np.float32),
        )

    if keep.size > RAW_TOP_K:
        top = np.argpartition(class_scores[keep], -RAW_TOP_K)[-RAW_TOP_K:]
        keep = keep[top]

    if anchors is not None:
        decoded_boxes = decode_efficientdet_boxes(boxes[keep], anchors[keep])
    else:
        decoded_boxes = boxes[keep]

    candidate_boxes = np.clip(decoded_boxes, 0.0, 1.0)
    candidate_scores = class_scores[keep]
    candidate_classes = class_ids[keep]

    selected = []
    for class_id in np.unique(candidate_classes):
        class_keep = np.where(candidate_classes == class_id)[0]
        nms_boxes = []
        nms_scores = []

        for box_index in class_keep:
            ymin, xmin, ymax, xmax = candidate_boxes[box_index]
            width = max(0.0, xmax - xmin)
            height = max(0.0, ymax - ymin)
            nms_boxes.append([float(xmin), float(ymin), float(width), float(height)])
            nms_scores.append(float(candidate_scores[box_index]))

        indices = cv2.dnn.NMSBoxes(
            nms_boxes,
            nms_scores,
            float(threshold),
            float(iou_threshold),
            top_k=max_detections,
        )

        for local_index in np.array(indices).reshape(-1):
            selected.append(class_keep[int(local_index)])

    if not selected:
        return (
            np.zeros((1, 0, 4), dtype=np.float32),
            np.zeros((1, 0), dtype=np.float32),
            np.zeros((1, 0), dtype=np.float32),
        )

    selected = sorted(selected, key=lambda index: candidate_scores[index], reverse=True)
    selected = selected[:max_detections]

    return (
        candidate_boxes[selected][np.newaxis, :, :].astype(np.float32),
        candidate_scores[selected][np.newaxis, :].astype(np.float32),
        candidate_classes[selected][np.newaxis, :].astype(np.float32),
    )


def run_candidate_nms(candidate_boxes, candidate_scores, candidate_classes, threshold, iou_threshold, max_detections):
    if candidate_scores.size == 0:
        return (
            np.zeros((1, 0, 4), dtype=np.float32),
            np.zeros((1, 0), dtype=np.float32),
            np.zeros((1, 0), dtype=np.float32),
        )

    nms_boxes = []
    for box in candidate_boxes:
        ymin, xmin, ymax, xmax = box
        width = max(0.0, xmax - xmin)
        height = max(0.0, ymax - ymin)
        nms_boxes.append([float(xmin), float(ymin), float(width), float(height)])

    indices = cv2.dnn.NMSBoxes(
        nms_boxes,
        candidate_scores.astype(float).tolist(),
        float(threshold),
        float(iou_threshold),
        top_k=max_detections,
    )
    selected = [int(index) for index in np.array(indices).reshape(-1)]
    selected = [
        index
        for index in selected
        if candidate_boxes[index][2] > candidate_boxes[index][0]
        and candidate_boxes[index][3] > candidate_boxes[index][1]
    ]

    if not selected:
        return (
            np.zeros((1, 0, 4), dtype=np.float32),
            np.zeros((1, 0), dtype=np.float32),
            np.zeros((1, 0), dtype=np.float32),
        )

    selected = sorted(selected, key=lambda index: candidate_scores[index], reverse=True)
    selected = selected[:max_detections]
    return (
        candidate_boxes[selected][np.newaxis, :, :].astype(np.float32),
        candidate_scores[selected][np.newaxis, :].astype(np.float32),
        candidate_classes[selected][np.newaxis, :].astype(np.float32),
    )


def run_raw_postprocess_fast(
    boxes,
    scores,
    threshold,
    iou_threshold,
    max_detections,
    anchors,
    labels,
    class_offset,
    ignore_na_labels,
    top_k,
    score_mode,
    candidate_multiplier,
):
    boxes = np.squeeze(boxes, axis=0)
    scores = np.squeeze(scores, axis=0)

    if boxes.ndim != 2 or boxes.shape[-1] != 4 or scores.ndim != 2:
        raise UnsupportedModelOutputError(
            f"Unsupported raw output shape: boxes={boxes.shape}, scores={scores.shape}"
        )

    keep, class_ids, class_scores = select_top_candidates(
        scores,
        labels or [],
        class_offset,
        ignore_na_labels,
        threshold,
        top_k,
        score_mode,
        candidate_multiplier,
    )

    if keep.size == 0:
        return (
            np.zeros((1, 0, 4), dtype=np.float32),
            np.zeros((1, 0), dtype=np.float32),
            np.zeros((1, 0), dtype=np.float32),
        )

    if anchors is not None:
        candidate_boxes = decode_efficientdet_boxes(boxes[keep], anchors[keep])
    else:
        candidate_boxes = boxes[keep]

    candidate_boxes = np.clip(candidate_boxes, 0.0, 1.0)
    return run_candidate_nms(
        candidate_boxes,
        class_scores,
        class_ids,
        threshold,
        iou_threshold,
        max_detections,
    )


def decode_raw_outputs(
    interpreter,
    output_details,
    threshold,
    iou_threshold,
    max_detections,
    anchors,
    labels,
    class_offset,
    ignore_na_labels,
    fast_postprocess,
    top_k,
    score_mode,
    candidate_multiplier,
):
    tensors = [(detail, get_output_array(interpreter, detail)) for detail in output_details]
    boxes = scores = None

    for detail, tensor in tensors:
        shape = list(tensor.shape)
        name = detail.get("name", "").lower()
        if len(shape) == 3 and shape[-1] == 4:
            boxes = tensor
        elif len(shape) == 3 and shape[-1] > 4:
            scores = tensor
        elif "boundingbox" in name or "box" in name:
            boxes = tensor
        elif "score" in name or "feature" in name:
            scores = tensor

    if boxes is None or scores is None:
        raise UnsupportedModelOutputError(
            "Unsupported model outputs. Expected either postprocessed outputs or raw boxes/scores outputs."
        )

    use_anchors = anchors if boxes.shape[1] == anchors.shape[0] else None
    if fast_postprocess:
        return run_raw_postprocess_fast(
            boxes,
            scores,
            threshold,
            iou_threshold,
            max_detections,
            use_anchors,
            labels,
            class_offset,
            ignore_na_labels,
            top_k,
            score_mode,
            candidate_multiplier,
        )

    return run_nms(
        boxes,
        scores,
        threshold,
        iou_threshold,
        max_detections,
        use_anchors,
        labels,
        class_offset,
        ignore_na_labels,
    )


def decode_outputs(
    interpreter,
    output_details,
    threshold,
    iou_threshold,
    max_detections,
    anchors,
    labels,
    class_offset,
    ignore_na_labels,
    fast_postprocess,
    top_k,
    score_mode,
    candidate_multiplier,
):
    try:
        return decode_nms_outputs(interpreter, output_details)
    except UnsupportedModelOutputError:
        return decode_raw_outputs(
            interpreter,
            output_details,
            threshold,
            iou_threshold,
            max_detections,
            anchors,
            labels,
            class_offset,
            ignore_na_labels,
            fast_postprocess,
            top_k,
            score_mode,
            candidate_multiplier,
        )


def preprocess_frame(frame, input_detail):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    input_data = np.expand_dims(resized, axis=0)
    dtype = input_detail["dtype"]

    if dtype == np.float32:
        return input_data.astype(np.float32)

    if dtype == np.uint8:
        return input_data.astype(np.uint8)

    if dtype == np.int8:
        scale, zero_point = input_detail.get("quantization", (0.0, 0))
        if scale and scale > 0:
            normalized = (input_data.astype(np.float32) - 127.5) / 127.5
            input_data = normalized / scale + zero_point
        input_data = np.clip(input_data, np.iinfo(dtype).min, np.iinfo(dtype).max)
        return input_data.astype(dtype)

    return input_data.astype(dtype)


def run_inference(
    interpreter,
    input_detail,
    output_details,
    frame,
    threshold,
    iou_threshold,
    max_detections,
    anchors,
    labels,
    class_offset,
    ignore_na_labels,
    fast_postprocess,
    top_k,
    score_mode,
    candidate_multiplier,
    profile=False,
):
    preprocess_start = time.perf_counter()
    input_data = preprocess_frame(frame, input_detail)
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0
    interpreter.set_tensor(input_detail["index"], input_data)

    start = time.perf_counter()
    interpreter.invoke()
    inference_ms = (time.perf_counter() - start) * 1000.0

    postprocess_start = time.perf_counter()
    boxes, scores, classes = decode_outputs(
        interpreter,
        output_details,
        threshold,
        iou_threshold,
        max_detections,
        anchors,
        labels,
        class_offset,
        ignore_na_labels,
        fast_postprocess,
        top_k,
        score_mode,
        candidate_multiplier,
    )
    postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
    if profile:
        return boxes, scores, classes, inference_ms, preprocess_ms, postprocess_ms
    return boxes, scores, classes, inference_ms


def draw_detections(frame, boxes, scores, classes, labels, colors, threshold):
    height, width = frame.shape[:2]
    boxes = np.squeeze(boxes, axis=0)
    scores = np.squeeze(scores, axis=0)
    classes = np.squeeze(classes, axis=0)
    detections = []

    for box, score, class_id in zip(boxes, scores, classes):
        if score < threshold:
            continue

        class_index = int(class_id)
        label = labels[class_index] if 0 <= class_index < len(labels) else str(class_index)
        color = colors[class_index % len(colors)]

        ymin, xmin, ymax, xmax = box
        left = max(0, min(width - 1, int(xmin * width)))
        top = max(0, min(height - 1, int(ymin * height)))
        right = max(0, min(width - 1, int(xmax * width)))
        bottom = max(0, min(height - 1, int(ymax * height)))

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        text_y = top - 8 if top > 24 else top + 22
        detections.append(
            {
                "box": (left, top, right, bottom),
                "text_origin": (left, text_y),
                "score": float(score),
                "class_id": class_index,
                "label": label,
                "color": color,
            }
        )

    return detections


def annotate_detection_labels(frame, detections):
    for detection in detections:
        cv2.putText(
            frame,
            detection_display_name(detection),
            detection["text_origin"],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            detection["color"],
            2,
        )


def detection_display_name(detection):
    label = detection.get("display_label", detection.get("label", ""))
    return label_to_chinese(label)


def is_traffic_light_detection(detection):
    label = str(detection.get("label", "")).lower().replace("_", " ").strip()
    label = " ".join(label.split())
    return label in ("traffic light", "traffic lights") or ("traffic" in label and "light" in label)


def classify_traffic_light_color(frame, box, min_pixels=TRAFFIC_LIGHT_MIN_PIXELS, min_ratio=TRAFFIC_LIGHT_MIN_RATIO):
    left, top, right, bottom = box
    width = max(1, right - left)
    height = max(1, bottom - top)
    pad_x = 2 if width > 10 else 0
    pad_y = 2 if height > 10 else 0
    roi = frame[top + pad_y : bottom - pad_y, left + pad_x : right - pad_x]
    if roi.size == 0:
        return None

    height = roi.shape[0]
    if height > 12:
        roi = roi[: max(1, int(height * 0.85)), :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask = (
        cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        | cv2.inRange(hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    )
    yellow_mask = cv2.inRange(hsv, np.array([15, 70, 90]), np.array([38, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([40, 60, 70]), np.array([95, 255, 255]))

    counts = {
        "red": int(cv2.countNonZero(red_mask)),
        "yellow": int(cv2.countNonZero(yellow_mask)),
        "green": int(cv2.countNonZero(green_mask)),
    }
    color, count = max(counts.items(), key=lambda item: item[1])
    if count < max(min_pixels, roi.shape[0] * roi.shape[1] * min_ratio):
        return None
    return color


def update_traffic_light_labels(
    frame,
    detections,
    enabled=True,
    min_pixels=TRAFFIC_LIGHT_MIN_PIXELS,
    min_ratio=TRAFFIC_LIGHT_MIN_RATIO,
    debug=False,
):
    if not enabled:
        return
    for detection in detections:
        if not is_traffic_light_detection(detection):
            continue
        light_color = classify_traffic_light_color(frame, detection["box"], min_pixels, min_ratio)
        if light_color:
            detection["display_label"] = f"traffic light {light_color}"
        else:
            detection["display_label"] = "traffic light unknown"
            if debug:
                print(f"traffic light color debug: box={detection['box']} no color matched")


def open_capture(source, width, height, fps):
    def configure_capture(capture):
        if width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            capture.set(cv2.CAP_PROP_FPS, fps)
        return capture

    try:
        camera_index = int(source)
    except ValueError:
        if "!" in source:
            capture = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        else:
            capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        configure_capture(capture)

        if not capture.isOpened() and str(source).startswith("/dev/video"):
            capture.release()
            try:
                camera_index = int(str(source).removeprefix("/dev/video"))
            except ValueError:
                raise RuntimeError(f"Could not open camera source: {source}")
            for backend in (cv2.CAP_V4L2, cv2.CAP_ANY):
                capture = configure_capture(cv2.VideoCapture(camera_index, backend))
                if capture.isOpened():
                    return capture
                capture.release()
            raise RuntimeError(f"Could not open camera source: {source}")
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")
        return capture

    if sys.platform.startswith("win"):
        backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
    else:
        backends = (cv2.CAP_V4L2, cv2.CAP_ANY)

    for backend in backends:
        capture = configure_capture(cv2.VideoCapture(camera_index, backend))
        if capture.isOpened():
            return capture
        capture.release()

    raise RuntimeError(f"Could not open camera source: {source}")


def default_camera_source():
    if sys.platform.startswith("win"):
        return "0"
    return "/dev/video0"


def annotate_stats(frame, fps_now, inference_ms, detections):
    cv2.putText(
        frame,
        f"{fps_now:.1f} FPS  {inference_ms:.1f} ms  objects:{detections}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


def split_stereo_frame(frame):
    left = frame[:STEREO_HEIGHT, :STEREO_LEFT_WIDTH]
    right = frame[:STEREO_HEIGHT, STEREO_LEFT_WIDTH:STEREO_FRAME_WIDTH]
    return left, right


def create_stereo_matcher(args):
    block_size = args.stereo_block_size
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(5, block_size)
    num_disparities = max(16, args.stereo_num_disparities)
    num_disparities = (num_disparities + 15) // 16 * 16

    if args.stereo_method == "bm":
        matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
        matcher.setMinDisparity(args.stereo_min_disparity)
        matcher.setPreFilterCap(31)
        matcher.setTextureThreshold(10)
        matcher.setUniquenessRatio(args.stereo_uniqueness)
        matcher.setSpeckleWindowSize(100)
        matcher.setSpeckleRange(32)
        matcher.setDisp12MaxDiff(1)
        return matcher

    return cv2.StereoSGBM_create(
        minDisparity=args.stereo_min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size * block_size,
        P2=32 * 3 * block_size * block_size,
        disp12MaxDiff=1,
        preFilterCap=63,
        uniquenessRatio=args.stereo_uniqueness,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity(left_frame, right_frame, stereo_matcher, depth_scale):
    left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    if depth_scale != 1.0:
        scaled_size = (
            max(1, int(left_gray.shape[1] * depth_scale)),
            max(1, int(left_gray.shape[0] * depth_scale)),
        )
        left_gray = cv2.resize(left_gray, scaled_size, interpolation=cv2.INTER_AREA)
        right_gray = cv2.resize(right_gray, scaled_size, interpolation=cv2.INTER_AREA)

    disparity_raw = stereo_matcher.compute(left_gray, right_gray).astype(np.float32)
    disparity = disparity_raw / 16.0
    if depth_scale != 1.0:
        disparity = cv2.resize(
            disparity,
            (left_frame.shape[1], left_frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        disparity = disparity / depth_scale
    return disparity


def render_disparity(disparity):
    valid = disparity[np.isfinite(disparity) & (disparity > 0.5)]
    if valid.size == 0:
        return np.zeros(disparity.shape, dtype=np.uint8)
    max_disp = max(16.0, float(np.percentile(valid, 95)))
    scaled = np.clip(disparity * (255.0 / max_disp), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


def render_depth(disparity, focal_px, baseline_m, disparity_offset, max_distance_m):
    depth = np.zeros(disparity.shape, dtype=np.float32)
    corrected_disparity = disparity - disparity_offset
    valid = np.isfinite(corrected_disparity) & (corrected_disparity > 0.5)
    depth[valid] = focal_px * baseline_m / corrected_disparity[valid]
    valid_depth = depth[(depth > 0.0) & (depth <= max_distance_m)]
    if valid_depth.size == 0:
        return np.zeros(disparity.shape, dtype=np.uint8)

    max_depth = min(max_distance_m, max(1.0, float(np.percentile(valid_depth, 95))))
    normalized = np.zeros(disparity.shape, dtype=np.uint8)
    normalized[valid] = np.clip((1.0 - depth[valid] / max_depth) * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = (0, 0, 0)
    return colored


def build_display_frame(detect_frame, cached_disparity, args):
    if not args.render_depth or cached_disparity is None:
        return detect_frame

    depth_view = render_depth(
        cached_disparity,
        args.stereo_focal_px,
        args.stereo_baseline_m,
        args.disparity_offset,
        args.max_distance_m,
    )
    if depth_view.shape[:2] != detect_frame.shape[:2]:
        depth_view = cv2.resize(
            depth_view,
            (detect_frame.shape[1], detect_frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    cv2.putText(depth_view, "Depth", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if args.depth_view == "only":
        return depth_view
    return np.hstack((detect_frame, depth_view))


def estimate_detection_distance(disparity, box, focal_px, baseline_m, disparity_offset, debug=False):
    left, top, right, bottom = box
    width = max(1, right - left)
    height = max(1, bottom - top)

    pad_x = max(1, int(width * 0.15))
    pad_y = max(1, int(height * 0.15))
    x1 = max(0, left + pad_x)
    y1 = max(0, top + pad_y)
    x2 = min(disparity.shape[1], right - pad_x)
    y2 = min(disparity.shape[0], bottom - pad_y)

    if x2 <= x1 or y2 <= y1:
        x1, y1 = max(0, left), max(0, top)
        x2, y2 = min(disparity.shape[1], right), min(disparity.shape[0], bottom)

    roi = disparity[y1:y2, x1:x2]
    valid = roi[np.isfinite(roi) & (roi > 0.5)]
    if valid.size < 10:
        if debug:
            print(f"distance debug: box={box} valid_disparity={valid.size}")
        return None

    disparity_px = float(np.median(valid))
    corrected_disparity = disparity_px - disparity_offset
    if corrected_disparity <= 0.5:
        if debug:
            print(
                f"distance debug: box={box} median_disparity={disparity_px:.3f} "
                f"corrected_disparity={corrected_disparity:.3f}"
            )
        return None
    distance_m = focal_px * baseline_m / corrected_disparity
    if debug:
        print(
            f"distance debug: box={box} valid_disparity={valid.size} "
            f"median_disparity={disparity_px:.3f} "
            f"corrected_disparity={corrected_disparity:.3f} "
            f"distance_m={distance_m:.3f}"
        )
    return distance_m


def format_distance(distance_m):
    if distance_m is None:
        return None
    return round(float(distance_m), 2)


def estimate_detection_distances(detections, disparity, args):
    distances = []
    for detection in detections:
        distance_m = estimate_detection_distance(
            disparity,
            detection["box"],
            args.stereo_focal_px,
            args.stereo_baseline_m,
            args.disparity_offset,
            args.debug_distance,
        )
        if distance_m is not None and distance_m > args.max_distance_m:
            distance_m = None
        distances.append(distance_m)
    return distances


def annotate_distances(frame, detections, disparity, args):
    distances = estimate_detection_distances(detections, disparity, args)
    for detection, distance_m in zip(detections, distances):
        if distance_m is None or distance_m > args.max_distance_m:
            distance_text = "-- m"
        else:
            distance_text = f"{distance_m:.2f} m"

        text = f"{detection_display_name(detection)} {distance_text}"
        cv2.putText(
            frame,
            text,
            detection["text_origin"],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            detection["color"],
            2,
        )


def compute_distance_labels(detections, disparity, args):
    labels = []
    distances = estimate_detection_distances(detections, disparity, args)
    for detection, distance_m in zip(detections, distances):
        if distance_m is None or distance_m > args.max_distance_m:
            distance_text = "-- m"
        else:
            distance_text = f"{distance_m:.2f} m"
        labels.append(f"{detection_display_name(detection)} {distance_text}")
    return labels


def extract_traffic_light_state(detections):
    for detection in detections:
        label = detection.get("display_label", detection.get("label", ""))
        normalized = str(label).lower()
        if "traffic light" not in normalized:
            continue
        if "red" in normalized:
            return "red"
        if "yellow" in normalized:
            return "yellow"
        if "green" in normalized:
            return "green"
        return "unknown"
    return None


def label_to_chinese(label):
    normalized = str(label).lower().strip()
    mapping = {
        "person": "行人",
        "bicycle": "自行车",
        "car": "汽车",
        "motorcycle": "摩托车",
        "airplane": "飞机",
        "bus": "公交车",
        "train": "火车",
        "truck": "卡车",
        "boat": "船",
        "fire hydrant": "消防栓",
        "stop sign": "停止标志",
        "parking meter": "停车计时器",
        "chair": "椅子",
        "bench": "长椅",
        "traffic light": "交通灯",
        "traffic light red": "红灯",
        "traffic light yellow": "黄灯",
        "traffic light green": "绿灯",
        "traffic light unknown": "交通灯",
        "bird": "鸟",
        "cat": "猫",
        "dog": "狗",
        "horse": "马",
        "sheep": "羊",
        "cow": "牛",
        "elephant": "大象",
        "bear": "熊",
        "zebra": "斑马",
        "giraffe": "长颈鹿",
        "sports ball": "球",
        "frisbee": "飞盘",
        "skis": "滑雪板",
        "snowboard": "单板滑雪板",
        "kite": "风筝",
        "baseball bat": "棒球棒",
        "baseball glove": "棒球手套",
        "skateboard": "滑板",
        "surfboard": "冲浪板",
        "tennis racket": "网球拍",
        "suitcase": "行李箱",
        "backpack": "背包",
        "umbrella": "雨伞",
        "handbag": "手提包",
        "tie": "领带",
        "bottle": "瓶子",
        "wine glass": "酒杯",
        "cup": "杯子",
        "fork": "叉子",
        "knife": "刀",
        "spoon": "勺子",
        "bowl": "碗",
        "banana": "香蕉",
        "apple": "苹果",
        "sandwich": "三明治",
        "orange": "橙子",
        "broccoli": "西兰花",
        "carrot": "胡萝卜",
        "hot dog": "热狗",
        "pizza": "披萨",
        "donut": "甜甜圈",
        "cake": "蛋糕",
        "couch": "沙发",
        "potted plant": "盆栽",
        "bed": "床",
        "dining table": "餐桌",
        "toilet": "马桶",
        "cell phone": "手机",
        "laptop": "电脑",
        "mouse": "鼠标",
        "remote": "遥控器",
        "keyboard": "键盘",
        "microwave": "微波炉",
        "oven": "烤箱",
        "toaster": "烤面包机",
        "sink": "水槽",
        "refrigerator": "冰箱",
        "book": "书",
        "clock": "时钟",
        "vase": "花瓶",
        "scissors": "剪刀",
        "teddy bear": "泰迪熊",
        "hair drier": "吹风机",
        "toothbrush": "牙刷",
        "tv": "电视",
    }
    return mapping.get(normalized, label)


def build_vision_state(detections, distances, args, frame_shape=None):
    objects = []
    for detection, distance_m in zip(detections, distances):
        label = detection.get("display_label", detection.get("label", "unknown"))
        item = {
            "label": label,
            "label_zh": label_to_chinese(label),
            "score": round(float(detection.get("score", 0.0)), 3),
            "class_id": int(detection.get("class_id", -1)),
            "box": [int(value) for value in detection.get("box", (0, 0, 0, 0))],
            "distance_m": format_distance(distance_m),
        }
        objects.append(item)

    nearest = None
    valid_distances = [item for item in objects if item["distance_m"] is not None]
    if valid_distances:
        nearest = min(valid_distances, key=lambda item: item["distance_m"])

    traffic_light = extract_traffic_light_state(detections)
    summary = build_vision_summary(objects, nearest, traffic_light)
    state = {
        "timestamp": time.time(),
        "objects": objects,
        "nearest_object": nearest,
        "traffic_light": traffic_light,
        "summary": summary,
    }
    if frame_shape is not None:
        state["frame_size"] = {
            "width": int(frame_shape[1]),
            "height": int(frame_shape[0]),
        }
    return state


def build_vision_summary(objects, nearest, traffic_light):
    parts = []
    if nearest:
        parts.append(f"前方{nearest['distance_m']:.2f}米有{nearest['label_zh']}")
    elif objects:
        names = []
        for item in objects[:3]:
            if item["label_zh"] not in names:
                names.append(item["label_zh"])
        parts.append("前方检测到" + "、".join(names))
    else:
        parts.append("前方暂未检测到明显物体")

    if traffic_light == "red":
        parts.append("检测到红灯")
    elif traffic_light == "yellow":
        parts.append("检测到黄灯")
    elif traffic_light == "green":
        parts.append("检测到绿灯")
    elif traffic_light == "unknown":
        parts.append("检测到交通灯")

    return "，".join(parts) + "。"


def write_vision_state(path, state):
    if not path:
        return
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as state_file:
        json.dump(state, state_file, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def annotate_distance_labels(frame, detections, labels):
    for detection, text in zip(detections, labels):
        cv2.putText(
            frame,
            text,
            detection["text_origin"],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            detection["color"],
            2,
        )


def run_image(args, interpreter, input_detail, output_details, labels, colors, anchors):
    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    result = run_inference(
        interpreter,
        input_detail,
        output_details,
        frame,
        args.threshold,
        args.iou_threshold,
        args.max_detections,
        anchors,
        labels,
        args.class_offset,
        not args.keep_na_labels,
        not args.slow_postprocess,
        args.top_k,
        args.raw_score_mode,
        args.candidate_multiplier,
        args.profile,
    )
    if args.profile:
        boxes, scores, classes, inference_ms, preprocess_ms, postprocess_ms = result
        print(f"preprocess_ms={preprocess_ms:.2f} inference_ms={inference_ms:.2f} postprocess_ms={postprocess_ms:.2f}")
    else:
        boxes, scores, classes, inference_ms = result
    detections = draw_detections(frame, boxes, scores, classes, labels, colors, args.threshold)
    update_traffic_light_labels(
        frame,
        detections,
        not args.disable_traffic_light_color,
        args.traffic_light_min_pixels,
        args.traffic_light_min_ratio,
        args.debug_traffic_light_color,
    )
    annotate_detection_labels(frame, detections)
    annotate_stats(frame, 0.0, inference_ms, len(detections))

    output_path = args.output or BASE_DIR / "camera_detect_output.jpg"
    cv2.imwrite(str(output_path), frame)
    print(f"Saved {output_path}")
    print(f"inference_ms={inference_ms:.2f} objects={len(detections)}")


def run_camera(args, interpreter, input_detail, input_details, output_details, labels, colors, anchors):
    capture = open_capture(args.source, args.width, args.height, args.fps)
    writer = None
    frame_count = 0
    fps_start = time.perf_counter()
    stereo_matcher = create_stereo_matcher(args) if args.stereo_distance else None
    cached_distance_labels = []
    cached_distances = []
    cached_disparity = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera frame read failed.", file=sys.stderr)
                break

            if args.stereo_distance:
                if frame.shape[1] < STEREO_FRAME_WIDTH or frame.shape[0] < STEREO_HEIGHT:
                    print(
                        f"Stereo mode requires at least {STEREO_FRAME_WIDTH}x{STEREO_HEIGHT}, "
                        f"got {frame.shape[1]}x{frame.shape[0]}.",
                        file=sys.stderr,
                    )
                    break
                left_frame, right_frame = split_stereo_frame(frame)
                detect_frame = left_frame.copy()
            else:
                detect_frame = frame

            try:
                result = run_inference(
                    interpreter,
                    input_detail,
                    output_details,
                    detect_frame,
                    args.threshold,
                    args.iou_threshold,
                    args.max_detections,
                    anchors,
                    labels,
                    args.class_offset,
                    not args.keep_na_labels,
                    not args.slow_postprocess,
                    args.top_k,
                    args.raw_score_mode,
                    args.candidate_multiplier,
                    args.profile,
                )
                if args.profile:
                    boxes, scores, classes, inference_ms, preprocess_ms, postprocess_ms = result
                    print(
                        f"preprocess_ms={preprocess_ms:.2f} "
                        f"inference_ms={inference_ms:.2f} "
                        f"postprocess_ms={postprocess_ms:.2f}"
                    )
                else:
                    boxes, scores, classes, inference_ms = result
            except UnsupportedModelOutputError as exc:
                print(str(exc), file=sys.stderr)
                print_model_io(input_details, output_details)
                break

            detections = draw_detections(
                detect_frame, boxes, scores, classes, labels, colors, args.threshold
            )
            update_traffic_light_labels(
                detect_frame,
                detections,
                not args.disable_traffic_light_color,
                args.traffic_light_min_pixels,
                args.traffic_light_min_ratio,
                args.debug_traffic_light_color,
            )

            if args.stereo_distance:
                should_update_depth = (
                    frame_count % max(1, args.depth_interval) == 0
                    or len(cached_distance_labels) != len(detections)
                )
                if should_update_depth:
                    cached_disparity = compute_disparity(
                        left_frame,
                        right_frame,
                        stereo_matcher,
                        args.depth_scale,
                    )
                    cached_distance_labels = compute_distance_labels(detections, cached_disparity, args)
                    cached_distances = estimate_detection_distances(detections, cached_disparity, args)

                if len(cached_distance_labels) == len(detections):
                    annotate_distance_labels(detect_frame, detections, cached_distance_labels)
                else:
                    annotate_distances(detect_frame, detections, cached_disparity, args)
                    cached_distances = estimate_detection_distances(detections, cached_disparity, args)

                if args.show_disparity and cached_disparity is not None and not args.no_display:
                    cv2.imshow("disparity", render_disparity(cached_disparity))
            else:
                annotate_detection_labels(detect_frame, detections)
                cached_distances = [None] * len(detections)

            if args.state_output and frame_count % max(1, args.state_interval) == 0:
                if len(cached_distances) != len(detections):
                    cached_distances = [None] * len(detections)
                write_vision_state(
                    args.state_output,
                    build_vision_state(detections, cached_distances, args, detect_frame.shape),
                )

            frame_count += 1
            elapsed = max(time.perf_counter() - fps_start, 1e-6)
            fps_now = frame_count / elapsed
            annotate_stats(detect_frame, fps_now, inference_ms, len(detections))

            display_frame = build_display_frame(detect_frame, cached_disparity, args)

            if args.output:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(args.output),
                        fourcc,
                        args.fps or 30,
                        (display_frame.shape[1], display_frame.shape[0]),
                    )
                writer.write(display_frame)

            if args.no_display:
                print(f"frame={frame_count} fps={fps_now:.2f} inference_ms={inference_ms:.2f} objects={len(detections)}")
            else:
                cv2.imshow("EfficientDet-Lite0", display_frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if args.max_frames and frame_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EfficientDet-Lite0 real-time object detection from a camera."
    )
    parser.add_argument("--model", type=Path, help="TFLite model path. Defaults to Vela+Ethos-U when available, else CPU model.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--image", type=Path, help="Run one image instead of opening a camera.")
    parser.add_argument(
        "--source",
        default=default_camera_source(),
        help="Camera index, /dev/videoX, RTSP URL, or GStreamer pipeline.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=DEFAULT_MAX_DETECTIONS)
    parser.add_argument("--top-k", type=int, default=FAST_RAW_TOP_K, help="Raw candidates decoded before NMS.")
    parser.add_argument("--candidate-multiplier", type=int, default=8, help="Global raw score candidates considered before filtering.")
    parser.add_argument(
        "--raw-score-mode",
        choices=("logits", "prob", "auto"),
        default="logits",
        help="Interpret raw EfficientDet scores as logits, probabilities, or auto-detect.",
    )
    parser.add_argument("--slow-postprocess", action="store_true", help="Use the old full raw postprocess path.")
    parser.add_argument("--profile", action="store_true", help="Print preprocess/inference/postprocess timings.")
    parser.add_argument("--stereo-distance", action="store_true", help="Enable stereo distance estimation from a 1280x480 side-by-side camera.")
    parser.add_argument("--stereo-baseline-m", type=float, default=STEREO_BASELINE_M)
    parser.add_argument("--stereo-focal-px", type=float, default=STEREO_FOCAL_PX)
    parser.add_argument("--disparity-offset", type=float, default=STEREO_DISPARITY_OFFSET)
    parser.add_argument("--stereo-method", choices=("bm", "sgbm"), default="bm")
    parser.add_argument("--depth-scale", type=float, default=STEREO_DEPTH_SCALE, help="Scale used for stereo matching resolution.")
    parser.add_argument("--depth-interval", type=int, default=STEREO_DEPTH_INTERVAL, help="Compute disparity every N frames.")
    parser.add_argument("--stereo-num-disparities", type=int, default=STEREO_NUM_DISPARITIES)
    parser.add_argument("--stereo-block-size", type=int, default=STEREO_BLOCK_SIZE)
    parser.add_argument("--stereo-min-disparity", type=int, default=STEREO_MIN_DISPARITY)
    parser.add_argument("--stereo-uniqueness", type=int, default=STEREO_UNIQUENESS)
    parser.add_argument("--max-distance-m", type=float, default=20.0)
    parser.add_argument("--debug-distance", action="store_true", help="Print distance estimation diagnostics.")
    parser.add_argument("--show-disparity", action="store_true", help="Show the computed disparity map.")
    parser.add_argument("--render-depth", action="store_true", help="Render the metric depth map in the main display/output.")
    parser.add_argument("--depth-view", choices=("side-by-side", "only"), default="side-by-side", help="Depth rendering layout.")
    parser.add_argument("--disable-traffic-light-color", action="store_true", help="Disable HSV traffic-light color classification.")
    parser.add_argument("--traffic-light-min-pixels", type=int, default=TRAFFIC_LIGHT_MIN_PIXELS)
    parser.add_argument("--traffic-light-min-ratio", type=float, default=TRAFFIC_LIGHT_MIN_RATIO)
    parser.add_argument("--debug-traffic-light-color", action="store_true", help="Print traffic-light color diagnostics.")
    parser.add_argument("--state-output", type=Path, help="Write the latest vision state as JSON for a voice agent.")
    parser.add_argument("--state-interval", type=int, default=3, help="Write vision state every N frames.")
    parser.add_argument("--class-offset", type=int, default=0, help="First raw score class index to consider.")
    parser.add_argument("--keep-na-labels", action="store_true", help="Allow N/A labels during raw class selection.")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--delegate-library", help="Optional TensorFlow Lite delegate shared library.")
    parser.add_argument(
        "--ethosu",
        choices=("auto", "on", "off"),
        default="auto",
        help="Use Ethos-U delegate automatically, require it, or disable it.",
    )
    parser.add_argument(
        "--delegate-option",
        action="append",
        default=[],
        help="Delegate option as KEY=VALUE. Can be passed multiple times.",
    )
    parser.add_argument("--no-display", action="store_true", help="Run headless and print detection stats.")
    parser.add_argument("--output", type=Path, help="Optional output video path.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 means run until interrupted.")
    parser.add_argument("--print-model-io", action="store_true", help="Print model input/output tensors and exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stereo_distance:
        args.width = STEREO_FRAME_WIDTH
        args.height = STEREO_HEIGHT
        args.depth_scale = min(1.0, max(0.1, args.depth_scale))
        args.depth_interval = max(1, args.depth_interval)

    labels = load_labels(args.labels)
    colors = make_colors(len(labels))
    model_path, delegate_library = resolve_runtime(args)

    print(f"Model: {model_path}")
    if delegate_library:
        print(f"Delegate: {delegate_library}")
    else:
        print("Delegate: CPU")

    interpreter = build_interpreter(
        model_path,
        delegate_library=str(delegate_library) if delegate_library else None,
        delegate_option=args.delegate_option,
        threads=args.threads,
    )
    input_detail = interpreter.get_input_details()[0]
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    anchors = generate_efficientdet_anchors()

    if args.print_model_io:
        print_model_io(input_details, output_details)
        print(f"Generated EfficientDet anchors: {anchors.shape[0]}")
        return

    if args.image:
        run_image(args, interpreter, input_detail, output_details, labels, colors, anchors)
    else:
        run_camera(args, interpreter, input_detail, input_details, output_details, labels, colors, anchors)


if __name__ == "__main__":
    main()
