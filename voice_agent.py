#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

import argparse
import asyncio
import json
import re
import shlex
import subprocess
import sys
import threading
import time
import tempfile
import wave
from pathlib import Path


DEFAULT_STATE_PATH = Path("/tmp/vision_state.json") if not sys.platform.startswith("win") else Path("vision_state.json")
LABEL_ZH = {
    "person": "行人",
    "bicycle": "自行车",
    "car": "汽车",
    "motorcycle": "摩托车",
    "airplane": "飞机",
    "bus": "公交车",
    "train": "火车",
    "truck": "卡车",
    "boat": "船",
    "traffic light": "交通灯",
    "traffic light red": "红灯",
    "traffic light yellow": "黄灯",
    "traffic light green": "绿灯",
    "traffic light unknown": "交通灯",
    "fire hydrant": "消防栓",
    "stop sign": "停止标志",
    "parking meter": "停车计时器",
    "bench": "长椅",
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
    "backpack": "背包",
    "umbrella": "雨伞",
    "handbag": "手提包",
    "tie": "领带",
    "suitcase": "行李箱",
    "frisbee": "飞盘",
    "skis": "滑雪板",
    "snowboard": "单板滑雪板",
    "sports ball": "球",
    "kite": "风筝",
    "baseball bat": "棒球棒",
    "baseball glove": "棒球手套",
    "skateboard": "滑板",
    "surfboard": "冲浪板",
    "tennis racket": "网球拍",
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
    "chair": "椅子",
    "couch": "沙发",
    "potted plant": "盆栽",
    "bed": "床",
    "dining table": "餐桌",
    "toilet": "马桶",
    "tv": "电视",
    "laptop": "电脑",
    "mouse": "鼠标",
    "remote": "遥控器",
    "keyboard": "键盘",
    "cell phone": "手机",
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
}

CACHE_PHRASES = {
    "start": "中文视觉助手已启动",
    "loading": "正在加载语音识别模型",
    "recording": "正在录音",
    "not_heard": "没有听清",
    "goodbye": "再见",
    "front": "前方",
    "left_front": "左前方",
    "center_front": "正前方",
    "right_front": "右前方",
    "has": "有",
    "detected": "检测到",
    "at": "在",
    "careful": "小心",
    "danger": "危险",
    "safe": "当前安全",
    "no_object": "前方暂未检测到明显物体",
    "not_found_object": "没有检测到这个物体",
    "no_distance": "目前没有可用的距离信息",
    "no_person": "前方暂未检测到行人",
    "person_ahead": "前方有行人",
    "red_light": "红灯",
    "yellow_light": "黄灯",
    "green_light": "绿灯",
    "unknown_light": "交通灯颜色不确定",
    "no_light": "目前没有检测到红绿灯",
    "wait_red": "检测到红灯，请等待",
}

IMPORTANT_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "chair",
    "bench",
    "backpack",
    "suitcase",
    "bottle",
    "cup",
    "cell phone",
    "laptop",
    "keyboard",
    "book",
]

OBJECT_QUERY_ALIASES = {
    "person": ("人", "行人", "人员"),
    "bicycle": ("自行车", "单车"),
    "car": ("汽车", "车", "车辆", "小车"),
    "motorcycle": ("摩托车", "摩托"),
    "bus": ("公交车", "公交", "巴士"),
    "truck": ("卡车", "货车"),
    "traffic light": ("红绿灯", "交通灯", "信号灯"),
    "traffic light red": ("红灯",),
    "traffic light yellow": ("黄灯",),
    "traffic light green": ("绿灯",),
    "bench": ("长椅", "椅子"),
    "backpack": ("背包", "书包"),
    "suitcase": ("行李箱", "箱子"),
    "bottle": ("瓶子", "水瓶"),
    "cup": ("杯子", "水杯", "杯"),
    "cell phone": ("手机", "电话"),
    "laptop": ("电脑", "笔记本", "笔记本电脑"),
    "keyboard": ("键盘",),
    "book": ("书", "书本"),
    "chair": ("椅子", "座椅"),
}


def load_state(path, max_age_s):
    try:
        with open(path, "r", encoding="utf-8") as state_file:
            state = json.load(state_file)
    except FileNotFoundError:
        return None, "还没有收到视觉信息。"
    except json.JSONDecodeError:
        return None, "视觉信息正在更新，请稍等。"

    timestamp = float(state.get("timestamp", 0.0))
    age = time.time() - timestamp
    if age > max_age_s:
        return state, "视觉信息已过期，请确认视觉程序正在运行。"
    return state, None


def object_name(item):
    label_zh = item.get("label_zh")
    if label_zh:
        return label_zh
    label = str(item.get("label") or "").lower().strip()
    return LABEL_ZH.get(label, item.get("label") or "物体")


def format_distance(distance_m):
    if distance_m is None:
        return None
    return f"{float(distance_m):.2f}米"


def format_distance_short(distance_m):
    if distance_m is None:
        return None
    return f"{float(distance_m):.1f}米"


def distance_key(distance_m):
    if distance_m is None:
        return None
    value = max(0.0, round(float(distance_m), 1))
    return f"dist_{int(round(value * 10)):03d}"


def distance_text_from_key(key):
    tenths = int(key.split("_", 1)[1])
    return normalize_tts_text(f"{tenths / 10.0:.1f}米")


def normalize_tts_text(text):
    digit_zh = {
        "0": "零",
        "1": "一",
        "2": "二",
        "3": "三",
        "4": "四",
        "5": "五",
        "6": "六",
        "7": "七",
        "8": "八",
        "9": "九",
    }

    def replace_decimal(match):
        decimal_digits = "".join(digit_zh.get(char, char) for char in match.group(2))
        return f"{match.group(1)}点{decimal_digits}"

    return re.sub(r"(?<!\d)(\d+)\.(\d+)(?=\s*(?:m|米|米|$))", replace_decimal, text)


def traffic_light_text(value):
    if value == "red":
        return "红灯"
    if value == "yellow":
        return "黄灯"
    if value == "green":
        return "绿灯"
    if value == "unknown":
        return "交通灯，但颜色不确定"
    return None


def traffic_light_key(value):
    if value == "red":
        return "red_light"
    if value == "yellow":
        return "yellow_light"
    if value == "green":
        return "green_light"
    if value == "unknown":
        return "unknown_light"
    return None


def label_cache_key(item):
    label = str(item.get("label") or "").lower().strip()
    normalized = label.replace(" ", "_")
    return f"label_{normalized}" if normalized else None


def object_direction_key(item, state=None):
    box = item.get("box") if item else None
    if not box or len(box) != 4:
        return "center_front"

    frame_width = None
    if state:
        frame_size = state.get("frame_size") or {}
        frame_width = frame_size.get("width")
    if not frame_width:
        frame_width = max(float(box[2]), 1.0)

    center_x = (float(box[0]) + float(box[2])) * 0.5
    left_boundary = float(frame_width) * 0.25
    right_boundary = float(frame_width) * 0.75
    if center_x < left_boundary:
        return "left_front"
    if center_x > right_boundary:
        return "right_front"
    return "center_front"


def object_direction_text(item, state=None):
    return CACHE_PHRASES.get(object_direction_key(item, state), "正前方")


def normalize_query_text(text):
    return re.sub(r"[\s,，。？！?！：:；;、\"'“”‘’]", "", text.lower())


def object_query_names(label):
    names = []
    label = str(label or "").lower().strip()
    if label:
        names.append(label)
        names.append(label.replace(" ", ""))
    zh = LABEL_ZH.get(label)
    if zh:
        names.append(zh)
    names.extend(OBJECT_QUERY_ALIASES.get(label, ()))
    return [normalize_query_text(name) for name in names if name]


def find_queried_object(question, objects):
    text = normalize_query_text(question)
    if not text or not objects:
        return None

    best_item = None
    best_score = 0
    for item in objects:
        labels = [
            item.get("label"),
            item.get("label_zh"),
            object_name(item),
        ]
        names = []
        for label in labels:
            if not label:
                continue
            label_text = str(label).lower().strip()
            names.extend(object_query_names(label_text))
            names.append(normalize_query_text(label_text))

        for name in set(names):
            if name and name in text and len(name) > best_score:
                best_item = item
                best_score = len(name)

    return best_item


def is_object_location_query(question):
    text = normalize_query_text(question)
    if not text:
        return False
    return any(keyword in text for keyword in ("在哪", "在哪里", "位置", "方位", "在什么地方", "在不在"))


def segment_text(segment):
    if not segment:
        return ""
    if segment in CACHE_PHRASES:
        return CACHE_PHRASES[segment]
    if segment.startswith("dist_"):
        return distance_text_from_key(segment)
    if segment.startswith("label_"):
        label = segment.removeprefix("label_").replace("_", " ")
        return LABEL_ZH.get(label, label)
    return str(segment)


def segments_text(segments):
    return "".join(segment_text(segment) for segment in segments if segment)


def answer_question(question, state, state_error, last_answer):
    text = question.strip()
    if not text:
        return ""

    if any(keyword in text for keyword in ("退出", "结束", "再见")):
        return "__quit__"

    if any(keyword in text for keyword in ("你是谁", "你能做什么", "介绍一下")):
        return "我是视觉辅助助手，可以告诉你前方物体、距离和红绿灯状态。"

    if any(keyword in text for keyword in ("再说一遍", "重复", "刚才")):
        return last_answer or "我还没有说过可重复的内容。"

    if state_error and any(keyword in text for keyword in ("前方", "前面", "有什么", "距离", "多远", "红灯", "绿灯", "安全吗")):
        return state_error

    state = state or {}
    objects = state.get("objects", [])
    nearest = state.get("nearest_object")
    light = traffic_light_text(state.get("traffic_light"))

    if is_object_location_query(text):
        target = find_queried_object(text, objects)
        if not target:
            return "没有检测到这个物体。"
        distance = format_distance(target.get("distance_m"))
        direction = object_direction_text(target, state)
        name = object_name(target)
        if distance:
            return f"{name}在{direction}{distance}。"
        return f"{name}在{direction}。"

    if any(keyword in text for keyword in ("前方有什么", "前面有什么", "有什么", "看到什么")):
        return describe_objects(objects, nearest, light, state)

    if any(keyword in text for keyword in ("最近", "离我最近", "多远", "距离")):
        if not nearest:
            return "目前没有可用的距离信息。"
        distance = format_distance(nearest.get("distance_m"))
        return f"{object_direction_text(nearest, state)}距离{distance}。"

    if any(keyword in text for keyword in ("有没有人", "有人吗", "行人")):
        people = [item for item in objects if item.get("label") == "person" or item.get("label_zh") == "行人"]
        if not people:
            return "前方暂未检测到行人。"
        person = min((item for item in people if item.get("distance_m") is not None), key=lambda item: item["distance_m"], default=people[0])
        distance = format_distance(person.get("distance_m"))
        if distance:
            return f"{object_direction_text(person, state)}有行人，距离{distance}。"
        return "前方有行人。"

    if any(keyword in text for keyword in ("红绿灯", "红灯", "绿灯", "黄灯", "交通灯")):
        if light:
            return f"当前检测到{light}。"
        return "目前没有检测到红绿灯。"

    if any(keyword in text for keyword in ("安全吗", "危险吗", "能走吗")):
        if nearest and nearest.get("distance_m") is not None:
            distance = float(nearest["distance_m"])
            if distance < 0.5:
                return f"不安全，{object_direction_text(nearest, state)}距离只有{format_distance(distance)}。"
            if distance < 0.8:
                return f"需要注意，{object_direction_text(nearest, state)}距离{format_distance(distance)}。"
        if state.get("traffic_light") == "red":
            return "检测到红灯，请等待。"
        return "当前没有发现明显近距离危险。"

    return "我可以回答前方有什么、最近物体多远、有没有行人、红绿灯状态和是否安全。"


def describe_objects(objects, nearest, light, state=None):
    parts = []
    if nearest:
        distance = format_distance(nearest.get("distance_m"))
        if distance:
            parts.append(f"{object_direction_text(nearest, state)}{distance}有{object_name(nearest)}")
    elif objects:
        names = []
        for item in objects[:3]:
            name = object_name(item)
            if name not in names:
                names.append(name)
        parts.append("前方检测到" + "、".join(names))
    else:
        parts.append("前方暂未检测到明显物体")

    if light:
        parts.append(f"检测到{light}")
    return "，".join(parts) + "。"


def cached_answer_segments(question, state, state_error):
    text = question.strip()
    if state_error and any(keyword in text for keyword in ("前方", "前面", "有什么", "距离", "多远", "红灯", "绿灯", "安全吗")):
        return None

    state = state or {}
    objects = state.get("objects", [])
    nearest = state.get("nearest_object")
    light_key = traffic_light_key(state.get("traffic_light"))

    if any(keyword in text for keyword in ("退出", "结束", "再见")):
        return "__quit__"

    if is_object_location_query(text):
        target = find_queried_object(text, objects)
        if not target:
            return ["not_found_object"]
        segments = []
        label_key = label_cache_key(target)
        if label_key:
            segments.append(label_key)
        segments.extend(["at", object_direction_key(target, state)])
        dist_key = distance_key(target.get("distance_m"))
        if dist_key:
            segments.append(dist_key)
        return segments

    if any(keyword in text for keyword in ("前方有什么", "前面有什么", "有什么", "看到什么")):
        if nearest:
            segments = [object_direction_key(nearest, state)]
            dist_key = distance_key(nearest.get("distance_m"))
            if dist_key:
                segments.append(dist_key)
            label_key = label_cache_key(nearest)
            if label_key:
                segments.extend(["has", label_key])
            if light_key:
                segments.extend(["detected", light_key])
            return segments
        if objects:
            segments = ["front", "detected"]
            used = set()
            for item in objects[:3]:
                key = label_cache_key(item)
                if key and key not in used:
                    segments.append(key)
                    used.add(key)
            return segments
        return ["no_object"]

    if any(keyword in text for keyword in ("最近", "离我最近", "多远", "距离")):
        if not nearest:
            return ["no_distance"]
        dist_key = distance_key(nearest.get("distance_m"))
        return [object_direction_key(nearest, state), dist_key] if dist_key else ["no_distance"]

    if any(keyword in text for keyword in ("有没有人", "有人吗", "行人")):
        people = [item for item in objects if item.get("label") == "person" or item.get("label_zh") == "行人"]
        if not people:
            return ["no_person"]
        person = min((item for item in people if item.get("distance_m") is not None), key=lambda item: item["distance_m"], default=people[0])
        dist_key = distance_key(person.get("distance_m"))
        return ["person_ahead", dist_key] if dist_key else ["person_ahead"]

    if any(keyword in text for keyword in ("红绿灯", "红灯", "绿灯", "黄灯", "交通灯")):
        return [light_key] if light_key else ["no_light"]

    if any(keyword in text for keyword in ("安全吗", "危险吗", "能走吗")):
        if nearest and nearest.get("distance_m") is not None:
            distance = float(nearest["distance_m"])
            dist_key = distance_key(distance)
            if distance <= 1.5:
                return ["danger", object_direction_key(nearest, state), dist_key]
            if distance <= 2.0:
                return ["careful", object_direction_key(nearest, state), dist_key]
        if state.get("traffic_light") == "red":
            return ["wait_red"]
        return ["safe"]

    return None


class Speaker:
    def __init__(self, args):
        self.mode = args.tts
        self.args = args
        self.engine = None
        self.lock = threading.Lock()
        if self.mode == "auto":
            self.mode = "pyttsx3"
        if self.mode == "pyttsx3":
            try:
                import pyttsx3

                self.engine = pyttsx3.init()
                self.configure_pyttsx3()
            except Exception as exc:
                print(f"TTS 初始化失败，改为文本输出：{exc}", file=sys.stderr)
                self.mode = "print"

    def configure_pyttsx3(self):
        if self.engine is None:
            return

        if self.args.tts_rate:
            self.engine.setProperty("rate", self.args.tts_rate)
        if self.args.tts_volume is not None:
            self.engine.setProperty("volume", self.args.tts_volume)

        voices = self.engine.getProperty("voices") or []
        selected_voice_id = self.args.tts_voice
        if not selected_voice_id and self.args.tts_prefer_chinese:
            selected_voice_id = self.find_chinese_voice(voices)

        if selected_voice_id:
            try:
                self.engine.setProperty("voice", selected_voice_id)
            except Exception as exc:
                print(f"设置 TTS voice 失败：{exc}", file=sys.stderr)

        if self.args.list_voices:
            for index, voice in enumerate(voices):
                languages = getattr(voice, "languages", "")
                print(f"{index}: id={voice.id} name={voice.name} languages={languages}")
            raise SystemExit(0)

    @staticmethod
    def find_chinese_voice(voices):
        keywords = ("zh", "chinese", "mandarin", "huihui", "xiaoxiao", "xiaoyi", "xiaobei", "xiaoni", "yunxi")
        for voice in voices:
            values = [
                str(getattr(voice, "id", "")),
                str(getattr(voice, "name", "")),
                str(getattr(voice, "languages", "")),
            ]
            haystack = " ".join(values).lower()
            if any(keyword in haystack for keyword in keywords):
                return voice.id
        return None

    def say(self, text):
        if not text:
            return
        if isinstance(text, (list, tuple)):
            if self.say_cached_segments(text):
                return
            text = segments_text(text)
        spoken_text = normalize_tts_text(text)
        with self.lock:
            print(f"助手：{text}")
            if self.mode == "pyttsx3" and self.engine is not None:
                self.engine.say(spoken_text)
                self.engine.runAndWait()
            elif self.mode == "edge":
                self.say_with_edge_tts(spoken_text)
            elif self.mode == "piper":
                self.say_with_piper(spoken_text)
            elif self.mode == "command":
                self.say_with_command(spoken_text)

    def say_cached_segments(self, segments):
        if self.mode != "cached":
            return False
        paths = []
        for segment in segments:
            if not segment:
                continue
            path = self.args.audio_cache_dir / f"{segment}.wav"
            if not path.exists():
                print(f"音频缓存缺失：{path}", file=sys.stderr)
                return False
            paths.append(path)
        if not paths:
            return False
        print(f"助手：{segments_text(segments)}")
        if len(paths) == 1:
            self.play_audio(paths[0])
        else:
            merged = self.merge_wav_files(paths)
            if merged:
                try:
                    self.play_audio(merged)
                finally:
                    try:
                        merged.unlink()
                    except OSError:
                        pass
            else:
                for path in paths:
                    self.play_audio(path)
        return True

    def merge_wav_files(self, paths):
        try:
            with wave.open(str(paths[0]), "rb") as first:
                params = first.getparams()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
                output_path = Path(output_file.name)

            with wave.open(str(output_path), "wb") as output:
                output.setparams(params)
                for path in paths:
                    with wave.open(str(path), "rb") as source:
                        if source.getparams()[:3] != params[:3]:
                            return None
                        while True:
                            frames = source.readframes(4096)
                            if not frames:
                                break
                            output.writeframes(frames)
            return output_path
        except (OSError, wave.Error):
            return None

    def close(self):
        pass

    def say_with_edge_tts(self, text):
        try:
            asyncio.run(self.edge_tts_once(text))
        except Exception as exc:
            print(f"edge-tts 播报失败：{exc}", file=sys.stderr)

    async def edge_tts_once(self, text):
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("缺少 edge-tts，请先执行 pip install edge-tts。") from exc

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio_file:
            audio_path = Path(audio_file.name)

        try:
            communicate = edge_tts.Communicate(text, self.args.edge_voice, rate=self.args.edge_rate, volume=self.args.edge_volume)
            await communicate.save(str(audio_path))
            self.play_audio(audio_path)
        finally:
            try:
                audio_path.unlink()
            except OSError:
                pass

    def play_audio(self, audio_path):
        if self.args.audio_player:
            command = [part.replace("{file}", str(audio_path)) for part in shlex.split(self.args.audio_player)]
            try:
                result = subprocess.run(command, check=False, timeout=self.args.audio_timeout)
                if result.returncode != 0:
                    print(f"音频播放命令退出码：{result.returncode}", file=sys.stderr)
            except subprocess.TimeoutExpired:
                print(f"音频播放超时，已跳过：{audio_path}", file=sys.stderr)
            return

        if sys.platform.startswith("win"):
            if str(audio_path).lower().endswith(".wav"):
                import winsound

                winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
                return

            ps_command = (
                "Add-Type -AssemblyName PresentationCore; "
                f"$p=New-Object System.Windows.Media.MediaPlayer; "
                f"$p.Open([Uri]'{audio_path.as_uri()}'); "
                "$p.Play(); "
                "while(-not $p.NaturalDuration.HasTimeSpan){ Start-Sleep -Milliseconds 50 }; "
                "$d=$p.NaturalDuration.TimeSpan.TotalMilliseconds; "
                "Start-Sleep -Milliseconds ([int]($d + 200)); "
                "$p.Close()"
            )
            try:
                subprocess.run(["powershell", "-NoProfile", "-Command", ps_command], check=False, timeout=self.args.audio_timeout)
            except subprocess.TimeoutExpired:
                print(f"音频播放超时，已跳过：{audio_path}", file=sys.stderr)
            return

        for player in ("ffplay", "mpg123", "mpv"):
            try:
                if player == "ffplay":
                    result = subprocess.run(
                        [player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)],
                        check=False,
                        timeout=self.args.audio_timeout,
                    )
                else:
                    result = subprocess.run([player, str(audio_path)], check=False, timeout=self.args.audio_timeout)
                if result.returncode == 0:
                    return
            except subprocess.TimeoutExpired:
                print(f"音频播放超时，已跳过：{audio_path}", file=sys.stderr)
                return
            except FileNotFoundError:
                continue
        raise RuntimeError("没有可用音频播放器。请安装 ffplay/mpg123/mpv，或用 --audio-player 指定播放命令。")

    def say_with_piper(self, text):
        command_parts = self.build_piper_command()
        if command_parts is None:
            return
        base_command, piper_exe = command_parts
        self.say_with_piper_once(text, base_command, piper_exe)

    def build_piper_command(self):
        if not self.args.piper_model:
            print("未设置 --piper-model，无法使用 piper 离线播报。", file=sys.stderr)
            return None
        piper_model = self.args.piper_model.resolve()
        if not piper_model.exists():
            print(f"piper 模型不存在：{piper_model}", file=sys.stderr)
            return None

        piper_config = self.args.piper_config
        if piper_config is None:
            default_config = Path(str(piper_model) + ".json")
            if default_config.exists():
                piper_config = default_config
            else:
                print(
                    f"piper 模型配置文件不存在：{default_config}。"
                    "请下载与 .onnx 同名的 .onnx.json 文件，或用 --piper-config 指定配置文件。",
                    file=sys.stderr,
                )
                return None
        else:
            piper_config = piper_config.resolve()

        piper_exe = Path(self.args.piper_exe).resolve()

        base_command = [
            str(piper_exe),
            "--model",
            str(piper_model),
        ]
        if piper_config:
            base_command.extend(["--config", str(piper_config)])
        if self.args.piper_speaker is not None:
            base_command.extend(["--speaker", str(self.args.piper_speaker)])
        if self.args.piper_length_scale is not None:
            base_command.extend(["--length_scale", str(self.args.piper_length_scale)])

        return base_command, piper_exe

    def say_with_piper_once(self, text, base_command, piper_exe):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_path = Path(audio_file.name).resolve()

        command = base_command + ["--output_file", str(audio_path)]
        try:
            piper_cwd = piper_exe.parent if piper_exe.exists() else None
            result = subprocess.run(
                command,
                input=text.encode("utf-8"),
                capture_output=True,
                check=False,
                cwd=piper_cwd,
            )
            if result.returncode != 0:
                stderr = (
                    result.stderr.decode("utf-8", errors="replace").strip()
                    or result.stdout.decode("utf-8", errors="replace").strip()
                    or f"returncode={result.returncode}"
                )
                print(f"piper 播报失败：{stderr}", file=sys.stderr)
                return
            self.play_audio(audio_path)
        except Exception as exc:
            print(f"piper 播报失败：{exc}", file=sys.stderr)
        finally:
            try:
                audio_path.unlink()
            except OSError:
                pass

    def say_with_command(self, text):
        if not self.args.tts_command:
            print("未设置 --tts-command，无法使用 command 播报。", file=sys.stderr)
            return
        command = [part.replace("{text}", text) for part in shlex.split(self.args.tts_command)]
        if not self.args.tts_stdin and not any(part == text for part in command):
            command.append(text)
        try:
            if self.args.tts_stdin:
                subprocess.run(command, input=text, text=True, check=False)
            else:
                subprocess.run(command, check=False)
        except Exception as exc:
            print(f"TTS command 执行失败：{exc}", file=sys.stderr)


class TextListener:
    def listen(self):
        return input("你：").strip()

    def close(self):
        pass


class SpeechRecognitionGoogleListener:
    def __init__(self, args):
        try:
            import speech_recognition as sr
        except ImportError as exc:
            raise RuntimeError("缺少 speech_recognition，请先安装 SpeechRecognition 和 PyAudio。") from exc

        self.sr = sr
        self.args = args
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(device_index=args.mic_index)

        print("正在校准环境噪声，请保持安静...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=args.ambient_duration)

    def listen(self):
        print("你：正在听...")
        with self.microphone as source:
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=self.args.listen_timeout,
                    phrase_time_limit=self.args.phrase_time_limit,
                )
            except self.sr.WaitTimeoutError:
                return ""

        try:
            text = self.recognizer.recognize_google(audio, language=self.args.language)
            print(f"你：{text}")
            return text.strip()
        except self.sr.UnknownValueError:
            print("你：（没有听清）")
            return ""
        except self.sr.RequestError as exc:
            print(f"语音识别请求失败：{exc}", file=sys.stderr)
            return ""

    def close(self):
        pass


class VoskListener:
    def __init__(self, args):
        if not args.vosk_model:
            raise RuntimeError("使用 vosk 输入模式时必须提供 --vosk-model。")

        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model
        except ImportError as exc:
            raise RuntimeError("缺少 vosk 或 sounddevice，请先安装 vosk sounddevice。") from exc

        self.json = json
        self.sd = sd
        self.args = args
        self.recognizer = KaldiRecognizer(Model(str(args.vosk_model)), args.sample_rate)
        if args.mic_index is None:
            print("未指定 --mic-index，使用系统默认输入设备。若无法持续监听，请先运行 --list-mics 选择麦克风。")
        self.stream = None
        if args.vosk_stream:
            self.stream = sd.RawInputStream(
                samplerate=args.sample_rate,
                blocksize=8000,
                device=args.mic_index,
                dtype="int16",
                channels=1,
            )
            self.stream.start()

    def listen(self):
        if not self.args.vosk_stream:
            return self.listen_record_once()

        print("你：正在听...")
        start = time.time()
        collected = []
        while True:
            try:
                data, overflowed = self.stream.read(4000)
            except Exception as exc:
                print(f"麦克风读取失败：{exc}", file=sys.stderr)
                time.sleep(0.5)
                return ""
            if overflowed:
                print("麦克风输入溢出，可能有音频丢失。", file=sys.stderr)

            if self.recognizer.AcceptWaveform(bytes(data)):
                result = self.json.loads(self.recognizer.Result())
                text = str(result.get("text", "")).replace(" ", "").strip()
                if text:
                    print(f"你：{text}")
                    return text

            partial = self.json.loads(self.recognizer.PartialResult()).get("partial", "")
            if partial:
                collected.append(partial)

            if time.time() - start >= self.args.phrase_time_limit:
                result = self.json.loads(self.recognizer.FinalResult())
                text = str(result.get("text", "")).replace(" ", "").strip()
                if text:
                    print(f"你：{text}")
                    return text
                if collected:
                    text = str(collected[-1]).replace(" ", "").strip()
                    print(f"你：{text}")
                    return text
                print("你：（没有听清，继续监听）")
                return ""

    def listen_record_once(self):
        print("你：正在听...")
        try:
            frames = int(self.args.sample_rate * self.args.phrase_time_limit)
            audio = self.sd.rec(
                frames,
                samplerate=self.args.sample_rate,
                channels=1,
                dtype="int16",
                device=self.args.mic_index,
            )
            self.sd.wait()
        except Exception as exc:
            print(f"麦克风录音失败：{exc}", file=sys.stderr)
            time.sleep(0.5)
            return ""

        self.recognizer.Reset()
        self.recognizer.AcceptWaveform(audio.tobytes())
        result = self.json.loads(self.recognizer.FinalResult())
        text = str(result.get("text", "")).replace(" ", "").strip()
        if text:
            print(f"你：{text}")
            return text
        print("你：（没有听清，继续监听）")
        return ""

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


class VoskWavListener:
    def __init__(self, args):
        if not args.vosk_model:
            raise RuntimeError("使用 vosk-wav 输入模式时必须提供 --vosk-model。")

        try:
            from vosk import KaldiRecognizer, Model
        except ImportError as exc:
            raise RuntimeError("缺少 vosk，请先安装 vosk。") from exc

        self.args = args
        self.recognizer = KaldiRecognizer(Model(str(args.vosk_model)), args.sample_rate)
        self.current_arecord_device = args.arecord_device
        if args.arecord_auto_detect:
            self.current_arecord_device = self.resolve_arecord_device("startup")
        if self.current_arecord_device == "auto":
            self.current_arecord_device = "default"

    def listen(self):
        print("你：正在录音...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_path = Path(audio_file.name)

        device = self.current_arecord_device
        result = None

        try:
            result = self.run_arecord(audio_path, device, self.args.record_seconds)
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace").strip()
                print(f"arecord 录音失败（{device}）：{stderr or result.returncode}", file=sys.stderr)
                retry_device = self.refresh_arecord_device(device, "record-error")
                if retry_device and retry_device != device:
                    result = self.run_arecord(audio_path, retry_device, self.args.record_seconds)
                    if result.returncode != 0:
                        retry_stderr = result.stderr.decode("utf-8", errors="replace").strip()
                        print(f"arecord 重试失败（{retry_device}）：{retry_stderr or result.returncode}", file=sys.stderr)
                        return ""
                else:
                    return ""
            return self.recognize_wav(audio_path)
        except subprocess.TimeoutExpired:
            print(f"arecord 录音超时（{device}），正在重新检查录音设备。", file=sys.stderr)
            retry_device = self.refresh_arecord_device(device, "record-timeout")
            if retry_device and retry_device != device:
                try:
                    result = self.run_arecord(audio_path, retry_device, self.args.record_seconds)
                    if result.returncode == 0:
                        return self.recognize_wav(audio_path)
                    stderr = result.stderr.decode("utf-8", errors="replace").strip()
                    print(f"arecord 重试失败（{retry_device}）：{stderr or result.returncode}", file=sys.stderr)
                except subprocess.TimeoutExpired:
                    print(f"arecord 重试超时（{retry_device}）。", file=sys.stderr)
            return ""
        finally:
            try:
                audio_path.unlink()
            except OSError:
                pass

    def run_arecord(self, audio_path, device, seconds):
        command = [
            self.args.arecord_exe,
            "-D",
            device,
            "-f",
            "S16_LE",
            "-r",
            str(self.args.sample_rate),
            "-c",
            "1",
            "-d",
            str(max(1, int(round(float(seconds))))),
            str(audio_path),
        ]
        timeout = max(1.0, float(seconds) + float(self.args.record_timeout_extra))
        return subprocess.run(command, capture_output=True, check=False, timeout=timeout)

    def refresh_arecord_device(self, failed_device, reason):
        if not self.args.arecord_auto_detect:
            return None
        detected = self.resolve_arecord_device(reason, skip_device=failed_device)
        if detected and detected != self.current_arecord_device:
            print(f"arecord device changed: {self.current_arecord_device} -> {detected}", file=sys.stderr)
            self.current_arecord_device = detected
        return detected

    def resolve_arecord_device(self, reason, skip_device=None):
        candidates = self.arecord_device_candidates()
        if skip_device:
            candidates = [device for device in candidates if device != skip_device]

        for device in candidates:
            if self.probe_arecord_device(device):
                print(f"arecord device selected ({reason}): {device}", file=sys.stderr)
                return device

        fallback = self.args.arecord_device
        if fallback == "auto":
            fallback = self.current_arecord_device if self.current_arecord_device != "auto" else "default"
        print(f"arecord auto-detect found no working device, using fallback: {fallback}", file=sys.stderr)
        return fallback

    def arecord_device_candidates(self):
        devices = []
        configured = str(self.args.arecord_device or "").strip()

        if configured and configured not in ("auto", "default"):
            devices.append(configured)

        try:
            result = subprocess.run(
                [self.args.arecord_exe, "-l"],
                capture_output=True,
                check=False,
                timeout=3.0,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"arecord -l failed during device detection: {exc}", file=sys.stderr)
            return self.dedupe_devices(devices or [configured or "default"])

        listing = (
            result.stdout.decode("utf-8", errors="replace")
            + "\n"
            + result.stderr.decode("utf-8", errors="replace")
        )
        for line in listing.splitlines():
            match = re.search(r"card\s+(\d+):.*device\s+(\d+):", line)
            if match:
                devices.append(f"plughw:{match.group(1)},{match.group(2)}")

        if configured == "default":
            devices.append("default")
        return self.dedupe_devices(devices or ["default"])

    def dedupe_devices(self, devices):
        unique = []
        for device in devices:
            if device and device not in unique:
                unique.append(device)
        return unique

    def probe_arecord_device(self, device):
        probe_seconds = max(0.0, float(self.args.arecord_probe_seconds))
        if probe_seconds <= 0:
            return True

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
            audio_path = Path(audio_file.name)
        try:
            result = self.run_arecord(audio_path, device, probe_seconds)
            if result.returncode == 0:
                return True
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            if stderr:
                print(f"arecord probe failed ({device}): {stderr}", file=sys.stderr)
            return False
        except subprocess.TimeoutExpired:
            print(f"arecord probe timed out ({device})", file=sys.stderr)
            return False
        finally:
            try:
                audio_path.unlink()
            except OSError:
                pass

    def recognize_wav(self, audio_path):
        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                    print("录音格式不正确，需要 16-bit 单声道 PCM。", file=sys.stderr)
                    return ""
                self.recognizer.Reset()
                while True:
                    data = wav_file.readframes(4000)
                    if not data:
                        break
                    self.recognizer.AcceptWaveform(data)
        except wave.Error as exc:
            print(f"读取录音文件失败：{exc}", file=sys.stderr)
            return ""

        result = json.loads(self.recognizer.FinalResult())
        text = str(result.get("text", "")).replace(" ", "").strip()
        if text:
            print(f"你：{text}")
            return text
        print("你：（没有听清，继续监听）")
        return ""

    def close(self):
        pass


class KeyboardTrigger:
    def wait(self):
        trigger = input("按回车开始识别，输入 q 退出：").strip().lower()
        if trigger in ("q", "quit", "exit"):
            return "quit"
        return "record"

    def close(self):
        pass


class GpioTrigger:
    def __init__(self, args):
        self.args = args
        self.backend = None
        self.chip = None
        self.line = None
        self.request = None
        self.gpiod = None
        self.gpiomon_process = None
        self.last_debug_print = 0.0
        self.setup_gpiod()

    def setup_gpiod(self):
        try:
            import gpiod
        except ImportError as exc:
            raise RuntimeError("缺少 python3-libgpiod，请先安装：apt install python3-libgpiod gpiod") from exc

        self.gpiod = gpiod
        if self.try_setup_v1(gpiod):
            return
        if self.try_setup_v2(gpiod):
            return
        if self.try_setup_gpiomon():
            return
        raise RuntimeError("无法初始化 GPIO。请确认 libgpiod Python 版本、gpiomon 命令和 GPIO 参数。")

    def try_setup_v1(self, gpiod):
        try:
            chip = gpiod.Chip(self.args.gpio_chip)
            if not hasattr(chip, "get_line"):
                return False
            line = chip.get_line(self.args.gpio_line)
            edge_type = {
                "falling": gpiod.LINE_REQ_EV_FALLING_EDGE,
                "rising": gpiod.LINE_REQ_EV_RISING_EDGE,
                "both": gpiod.LINE_REQ_EV_BOTH_EDGES,
            }[self.args.gpio_edge]
            flags = 0
            if self.args.gpio_active_low and hasattr(gpiod, "LINE_REQ_FLAG_ACTIVE_LOW"):
                flags |= gpiod.LINE_REQ_FLAG_ACTIVE_LOW
            line.request(consumer="voice_agent", type=edge_type, flags=flags)
            self.backend = "v1"
            self.chip = chip
            self.line = line
            print(f"GPIO 按钮已启用：{self.args.gpio_chip} line {self.args.gpio_line}", flush=True)
            return True
        except Exception:
            return False

    def try_setup_v2(self, gpiod):
        try:
            from datetime import timedelta
            from gpiod.line import Bias, Direction, Edge, LineSettings

            edge = {
                "falling": Edge.FALLING,
                "rising": Edge.RISING,
                "both": Edge.BOTH,
            }[self.args.gpio_edge]
            settings_kwargs = {
                "direction": Direction.INPUT,
                "edge_detection": edge,
            }
            if self.args.gpio_bias != "default":
                settings_kwargs["bias"] = {
                    "pull-up": Bias.PULL_UP,
                    "pull-down": Bias.PULL_DOWN,
                    "disabled": Bias.DISABLED,
                }[self.args.gpio_bias]
            if self.args.gpio_active_low:
                settings_kwargs["active_low"] = True

            self.request = gpiod.request_lines(
                self.args.gpio_chip,
                consumer="voice_agent",
                config={self.args.gpio_line: LineSettings(**settings_kwargs)},
            )
            self.wait_timeout = timedelta(seconds=1)
            self.backend = "v2"
            print(f"GPIO 按钮已启用：{self.args.gpio_chip} line {self.args.gpio_line}", flush=True)
            return True
        except Exception:
            return False

    def try_setup_gpiomon(self):
        try:
            edge_short_flag = {
                "falling": "-f",
                "rising": "-r",
                "both": "-b",
            }[self.args.gpio_edge]
            result = subprocess.run(
                [self.args.gpiomon_exe, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode not in (0, 1):
                return False
            help_text = f"{result.stdout}\n{result.stderr}"
            use_long_options = self.args.gpiomon_style == "long-option" or (
                self.args.gpiomon_style == "auto"
                and "--chip" in help_text
                and "--edges" in help_text
                and "--debounce-period" in help_text
            )
            use_chip_option = self.args.gpiomon_style == "chip-option" or (
                self.args.gpiomon_style == "auto" and "-c" in help_text and "chip" in help_text.lower()
            )
            if use_long_options:
                # libgpiod v2 long-option style, verified command:
                #   gpiomon --chip=0 --edges=falling --bias=pull-down --debounce-period=8ms 2
                chip_arg = self.gpiomon_chip_arg()
                self.gpiomon_command = [
                    self.args.gpiomon_exe,
                    f"--chip={chip_arg}",
                    f"--edges={self.args.gpio_edge}",
                    f"--debounce-period={self.args.gpio_debounce_ms}ms",
                ]
                if self.args.gpio_bias != "default":
                    self.gpiomon_command.append(f"--bias={self.args.gpio_bias}")
                self.gpiomon_command.append(str(self.args.gpio_line))
                self.gpiomon_label = "gpiomon long-option"
            elif use_chip_option:
                # libgpiod v2 style, verified on the FRDM i.MX93 Debian image:
                #   gpiomon -c /dev/gpiochip0 2
                self.gpiomon_command = [
                    self.args.gpiomon_exe,
                    "-c",
                    self.args.gpio_chip,
                    str(self.args.gpio_line),
                ]
                self.gpiomon_label = "gpiomon -c"
            else:
                # libgpiod v1 style.
                self.gpiomon_command = [
                    self.args.gpiomon_exe,
                    "-n",
                    "1",
                    edge_short_flag,
                    self.args.gpio_chip,
                    str(self.args.gpio_line),
                ]
                self.gpiomon_label = "gpiomon"
            self.backend = "gpiomon"
            print(
                f"GPIO 按钮已启用：{self.args.gpio_chip} line {self.args.gpio_line} "
                f"({self.gpiomon_label})",
                flush=True,
            )
            return True
        except Exception:
            return False

    def gpiomon_chip_arg(self):
        chip = str(self.args.gpio_chip).strip()
        match = re.search(r"gpiochip(\d+)$", chip)
        if match:
            return match.group(1)
        return chip

    def debug_gpio_state(self, force=False):
        if not self.args.gpio_debug:
            return
        now = time.time()
        if not force and now - self.last_debug_print < self.args.gpio_debug_interval:
            return
        self.last_debug_print = now
        value = self.read_gpio_value()
        print(
            f"GPIO DEBUG chip={self.args.gpio_chip} line={self.args.gpio_line} "
            f"value={value} backend={self.backend}",
            flush=True,
        )

    def read_gpio_value(self):
        if self.backend == "v1" and self.line is not None:
            try:
                return self.line.get_value()
            except Exception:
                return None
        try:
            chip_arg = self.gpiomon_chip_arg()
            commands = [
                ["gpioget", f"--chip={chip_arg}", str(self.args.gpio_line)],
                ["gpioget", f"--chip={chip_arg}", "--numeric", str(self.args.gpio_line)],
                ["gpioget", "--chip", str(chip_arg), str(self.args.gpio_line)],
                ["gpioget", "-c", self.args.gpio_chip, str(self.args.gpio_line)],
                ["gpioget", self.args.gpio_chip, str(self.args.gpio_line)],
            ]
            for command in commands:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=1,
                    check=False,
                )
                value = result.stdout.strip()
                if result.returncode == 0 and value in ("0", "1"):
                    return int(value)
        except Exception:
            return None
        return None

    def wait(self):
        print("等待按钮按下开始识别，按 Ctrl+C 退出。", flush=True)
        self.debug_gpio_state(force=True)
        while True:
            self.debug_gpio_state()
            if self.backend == "v1":
                if self.line.event_wait(sec=1):
                    self.line.event_read()
                    time.sleep(self.args.gpio_debounce_ms / 1000.0)
                    return "record"
            elif self.backend == "v2":
                if self.request.wait_edge_events(timeout=self.wait_timeout):
                    self.request.read_edge_events()
                    time.sleep(self.args.gpio_debounce_ms / 1000.0)
                    return "record"
            elif self.backend == "gpiomon":
                if self.wait_gpiomon_event():
                    time.sleep(self.args.gpio_debounce_ms / 1000.0)
                    return "record"

    def wait_gpiomon_event(self):
        if self.gpiomon_process is None or self.gpiomon_process.poll() is not None:
            if self.gpiomon_process is not None and self.args.gpio_debug:
                print(f"GPIO DEBUG gpiomon exited: {self.gpiomon_process.returncode}", flush=True)
            if self.args.gpio_debug:
                print("GPIO DEBUG gpiomon command: " + " ".join(self.gpiomon_command), flush=True)
            self.gpiomon_process = subprocess.Popen(
                self.gpiomon_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )

        self.debug_gpio_state()
        try:
            import select

            readable, _, _ = select.select([self.gpiomon_process.stdout], [], [], self.args.gpio_event_timeout)
            if not readable:
                return False
            line = self.gpiomon_process.stdout.readline().strip()
            if self.args.gpio_debug and line:
                print(f"GPIO DEBUG event: {line.strip()}", flush=True)
            return bool(line)
        except Exception as exc:
            if self.args.gpio_debug:
                print(f"GPIO DEBUG gpiomon read failed: {exc}", flush=True)
            self.stop_gpiomon()
            return False

    def stop_gpiomon(self):
        process = self.gpiomon_process
        self.gpiomon_process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()

    def close(self):
        try:
            if self.line is not None:
                self.line.release()
        except Exception:
            pass
        try:
            if self.request is not None:
                self.request.release()
        except Exception:
            pass
        try:
            if self.chip is not None:
                self.chip.close()
        except Exception:
            pass
        self.stop_gpiomon()


def create_trigger(args):
    if args.trigger_mode == "keyboard":
        return KeyboardTrigger()
    if args.trigger_mode == "gpio":
        return GpioTrigger(args)
    raise RuntimeError(f"不支持的触发模式：{args.trigger_mode}")


def create_listener(args):
    if args.input_mode == "text":
        return TextListener()
    if args.input_mode == "sr-google":
        return SpeechRecognitionGoogleListener(args)
    if args.input_mode == "vosk":
        return VoskListener(args)
    if args.input_mode == "vosk-wav":
        return VoskWavListener(args)
    raise RuntimeError(f"不支持的输入模式：{args.input_mode}")


def list_microphones():
    printed = False
    try:
        import speech_recognition as sr

        print("speech_recognition 麦克风：")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"{index}: {name}")
            printed = True
    except ImportError:
        print("缺少 speech_recognition，无法列出麦克风。", file=sys.stderr)

    try:
        import sounddevice as sd

        print("sounddevice 设备：")
        for index, device in enumerate(sd.query_devices()):
            marker = "input" if device.get("max_input_channels", 0) > 0 else "     "
            print(
                f"{index}: {marker} {device.get('name')} "
                f"in={device.get('max_input_channels')} out={device.get('max_output_channels')} "
                f"rate={device.get('default_samplerate')}"
            )
            printed = True
    except ImportError:
        print("缺少 sounddevice，无法列出 Vosk 麦克风设备。", file=sys.stderr)

    if not printed:
        print("没有列出任何麦克风设备。", file=sys.stderr)


def item_box(item):
    box = item.get("box")
    if not box or len(box) != 4:
        return None
    return tuple(float(value) for value in box)


def box_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def center_distance_ratio(box_a, box_b):
    if box_a is None or box_b is None:
        return 1.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    acx = (ax1 + ax2) * 0.5
    acy = (ay1 + ay2) * 0.5
    bcx = (bx1 + bx2) * 0.5
    bcy = (by1 + by2) * 0.5
    scale = max(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1, 1.0)
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / scale)


def same_warning_target(item, alerted, args):
    label = item.get("label") or item.get("label_zh") or object_name(item)
    if label != alerted.get("label"):
        return False
    current_box = item_box(item)
    alerted_box = alerted.get("box")
    if current_box is None or alerted_box is None:
        return True
    if box_iou(current_box, alerted_box) >= args.warning_same_iou:
        return True
    return center_distance_ratio(current_box, alerted_box) <= args.warning_same_center_ratio


def warning_segments(distance, nearest=None, state=None):
    dist_key = distance_key(distance)
    if distance <= 1.5:
        return ["danger", object_direction_key(nearest, state), dist_key]
    if distance <= 2.0:
        return ["careful", object_direction_key(nearest, state), dist_key]
    return None


def warning_loop(args, speaker, stop_event, interaction_event):
    alerted_targets = []

    while not stop_event.is_set():
        if interaction_event.is_set():
            time.sleep(args.warning_poll_interval)
            continue

        state, error = load_state(args.state_input, args.max_state_age)
        now = time.time()
        alerted_targets = [
            target for target in alerted_targets
            if now - target.get("last_seen", 0.0) <= args.warning_forget_seconds
        ]
        if error or not state:
            time.sleep(args.warning_poll_interval)
            continue

        nearest = state.get("nearest_object")
        if not nearest or nearest.get("distance_m") is None:
            time.sleep(args.warning_poll_interval)
            continue

        distance = float(nearest["distance_m"])
        warning = ""
        segments = None

        if distance <= args.danger_distance:
            warning = f"危险，{object_direction_text(nearest, state)}{format_distance_short(distance)}。"
            segments = warning_segments(distance, nearest, state)
        elif distance <= args.warning_distance:
            warning = f"小心，{object_direction_text(nearest, state)}{format_distance_short(distance)}。"
            segments = warning_segments(distance, nearest, state)

        matching_target = None
        for target in alerted_targets:
            if same_warning_target(nearest, target, args):
                matching_target = target
                target["last_seen"] = now
                target["box"] = item_box(nearest)
                break

        should_say = False
        if warning and matching_target is None:
            should_say = True
        elif warning and matching_target is not None:
            last_distance = matching_target.get("last_distance")
            if last_distance is None or abs(distance - last_distance) >= args.warning_distance_delta:
                should_say = True

        if should_say:
            speaker.say(segments if args.tts == "cached" and segments else warning)
            if matching_target is not None:
                matching_target["last_distance"] = distance
                matching_target["last_seen"] = now
                matching_target["box"] = item_box(nearest)
            else:
                alerted_targets.append(
                    {
                        "label": nearest.get("label") or nearest.get("label_zh") or object_name(nearest),
                        "box": item_box(nearest),
                        "last_seen": now,
                        "last_distance": distance,
                    }
                )

        time.sleep(args.warning_poll_interval)


def interactive_loop(args, speaker, listener, trigger):
    last_answer = ""
    stop_event = threading.Event()
    interaction_event = threading.Event()
    monitor = threading.Thread(target=warning_loop, args=(args, speaker, stop_event, interaction_event), daemon=True)
    if not args.no_warning:
        monitor.start()

    speaker.say(["start"] if args.tts == "cached" else "中文视觉助手已启动。你可以问：前方有什么？")
    try:
        while True:
            try:
                if args.input_mode != "text" and args.push_to_talk:
                    trigger_event = trigger.wait()
                    if trigger_event == "quit":
                        interaction_event.set()
                        speaker.say(["goodbye"] if args.tts == "cached" else "再见。")
                        break
                interaction_event.set()
                if args.tts == "cached" and args.input_mode != "text":
                    if args.recording_prompt:
                        speaker.say(["recording"])
                    else:
                        print("你：正在录音...")
                question = listener.listen()
            except KeyboardInterrupt:
                interaction_event.set()
                speaker.say(["goodbye"] if args.tts == "cached" else "再见。")
                break
            except Exception as exc:
                print(f"监听失败：{exc}", file=sys.stderr)
                interaction_event.clear()
                time.sleep(0.5)
                continue
            if not question:
                interaction_event.clear()
                continue
            state, error = load_state(args.state_input, args.max_state_age)
            answer = None
            if args.tts == "cached":
                answer = cached_answer_segments(question, state, error)
            if answer is None:
                answer = answer_question(question, state, error, last_answer)
            if answer == "__quit__":
                speaker.say(["goodbye"] if args.tts == "cached" else "再见。")
                break
            if answer:
                last_answer = answer
                speaker.say(answer)
            interaction_event.clear()
    finally:
        interaction_event.clear()
        stop_event.set()
        listener.close()
        trigger.close()
        speaker.close()


def cache_entries(args):
    entries = dict(CACHE_PHRASES)
    max_tenths = int(round(args.cache_max_distance * 10))
    for tenths in range(0, max_tenths + 1):
        key = f"dist_{tenths:03d}"
        entries[key] = distance_text_from_key(key)

    labels = set(IMPORTANT_LABELS)
    labels.update(LABEL_ZH.keys() if args.cache_all_labels else [])
    for label in sorted(labels):
        text = LABEL_ZH.get(label, label)
        key = f"label_{label.replace(' ', '_')}"
        entries[key] = text
    return entries


def build_audio_cache(args):
    args.audio_cache_dir.mkdir(parents=True, exist_ok=True)
    speaker_args = argparse.Namespace(**vars(args))
    speaker_args.tts = "piper"
    speaker = Speaker(speaker_args)
    entries = cache_entries(args)
    total = len(entries)
    for index, (key, text) in enumerate(entries.items(), start=1):
        output = args.audio_cache_dir / f"{key}.wav"
        if output.exists() and not args.rebuild_audio_cache:
            continue
        command_parts = speaker.build_piper_command()
        if command_parts is None:
            raise RuntimeError("无法构建 Piper 命令。")
        base_command, piper_exe = command_parts
        command = base_command + ["--output_file", str(output.resolve())]
        result = subprocess.run(
            command,
            input=normalize_tts_text(text).encode("utf-8"),
            capture_output=True,
            check=False,
            cwd=piper_exe.parent if piper_exe.exists() else None,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"生成 {key} 失败：{stderr or result.returncode}")
        if index % 20 == 0 or index == total:
            print(f"已生成 {index}/{total}")
    speaker.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Chinese voice/text agent for the vision detector.")
    parser.add_argument("--state-input", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--max-state-age", type=float, default=2.0)
    parser.add_argument("--input-mode", choices=("text", "sr-google", "vosk", "vosk-wav"), default="text")
    parser.add_argument("--language", default="zh-CN")
    parser.add_argument("--mic-index", type=int)
    parser.add_argument("--list-mics", action="store_true")
    parser.add_argument("--push-to-talk", dest="push_to_talk", action="store_true", default=True)
    parser.add_argument("--no-push-to-talk", dest="push_to_talk", action="store_false")
    parser.add_argument("--recording-prompt", action="store_true", help="Speak a prompt before recording. Disabled by default to avoid blocking arecord.")
    parser.add_argument("--trigger-mode", choices=("keyboard", "gpio"), default="keyboard")
    parser.add_argument("--gpio-chip", default="/dev/gpiochip0")
    parser.add_argument("--gpio-line", type=int, default=2)
    parser.add_argument("--gpio-edge", choices=("falling", "rising", "both"), default="falling")
    parser.add_argument("--gpio-active-low", action="store_true")
    parser.add_argument("--gpio-bias", choices=("default", "pull-up", "pull-down", "disabled"), default="default")
    parser.add_argument("--gpio-debounce-ms", type=int, default=80)
    parser.add_argument("--gpiomon-exe", default="gpiomon")
    parser.add_argument("--gpiomon-style", choices=("auto", "long-option", "chip-option", "positional"), default="auto")
    parser.add_argument("--gpio-debug", action="store_true")
    parser.add_argument("--gpio-debug-interval", type=float, default=1.0)
    parser.add_argument("--gpio-event-timeout", type=float, default=2.0, help="Seconds to wait for one gpiomon event before restarting the watcher.")
    parser.add_argument("--listen-timeout", type=float, default=5.0)
    parser.add_argument("--phrase-time-limit", type=float, default=4.0)
    parser.add_argument("--ambient-duration", type=float, default=0.5)
    parser.add_argument("--vosk-model", type=Path)
    parser.add_argument("--vosk-stream", action="store_true", help="Use continuous RawInputStream mode for Vosk.")
    parser.add_argument("--arecord-exe", default="arecord")
    parser.add_argument("--arecord-device", default="default")
    parser.add_argument("--arecord-auto-detect", dest="arecord_auto_detect", action="store_true", default=True)
    parser.add_argument("--no-arecord-auto-detect", dest="arecord_auto_detect", action="store_false")
    parser.add_argument("--arecord-probe-seconds", type=float, default=1.0)
    parser.add_argument("--record-seconds", type=int, default=4)
    parser.add_argument("--record-timeout-extra", type=float, default=3.0, help="Extra seconds allowed for arecord beyond --record-seconds.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--tts", choices=("auto", "print", "pyttsx3", "edge", "piper", "cached", "command"), default="auto")
    parser.add_argument("--tts-voice", help="pyttsx3 voice id. Use --list-voices to inspect available voices.")
    parser.add_argument("--tts-rate", type=int, default=170)
    parser.add_argument("--tts-volume", type=float, default=1.0)
    parser.add_argument("--tts-prefer-chinese", dest="tts_prefer_chinese", action="store_true", default=True)
    parser.add_argument("--no-tts-prefer-chinese", dest="tts_prefer_chinese", action="store_false")
    parser.add_argument("--edge-voice", default="zh-CN-XiaoxiaoNeural")
    parser.add_argument("--edge-rate", default="+0%")
    parser.add_argument("--edge-volume", default="+0%")
    parser.add_argument("--piper-exe", default="piper")
    parser.add_argument("--piper-model", type=Path)
    parser.add_argument("--piper-config", type=Path)
    parser.add_argument("--piper-speaker", type=int)
    parser.add_argument("--piper-length-scale", type=float, default=1.0)
    parser.add_argument("--audio-player", help="Audio player command for generated files. Use {file} as placeholder.")
    parser.add_argument("--audio-timeout", type=float, default=8.0, help="Maximum seconds allowed for one audio playback command.")
    parser.add_argument("--audio-cache-dir", type=Path, default=Path("audio_cache"))
    parser.add_argument("--build-audio-cache", action="store_true")
    parser.add_argument("--rebuild-audio-cache", action="store_true")
    parser.add_argument("--cache-max-distance", type=float, default=5.0)
    parser.add_argument("--cache-all-labels", action="store_true")
    parser.add_argument("--tts-command", help="External TTS command. Use {text} as placeholder, or text is appended.")
    parser.add_argument("--tts-stdin", action="store_true", help="Send text to --tts-command stdin instead of argv.")
    parser.add_argument("--list-voices", action="store_true")
    parser.add_argument("--test-tts", help="Speak this text once and exit.")
    parser.add_argument("--warning-distance", type=float, default=2.0)
    parser.add_argument("--danger-distance", type=float, default=1.5)
    parser.add_argument("--warning-cooldown", type=float, default=3.0)
    parser.add_argument("--danger-cooldown", type=float, default=1.0)
    parser.add_argument("--warning-poll-interval", type=float, default=0.3)
    parser.add_argument("--warning-forget-seconds", type=float, default=2.0)
    parser.add_argument("--warning-same-iou", type=float, default=0.25)
    parser.add_argument("--warning-same-center-ratio", type=float, default=0.6)
    parser.add_argument("--warning-distance-delta", type=float, default=0.3)
    parser.add_argument("--no-warning", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_mics:
        list_microphones()
        return
    if args.build_audio_cache:
        build_audio_cache(args)
        return
    speaker = Speaker(args)
    if args.list_voices:
        return
    if args.test_tts:
        speaker.say(args.test_tts)
        speaker.close()
        return
    print("助手：正在加载语音识别模型...")
    listener = create_listener(args)
    trigger = create_trigger(args) if args.input_mode != "text" and args.push_to_talk else KeyboardTrigger()
    interactive_loop(args, speaker, listener, trigger)


if __name__ == "__main__":
    main()
