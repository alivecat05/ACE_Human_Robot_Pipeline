"""Stream Galbot protobuf recordings using their embedded schemas."""

import csv
import json
import math
import re
from contextlib import ExitStack
from pathlib import Path

from google.protobuf.json_format import MessageToDict
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

STATE_TOPIC = "singorix/wbcs/sensor"
GROUPS = ("left_arm", "right_arm", "left_gripper", "right_gripper", "head", "chassis", "leg")
CAMERAS = {
    "/front_head_camera/left_color/image_raw": "head_left",
    "/front_head_camera/right_color/image_raw": "head_right",
    "/left_arm_camera/color/image_raw": "left_arm",
    "/right_arm_camera/color/image_raw": "right_arm",
}


def topic_filename(topic):
    return re.sub(r"[^\w.-]+", "_", topic.strip("/")) or "topic"


def joint_values(message, groups=GROUPS):
    if not hasattr(message, "joint_sensor_map"):
        raise ValueError("关节 topic 必须是包含 joint_sensor_map 的 SingoriXSensor")
    values = {}
    for group in groups:
        if group not in message.joint_sensor_map:
            continue
        joints = message.joint_sensor_map[group]
        if len(joints.name) != len(joints.position):
            raise ValueError(f"{group}: 关节名称和 position 数量不一致")
        for name, position in zip(joints.name, joints.position):
            column = f"{group}/{name}"
            if column in values or not math.isfinite(position):
                raise ValueError(f"无效或重复的关节值: {column}")
            values[column] = f"{position:.6f}"
    if not values:
        raise ValueError("关节消息中没有可导出的关节")
    return values


class JsonArray:
    """Write JSON arrays without keeping all messages in memory."""

    def __init__(self, file):
        self.file = file
        self.count = 0
        file.write("[\n")

    def write(self, value):
        if self.count:
            self.file.write(",\n")
        json.dump(value, self.file, ensure_ascii=False)
        self.count += 1

    def close(self):
        self.file.write("\n]\n")


def extract_recording(source, output, *, videos=True, fps=None, all_json=False,
                      state_topic=STATE_TOPIC, train=False, head="left"):
    """Extract into a new/temporary episode directory supplied by the caller."""
    if head not in ("left", "right"):
        raise ValueError("head 必须是 left 或 right")
    if train and not videos:
        raise ValueError("训练模式需要导出三路视频")
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stats = {}
    json_streams = {}
    column_names = None
    rows = 0
    # First pass writes states/metadata and measures each camera's actual rate.
    with ExitStack() as stack:
        file = stack.enter_context(source.open("rb"))
        reader = make_reader(file)
        decoder = DecoderFactory()
        csv_file = stack.enter_context((output / "action.csv").open("w", newline="", encoding="utf-8"))
        writer = csv.writer(csv_file)
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            if topic in CAMERAS:
                stat = stats.setdefault(topic, {"messages": 0, "first_ns": message.log_time,
                                                "last_ns": message.log_time})
                stat["messages"] += 1
                stat["last_ns"] = message.log_time
                continue
            if train and topic != state_topic:
                continue
            is_image = topic.endswith("/image_raw")
            is_urdf = topic.strip("/") == "robot_description"
            is_metadata = topic.endswith("/camera_info")
            if is_image or not (topic == state_topic or is_urdf or is_metadata or all_json):
                continue
            decode = decoder.decoder_for(channel.message_encoding, schema)
            if decode is None:
                raise ValueError(f"无法解码 topic {topic}: {channel.message_encoding}")
            decoded = decode(message.data)
            if topic == state_topic:
                values = joint_values(decoded)
                if column_names is None:
                    column_names = list(values)
                    writer.writerow(["timestamp", *column_names])
                if set(values) != set(column_names):
                    raise ValueError(f"{message.log_time}: 关节集合发生变化，停止以避免列错位")
                writer.writerow([str(message.log_time), *(values[name] for name in column_names)])
                rows += 1
            if is_urdf:
                urdf = getattr(decoded, "value", None)
                if not isinstance(urdf, str):
                    raise ValueError("robot_description.value 不是 URDF 字符串")
                (output / "robot_description.urdf").write_text(urdf, encoding="utf-8")
            elif not train and (is_metadata or all_json):
                if topic not in json_streams:
                    # Channel id prevents sanitized topic filename collisions.
                    name = f"{channel.id}_{topic_filename(topic)}.json"
                    path = output / "metadata" / name
                    path.parent.mkdir(exist_ok=True)
                    stream = JsonArray(stack.enter_context(path.open("w", encoding="utf-8")))
                    stack.callback(stream.close)
                    json_streams[topic] = (stream, str(path.relative_to(output)))
                data = MessageToDict(decoded, preserving_proto_field_name=True)
                data["_timestamp_ns"] = message.log_time
                json_streams[topic][0].write(data)
        if rows == 0:
            raise ValueError(f"没有找到关节状态消息: {state_topic}")

    expected_cameras = set(CAMERAS)
    if train:
        available = {CAMERAS[topic]: topic for topic in stats}
        selected = (f"head_{head}", "left_arm", "right_arm")
        missing = [name for name in selected if name not in available]
        if missing:
            raise ValueError(f"训练模式缺少必需相机: {', '.join(missing)}（指定头部相机为 {head}）")
        stats = {available[name]: stats[available[name]] for name in selected}
        expected_cameras = set(stats)
    for topic, stat in stats.items():
        duration = (stat["last_ns"] - stat["first_ns"]) / 1e9
        stat["fps"] = fps if fps is not None else ((stat["messages"] - 1) / duration if duration > 0 else 30.0)
        stat["name"] = CAMERAS[topic]
        stat["frames"] = 0
    if videos:
        extract_videos(source, output, stats, timestamps=not train)
    manifest = {
        "source": str(source.resolve()), "state_topic": state_topic,
        "action_semantics": "measured joint positions (sensor state), not target commands",
        "timestamp_unit": "nanoseconds", "timestamp_source": "MCAP log_time",
        "action_rows": rows, "columns": ["timestamp", *column_names],
        "videos_exported": videos, "cameras": stats,
        "missing_camera_topics": sorted(expected_cameras - set(stats)),
        "metadata": {topic: path for topic, (_, path) in json_streams.items()},
    }
    if not train:
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def extract_videos(source, output, stats, *, timestamps=True):
    if not stats:
        return
    import cv2
    import numpy as np

    writers = {}
    sizes = {}
    with ExitStack() as stack:
        source_file = stack.enter_context(source.open("rb"))
        reader = make_reader(source_file, decoder_factories=[DecoderFactory()])
        timestamp_writers = {}
        for topic, stat in stats.items():
            if not timestamps:
                continue
            file = stack.enter_context((output / f"{stat['name']}_timestamps.csv").open("w", newline="", encoding="utf-8"))
            timestamp_writers[topic] = csv.writer(file)
            timestamp_writers[topic].writerow(["frame_index", "timestamp"])
        for _, channel, message, decoded in reader.iter_decoded_messages(topics=list(stats)):
            topic = channel.topic
            frame = cv2.imdecode(np.frombuffer(decoded.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"图像无法解码: {topic} @ {message.log_time}")
            height, width = frame.shape[:2]
            if topic not in writers:
                stat = stats[topic]
                path = output / f"{stat['name']}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), stat["fps"], (width, height))
                stack.callback(writer.release)
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建视频: {path}")
                writers[topic] = writer
                sizes[topic] = (width, height)
                stat.update(width=width, height=height)
            if sizes[topic] != (width, height):
                raise ValueError(f"视频分辨率发生变化: {topic}")
            writers[topic].write(frame)
            if timestamps:
                timestamp_writers[topic].writerow([stats[topic]["frames"], str(message.log_time)])
            stats[topic]["frames"] += 1
