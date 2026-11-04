"""Read-only validation of the native contract, including actual source timing."""

import csv
import json
import math
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .media import decoded_frames
from .model import ACTION_SEMANTICS, CAMERAS, SCHEMA, VECTOR_NAMES, gripper_bounds_native


@dataclass(frozen=True)
class Episode:
    path: Path
    manifest: dict
    segments: tuple[tuple[int, int], ...]  # half-open saved frame ranges
    fingerprint: tuple = ()
    camera_stats: tuple | None = None

    def assert_current(self):
        if (self.path / ".workbench-deleted.json").exists():
            raise ValueError("Episode was deleted by operator")
        if not self.fingerprint or self.fingerprint != episode_fingerprint(
            self.path, self.manifest
        ):
            raise ValueError("Source files changed after validation")


def episode_fingerprint(path, manifest):
    files = [path / n for n in ("manifest.json", "samples.jsonl", "action.csv")]
    files += [video_path(path, manifest["videos"][n]) for n in CAMERAS]
    result = []
    for file in files:
        stat = file.stat()
        result.append(
            (str(file), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        )
    return tuple(result)


def rows(path: Path):
    with (path / "samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def video_path(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Video path escapes its episode")
    return result


def validate_episode(
    path: Path, *, allow_mock: bool = False, _pending: bool = False, _collect_stats: bool = False
) -> Episode:
    path = path.resolve()
    if (path / ".workbench-deleted.json").exists():
        raise ValueError("Episode was deleted by operator")
    meta = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if meta["schema"] != SCHEMA or meta["robot"] != "arx_x5_2023" or meta["sdk_type"] != 0:
        raise ValueError("Wrong native format or robot model")
    if meta["status"] != "complete" and not (_pending and meta["status"] == "pending_validation"):
        raise ValueError(f"Episode is {meta['status']}, not complete")
    if type(meta["mock"]) is not bool or (meta["mock"] and not allow_mock):
        raise ValueError("Mock data is excluded; use --allow-mock only for testing")
    if meta["vector_names"] != list(VECTOR_NAMES) or meta["action_semantics"] != ACTION_SEMANTICS:
        raise ValueError("State/action definition changed")
    if type(meta["frames"]) is not int or meta["frames"] < 2:
        raise ValueError("At least two saved frames are required")
    fps = meta["fps"]
    if type(fps) is not int or not 1 <= fps <= 30:
        raise ValueError("Invalid recording frame rate")
    fingerprint = episode_fingerprint(path, meta)
    config = meta["config"]
    # Older recordings without this declaration retain their strict 0..5 contract.
    feedback_margin = config.get("gripper_feedback_margin_native", 0.0)
    state_low, state_high = gripper_bounds_native(command=False, feedback_margin=feedback_margin)
    if config["fps"] != fps or set(config["cameras"]) != set(CAMERAS):
        raise ValueError("Manifest/config mismatch")
    if set(meta["videos"]) != set(CAMERAS):
        raise ValueError("Three named RGB videos are required")
    stats_collector = None
    if _collect_stats:
        from .video_stats import ValidationStats

        stats_collector = ValidationStats(path)
    sync = config["sync"]
    segments = []
    segment_start = 0
    previous = None
    previous_camera_ids = dict.fromkeys(CAMERAS, -1)
    previous_robot_time = float("-inf")
    with ExitStack() as stack:
        videos = {}
        for name in CAMERAS:
            camera = config["cameras"][name]
            iterator = decoded_frames(
                video_path(path, meta["videos"][name]),
                camera["width"],
                camera["height"],
                fps,
                rgb=False,
            )
            stack.callback(iterator.close)
            videos[name] = iterator
        csv_file = stack.enter_context((path / "action.csv").open(encoding="utf-8", newline=""))
        actions = csv.reader(csv_file)
        if next(actions, None) != ["timestamp_unix_s", *VECTOR_NAMES]:
            raise ValueError("CSV columns differ from the data contract")
        count = 0
        for index, row in enumerate(rows(path)):
            if type(row["grid_index"]) is not int or row["grid_index"] < 0:
                raise ValueError("Invalid grid index")
            if row["frame_index"] != index:
                raise ValueError("Saved frame indexes are not consecutive")
            stamp = row["timestamp_monotonic_s"]
            expected = meta["start_monotonic_s"] + row["grid_index"] / fps
            if not math.isfinite(stamp) or abs(stamp - expected) > 1e-6:
                raise ValueError("Acquisition timestamp differs from the sampling grid")
            unix = stamp + meta["monotonic_to_unix_offset_s"]
            if not math.isfinite(unix) or abs(row["timestamp_unix_s"] - unix) > 1e-6:
                raise ValueError("Unix and monotonic clocks disagree")
            if previous is not None:
                if row["grid_index"] <= previous["grid_index"]:
                    raise ValueError("Duplicate or decreasing sampling grid")
                if row["grid_index"] != previous["grid_index"] + 1:
                    if index - segment_start >= 2:
                        segments.append((segment_start, index))
                    segment_start = index
            for key in ("observation.state", "action"):
                vector = np.asarray(row[key], dtype=float)
                if vector.shape != (14,) or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid {key}")
                low, high = (0.0, 1.0) if key == "action" else (state_low / 5, state_high / 5)
                if any(not low <= vector[i] <= high for i in (6, 13)):
                    raise ValueError(f"{key}: normalized gripper value outside {low}..{high}")
            packet_time = row["telemetry_timestamp_monotonic_s"]
            if (
                not math.isfinite(packet_time)
                or abs(packet_time - stamp) > sync["telemetry_tolerance_ms"] / 1000 + 1e-8
                or packet_time <= previous_robot_time
            ):
                raise ValueError("Telemetry is reused or outside the matching window")
            previous_robot_time = packet_time
            if len(row["feedback_oldest_monotonic_s"]) != 2 or any(
                not math.isfinite(t)
                or not 0 <= packet_time - t <= sync["feedback_max_age_ms"] / 1000 + 1e-8
                for t in row["feedback_oldest_monotonic_s"]
            ):
                raise ValueError("Robot feedback is stale")
            camera_stamps = []
            if set(row["cameras"]) != set(CAMERAS):
                raise ValueError("Row camera names changed")
            for name in CAMERAS:
                camera = row["cameras"][name]
                t = camera["timestamp_monotonic_s"]
                allowed_clock = "mock_monotonic" if meta["mock"] else "v4l2_monotonic_pts"
                if (
                    not math.isfinite(t)
                    or abs(t - stamp) > sync["camera_tolerance_ms"] / 1000 + 1e-8
                    or camera["capture_index"] <= previous_camera_ids[name]
                    or camera["clock"] != allowed_clock
                ):
                    raise ValueError(f"{name}: reused, mistimed or unverified camera frame")
                previous_camera_ids[name] = camera["capture_index"]
                camera_stamps.append(t)
                frame = next(videos[name], None)
                if frame is None:
                    raise ValueError(f"{name}: truncated video")
                if stats_collector is not None:
                    stats_collector.add(name, index, frame)
            if max(camera_stamps) - min(camera_stamps) > sync["camera_span_ms"] / 1000 + 1e-8:
                raise ValueError("Camera span exceeds the configured limit")
            csv_row = next(actions, None)
            if csv_row is None or len(csv_row) != 15:
                raise ValueError("CSV/journal row count or shape differs")
            csv_stamp = float(csv_row[0])
            if (
                not math.isfinite(csv_stamp)
                or abs(csv_stamp - row["timestamp_unix_s"]) > 1e-6
                or not np.array_equal(np.array(csv_row[1:], dtype=float), row["action"])
            ):
                raise ValueError("CSV values differ from the journal")
            previous = row
            count += 1
        if count != meta["frames"] or next(actions, None) is not None:
            raise ValueError("Manifest, CSV and journal counts disagree")
        if any(next(video, None) is not None for video in videos.values()):
            raise ValueError("Video contains extra frames")
    if count - segment_start >= 2:
        segments.append((segment_start, count))
    if not segments:
        raise ValueError("No contiguous segment contains at least two frames")
    camera_stats = stats_collector.finish(tuple(segments)) if stats_collector else None
    episode = Episode(path, meta, tuple(segments), fingerprint, camera_stats)
    episode.assert_current()
    return episode


def discover_episodes(source: Path, *, allow_mock: bool = False) -> list[Episode]:
    source = source.resolve()
    if (source / "manifest.json").is_file():
        return [validate_episode(source, allow_mock=allow_mock)]
    result = []
    for manifest in sorted(source.rglob("manifest.json")):
        if (manifest.parent / ".workbench-deleted.json").exists():
            continue
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if meta.get("status") in {"incomplete", "recording", "cancelled"}:
            continue
        result.append(validate_episode(manifest.parent, allow_mock=allow_mock))
    if not result:
        raise ValueError("No complete episodes found")
    return result
