"""Bounded background writes and explicit complete/incomplete/cancelled episodes."""

import csv
import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from queue import Empty, Full, Queue

from .config import Config
from .media import VideoWriter, decoded_frames
from .model import ACTION_SEMANTICS, CAMERAS, SCHEMA, VECTOR_NAMES, AlignedSample


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def reserve_episode(root: Path, task: str) -> Path:
    if (
        not task.strip()
        or task != task.strip()
        or len(task) > 100
        or re.search(r'[<>:"/\\|?*\x00-\x1f]', task)
        or task in {".", ".."}
    ):
        raise ValueError("Task must be a nonempty directory name without path separators")
    folder = root.resolve() / task
    if folder.resolve().parent != root.resolve():
        raise ValueError("Task directory escapes the recording root")
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1_000_000):
        path = folder / f"episode_{index:06d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("No free episode index")


class EpisodeWriter:
    def __init__(
        self, config: Config, task: str, start: float, *, mock: bool, background: bool = False
    ):
        self.config, self.task, self.start, self.mock = config, task, start, mock
        self.background = background
        self.timings = {}
        self._finish_at = None
        self.path = reserve_episode(config.output, task)
        self.offset = time.time() - time.monotonic()
        self.frames = 0
        self.drops: Counter[str] = Counter()
        self.error: Exception | None = None
        self.result = "recording"
        self.reason = ""
        self._requested_status = "complete"
        self._queue: Queue[AlignedSample] = Queue(config.writer_queue_size)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_grid = -1
        self._thread = threading.Thread(target=self._run, name="episode-writer", daemon=True)
        self._manifest("recording")
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._thread.is_alive()

    @property
    def accepting(self) -> bool:
        return self.busy and not self._stop.is_set() and self.error is None

    def submit(self, sample: AlignedSample) -> None:
        if self.error:
            raise RuntimeError(f"Disk writer failed: {self.error}") from self.error
        if self._stop.is_set() or not self.busy:
            raise RuntimeError("Writer no longer accepts samples")
        if sample.grid_index <= self._last_grid:
            raise ValueError("Recording grid must increase")
        try:
            self._queue.put_nowait(sample)
        except Full as error:
            raise RuntimeError("Video writer queue full; recording must stop") from error
        self._last_grid = sample.grid_index

    def dropped(self, reason: str) -> None:
        with self._lock:
            self.drops[reason] += 1

    def finish(self, status: str = "complete", reason: str = "operator_stop") -> None:
        if status not in {"complete", "incomplete", "cancelled"}:
            raise ValueError("Unknown recording outcome")
        if not self._stop.is_set():
            self._finish_at = time.monotonic()
            self._requested_status, self.reason = status, reason
            self._stop.set()

    def wait(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)
        if self.busy:
            raise TimeoutError("Video writer is still saving")

    def _row(self, sample: AlignedSample) -> dict:
        return {
            "frame_index": self.frames,
            "grid_index": sample.grid_index,
            "timestamp_monotonic_s": sample.timestamp,
            "timestamp_unix_s": sample.timestamp + self.offset,
            "observation.state": sample.robot.state,
            "action": sample.robot.action,
            "telemetry_timestamp_monotonic_s": sample.robot.timestamp,
            "feedback_oldest_monotonic_s": sample.robot.feedback_times,
            "telemetry_source": sample.robot.source,
            "cameras": {
                name: {
                    "timestamp_monotonic_s": frame.timestamp,
                    "received_monotonic_s": frame.received_at,
                    "capture_index": frame.sequence,
                    "clock": frame.clock,
                    "discarded_invalid_packets": frame.discarded_invalid_packets,
                }
                for name, frame in sample.cameras.items()
            },
        }

    def _manifest(self, status: str) -> None:
        with self._lock:
            drops = dict(self.drops)
        atomic_json(
            self.path / "manifest.json",
            {
                "timings": self.timings,
                "schema": SCHEMA,
                "robot": "arx_x5_2023",
                "sdk_type": 0,
                "task": self.task,
                "status": status,
                "reason": self.reason,
                "error": str(self.error) if self.error else None,
                "mock": self.mock,
                "fps": self.config.fps,
                "frames": self.frames,
                "start_monotonic_s": self.start,
                "monotonic_to_unix_offset_s": self.offset,
                "dropped_grid_ticks": sum(drops.values()),
                "drop_reasons": drops,
                "vector_names": VECTOR_NAMES,
                "action_semantics": ACTION_SEMANTICS,
                "action_time_basis": (
                    "teleop publication after SDK submission; not motor emission time"
                ),
                "feedback_time_basis": "oldest motor host-receive Unix time mapped per packet",
                "video_timeline": (
                    "constant fps; real gaps preserved in grid_index and source clocks"
                ),
                "camera_sequence_basis": "V4L2 packet index in this stream, not driver sequence",
                "videos": {name: f"{name}.mp4" for name in CAMERAS},
                "config": self.config.snapshot(),
            },
        )

    def _run(self) -> None:
        videos = {}
        try:
            for name in CAMERAS:
                camera = self.config.cameras[name]
                videos[name] = VideoWriter(
                    self.path / f"{name}.mp4",
                    camera.width,
                    camera.height,
                    self.config.fps,
                    self.config.crf,
                    self.config.preset,
                )
            with (
                (self.path / "samples.jsonl").open("x", encoding="utf-8") as journal,
                (self.path / "action.csv").open("x", encoding="utf-8", newline="") as csv_file,
            ):
                actions = csv.writer(csv_file)
                actions.writerow(["timestamp_unix_s", *VECTOR_NAMES])
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        sample = self._queue.get(timeout=0.05)
                    except Empty:
                        continue
                    row = self._row(sample)
                    encoded = json.dumps(row, allow_nan=False) + "\n"
                    for name in CAMERAS:
                        videos[name].write(sample.cameras[name].rgb)
                    journal.write(encoded)
                    actions.writerow([format(row["timestamp_unix_s"], ".6f"), *sample.robot.action])
                    journal.flush()
                    csv_file.flush()
                    self.frames += 1
                os.fsync(journal.fileno())
                os.fsync(csv_file.fileno())
        except Exception as error:
            self.error = error
        finally:
            for video in videos.values():
                try:
                    video.close()
                except Exception as error:
                    self.error = self.error or error
        try:
            if not self.error and self.frames and not self.background:
                for name in CAMERAS:
                    camera = self.config.cameras[name]
                    count = sum(
                        1
                        for _ in decoded_frames(
                            self.path / f"{name}.mp4", camera.width, camera.height, self.config.fps
                        )
                    )
                    if count != self.frames:
                        raise ValueError(f"{name}: encoded frame count differs from the journal")
            if not self.error and self._requested_status == "complete" and self.frames < 2:
                self.error = ValueError(
                    "At least two frames are required to save a complete episode"
                )
            self.result = self._requested_status if not self.error else "incomplete"
            for name in videos if self.frames else {}:
                with (self.path / f"{name}.mp4").open("rb") as stream:
                    os.fsync(stream.fileno())
            if self._finish_at is not None:
                self.timings["close_s"] = time.monotonic() - self._finish_at
            if self.background and self.result == "complete":
                self.result = "pending_validation"
                self._manifest(self.result)
                atomic_json(self.path / "job.json", {"state": "queued", "attempts": 0})
            else:
                self._manifest(self.result)
        except Exception as error:
            self.error = self.error or error
            self.result = "incomplete"
            try:
                self._manifest("incomplete")
            except OSError:
                pass
