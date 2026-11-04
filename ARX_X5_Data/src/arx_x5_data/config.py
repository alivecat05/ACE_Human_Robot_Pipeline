"""Portable TOML configuration. Local serial numbers stay outside version control."""

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from .model import CAMERAS, DEFAULT_GRIPPER_FEEDBACK_MARGIN_NATIVE, gripper_bounds_native


@dataclass(frozen=True)
class CameraConfig:
    device: str
    width: int
    height: int
    fps: int = 30


@dataclass(frozen=True)
class SyncConfig:
    delay_ms: float = 100
    camera_tolerance_ms: float = 20
    camera_span_ms: float = 25
    telemetry_tolerance_ms: float = 20
    feedback_max_age_ms: float = 100
    source_timeout_s: float = 0.5


@dataclass(frozen=True)
class Config:
    output: Path
    socket: Path
    cameras: dict[str, CameraConfig]
    fps: int = 30
    sync: SyncConfig = SyncConfig()
    writer_queue_size: int = 8
    crf: int = 18
    preset: str = "veryfast"
    gripper_feedback_margin_native: float = DEFAULT_GRIPPER_FEEDBACK_MARGIN_NATIVE

    def snapshot(self) -> dict:
        value = asdict(self)
        value["output"], value["socket"] = str(self.output), str(self.socket)
        return value


def load_config(path: Path) -> Config:
    path = path.resolve()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {"recording", "sync", "cameras"}:
        raise ValueError("Config requires exactly recording, sync and cameras sections")
    rec = dict(raw["recording"])
    if set(raw["cameras"]) != set(CAMERAS):
        raise ValueError(f"Configure exactly these cameras: {CAMERAS}")
    cameras = {name: CameraConfig(**raw["cameras"][name]) for name in CAMERAS}
    devices = [camera.device for camera in cameras.values()]
    if len(set(devices)) != 3:
        raise ValueError("Three distinct camera devices are required")
    for camera in cameras.values():
        if not camera.device or not all(
            type(v) is int and v > 0 for v in (camera.width, camera.height, camera.fps)
        ):
            raise ValueError("Invalid camera device, dimensions or frame rate")
        if camera.width % 2 or camera.height % 2:
            raise ValueError("Camera dimensions must be even")
    for key in ("output", "socket"):
        value = Path(rec[key]).expanduser()
        rec[key] = value if value.is_absolute() else path.parent / value
    cfg = Config(cameras=cameras, sync=SyncConfig(**raw["sync"]), **rec)
    gripper_bounds_native(command=False, feedback_margin=cfg.gripper_feedback_margin_native)
    if type(cfg.fps) is not int or not 1 <= cfg.fps <= 30:
        raise ValueError("Recording fps must be an integer in 1..30")
    if any(camera.fps < cfg.fps for camera in cameras.values()):
        raise ValueError("Camera fps must be at least the recording fps")
    if type(cfg.writer_queue_size) is not int or not 1 <= cfg.writer_queue_size <= 60:
        raise ValueError("writer_queue_size must be in 1..60")
    if type(cfg.crf) is not int or not 0 <= cfg.crf <= 30:
        raise ValueError("crf must be in 0..30")
    if cfg.preset not in {"ultrafast", "superfast", "veryfast", "faster", "fast"}:
        raise ValueError("Unsupported encoder preset")
    if not all(math.isfinite(v) and v > 0 for v in asdict(cfg.sync).values()):
        raise ValueError("Sync tolerances must be finite and positive")
    if cfg.sync.delay_ms < max(cfg.sync.camera_tolerance_ms, cfg.sync.telemetry_tolerance_ms):
        raise ValueError("Buffer delay must cover the matching windows")
    if cfg.sync.delay_ms / 1000 >= cfg.sync.source_timeout_s:
        raise ValueError("Buffer delay must be shorter than the source timeout")
    return cfg
