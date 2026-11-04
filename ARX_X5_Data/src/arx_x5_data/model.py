"""One explicit data contract for acquisition, validation and export."""

import math
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar

import numpy as np

CAMERAS = ("head", "left_wrist", "right_wrist")
ARMS = ("left", "right")
DEFAULT_GRIPPER_FEEDBACK_MARGIN_NATIVE = 0.1


def gripper_bounds_native(
    *, command: bool, feedback_margin: float = DEFAULT_GRIPPER_FEEDBACK_MARGIN_NATIVE
) -> tuple[float, float]:
    """Feedback tolerance only; never expand the SDK command range or clip data."""
    if not math.isfinite(feedback_margin) or not 0 <= feedback_margin <= 0.5:
        raise ValueError("gripper_feedback_margin_native must be finite and in 0..0.5")
    return (0.0, 5.0) if command else (-feedback_margin, 5.0 + feedback_margin)


VECTOR_NAMES = tuple(
    f"{side}.{field}"
    for side in ARMS
    for field in (*(f"joint_{i}_rad" for i in range(1, 7)), "gripper_0_to_1")
)
SCHEMA = "arx.x5_2023.episode.v1"
ACTION_SEMANTICS = (
    "SDK-accepted joint targets (rad) and normalized gripper targets from the same "
    "teleop cycle; not emitted motor setpoints or next-step measured states"
)


@dataclass(frozen=True)
class CameraFrame:
    timestamp: float
    sequence: int
    rgb: np.ndarray
    received_at: float
    clock: str = "v4l2_monotonic_pts"
    discarded_invalid_packets: int = 0


@dataclass(frozen=True)
class RobotSample:
    timestamp: float
    state: tuple[float, ...]
    action: tuple[float, ...]
    feedback_times: tuple[float, float]
    source: str = "legacy_teleop"


@dataclass(frozen=True)
class AlignedSample:
    timestamp: float
    grid_index: int
    cameras: dict[str, CameraFrame]
    robot: RobotSample


T = TypeVar("T")


class History(Generic[T]):
    """Bounded history with strictly increasing acquisition timestamps."""

    def __init__(self, capacity: int):
        self._items: deque[T] = deque(maxlen=capacity)
        self._lock = Lock()

    def add(self, item: T) -> None:
        with self._lock:
            if self._items and item.timestamp <= self._items[-1].timestamp:
                raise ValueError("Acquisition timestamp did not increase")
            self._items.append(item)

    def snapshot(self) -> list[T]:
        with self._lock:
            return list(self._items)

    def latest(self) -> T | None:
        with self._lock:
            return self._items[-1] if self._items else None
