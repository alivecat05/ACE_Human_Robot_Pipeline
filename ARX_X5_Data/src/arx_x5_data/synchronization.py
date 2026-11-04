"""Match acquired frames and genuine accepted targets, without inventing actions."""

from itertools import product
from math import ceil

from .config import SyncConfig
from .model import CAMERAS, AlignedSample, CameraFrame, History, RobotSample


class Synchronizer:
    def __init__(
        self,
        cameras: dict[str, History[CameraFrame]],
        robot: History[RobotSample],
        config: SyncConfig,
    ):
        self.cameras, self.robot, self.config = cameras, robot, config
        self.last_sequences = dict.fromkeys(CAMERAS, -1)
        self.last_robot_time = float("-inf")
        self.rejection = ""

    def grid_start(self, not_before: float, fps: int) -> float:
        """Choose a future grid phase from a recent, tightly grouped camera tuple.

        A wall-clock click can put each grid tick between unrelated camera cycles.
        Keep genuine source timestamps and all tolerances; only phase the new grid.
        """
        histories = {name: self.cameras[name].snapshot() for name in CAMERAS}
        if not all(histories.values()):
            return not_before
        period = 1 / fps
        anchor = histories["head"][-1].timestamp
        candidates = [
            [frame for frame in histories[name] if abs(frame.timestamp - anchor) <= period]
            for name in CAMERAS
        ]
        if not all(candidates):
            return not_before

        def score(frames):
            stamps = [frame.timestamp for frame in frames]
            return max(stamps) - min(stamps), -max(stamps)

        frames = min(product(*candidates), key=score)
        span = score(frames)[0]
        if span > min(self.config.camera_span_ms, 2 * self.config.camera_tolerance_ms) / 1000:
            return not_before
        stamps = [frame.timestamp for frame in frames]
        origin = (min(stamps) + max(stamps)) / 2
        return max(not_before, origin + ceil((not_before - origin) / period) * period)

    def match(self, timestamp: float, grid_index: int) -> AlignedSample | None:
        candidates = {}
        for name in CAMERAS:
            candidates[name] = [
                frame
                for frame in self.cameras[name].snapshot()
                if abs(frame.timestamp - timestamp) <= self.config.camera_tolerance_ms / 1000
                and frame.sequence > self.last_sequences[name]
            ]
            if not candidates[name]:
                self.rejection = f"camera_missing_or_reused:{name}"
                return None
        packets = [
            packet
            for packet in self.robot.snapshot()
            if abs(packet.timestamp - timestamp) <= self.config.telemetry_tolerance_ms / 1000
            and packet.timestamp > self.last_robot_time
            and all(
                0 <= packet.timestamp - stamp <= self.config.feedback_max_age_ms / 1000
                for stamp in packet.feedback_times
            )
        ]
        if not packets:
            self.rejection = "telemetry_missing_reused_or_stale"
            return None

        def score(frames: tuple[CameraFrame, ...]) -> tuple[float, float]:
            stamps = [frame.timestamp for frame in frames]
            return max(stamps) - min(stamps), sum(abs(stamp - timestamp) for stamp in stamps)

        frames = min(product(*(candidates[name] for name in CAMERAS)), key=score)
        if score(frames)[0] > self.config.camera_span_ms / 1000:
            self.rejection = "camera_span"
            return None
        packet = min(packets, key=lambda item: abs(item.timestamp - timestamp))
        self.last_sequences = {
            name: frame.sequence for name, frame in zip(CAMERAS, frames, strict=False)
        }
        self.last_robot_time = packet.timestamp
        self.rejection = ""
        return AlignedSample(
            timestamp, grid_index, dict(zip(CAMERAS, frames, strict=False)), packet
        )
