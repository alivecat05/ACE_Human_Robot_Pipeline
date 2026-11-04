"""Independent camera workers and telemetry input. No motor-control imports."""

import threading
import time

import av
import numpy as np

from .config import CameraConfig, Config
from .model import CAMERAS, CameraFrame, History, RobotSample
from .telemetry import TelemetryReceiver


def capture_frames(config: CameraConfig, stop: threading.Event):
    """Keep V4L2's original monotonic PTS; reject silently negotiated depth streams."""
    options = {
        "input_format": "yuyv422",
        "video_size": f"{config.width}x{config.height}",
        "framerate": str(config.fps),
        "timestamps": "default",
    }
    with av.open(config.device, format="v4l2", options=options, timeout=(3, 2)) as container:
        stream = container.streams.video[0]
        actual = stream.codec_context
        if (
            actual.width != config.width
            or actual.height != config.height
            or actual.format.name != "yuyv422"
        ):
            raise ValueError(
                f"{config.device}: negotiated {actual.width}x{actual.height} "
                f"{actual.format.name}, expected {config.width}x{config.height} YUYV RGB source"
            )
        if stream.average_rate is not None and abs(float(stream.average_rate) - config.fps) > 0.1:
            raise ValueError(f"{config.device}: negotiated frame rate differs from configuration")
        discarded = 0
        stale_since = None
        expected_bytes = config.width * config.height * 2
        for sequence, packet in enumerate(container.demux(video=0)):
            if stop.is_set():
                return
            # UVC can deliver incomplete startup/error buffers. Never send them
            # into the rawvideo decoder or pad missing bytes into training data.
            if packet.size != expected_bytes or packet.is_corrupt:
                discarded += 1
                continue
            for frame in packet.decode():
                received = time.monotonic()
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("V4L2 frame has no timestamp")
                stamp = float(frame.pts * frame.time_base)
                age = received - stamp
                if not np.isfinite(stamp) or age < -0.01:
                    raise ValueError(
                        f"V4L2 PTS is not in the host monotonic clock (age={age:.3f}s)"
                    )
                if age > 0.5:
                    # Drain delayed UVC buffers without RGB conversion. Never replace PTS
                    # with receipt time or let these frames enter the synchronizer.
                    discarded += 1
                    if stale_since is None:
                        stale_since = received
                    if received - stale_since > 2:
                        raise ValueError(f"V4L2 PTS remains stale for >2s (age={age:.3f}s)")
                    continue
                stale_since = None
                rgb = frame.to_ndarray(format="rgb24")
                rgb.setflags(write=False)
                yield CameraFrame(
                    stamp,
                    sequence,
                    rgb,
                    received,
                    discarded_invalid_packets=discarded,
                )


class Sources:
    def __init__(self, config: Config, *, mock: bool = False, cameras_only: bool = False):
        self.config, self.mock, self.cameras_only = config, mock, cameras_only
        # 0.4 s of full-resolution images, independent of run duration.
        self.cameras = {
            name: History[CameraFrame](max(8, int(camera.fps * 0.4)))
            for name, camera in config.cameras.items()
        }
        self.robot = History[RobotSample](120)
        self.stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()
        self.receiver = None

    def start(self) -> None:
        try:
            if not self.mock and not self.cameras_only:
                self.receiver = TelemetryReceiver(
                    self.config.socket,
                    self.robot,
                    self.config.sync.feedback_max_age_ms / 1000,
                    gripper_feedback_margin_native=self.config.gripper_feedback_margin_native,
                )
                self.receiver.start()
            for name in CAMERAS:
                self._launch(name, lambda name=name: self._camera(name))
            if self.mock and not self.cameras_only:
                self._launch("mock-robot", self._mock_robot)
        except BaseException:
            self.close()
            raise

    def _launch(self, name, function) -> None:
        def run():
            try:
                function()
            except Exception as error:
                with self._lock:
                    self._errors[name] = f"{type(error).__name__}: {error}"

        thread = threading.Thread(target=run, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _camera(self, name: str) -> None:
        config = self.config.cameras[name]
        if not self.mock:
            for frame in capture_frames(config, self.stop):
                self.cameras[name].add(frame)
            return
        sequence = 0
        deadline = time.monotonic()
        while not self.stop.is_set():
            stamp = time.monotonic()
            rgb = np.zeros((config.height, config.width, 3), dtype=np.uint8)
            rgb[..., CAMERAS.index(name)] = 64 + sequence % 128
            rgb.setflags(write=False)
            self.cameras[name].add(CameraFrame(stamp, sequence, rgb, stamp, "mock_monotonic"))
            sequence += 1
            deadline += 1 / config.fps
            self.stop.wait(max(0, deadline - time.monotonic()))

    def _mock_robot(self) -> None:
        while not self.stop.is_set():
            stamp = time.monotonic()
            state = (0.0,) * 6 + (0.5,) + (0.0,) * 6 + (0.5,)
            action = (0.01,) * 6 + (0.5,) + (0.01,) * 6 + (0.5,)
            self.robot.add(RobotSample(stamp, state, action, (stamp, stamp), "mock"))
            self.stop.wait(0.01)

    def check(self, *, require_ready: bool = True) -> None:
        with self._lock:
            if self._errors:
                raise RuntimeError(str(self._errors))
        if self.receiver and self.receiver.error:
            raise RuntimeError(f"Telemetry failed: {self.receiver.error}")
        if require_ready:
            now = time.monotonic()
            histories = dict(self.cameras)
            if not self.cameras_only:
                histories["telemetry"] = self.robot
            for name, history in histories.items():
                latest = history.latest()
                if latest is None or not -0.01 <= now - latest.timestamp <= (
                    self.config.sync.source_timeout_s
                ):
                    raise RuntimeError(f"{name}: no fresh data")

    def wait_ready(self, timeout: float = 12) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.check(require_ready=False)
            ready = all(history.latest() for history in self.cameras.values())
            if ready and (self.cameras_only or self.robot.latest()):
                self.check()
                return
            if self.stop.wait(0.02):
                raise RuntimeError("Source startup cancelled")
        raise TimeoutError("Waiting for three cameras and teleop telemetry timed out")

    def close(self) -> None:
        self.stop.set()
        if self.receiver:
            self.receiver.close()
        for thread in self._threads:
            thread.join(timeout=4)
        alive = [thread.name for thread in self._threads if thread.is_alive()]
        if alive:
            raise RuntimeError(f"Acquisition workers did not exit: {alive}")
