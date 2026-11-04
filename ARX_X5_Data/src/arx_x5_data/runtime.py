"""Small recording state machine shared by the terminal and optional desktop UI."""

import threading
import time

from .config import Config
from .jobs import BackgroundJobs, retry_failed, summary
from .recording import EpisodeWriter
from .sources import Sources
from .synchronization import Synchronizer


class Runtime:
    def __init__(self, config: Config, *, mock: bool = False):
        self.config, self.mock = config, mock
        self.sources = Sources(config, mock=mock)
        self.jobs = BackgroundJobs(config.output)
        self.writer: EpisodeWriter | None = None
        self.mode, self.message = "warming", "Waiting for cameras and teleop telemetry"
        self._lock = threading.RLock()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._end: float | None = None
        self._cancel = False
        self._job_status_at = 0.0
        self._job_status = {}

    def start(self):
        self.sources.start()
        self.jobs.start()
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="recording-runtime", daemon=True)
        self._thread.start()
        return self

    def record(self, task: str) -> None:
        with self._lock:
            if self.mode != "idle":
                raise RuntimeError(f"Recorder is not ready: {self.mode}")
            self.sources.check()
            self._start, self._grid = time.monotonic(), 0
            self._end, self._cancel = None, False
            self._sync = Synchronizer(self.sources.cameras, self.sources.robot, self.config.sync)
            self._start = self._sync.grid_start(self._start, self.config.fps)
            self.writer = EpisodeWriter(
                self.config, task, self._start, mock=self.mock, background=True
            )
            self.mode, self.message = "recording", f"Recording {task}"

    def stop_recording(self, *, cancel: bool = False) -> None:
        with self._lock:
            if self.mode == "recording":
                self._end, self._cancel = time.monotonic(), cancel
                self.mode, self.message = "saving", "Draining matched samples and finalizing videos"

    def retry_background(self):
        retry_failed(self.config.output)
        if self.jobs.process is None or self.jobs.process.poll() is not None:
            if hasattr(self.jobs, "log"):
                self.jobs.log.close()
            self.jobs.start()

    def _can_recover(self) -> bool:
        if self.mode != "fault" or not self.writer or self.writer.busy:
            return False
        try:
            self.sources.check()
        except Exception:
            return False
        return self.writer.result == "incomplete"

    def recover(self) -> None:
        """Acknowledge a finalized failed take; never restart sources or teleop."""
        with self._lock:
            if not self._can_recover():
                raise RuntimeError("请等待本条写盘完成，并确认相机与遥操数据已恢复。")
            self.mode, self.message = "warming", "Failed take retained; checking inputs"

    def status(self) -> dict:
        with self._lock:
            writer = self.writer
            now = time.monotonic()

            def fresh(history) -> bool:
                sample = history.latest()
                return sample is not None and (
                    -0.01 <= now - sample.timestamp <= self.config.sync.source_timeout_s
                )

            if now - self._job_status_at >= 1:
                self._job_status = summary(self.config.output)
                self._job_status_at = now
            return {
                "background": self._job_status,
                "worker_alive": (
                    self.jobs.process is not None and self.jobs.process.poll() is None
                ),
                "mode": self.mode,
                "recoverable": self._can_recover(),
                "message": self.message,
                "mock": self.mock,
                "frames": writer.frames if writer else 0,
                "dropped": sum(writer.drops.values()) if writer else 0,
                "episode": str(writer.path) if writer else None,
                "result": writer.result if writer else None,
                "telemetry_rejections": (
                    {
                        "count": self.sources.receiver.rejected_packets,
                        "last": self.sources.receiver.last_rejection,
                    }
                    if self.sources.receiver
                    else {"count": 0, "last": ""}
                ),
                "inputs": {
                    "cameras": {
                        name: fresh(history) for name, history in self.sources.cameras.items()
                    },
                    "telemetry": fresh(self.sources.robot),
                },
            }

    def _tick(self, now: float) -> None:
        if self.mode in {"warming", "idle"}:
            self.sources.check(require_ready=False)
            inputs = self.status()["inputs"]
            cameras_ready = all(inputs["cameras"].values())
            if cameras_ready and inputs["telemetry"]:
                self.mode, self.message = "idle", "Three cameras and teleop telemetry ready"
            elif now - self._started > 12 and not all(
                history.latest() for history in self.sources.cameras.values()
            ):
                raise TimeoutError("Three cameras did not initialize within 12 seconds")
            else:
                self.mode = "warming"
                self.message = "Waiting for fresh cameras and teleop telemetry"
            return
        if self.mode not in {"recording", "saving"}:
            return
        self.sources.check()
        writer = self.writer
        if writer and writer.error:
            raise RuntimeError(f"Recording failed: {writer.error}")
        if writer.accepting:
            limit = now - self.config.sync.delay_ms / 1000
            if self._end is not None:
                limit = min(limit, self._end)
            while (stamp := self._start + self._grid / self.config.fps) <= limit:
                sample = self._sync.match(stamp, self._grid)
                if sample is None:
                    writer.dropped(self._sync.rejection)
                elif (
                    self._end is not None
                    and max(
                        sample.robot.timestamp,
                        *(frame.timestamp for frame in sample.cameras.values()),
                    )
                    > self._end
                ):
                    writer.dropped("after_operator_stop")
                else:
                    writer.submit(sample)
                self._grid += 1
            if self._end is not None and now >= (self._end + self.config.sync.delay_ms / 1000):
                writer.finish(
                    "cancelled" if self._cancel else "complete",
                    "operator_cancel" if self._cancel else "operator_stop",
                )
        if self.mode == "saving" and not writer.busy:
            if writer.result not in {"complete", "cancelled", "pending_validation"}:
                raise RuntimeError(f"Episode could not be completed: {writer.error}")
            self.mode, self.message = "idle", f"{writer.result}: {writer.path}"

    def _run(self):
        while not self._done.is_set():
            with self._lock:
                if self.mode != "fault":
                    try:
                        self._tick(time.monotonic())
                    except Exception as error:
                        self.mode, self.message = "fault", f"{type(error).__name__}: {error}"
                        if self.writer:
                            self.writer.finish("incomplete", self.message)
            self._done.wait(0.005)

    def close(self):
        self._done.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            if self.writer:
                self.writer.finish("incomplete", "closed_without_explicit_save")
                self.writer.wait()
        finally:
            try:
                self.jobs.close()
            finally:
                self.sources.close()
            self.mode = "closed"

    def __enter__(self):
        try:
            return self.start()
        except BaseException:
            self.close()
            raise

    def __exit__(self, *args):
        self.close()
