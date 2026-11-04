"""Shared latest-frame JPEG cache, independent of recording and page updates."""

import threading
import time
from collections import deque
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from .model import CAMERAS


@dataclass(frozen=True)
class PreviewFrame:
    sequence: int
    captured_at: float
    encoded_at: float
    jpeg: bytes


class Preview:
    """Encode each source frame once; slow clients skip to the latest cached frame."""

    def __init__(self, sources, fps: int = 30):
        self.sources, self.fps = sources, fps
        self._frames: dict[str, PreviewFrame] = {}
        self._times = {name: deque(maxlen=120) for name in CAMERAS}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.error = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name="preview-encoder", daemon=True)
        self._thread.start()

    def latest(self, name: str, *, fresh: bool = False) -> PreviewFrame | None:
        with self._lock:
            frame = self._frames.get(name)
        if fresh and frame is not None and not -0.01 <= time.monotonic() - frame.captured_at <= 0.5:
            return None
        return frame

    def stats(self) -> dict:
        now = time.monotonic()
        with self._lock:
            result = {}
            for name in CAMERAS:
                times = [t for t in self._times[name] if now - t < 2]
                frame = self._frames.get(name)
                result[name] = {
                    "fps": (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0,
                    "source_age_ms": (now - frame.captured_at) * 1000 if frame else None,
                }
            return result

    def _run(self):
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                for name in CAMERAS:
                    source = self.sources.cameras[name].latest()
                    previous = self.latest(name)
                    if source is None or (previous and source.sequence == previous.sequence):
                        continue
                    image = Image.fromarray(source.rgb)
                    image.thumbnail((800, 500))
                    stream = BytesIO()
                    image.save(stream, format="JPEG", quality=80)
                    now = time.monotonic()
                    frame = PreviewFrame(source.sequence, source.timestamp, now, stream.getvalue())
                    with self._lock:
                        self._frames[name] = frame
                        self._times[name].append(now)
                self._stop.wait(max(0, 1 / self.fps - (time.monotonic() - started)))
        except Exception as error:
            self.error = error

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
