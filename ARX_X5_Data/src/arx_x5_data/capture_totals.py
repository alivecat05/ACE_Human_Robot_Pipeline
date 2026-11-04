"""Saved, validated demonstration duration; no video decoding or robot access."""

import json
import threading
import time
from fractions import Fraction
from pathlib import Path


def capture_totals(root: Path) -> dict:
    duration = Fraction(0)
    count = 0
    for file in root.glob("*/episode_*/manifest.json"):
        try:
            meta = json.loads(file.read_text(encoding="utf-8"))
            if (
                (file.parent / ".workbench-deleted.json").exists()
                or meta.get("status") != "complete"
                or meta.get("mock") is not False
                or str(meta.get("task", "")).startswith("试采")
                or file.parent.parent.name.startswith("试采")
            ):
                continue
            frames, fps = meta.get("frames"), meta.get("fps")
            if type(frames) is not int or frames < 2 or type(fps) is not int or not 1 <= fps <= 30:
                continue
            duration += Fraction(frames, fps)
            count += 1
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    seconds = int(duration)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return {
        "seconds": float(duration),
        "episodes": count,
        "formatted": f"{hours:02d}:{minutes:02d}:{seconds_part:02d}",
    }


class CaptureTotals:
    def __init__(self, root: Path):
        self.root = root
        self._lock = threading.Lock()
        self._updated = float("-inf")
        self._value = {}

    def snapshot(self):
        with self._lock:
            now = time.monotonic()
            if now - self._updated >= 1:
                self._value = capture_totals(self.root)
                self._updated = now
            return dict(self._value)
