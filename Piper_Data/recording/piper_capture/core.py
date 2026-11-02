"""Thread-safe timestamp histories and robot pose conversion."""
from collections import deque
from dataclasses import dataclass
from threading import Lock

from scipy.spatial.transform import Rotation

from .schema import ACTION_SEMANTICS, ARMS, CAMERAS, VECTOR_NAMES


@dataclass(frozen=True)
class Sample:
    t: float
    value: object


class History:
    def __init__(self, capacity=240):
        self.items = deque(maxlen=capacity)
        self.lock = Lock()

    def add(self, t, value):
        with self.lock:
            if not self.items or t > self.items[-1].t:
                self.items.append(Sample(t, value))

    def snapshot(self):
        with self.lock:
            return list(self.items)

    def latest(self):
        items = self.snapshot()
        return items[-1] if items else None

    def fps(self, now):
        items = [x for x in self.snapshot() if now - 2 <= x.t <= now]
        if len(items) < 2 or now - items[-1].t > .5:
            return 0.
        return (len(items) - 1) / (items[-1].t - items[0].t)


def pose_to_vector(pose, width):
    return [*pose[:3], *Rotation.from_euler("xyz", pose[3:]).as_quat(), width]
