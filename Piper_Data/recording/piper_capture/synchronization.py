"""Match RGB and robot feedback before any episode frames are written to disk."""
from itertools import product

import numpy as np

from .schema import ARMS, CAMERAS


class Synchronizer:
    def __init__(self, cameras, states, cfg, period=1/30):
        self.cameras, self.states, self.cfg = cameras, states, cfg
        self.period = period
        self.last_frames = {}
        self.rejection = None

    def _states(self, t, after=None):
        tolerance = self.cfg["state_tolerance_ms"] / 1000
        result = {}
        for side in ARMS:
            candidates = [x for x in self.states[side].snapshot()
                          if abs(x.t - t) <= tolerance
                          and all(abs(stamp - t) <= tolerance
                                  for stamp in x.value["source_timestamps"].values())
                          and (after is None or all(stamp > after
                               for field, stamp in x.value["source_timestamps"].items()
                               if field in ("tcp", "gripper") or field.startswith("end_pose_")))]
            result[side] = min(candidates, key=lambda x: abs(x.t - t), default=None)
        return result if all(result.values()) else None

    def _reject(self, reason):
        self.rejection = reason
        return None

    def sample(self, t):
        """Use an actual head-camera timestamp; never impose a wall-clock grid.

        Each saved group contains three different, previously unused frames.
        The pairwise span is the hard limit; there is no extra grid-phase gate.
        """
        self.rejection = None
        snapshots = {name: self.cameras[name].snapshot() for name in CAMERAS}
        span_limit = self.cfg["camera_span_ms"] / 1000
        reference = CAMERAS[0]
        near = [x for x in snapshots[reference] if abs(x.t - t) <= span_limit]
        if not near:
            return self._reject(f"camera_missing:{reference}")
        available = [x for x in near if x.value["frame_number"] > self.last_frames.get(reference, -1)]
        if not available:
            return self._reject(f"camera_reused:{reference}")
        anchor = min(available, key=lambda x: abs(x.t - t))
        t = anchor.t
        candidates = {}
        for name in CAMERAS:
            near = [anchor] if name == reference else [x for x in snapshots[name]
                                                       if abs(x.t - t) <= span_limit]
            if not near:
                return self._reject(f"camera_span:{name}" if snapshots[name] else f"camera_missing:{name}")
            candidates[name] = [x for x in near if x.value["frame_number"] > self.last_frames.get(name, -1)]
            if not candidates[name]:
                return self._reject(f"camera_reused:{name}")
        # Filter by the physical timing limit first. Prefer continuing frame
        # sequences, then proximity to the reference, instead of greedily taking
        # a later frame just because it makes a slightly tighter group.
        def score(group):
            stamps = [x.t for x in group]
            skipped = sum(x.value["frame_number"] - self.last_frames.get(n, x.value["frame_number"]-1) - 1
                          for n, x in zip(CAMERAS, group))
            return skipped, sum(abs(stamp-t) for stamp in stamps), max(stamps)-min(stamps)
        valid = [group for group in product(*(candidates[n] for n in CAMERAS))
                 if max(x.t for x in group) - min(x.t for x in group) <= span_limit]
        if not valid:
            return self._reject("camera_span")
        group = min(valid, key=score)
        frames = dict(zip(CAMERAS, group))
        states = self._states(t)
        if states is None:
            return self._reject("state_missing_or_stale")
        future = self._states(t + self.period, after=t)
        if future is None:
            return self._reject("action_missing_or_stale")
        vector = [v for side in ARMS for v in states[side].value["vector"]]
        action = [v for side in ARMS for v in future[side].value["vector"]]
        for offset in (0, 8):
            if np.dot(action[offset+3:offset+7], vector[offset+3:offset+7]) < 0:
                action[offset+3:offset+7] = [-v for v in action[offset+3:offset+7]]
        ids = {n: x.value["frame_number"] for n, x in frames.items()}
        self.last_frames = ids
        return {
            "timestamp_monotonic_s": t,
            "observation.state": vector,
            "action": action,
            "action_timestamp_monotonic_s": t + self.period,
            "joints": {side: states[side].value["joints"] for side in ARMS},
            "source_timestamps": {
                "cameras": {n: x.t for n, x in frames.items()},
                "states": {n: x.value["source_timestamps"] for n, x in states.items()},
                "actions": {n: x.value["source_timestamps"] for n, x in future.items()},
            },
            "camera_metadata": {n: {k: v for k, v in x.value.items() if k != "bgr"} for n, x in frames.items()},
            "images": {n: x.value["bgr"] for n, x in frames.items()},
        }
