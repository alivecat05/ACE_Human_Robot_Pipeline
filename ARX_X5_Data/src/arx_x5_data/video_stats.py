"""Collect identical official samples during the mandatory source validation pass."""

import numpy as np

from .model import CAMERAS
from .validation import rows

MAX_STATS_BYTES = 64 * 1024**2


class ValidationStats:
    def __init__(self, path):
        from lerobot.datasets.compute_stats import sample_indices

        # Read only small journal rows to determine segment boundaries before decoding.
        segments, begin, previous, count = [], 0, None, 0
        for index, row in enumerate(rows(path)):
            if previous is not None and row["grid_index"] != previous + 1:
                if index - begin >= 2:
                    segments.append((begin, index))
                begin = index
            previous, count = row["grid_index"], index + 1
        if count - begin >= 2:
            segments.append((begin, count))
        self.segments = tuple(segments)
        self.wanted = {
            begin + i: segment
            for segment, (begin, end) in enumerate(segments)
            for i in sample_indices(end - begin)
        }
        self.images = [{name: [] for name in CAMERAS} for _ in segments]
        self.cached_bytes = 0

    def add(self, name, index, frame):
        if self.images is None or index not in self.wanted:
            return
        from lerobot.datasets.compute_stats import auto_downsample_height_width

        rgb = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
        small = auto_downsample_height_width(rgb).copy()
        self.cached_bytes += small.nbytes
        if self.cached_bytes > MAX_STATS_BYTES:
            self.images = None  # Fall back to sequential statistics for long episodes.
            return
        self.images[self.wanted[index]][name].append(small)

    def finish(self, segments):
        from lerobot.datasets.compute_stats import get_feature_stats

        if segments != self.segments:
            raise ValueError("Statistics segment plan differs from validated journal")
        if self.images is None:
            return None
        result = []
        for images in self.images:
            stats = {}
            for name, samples in images.items():
                values = get_feature_stats(np.stack(samples), axis=(0, 2, 3), keepdims=True)
                stats[f"observation.images.{name}"] = {
                    k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
                    for k, v in values.items()
                }
            result.append(stats)
        return tuple(result)
