"""Sequential RGB video I/O with explicit dimensions, frame order and finalization."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np


class VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int, crf: int, preset: str):
        self.width, self.height = width, height
        self.frame_time_base = Fraction(1, fps)
        self.container = av.open(
            str(path), "w", options={"movflags": "frag_keyframe+empty_moov+default_base_moof"}
        )
        try:
            self.stream = self.container.add_stream("libx264", rate=fps)
            self.stream.width, self.stream.height = width, height
            self.stream.pix_fmt = "yuv422p"
            self.stream.time_base = self.frame_time_base
            self.stream.options = {"crf": str(crf), "preset": preset, "bf": "0"}
            self.index = 0
        except BaseException:
            self.container.close()
            raise

    def write(self, rgb: np.ndarray) -> None:
        if rgb.dtype != np.uint8 or rgb.shape != (self.height, self.width, 3):
            raise ValueError("RGB shape or dtype does not match the recording configuration")
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        # The MP4 muxer may change stream.time_base after the first packet.
        # Frame timestamps always stay on the original recording grid.
        frame.pts, frame.time_base = self.index, self.frame_time_base
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.index += 1

    def close(self) -> None:
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


def decoded_frames(path: Path, width: int, height: int, fps: int, *, rgb: bool = True):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if (stream.width, stream.height) != (width, height):
            raise ValueError(f"{path}: wrong video dimensions")
        for index, frame in enumerate(container.decode(video=0)):
            if frame.pts is None or abs(float(frame.pts * frame.time_base) - index / fps) > 0.002:
                raise ValueError(f"{path}: incorrect video presentation timestamp")
            if (frame.width, frame.height) != (width, height):
                raise ValueError(f"{path}: video frame dimensions changed")
            yield frame.to_ndarray(format="rgb24") if rgb else frame
