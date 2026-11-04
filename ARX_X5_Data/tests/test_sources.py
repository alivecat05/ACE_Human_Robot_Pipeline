import threading
import time
from fractions import Fraction
from types import SimpleNamespace

import av
import numpy as np
import pytest

from arx_x5_data.sources import capture_frames


class FakeContainer:
    def __init__(self, packets, *, format_name="yuyv422"):
        self.streams = SimpleNamespace(
            video=[
                SimpleNamespace(
                    codec_context=SimpleNamespace(
                        width=64, height=48, format=SimpleNamespace(name=format_name)
                    ),
                    average_rate=30,
                )
            ]
        )
        self.packets = packets
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def demux(self, **kwargs):
        yield from self.packets


def test_truncated_uvc_packet_is_discarded_without_decoding(config, monkeypatch):
    frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="rgb24")
    frame.pts, frame.time_base = int(time.monotonic() * 1_000_000), Fraction(1, 1_000_000)

    def fail():
        pytest.fail("Incomplete raw frames must not reach the decoder")

    packets = [
        SimpleNamespace(size=4, is_corrupt=False, decode=fail),
        SimpleNamespace(size=64 * 48 * 2, is_corrupt=True, decode=fail),
        SimpleNamespace(size=64 * 48 * 2, is_corrupt=False, decode=lambda: [frame]),
    ]
    container = FakeContainer(packets)
    monkeypatch.setattr("arx_x5_data.sources.av.open", lambda *a, **k: container)
    result = list(capture_frames(config.cameras["head"], threading.Event()))
    assert len(result) == 1
    assert result[0].discarded_invalid_packets == 2
    assert result[0].sequence == 2
    assert container.closed


def test_wrong_depth_or_ir_node_is_rejected(config, monkeypatch):
    container = FakeContainer([], format_name="gray16le")
    monkeypatch.setattr("arx_x5_data.sources.av.open", lambda *a, **k: container)
    with pytest.raises(ValueError, match="negotiated"):
        list(capture_frames(config.cameras["head"], threading.Event()))
    assert container.closed


def timed_packet(stamp):
    frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="rgb24")
    frame.pts, frame.time_base = int(stamp * 1_000_000), Fraction(1, 1_000_000)
    return SimpleNamespace(size=64 * 48 * 2, is_corrupt=False, decode=lambda: [frame])


def test_stale_buffer_is_drained_without_relabelling_its_timestamp(config, monkeypatch):
    container = FakeContainer([timed_packet(98), timed_packet(99.99)])
    monkeypatch.setattr("arx_x5_data.sources.av.open", lambda *a, **k: container)
    monkeypatch.setattr("arx_x5_data.sources.time", SimpleNamespace(monotonic=lambda: 100))
    frames = list(capture_frames(config.cameras["head"], threading.Event()))
    assert len(frames) == 1 and frames[0].timestamp == 99.99
    assert frames[0].sequence == 1 and frames[0].discarded_invalid_packets == 1


def test_persistently_stale_or_wrong_clock_still_fails_closed(config, monkeypatch):
    container = FakeContainer([timed_packet(98), timed_packet(98)])
    now = iter((100, 102.1))
    monkeypatch.setattr("arx_x5_data.sources.av.open", lambda *a, **k: container)
    monkeypatch.setattr("arx_x5_data.sources.time", SimpleNamespace(monotonic=lambda: next(now)))
    with pytest.raises(ValueError, match="remains stale"):
        list(capture_frames(config.cameras["head"], threading.Event()))
    container = FakeContainer([timed_packet(110)])
    monkeypatch.setattr("arx_x5_data.sources.time", SimpleNamespace(monotonic=lambda: 100))
    with pytest.raises(ValueError, match="not in the host monotonic clock"):
        list(capture_frames(config.cameras["head"], threading.Event()))
