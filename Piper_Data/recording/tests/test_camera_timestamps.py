import pytest

from piper_capture.cameras import frame_timestamp


@pytest.mark.parametrize("domain", ["global_time", "system_time"])
def test_sdk_host_clock_uses_frame_time_not_python_receive(domain):
    stamp, basis = frame_timestamp(1_700_000_000_000, f"timestamp_domain.{domain}",
                                  receive=100.07, wall_offset=100-1_700_000_000)
    assert stamp == 100
    assert basis == domain


def test_unmapped_hardware_clock_is_not_treated_as_unix():
    stamp, basis = frame_timestamp(1234, "timestamp_domain.hardware_clock", 100,
                                  wall_offset=100-1_700_000_000, latency_ms=10)
    assert stamp == pytest.approx(99.99)
    assert basis == "host_receive"


def test_invalid_host_clock_mapping_is_rejected():
    with pytest.raises(RuntimeError, match="timestamp mapping"):
        frame_timestamp(1234, "system_time", 100, wall_offset=100-1_700_000_000)


def test_sdk_wait_timeout_does_not_kill_camera_worker(monkeypatch):
    import sys
    from types import SimpleNamespace
    import numpy as np
    from piper_capture.cameras import Cameras

    cameras = Cameras({"cameras": {"head": {"serial": "test", "model": "D555",
                      "width": 64, "height": 48, "fps": 30, "latency_ms": 0}}})
    frame = SimpleNamespace(get_frame_timestamp_domain=lambda: "hardware_clock",
                            get_timestamp=lambda: 1234, get_frame_number=lambda: 100,
                            get_data=lambda: np.zeros((48, 64, 3), dtype=np.uint8))
    class Pipe:
        calls = 0
        stopped = False
        def start(self, cfg):
            return SimpleNamespace(get_device=lambda: SimpleNamespace(
                get_info=lambda key: "D555", query_sensors=lambda: []))
        def wait_for_frames(self, timeout):
            return None  # Warm-up only.
        def try_wait_for_frames(self, timeout):
            self.calls += 1
            if self.calls <= 2:
                return False, None
            return True, SimpleNamespace(get_color_frame=lambda: frame)
        def stop(self):
            self.stopped = True
    pipe = Pipe()
    rs = SimpleNamespace(pipeline=lambda ctx: pipe, config=lambda: SimpleNamespace(
        enable_device=lambda *args: None, enable_stream=lambda *args: None),
        stream=SimpleNamespace(color=1), format=SimpleNamespace(bgr8=1),
        camera_info=SimpleNamespace(name=1))
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    original = cameras.histories["head"].add
    def add(t, value):
        original(t, value)
        cameras.stop_event.set()
    monkeypatch.setattr(cameras.histories["head"], "add", add)
    cameras._worker("head")
    assert not cameras.errors
    assert pipe.calls == 3 and pipe.stopped
    assert cameras.histories["head"].latest().value["frame_number"] == 100
