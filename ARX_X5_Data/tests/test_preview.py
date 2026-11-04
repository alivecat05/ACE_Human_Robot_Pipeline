import time
from io import BytesIO

import numpy as np
from PIL import Image

from arx_x5_data.preview import Preview
from arx_x5_data.sources import Sources


def test_preview_encodes_independently_without_modifying_acquisition(config):
    sources = Sources(config, mock=True, cameras_only=True)
    preview = Preview(sources)
    try:
        sources.start()
        sources.wait_ready()
        original = sources.cameras["head"].latest()
        rgb = original.rgb.copy()
        preview.start()
        deadline = time.monotonic() + 2
        while preview.latest("head") is None and time.monotonic() < deadline:
            time.sleep(0.01)
        frame = preview.latest("head")
        assert frame is not None
        assert Image.open(BytesIO(frame.jpeg)).size == (64, 48)
        assert np.array_equal(original.rgb, rgb)
        assert not original.rgb.flags.writeable
        assert preview.error is None
    finally:
        preview.close()
        sources.close()


def test_slow_stream_client_skips_to_current_frame(config):
    sources = Sources(config, mock=True, cameras_only=True)
    preview = Preview(sources)
    try:
        sources.start()
        sources.wait_ready()
        preview.start()
        deadline = time.monotonic() + 2
        while preview.latest("head") is None and time.monotonic() < deadline:
            time.sleep(0.01)
        first = preview.latest("head")
        time.sleep(0.3)  # Consumer stalls while acquisition and encoding continue.
        latest = preview.latest("head")
        assert latest.sequence >= first.sequence + 5
        assert latest.encoded_at > first.encoded_at
        assert latest.jpeg != first.jpeg
    finally:
        preview.close()
        sources.close()


def test_old_preview_is_not_served_as_current(config, monkeypatch):
    from types import SimpleNamespace

    from arx_x5_data.preview import PreviewFrame

    preview = Preview(Sources(config, mock=True, cameras_only=True))
    old = PreviewFrame(1, 100, 100.01, b"old")
    preview._frames["right_wrist"] = old
    monkeypatch.setattr("arx_x5_data.preview.time", SimpleNamespace(monotonic=lambda: 101))
    assert preview.latest("right_wrist") is old
    assert preview.latest("right_wrist", fresh=True) is None
    current = PreviewFrame(2, 100.98, 100.99, b"current")
    preview._frames["right_wrist"] = current
    assert preview.latest("right_wrist", fresh=True) is current


def test_large_preview_uses_800_by_500_without_modifying_source():
    from types import SimpleNamespace

    source = SimpleNamespace(
        sequence=1, timestamp=time.monotonic(), rgb=np.zeros((800, 1280, 3), dtype=np.uint8)
    )
    source.rgb.flags.writeable = False

    class History:
        def latest(self):
            return source

    sources = SimpleNamespace(
        cameras={name: History() for name in ("head", "left_wrist", "right_wrist")}
    )
    preview = Preview(sources)
    try:
        preview.start()
        deadline = time.monotonic() + 2
        while preview.latest("head") is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert Image.open(BytesIO(preview.latest("head").jpeg)).size == (800, 500)
        assert source.rgb.shape == (800, 1280, 3) and not source.rgb.flags.writeable
    finally:
        preview.close()
