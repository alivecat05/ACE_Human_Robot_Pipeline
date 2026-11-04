import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from arx_x5_data.h264_encoder import encode_h264, verify_decode_log


@pytest.mark.parametrize("change", ["count", "pts", "dimensions", "timebase"])
def test_hardware_decode_proof_rejects_missing_or_changed_frames(change):
    log = "config in time_base: 1/30\nn: 0 pts: 0 s:64x48\nn: 1 pts: 1 s:64x48\n"
    kwargs = dict(frames=2, fps=30, width=64, height=48)
    verify_decode_log(log, **kwargs)
    if change == "count":
        kwargs["frames"] = 3
    elif change == "pts":
        log = log.replace("pts: 1", "pts: 2")
    elif change == "dimensions":
        log = log.replace("64x48", "64x49")
    else:
        log = log.replace("config in time_base:", "missing:")
    with pytest.raises(ValueError):
        verify_decode_log(log, **kwargs)


def test_hardware_command_is_bounded_and_uses_fixed_device(tmp_path, monkeypatch):
    source, dest = tmp_path / "source.mp4", tmp_path / "dest.mp4"

    def run(args, **kwargs):
        assert args[:3] == [sys.executable, "-m", "arx_x5_data.h264_encoder"]
        assert args[args.index("-vaapi_device") + 1] == "/dev/dri/renderD128"
        assert args[args.index("-threads") + 1] == "2"
        assert args[args.index("-filter_threads") + 1] == "1"
        assert kwargs["check"] and kwargs["timeout"] == 600

    monkeypatch.setattr(subprocess, "run", run)
    assert encode_h264(source, dest, gop=8, threads=2, backend="vaapi")["backend"] == "vaapi"


def test_hardware_failure_discards_partial_before_software_retry(tmp_path, monkeypatch):
    source, dest = tmp_path / "source.mp4", tmp_path / "dest.mp4"

    def failed(*args, **kwargs):
        dest.write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, "ffmpeg")

    def software(source, target, **kwargs):
        assert not target.exists()
        target.write_bytes(b"software")

    monkeypatch.setattr(subprocess, "run", failed)
    monkeypatch.setitem(
        sys.modules,
        "lerobot.configs.video",
        SimpleNamespace(RGBEncoderConfig=lambda **kwargs: kwargs),
    )
    monkeypatch.setitem(
        sys.modules, "lerobot.datasets.video_utils", SimpleNamespace(reencode_video=software)
    )
    result = encode_h264(source, dest, gop=8, threads=2, backend="vaapi")
    assert result["backend"] == "software" and result["fallback_reason"]
    assert dest.read_bytes() == b"software"


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("ARX_TEST_LEROBOT") != "1", reason="Opt in")
def test_fused_stats_exactly_match_old_path_across_gaps(make_episode, tmp_path):
    from arx_x5_data.export import convert
    from arx_x5_data.validation import validate_episode

    source = make_episode((0, 1, 4, 5, 8))
    proof = validate_episode(source, allow_mock=True, _collect_stats=True)
    assert proof.segments == ((0, 2), (2, 4))
    for label, verified in (("old", None), ("fused", proof)):
        result = convert(
            source,
            tmp_path / label,
            "local/stats",
            allow_mock=True,
            mode="reuse_mp4",
            _validated_episode=verified,
        )
        if verified:
            assert result["metrics"]["statistics_reused"]
    old = (tmp_path / "old/meta/episodes").rglob("*.parquet")
    import pandas as pd

    a = pd.read_parquet(next(old))
    b = pd.read_parquet(next((tmp_path / "fused/meta/episodes").rglob("*.parquet")))
    columns = [c for c in a.columns if c.startswith("stats/")]
    assert columns
    for column in columns:
        for x, y in zip(a[column], b[column], strict=True):
            np.testing.assert_array_equal(x, y)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("ARX_TEST_VAAPI") != "1", reason="Requires local Intel GPU")
def test_real_hardware_segmented_export(make_episode, tmp_path):
    from arx_x5_data.export import convert

    result = convert(
        make_episode((0, 1, 4, 5)),
        tmp_path / "hardware",
        "local/hardware",
        allow_mock=True,
        h264_backend="vaapi",
    )
    assert result["episodes"] == 2 and result["frames"] == 4
    assert result["metrics"]["statistics_reused"]
    assert all(v["backend"] == "vaapi" for v in result["metrics"]["videos"])
    assert result["metrics"]["output_verifiers"] == ["vaapi"] * 3
    info = json.loads((tmp_path / "hardware/meta/info.json").read_text())
    assert info["features"]["observation.images.head"]["info"]["video.codec"] == "h264"


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("ARX_TEST_LEROBOT") != "1", reason="Opt in")
def test_stats_cache_has_fixed_memory_bound(make_episode, monkeypatch):
    from arx_x5_data.validation import validate_episode

    monkeypatch.setattr("arx_x5_data.video_stats.MAX_STATS_BYTES", 1)
    proof = validate_episode(make_episode(), allow_mock=True, _collect_stats=True)
    assert proof.camera_stats is None
