import json
import time

import pytest

from arx_x5_data.jobs import retry_failed, run_one, summary
from arx_x5_data.recording import EpisodeWriter
from arx_x5_data.validation import validate_episode

from .conftest import sample


def pending(config):
    stamp = time.monotonic()
    writer = EpisodeWriter(config, "background", stamp, mock=True, background=True)
    for i in range(4):
        writer.submit(sample(config, stamp + i / config.fps, i, i))
    writer.finish()
    writer.wait(3)
    assert writer.result == "pending_validation"
    return writer.path


def test_pending_is_not_exportable_and_failed_validation_can_retry(config, monkeypatch):
    path = pending(config)
    with pytest.raises(ValueError, match="pending_validation"):
        validate_episode(path, allow_mock=True)
    original = (path / "head.mp4").read_bytes()
    (path / "head.mp4").write_bytes(b"broken")
    run_one(path / "job.json")
    assert summary(config.output)["counts"] == {"failed": 1}
    assert json.loads((path / "manifest.json").read_text())["status"] == "pending_validation"
    (path / "head.mp4").write_bytes(original)
    retry_failed(config.output)
    assert summary(config.output)["counts"] == {"queued": 1}
    monkeypatch.setattr("arx_x5_data.export.convert", lambda *a, **k: {"metrics": {}})
    run_one(path / "job.json")
    assert summary(config.output)["counts"] == {"done": 1}
    validate_episode(path, allow_mock=True)


@pytest.mark.integration
@pytest.mark.skipif(__import__("os").environ.get("ARX_TEST_LEROBOT") != "1", reason="Opt in")
def test_reuse_gap_mapping_and_pixels(make_episode, tmp_path):
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from arx_x5_data.export import convert
    from arx_x5_data.media import decoded_frames
    from arx_x5_data.model import CAMERAS

    source = make_episode((0, 1, 4, 5))
    target = tmp_path / "reuse"
    convert(source, target, "local/reuse", allow_mock=True, mode="reuse_mp4")
    dataset = LeRobotDataset("local/reuse", root=target, video_backend="pyav")
    assert dataset.num_frames == 4 and dataset.num_episodes == 2
    for name in CAMERAS:
        copied = next((target / "videos" / f"observation.images.{name}").rglob("*.mp4"))
        assert copied.read_bytes() == (source / f"{name}.mp4").read_bytes()
        for i, image in enumerate(decoded_frames(source / f"{name}.mp4", 64, 48, 30)):
            actual = dataset[i][f"observation.images.{name}"].numpy()
            assert np.max(np.abs(actual - image.transpose(2, 0, 1) / 255)) < 1e-6


@pytest.mark.integration
@pytest.mark.skipif(__import__("os").environ.get("ARX_TEST_LEROBOT") != "1", reason="Opt in")
@pytest.mark.parametrize("gop", [2, 8, 15])
def test_direct_h264_alignment_without_png(make_episode, tmp_path, monkeypatch, gop):
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from arx_x5_data.export import convert
    from arx_x5_data.media import decoded_frames
    from arx_x5_data.model import CAMERAS

    source = make_episode((0, 1, 4, 5))
    target = tmp_path / "direct"

    def no_png(*args, **kwargs):
        raise AssertionError("Direct H.264 path must never write PNG images")

    monkeypatch.setattr(LeRobotDataset, "add_frame", no_png)
    result = convert(
        source, target, "local/direct", allow_mock=True, mode="reencode_h264", h264_gop=gop
    )
    assert result["frames"] == 4 and result["episodes"] == 2
    dataset = LeRobotDataset("local/direct", root=target, video_backend="pyav")
    assert not list(target.rglob("*.png"))
    assert len(result["metrics"]["videos"]) == 3
    for name in CAMERAS:
        copied = next((target / "videos" / f"observation.images.{name}").rglob("*.mp4"))
        assert len(list(decoded_frames(copied, 64, 48, 30))) == 4
        for i, expected in enumerate(decoded_frames(copied, 64, 48, 30)):
            actual = dataset[i][f"observation.images.{name}"].numpy()
            assert np.max(np.abs(actual - expected.transpose(2, 0, 1) / 255)) < 1e-6
        assert np.allclose(dataset[2]["action"].numpy()[:6], 0.2)
    assert result["metrics"]["encoding_s"] > 0


@pytest.mark.integration
@pytest.mark.skipif(__import__("os").environ.get("ARX_TEST_LEROBOT") != "1", reason="Opt in")
def test_actual_background_default_uses_official_main_h264(config, monkeypatch):
    monkeypatch.delenv("ARX_EXPORT_MODE", raising=False)
    path = pending(config)
    run_one(path / "job.json")
    job = json.loads((path / "job.json").read_text())
    assert job["state"] == "done", job
    assert job["metrics"]["mode"] == "reencode_h264"
    assert job["metrics"]["lerobot_version"] == "0.6.2"
    assert (path.parents[1] / "lerobot" / path.parent.name / path.name).exists()
