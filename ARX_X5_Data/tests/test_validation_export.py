import json
import os

import av
import numpy as np
import pytest

from arx_x5_data.export import convert
from arx_x5_data.validation import validate_episode

from .conftest import edit_jsonl


def boundary_feedback(rows):
    for row in rows:
        row["observation.state"][6] = -0.02
        row["observation.state"][13] = 1.02


@pytest.mark.parametrize(
    "change",
    [
        lambda rs: rs[1].update(grid_index=0),
        lambda rs: rs[1].update(timestamp_monotonic_s=0),
        lambda rs: rs[1]["cameras"]["head"].update(timestamp_monotonic_s=0),
        lambda rs: rs[1]["cameras"]["head"].update(capture_index=0),
        lambda rs: rs[1].update(feedback_oldest_monotonic_s=[0, 0]),
        lambda rs: rs[1]["action"].__setitem__(6, 5.0),
        lambda rs: rs[1]["observation.state"].__setitem__(6, -0.02001),
        lambda rs: rs[1]["observation.state"].__setitem__(13, 1.02001),
    ],
)
def test_invalid_timing_and_semantics_are_rejected(make_episode, change):
    path = make_episode()
    edit_jsonl(path, change)
    with pytest.raises(ValueError):
        validate_episode(path, allow_mock=True)


def test_mock_data_is_never_silently_accepted(make_episode):
    with pytest.raises(ValueError, match="Mock"):
        validate_episode(make_episode())


def test_corrupted_video_and_csv_are_rejected(make_episode):
    path = make_episode()
    (path / "head.mp4").write_bytes(b"broken")
    with pytest.raises(av.error.FFmpegError):
        validate_episode(path, allow_mock=True)
    path = make_episode()
    with (path / "action.csv").open("a") as stream:
        stream.write("0," + ",".join(["0"] * 14) + "\n")
    with pytest.raises(ValueError, match="counts"):
        validate_episode(path, allow_mock=True)


class FakeDataset:
    fail = False

    @classmethod
    def create(cls, **kwargs):
        (kwargs["root"] / "meta").mkdir(parents=True)
        return cls()

    def add_frame(self, frame):
        if self.fail:
            raise OSError("simulated disk failure")

    def save_episode(self, *, parallel_encoding):
        pass

    def finalize(self):
        pass


def test_failed_export_is_transactional_and_retryable(make_episode, tmp_path):
    path = make_episode((0, 1, 4, 5))
    original = (path / "samples.jsonl").read_bytes()
    target = tmp_path / "dataset"

    class Failing(FakeDataset):
        fail = True

    with pytest.raises(OSError, match="disk failure"):
        convert(path, target, "local/test", allow_mock=True, factory=Failing, mode="reencode_av1")
    assert not target.exists()
    assert not list(tmp_path.glob(".dataset.export-*"))
    assert not list(tmp_path.glob("*.lock"))
    result = convert(
        path, target, "local/test", allow_mock=True, factory=FakeDataset, mode="reencode_av1"
    )
    assert result["episodes"] == 2 and result["frames"] == 4
    assert original == (path / "samples.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        convert(
            path, target, "local/test", allow_mock=True, factory=FakeDataset, mode="reencode_av1"
        )


def test_boundary_feedback_survives_validation_and_export(make_episode, tmp_path):
    path = make_episode()
    edit_jsonl(path, boundary_feedback)
    frames = []

    class Inspecting(FakeDataset):
        def add_frame(self, frame):
            frames.append(frame)

    convert(
        path,
        tmp_path / "dataset",
        "local/test",
        allow_mock=True,
        factory=Inspecting,
        mode="reencode_av1",
    )
    assert len(frames) == 4
    assert frames[0]["observation.state"][[6, 13]] == pytest.approx([-0.02, 1.02])


def test_legacy_manifest_retains_strict_feedback_bounds(make_episode):
    path = make_episode()
    manifest = path / "manifest.json"
    meta = json.loads(manifest.read_text())
    meta["config"].pop("gripper_feedback_margin_native")
    manifest.write_text(json.dumps(meta))
    edit_jsonl(path, lambda rows: rows[0]["observation.state"].__setitem__(6, -0.001))
    with pytest.raises(ValueError, match="observation.state"):
        validate_episode(path, allow_mock=True)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("ARX_TEST_LEROBOT") != "1", reason="Opt-in real LeRobot test")
def test_official_lerobot_write_load_and_decode(make_episode, tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    path = make_episode((0, 1, 4, 5))
    edit_jsonl(path, boundary_feedback)
    target = tmp_path / "lerobot"
    result = convert(path, target, "local/arx-test", allow_mock=True, mode="reencode_av1")
    assert result["episodes"] == 2 and result["frames"] == 4
    dataset = LeRobotDataset("local/arx-test", root=target, video_backend="pyav")
    assert dataset.num_frames == 4 and dataset.num_episodes == 2
    first = dataset[0]
    assert first["observation.state"].shape == (14,)
    assert first["action"].shape == (14,)
    assert first["observation.images.head"].shape == (3, 48, 64)
    assert np.allclose(first["action"].numpy()[:6], 0.2)
    assert np.allclose(first["observation.state"].numpy()[:6], 0.1)
    assert np.allclose(first["observation.state"].numpy()[[6, 13]], [-0.02, 1.02])
    provenance = json.loads((target / "meta/arx_sources.json").read_text())
    assert len(provenance["episodes"]) == 2
