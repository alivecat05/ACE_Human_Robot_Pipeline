import json

import pytest

from arx_x5_data.capture_totals import capture_totals


def episode(root, task, index, *, status="complete", mock=False, frames=30, fps=30):
    path = root / task / f"episode_{index:06d}"
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(
        json.dumps(dict(task=task, status=status, mock=mock, frames=frames, fps=fps))
    )
    return path


def test_sums_frames_without_multiplying_cameras_or_counting_reset_time(tmp_path):
    episode(tmp_path, "a", 1, frames=731)
    episode(tmp_path, "b", 1, frames=108000)
    total = capture_totals(tmp_path)
    assert total["episodes"] == 2
    assert total["seconds"] == pytest.approx(3600 + 731 / 30)
    assert total["formatted"] == "01:00:24"


def test_excludes_trials_invalid_unvalidated_mock_and_bad_manifests(tmp_path):
    episode(tmp_path, "a", 1)
    invalid = episode(tmp_path, "a", 2)
    (invalid / ".workbench-deleted.json").write_text("{}")
    episode(tmp_path, "试采_a", 1)
    episode(tmp_path, "mock", 1, mock=True)
    for i, status in enumerate(("recording", "pending_validation", "incomplete", "cancelled"), 3):
        episode(tmp_path, "a", i, status=status)
    bad = episode(tmp_path, "a", 7)
    (bad / "manifest.json").write_text("broken")
    assert capture_totals(tmp_path) == dict(seconds=1.0, episodes=1, formatted="00:00:01")


def test_fractional_durations_are_added_before_rounding(tmp_path):
    for i in range(15):
        episode(tmp_path, "a", i, frames=2)
    assert capture_totals(tmp_path)["formatted"] == "00:00:01"
