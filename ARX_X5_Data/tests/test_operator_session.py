import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arx_x5_data.operator_session import OperatorSession


class Recorder:
    def __init__(self, root):
        self.config = SimpleNamespace(output=root)
        self.mode = "idle"
        self.episode = None
        self.stops = []
        self.starts = 0

    def status(self):
        return dict(mode=self.mode, episode=self.episode)

    def record(self, task):
        self.starts += 1
        path = self.config.output / task / f"episode_{self.starts:06}"
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(json.dumps({"status": "recording"}))
        self.episode = str(path)
        self.mode = "recording"

    def stop_recording(self, cancel=False):
        self.mode = "saving"
        self.stops.append(cancel)

    def finish(self, status="pending_validation"):
        (Path(self.episode) / "manifest.json").write_text(json.dumps({"status": status}))
        self.mode = "idle"


def test_next_take_requires_reset_even_after_page_or_service_refresh(tmp_path):
    runtime = Recorder(tmp_path)
    session = OperatorSession(runtime)
    session.start("task")
    session.stop()
    runtime.finish()
    reopened = OperatorSession(runtime)
    with pytest.raises(RuntimeError, match="场景 reset"):
        reopened.start("task")
    reopened.reset()
    reopened.start("task")
    assert runtime.starts == 2
    assert reopened.status()["needs_reset"]


def test_duplicate_start_and_reset_during_recording_are_rejected(tmp_path):
    runtime = Recorder(tmp_path)
    session = OperatorSession(runtime)
    session.start("task")
    with pytest.raises(RuntimeError):
        session.start("task")
    with pytest.raises(RuntimeError):
        session.reset()
    assert runtime.starts == 1


def test_invalid_recording_uses_cancel_without_touching_teleop(tmp_path):
    runtime = Recorder(tmp_path)
    session = OperatorSession(runtime)
    session.start("task")
    session.invalidate()
    assert runtime.stops == [True]
    runtime.finish("cancelled")
    assert session.status()["invalid"]
    assert session.status()["needs_reset"]
    session.reset()
    session.start("task")
    assert runtime.starts == 2


def test_saved_invalid_marks_only_current_episode_and_preserves_files(tmp_path):
    runtime = Recorder(tmp_path)
    session = OperatorSession(runtime)
    session.start("task")
    session.stop()
    runtime.finish()
    path = Path(runtime.episode)
    raw = path / "head.mp4"
    raw.write_bytes(b"original")
    session.invalidate()
    assert (path / ".workbench-deleted.json").exists()
    assert raw.read_bytes() == b"original"
    assert session.status()["invalid"]
    assert runtime.stops == [False]


def test_saving_blocks_reset_and_invalidation(tmp_path):
    runtime = Recorder(tmp_path)
    session = OperatorSession(runtime)
    session.start("task")
    session.stop()
    with pytest.raises(RuntimeError):
        session.reset()
    with pytest.raises(RuntimeError):
        session.invalidate()


def test_background_does_not_gate_next_take(tmp_path):
    runtime = Recorder(tmp_path)
    session = OperatorSession(runtime)
    session.start("task")
    session.stop()
    runtime.finish()
    (Path(runtime.episode) / "job.json").write_text(json.dumps({"state": "queued"}))
    session.reset()
    session.start("task")
    assert runtime.starts == 2


def test_primary_cycles_without_auto_start_and_ignores_double_click(tmp_path, monkeypatch):
    runtime = Recorder(tmp_path)
    session = OperatorSession(runtime)
    clock = [10.0]
    monkeypatch.setattr("arx_x5_data.operator_session.time.monotonic", lambda: clock[0])
    assert session.primary_action() == "start"
    session.advance("start", "task")
    assert session.primary_action() == "stop"
    session.advance("stop", "task")  # A second click in the same burst must not stop the new take.
    assert runtime.mode == "recording" and runtime.stops == []
    clock[0] += 1
    session.advance("stop", "task")
    assert runtime.mode == "saving" and session.primary_action() is None
    runtime.finish()
    clock[0] += 1
    assert session.primary_action() == "reset"
    session.advance("reset", "task")
    assert runtime.starts == 1 and session.primary_action() == "start"
    clock[0] += 1
    session.advance("start", "task")
    assert runtime.starts == 2


def test_old_primary_action_cannot_operate_a_new_stage(tmp_path):
    runtime = Recorder(tmp_path)
    session = OperatorSession(runtime)
    session.start("task")
    with pytest.raises(RuntimeError, match="步骤已更新"):
        session.advance("start", "task")
    assert runtime.mode == "recording" and runtime.starts == 1
