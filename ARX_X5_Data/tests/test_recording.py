import json
import time

import pytest

from arx_x5_data.recording import EpisodeWriter, reserve_episode
from arx_x5_data.runtime import Runtime
from arx_x5_data.validation import validate_episode

from .conftest import sample


def test_idle_input_gap_waits_and_recovers_without_locking_ui(config, monkeypatch):
    runtime = Runtime(config, mock=True)
    runtime._started = 10.0
    clock = [10.0]
    monkeypatch.setattr("arx_x5_data.runtime.time.monotonic", lambda: clock[0])

    def add_inputs(stamp, sequence):
        aligned = sample(config, stamp, sequence, sequence)
        for name, frame in aligned.cameras.items():
            runtime.sources.cameras[name].add(frame)
        runtime.sources.robot.add(aligned.robot)

    add_inputs(10.0, 0)
    runtime._tick(clock[0])
    assert runtime.mode == "idle"
    clock[0] = 11.0
    runtime._tick(clock[0])
    assert runtime.mode == "warming"
    assert runtime.writer is None
    add_inputs(11.0, 1)
    runtime._tick(clock[0])
    assert runtime.mode == "idle"


def test_video_csv_journal_round_trip_and_gap_split(make_episode):
    path = make_episode((0, 1, 4, 5))
    episode = validate_episode(path, allow_mock=True)
    assert episode.segments == ((0, 2), (2, 4))
    assert episode.manifest["frames"] == 4


def test_existing_task_data_is_never_overwritten(config):
    first = reserve_episode(config.output, "拿起杯子")
    (first / "sentinel").write_bytes(b"keep")
    second = reserve_episode(config.output, "拿起杯子")
    assert first != second
    assert (first / "sentinel").read_bytes() == b"keep"


@pytest.mark.parametrize("task", ["../outside", ".", "..", "", "bad/name", "bad\\name"])
def test_invalid_task_path_is_rejected(config, task):
    with pytest.raises(ValueError):
        reserve_episode(config.output, task)


def test_encoder_error_cannot_mark_recording_complete(config, monkeypatch):
    class BrokenEncoder:
        def __init__(self, *args):
            raise OSError("disk unavailable")

    monkeypatch.setattr("arx_x5_data.recording.VideoWriter", BrokenEncoder)
    writer = EpisodeWriter(config, "test", time.monotonic(), mock=True)
    writer.wait(2)
    assert writer.result == "incomplete"
    meta = json.loads((writer.path / "manifest.json").read_text())
    assert meta["status"] == "incomplete" and "disk unavailable" in meta["error"]


def test_cancelled_data_is_retained_but_cannot_be_exported(make_episode):
    path = make_episode(status="cancelled")
    assert (path / "head.mp4").exists()
    with pytest.raises(ValueError, match="cancelled"):
        validate_episode(path, allow_mock=True)


def test_immediate_cancel_is_a_valid_operator_outcome(config):
    writer = EpisodeWriter(config, "cancel", time.monotonic(), mock=True)
    writer.finish("cancelled")
    writer.wait(2)
    assert writer.result == "cancelled", writer.error
    assert writer.path.exists()


def wait_mode(runtime, mode):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = runtime.status()
        if state["mode"] == mode:
            return
        if state["mode"] == "fault":
            pytest.fail(state["message"])
        time.sleep(0.01)
    pytest.fail(str(runtime.status()))


def test_mock_runtime_lifecycle_and_read_only_validation(config):
    with Runtime(config, mock=True) as runtime:
        wait_mode(runtime, "idle")
        runtime.record("模拟示教")
        time.sleep(0.35)
        runtime.stop_recording()
        wait_mode(runtime, "idle")
        path = runtime.writer.path
        assert runtime.writer.result == "pending_validation", runtime.writer.error
    episode = validate_episode(path, allow_mock=True, _pending=True)
    assert episode.manifest["frames"] >= 3


def test_close_without_save_marks_incomplete(config):
    with Runtime(config, mock=True) as runtime:
        wait_mode(runtime, "idle")
        runtime.record("interrupted")
        time.sleep(0.25)
        path = runtime.writer.path
    assert json.loads((path / "manifest.json").read_text())["status"] == "incomplete"


def test_writer_rejects_out_of_order_grid(config):
    stamp = time.monotonic()
    writer = EpisodeWriter(config, "order", stamp, mock=True)
    try:
        writer.submit(sample(config, stamp, 0, 0))
        with pytest.raises(ValueError, match="increase"):
            writer.submit(sample(config, stamp, 0, 1))
    finally:
        writer.finish("incomplete")
        writer.wait(2)


def test_zero_frame_failure_can_recover_and_save_next_take(config, monkeypatch):
    from arx_x5_data.operator_session import OperatorSession

    with Runtime(config, mock=True) as runtime:
        wait_mode(runtime, "idle")
        session = OperatorSession(runtime)
        session.start("failed-and-recovered")
        monkeypatch.setattr(runtime._sync, "match", lambda *args: None)
        time.sleep(0.15)
        session.stop()
        deadline = time.monotonic() + 5
        while runtime.status()["mode"] != "fault" and time.monotonic() < deadline:
            time.sleep(0.01)
        failed = runtime.writer.path
        assert runtime.mode == "fault"
        runtime.writer.wait(2)
        assert json.loads((failed / "manifest.json").read_text())["status"] == "incomplete"
        assert session.primary_action() == "recover"
        assert session.status()["incomplete"]
        session.advance("recover", "failed-and-recovered")
        wait_mode(runtime, "idle")
        assert session.primary_action() == "reset"
        session.reset()
        session.start("failed-and-recovered")
        time.sleep(0.35)
        session.stop()
        wait_mode(runtime, "idle")
        assert runtime.writer.path != failed
        assert runtime.writer.result == "pending_validation"
        assert runtime.writer.frames >= 2
        assert json.loads((failed / "manifest.json").read_text())["status"] == "incomplete"


def test_recovery_rejects_busy_writer_or_failed_sources(config):
    from types import SimpleNamespace

    runtime = Runtime(config, mock=True)
    runtime.mode = "fault"
    runtime.writer = SimpleNamespace(busy=True, result="incomplete")
    assert not runtime._can_recover()
    with pytest.raises(RuntimeError):
        runtime.recover()
    runtime.writer.busy = False
    runtime.sources._errors["head"] = "camera disconnected"
    with pytest.raises(RuntimeError):
        runtime.recover()
    assert runtime.mode == "fault"
