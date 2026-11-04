import json
from pathlib import Path
import threading
import time

import pytest

from piper_capture.capture import Capture
from piper_capture.core import CAMERAS
from test_capture import cfg, wait_mode


def wait_until(predicate, timeout=4):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(.01)
    pytest.fail("Timed out waiting for expected state")


def pause_camera(cap, monkeypatch, name="right_wrist"):
    paused = threading.Event()
    history = cap.cameras.histories[name]
    original = history.add

    def add(t, value):
        if not paused.is_set():
            original(t, value)

    monkeypatch.setattr(history, "add", add)
    return paused


def test_thirty_episodes_with_transient_stalls_preserve_sync_and_release_writers(cfg, monkeypatch):
    with Capture(cfg, mock=True) as cap:
        wait_mode(cap, "idle")
        paused = pause_camera(cap, monkeypatch)
        workers = tuple(cap.cameras.threads)
        for index in range(1, 31):
            cap.record(task="continuous", capture_index=index)
            wait_until(lambda: cap.recorder.count >= 3)
            if index % 6 == 0:
                paused.set()
                wait_until(lambda: cap.status()["camera_waiting"])
                assert cap.status()["mode"] == "recording"
                time.sleep(.1)
                paused.clear()
                wait_until(lambda: not cap.status()["camera_waiting"])
            cap.stop()
            wait_mode(cap, "idle")
            assert not cap.recorder.busy
            assert cap.recorder.queue.empty()
            meta = json.loads((cap.recorder.path / "manifest.json").read_text(encoding="utf-8"))
            assert meta["status"] == "complete"
            rows = [json.loads(line) for line in (cap.recorder.path / "samples.jsonl").read_text().splitlines()]
            for name in CAMERAS:
                ids = [row["camera_metadata"][name]["frame_number"] for row in rows]
                assert len(ids) == len(set(ids))
            for row in rows:
                stamps = list(row["source_timestamps_unix_s"]["cameras"].values())
                assert max(stamps)-min(stamps) <= cfg["sync"]["camera_span_ms"]/1000 + 1e-6
                t = row["timestamp_unix_s"]
                for side in ("left", "right"):
                    state = row["source_timestamps_unix_s"]["states"][side]["tcp"]
                    action = row["source_timestamps_unix_s"]["actions"][side]["tcp"]
                    assert abs(state-t) <= .04 + 1e-6
                    assert abs(action-(t+1/30)) <= .04 + 1e-6
                    assert action > t
            assert tuple(cap.cameras.threads) == workers
        assert all(worker.is_alive() for worker in workers)
        assert not any(robot.commands for robot in cap.robots.values())
    assert not any(worker.is_alive() for worker in workers)


def test_persistent_gap_finishes_incomplete_then_recovers_without_autorecord(cfg, monkeypatch):
    cfg["camera_recovery_s"] = .15
    with Capture(cfg, mock=True) as cap:
        wait_mode(cap, "idle")
        paused = pause_camera(cap, monkeypatch)
        cap.record()
        wait_until(lambda: cap.recorder.count >= 3)
        path = cap.recorder.path
        paused.set()
        wait_until(lambda: cap.status()["mode"] == "fault")
        assert "right_wrist" in cap.status()["message"]
        paused.clear()
        wait_until(lambda: cap.status()["mode"] == "idle")
        assert not cap.recorder.active
        assert json.loads((path / "manifest.json").read_text())["status"] == "incomplete"
        events = [json.loads(line) for line in (Path(cfg["output"])/"capture_events.jsonl").read_text(encoding="utf-8").splitlines()]
        assert {e["event"] for e in events} >= {"camera_stall", "fault", "camera_recovered"}
        cap.record()
        assert cap.recorder.path != path
        cap.stop()
        wait_mode(cap, "idle")
        assert not any(robot.commands for robot in cap.robots.values())


def test_stop_during_gap_does_not_hang(cfg, monkeypatch):
    with Capture(cfg, mock=True) as cap:
        wait_mode(cap, "idle")
        paused = pause_camera(cap, monkeypatch)
        cap.record()
        wait_until(lambda: cap.recorder.count >= 3)
        paused.set()
        wait_until(lambda: cap.status()["camera_waiting"])
        cap.stop()
        wait_mode(cap, "idle")
        assert not cap.recorder.busy
        paused.clear()


def test_reconnect_replaces_faulted_devices_and_can_record_again(cfg):
    with Capture(cfg, mock=True) as cap:
        wait_mode(cap, "idle")
        old_cameras, old_robots = cap.cameras, cap.robots
        old_workers = list(old_cameras.threads)
        cap.cameras.errors["head"] = "injected SDK failure"
        wait_until(lambda: cap.status()["mode"] == "fault")
        cap.reconnect()
        wait_mode(cap, "idle")
        assert cap.cameras is not old_cameras
        assert not any(t.is_alive() for t in old_workers)
        assert not any(r.commands for r in old_robots.values())
        cap.record()
        with pytest.raises(RuntimeError, match="保存"):
            cap.reconnect()
        wait_until(lambda: cap.recorder.count >= 3)
        cap.stop()
        wait_mode(cap, "idle")
        assert cap.recorder.result == "complete"
        assert not any(r.commands for r in cap.robots.values())


def test_close_during_reconnect_leaves_no_workers(cfg):
    cap = Capture(cfg, mock=True).open()
    wait_mode(cap, "idle")
    old_workers = list(cap.cameras.threads)
    cap.reconnect()
    cap.close()
    assert cap.mode == "closed"
    assert not cap.reconnect_thread.is_alive()
    assert not any(t.is_alive() for t in old_workers + cap.cameras.threads)
    assert cap.thread is None or not cap.thread.is_alive()


def test_failed_reconnect_can_be_retried_and_repeated_without_old_workers(cfg, monkeypatch):
    import piper_capture.capture as module
    factory = module.Cameras
    def failing_factory(*args):
        raise RuntimeError("injected reconnect failure")
    with Capture(cfg, mock=True) as cap:
        wait_mode(cap, "idle")
        monkeypatch.setattr(module, "Cameras", failing_factory)
        cap.reconnect()
        wait_until(lambda: cap.status()["mode"] == "fault")
        cap.reconnect_thread.join()
        assert "injected reconnect failure" in cap.status()["message"]
        monkeypatch.setattr(module, "Cameras", factory)
        for _ in range(12):
            old_workers = [cap.thread, *cap.cameras.threads]
            cap.reconnect()
            wait_mode(cap, "idle")
            cap.reconnect_thread.join()
            assert not any(worker.is_alive() for worker in old_workers)
            assert not any(robot.commands for robot in cap.robots.values())
