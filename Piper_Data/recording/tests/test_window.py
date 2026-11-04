import json
from pathlib import Path
import time
from types import SimpleNamespace
import tkinter as tk

import pytest

from piper_capture.app import CaptureWindow, RECORDING_GREEN, STOPPED_RED, WAITING_AMBER
from piper_capture.capture import Capture
from test_capture import cfg, wait_mode


@pytest.fixture(scope="module", autouse=True)
def require_display():
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk display unavailable: {error}")
    else:
        root.destroy()


def dispose(window):
    if not window.closed:
        window.root.after_cancel(window.timer)
        window.root.destroy()


def test_reconnect_button_and_camera_waiting_indicator(cfg):
    with Capture(cfg, mock=True) as cap:
        wait_mode(cap, "idle")
        window = CaptureWindow(cap)
        window.root.withdraw()
        try:
            old = cap.cameras
            window.buttons["d"].invoke()
            wait_mode(cap, "idle")
            assert cap.cameras is not old
            window.buttons["r"].invoke()
            window.refresh()
            assert window.buttons["d"].instate(["disabled"])
            state = cap.status()
            state["camera_waiting"] = True
            window.update_capture_indicator(state)
            assert window.capture_indicator.cget("background") == WAITING_AMBER
            assert "等待恢复" in window.capture_indicator.cget("text")
            cap.stop()
            wait_mode(cap, "idle")
        finally:
            dispose(window)


def test_native_buttons_record_save_set_home_return_cancel(cfg, tmp_path):
    cfg["topology"] = "independent"
    with Capture(cfg, mock=True, home_path=tmp_path/"home.json") as cap:
        wait_mode(cap, "idle")
        window = CaptureWindow(cap)
        window.root.withdraw()
        try:
            assert cap.status()["mode"] == "idle", cap.status()
            assert window.capture_indicator.cget("background") == STOPPED_RED
            assert len(window.buttons) == 7
            window.buttons["g"].invoke()
            assert "请先" in window.result_text.get()
            window.buttons["h"].invoke()
            assert cap.home_path.exists()
            window.task_name.set("把杯子放入托盘")
            window.capture_index.set("12")
            for key in ("R", "S", "G", "H", "C", "Q"):
                window.on_key(SimpleNamespace(keysym=key, widget=window.task_entry))
            assert cap.status()["mode"] == "idle"
            assert not window.exit_requested
            window.buttons["r"].invoke()
            assert cap.status()["recording"]
            assert window.capture_indicator.cget("background") == RECORDING_GREEN
            assert "正在采集" in window.capture_indicator.cget("text")
            assert str(window.task_entry.cget("state")) == "disabled"
            assert str(window.capture_index_entry.cget("state")) == "disabled"
            window.buttons["h"].invoke()
            assert "未执行" in window.result_text.get()
            time.sleep(.35)
            window.buttons["s"].invoke()
            assert window.capture_indicator.cget("background") == STOPPED_RED
            assert "正在保存" in window.capture_indicator.cget("text")
            wait_mode(cap, "idle")
            assert cap.status()["frames"] > 0
            meta = json.loads((Path(cap.status()["episode"])/"manifest.json").read_text(encoding="utf-8"))
            assert Path(cap.status()["episode"]).parts[-2:] == ("把杯子放入托盘", "episode_012")
            assert "task" not in meta
            window.buttons["g"].invoke()
            assert cap.status()["mode"] == "returning"
            window.buttons["c"].invoke()
            assert cap.status()["mode"] == "idle"
            window.buttons["s"].invoke()
            assert "当前没有录制" in window.result_text.get()
        finally:
            dispose(window)


def test_uppercase_key_and_exit_button_save_before_closing(cfg, tmp_path):
    with Capture(cfg, mock=True, home_path=tmp_path/"home.json") as cap:
        wait_mode(cap, "idle")
        window = CaptureWindow(cap)
        window.root.withdraw()
        try:
            window.on_key(SimpleNamespace(keysym="R"))
            assert cap.status()["recording"]
            time.sleep(.35)
            episode = Path(cap.status()["episode"])
            window.buttons["q"].invoke()
            assert window.exit_requested and not window.closed
            deadline = time.perf_counter() + 3
            while not window.closed and time.perf_counter() < deadline:
                window.root.update()
                time.sleep(.01)
            assert window.closed
            assert json.loads((episode/"manifest.json").read_text())["status"] == "complete"
            assert not any(robot.commands for robot in cap.robots.values())
        finally:
            dispose(window)


def test_shared_can_hides_home_buttons_and_ignores_their_shortcuts(cfg, tmp_path):
    cfg["topology"] = "leader_follower_shared_can"
    with Capture(cfg, mock=True, home_path=tmp_path/"home.json") as cap:
        wait_mode(cap, "idle")
        cap.mock = False
        window = CaptureWindow(cap)
        window.root.withdraw()
        try:
            assert set(window.buttons) == {"r", "s", "c", "d", "q"}
            for key in ("G", "H"):
                window.on_key(SimpleNamespace(keysym=key))
            assert not cap.home_path.exists()
            assert cap.status()["mode"] == "idle"
            assert not any(robot.commands for robot in cap.robots.values())
        finally:
            dispose(window)


def test_pedal_records_saves_ignores_saving_and_advances_episode(cfg, tmp_path, monkeypatch):
    class FakePedal:
        def __init__(self, *args):
            self.events = [("status", "脚踏已连接")]
            self.closed = False

        def start(self):
            return self

        def poll(self):
            events, self.events = self.events, []
            return events

        def close(self):
            self.closed = True

    monkeypatch.setattr("piper_capture.app.RawPedal", FakePedal)
    cfg["foot_pedal"]["enabled"] = True
    with Capture(cfg, mock=True, home_path=tmp_path/"home.json") as cap:
        wait_mode(cap, "idle")
        window = CaptureWindow(cap)
        window.root.withdraw()
        listener = window.pedal
        try:
            assert "脚踏已连接" in window.pedal_text.get()
            window.task_name.set("脚踏录制")
            window.task_entry.focus_set()
            window.on_key(SimpleNamespace(keysym="F8"))
            assert not cap.status()["recording"]
            listener.events.append(("press", 119))
            window.poll_pedal()
            assert cap.status()["recording"]
            assert window.capture_indicator.cget("background") == RECORDING_GREEN
            assert "正在采集" in window.capture_indicator.cget("text")
            episode = Path(cap.status()["episode"])
            assert episode.name == "episode_001"
            time.sleep(.35)
            listener.events.append(("press", 119))
            window.poll_pedal()
            assert cap.status()["saving"]
            assert window.capture_indicator.cget("background") == STOPPED_RED
            assert "正在保存" in window.capture_indicator.cget("text")
            listener.events.append(("press", 119))
            window.poll_pedal()
            assert "正在保存" in window.result_text.get()
            wait_mode(cap, "idle")
            meta = json.loads((episode/"manifest.json").read_text(encoding="utf-8"))
            assert meta["status"] == "complete"
            assert meta["frames"] > 0
            listener.events.append(("press", 119))
            window.poll_pedal()
            assert cap.status()["recording"]
            assert Path(cap.status()["episode"]).name == "episode_002"
            window.request_close()
            listener.events.append(("press", 119))
            window.poll_pedal()
            assert window.exit_requested
            assert not any(robot.commands for robot in cap.robots.values())
        finally:
            dispose(window)
        assert listener.closed


@pytest.mark.parametrize("mode", ["warming", "returning", "fault"])
def test_pedal_cannot_record_before_ready_or_during_return_or_fault(cfg, tmp_path, mode):
    with Capture(cfg, mock=True, home_path=tmp_path/"home.json") as cap:
        wait_mode(cap, "idle")
        window = CaptureWindow(cap)
        window.root.withdraw()
        try:
            with cap.lock:
                cap.mode = mode
                window.toggle_recording()
                window.update_capture_indicator(cap.status())
                assert cap.mode == mode
                assert window.capture_indicator.cget("background") == STOPPED_RED
                assert "脚踏未执行" in window.result_text.get()
                cap.mode = "idle"
            assert not any(robot.commands for robot in cap.robots.values())
        finally:
            dispose(window)


@pytest.mark.parametrize("scenario", ["missing_config", "disabled", "absent", "start_error",
                                     "disconnect", "error_event", "poll_error", "close_error"])
def test_gui_buttons_record_and_save_when_pedal_is_unavailable(cfg, tmp_path, monkeypatch, scenario):
    listeners = []

    class UnavailablePedal:
        def __init__(self, *args):
            self.events = [("status", "脚踏未连接" if scenario == "absent" else "脚踏已连接")]
            self.closed = False
            self.fail_poll = False
            listeners.append(self)

        def start(self):
            if scenario == "start_error":
                raise OSError("测试：输入接口不可用")
            return self

        def poll(self):
            if self.fail_poll:
                raise OSError("测试：读取脚踏失败")
            events, self.events = self.events, []
            return events

        def close(self):
            self.closed = True
            if scenario == "close_error":
                raise OSError("测试：关闭脚踏失败")

    monkeypatch.setattr("piper_capture.app.RawPedal", UnavailablePedal)
    cfg["foot_pedal"]["enabled"] = scenario != "disabled"
    if scenario == "missing_config":
        cfg.pop("foot_pedal")
    with Capture(cfg, mock=True, home_path=tmp_path/"home.json") as cap:
        wait_mode(cap, "idle")
        window = CaptureWindow(cap)
        window.root.withdraw()
        try:
            if scenario in ("missing_config", "disabled"):
                assert not listeners  # No native listener is created.
                assert window.pedal is None
            elif scenario == "start_error":
                assert "脚踏不可用" in window.pedal_text.get()
                assert window.pedal is None and listeners[0].closed
            elif scenario == "absent":
                assert "脚踏未连接" in window.pedal_text.get()
                assert "界面按钮" in window.pedal_text.get()

            assert "disabled" not in window.buttons["r"].state()
            window.buttons["r"].invoke()
            assert cap.status()["recording"]
            episode = Path(cap.status()["episode"])

            if scenario == "disconnect":
                listeners[0].events.append(("status", "脚踏未连接"))
            elif scenario == "error_event":
                listeners[0].events.append(("error", "测试：脚踏监听退出"))
            elif scenario in ("poll_error", "close_error"):
                listeners[0].fail_poll = True
            # Use the real GUI refresh path: a pedal failure must not stop it.
            window.root.update()
            time.sleep(.25)
            window.root.update()
            assert cap.status()["recording"]
            assert window.timer is not None
            if scenario in ("poll_error", "close_error"):
                assert window.pedal is None and listeners[0].closed
                assert "脚踏不可用" in window.pedal_text.get()

            assert "disabled" not in window.buttons["s"].state()
            window.buttons["s"].invoke()
            wait_mode(cap, "idle")
            window.root.update()
            meta = json.loads((episode/"manifest.json").read_text(encoding="utf-8"))
            assert meta["status"] == "complete"
            assert meta["frames"] > 0
            assert window.capture_index.get() == "2"
            assert not any(robot.commands for robot in cap.robots.values())
        finally:
            dispose(window)
        assert all(listener.closed for listener in listeners)


def test_gui_buttons_with_real_listener_and_no_matching_usb_pedal(cfg, tmp_path):
    cfg["foot_pedal"].update(enabled=True, device_id="VID_FFFF&PID_FFFF")
    with Capture(cfg, mock=True, home_path=tmp_path/"home.json") as cap:
        wait_mode(cap, "idle")
        window = CaptureWindow(cap)
        window.root.withdraw()
        try:
            assert "脚踏未连接" in window.pedal_text.get() or "脚踏不可用" in window.pedal_text.get()
            window.buttons["r"].invoke()
            assert cap.status()["recording"]
            time.sleep(.25)
            window.buttons["s"].invoke()
            wait_mode(cap, "idle")
            episode = Path(cap.status()["episode"])
            meta = json.loads((episode/"manifest.json").read_text(encoding="utf-8"))
            assert meta["status"] == "complete" and meta["frames"] > 0
        finally:
            dispose(window)
