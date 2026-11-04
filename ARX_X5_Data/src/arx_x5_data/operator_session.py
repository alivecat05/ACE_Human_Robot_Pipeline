"""Shared primary-button workflow; reset confirms the scene and never commands motion."""

import json
import threading
import time
from pathlib import Path

from .recording import atomic_json


class OperatorSession:
    def __init__(self, runtime):
        self.runtime = runtime
        self.lock = threading.RLock()
        self.primary_until = 0.0
        self.path = runtime.config.output / ".operator-session.json"
        self.data = dict(needs_reset=False, task="数采任务", episode=None)
        if self.path.exists():
            self.data.update(json.loads(self.path.read_text()))

    def persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.path, self.data)

    def status(self):
        with self.lock:
            state = self.runtime.status()
            episode = state.get("episode")
            if episode and episode != self.data["episode"]:
                self.data.update(episode=episode, needs_reset=True)
                self.persist()
            current = episode or self.data["episode"]
            invalid = False
            incomplete = False
            if current:
                path = Path(current)
                try:
                    manifest = json.loads((path / "manifest.json").read_text())
                    incomplete = manifest.get("status") == "incomplete"
                    invalid = (
                        manifest.get("status") == "cancelled"
                        or (path / ".workbench-deleted.json").exists()
                    )
                except (OSError, ValueError):
                    pass
            return {
                **state,
                "needs_reset": self.data["needs_reset"],
                "last_episode": current,
                "invalid": invalid,
                "incomplete": incomplete,
                "task": self.data["task"],
            }

    def primary_action(self, state=None):
        state = self.status() if state is None else state
        if state["mode"] == "recording":
            return "stop"
        if state["mode"] == "fault" and state.get("recoverable"):
            return "recover"
        if state["mode"] in ("idle", "warming") and state["needs_reset"]:
            return "reset"
        if state["mode"] == "idle":
            return "start"
        return None

    def advance(self, expected, task):
        with self.lock:
            now = time.monotonic()
            if now < self.primary_until:
                return  # Ignore a double-click instead of triggering the next step.
            current = self.primary_action()
            if expected != current or current is None:
                raise RuntimeError("当前步骤已更新，请按按钮显示的操作继续。")
            self.primary_until = now + 0.7
            {
                "start": lambda: self.start(task),
                "stop": self.stop,
                "reset": self.reset,
                "recover": lambda: self.runtime.recover(),
            }[current]()

    def start(self, task):
        with self.lock:
            state = self.status()
            if state["mode"] != "idle":
                raise RuntimeError("设备尚未就绪，请查看相机与遥操连接状态。")
            if state["needs_reset"]:
                raise RuntimeError("请恢复场景后，先点击“场景 reset”。")
            task = (task or "").strip()
            if not task or len(task) > 100:
                raise ValueError("请填写 1～100 字的任务名称。")
            self.runtime.record(task)
            self.data.update(task=task, needs_reset=True, episode=self.runtime.status()["episode"])
            self.persist()

    def stop(self):
        with self.lock:
            if self.status()["mode"] != "recording":
                raise RuntimeError("当前没有正在录制的条目。")
            self.runtime.stop_recording()

    def reset(self):
        with self.lock:
            state = self.status()
            if state["mode"] not in ("idle", "warming") or not state["needs_reset"]:
                raise RuntimeError("请先结束本条并等待文件写盘完成。")
            self.data["needs_reset"] = False
            self.persist()

    def invalidate(self):
        with self.lock:
            state = self.status()
            if state["mode"] == "recording":
                self.runtime.stop_recording(cancel=True)
                return
            if (
                state["mode"] not in ("idle", "warming")
                or not state["last_episode"]
                or state["invalid"]
            ):
                raise RuntimeError("当前条目不能标记无效，请等待文件写盘完成。")
            path = Path(state["last_episode"]).resolve()
            if (
                not path.is_relative_to(self.runtime.config.output.resolve())
                or path == self.runtime.config.output.resolve()
            ):
                raise ValueError("条目不在本地数据目录内。")
            manifest = json.loads((path / "manifest.json").read_text())
            if manifest.get("status") not in (
                "complete",
                "pending_validation",
                "incomplete",
                "cancelled",
            ):
                raise RuntimeError("本条尚未写盘完成。")
            atomic_json(
                path / ".workbench-deleted.json",
                {"deleted_at": time.time(), "reason": "operator_invalid"},
            )
