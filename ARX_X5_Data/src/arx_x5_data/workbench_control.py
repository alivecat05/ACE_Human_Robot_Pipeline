"""Idempotent local recording operations for the operator workbench. No robot control."""

import json
import threading
from pathlib import Path


class RecordingControl:
    def __init__(self, runtime, journal):
        self.runtime = runtime
        self.journal = Path(journal)
        self.lock = threading.Lock()
        self.requests = json.loads(self.journal.read_text()) if self.journal.exists() else {}

    def persist(self):
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        temp = self.journal.with_suffix(".tmp")
        temp.write_text(json.dumps(self.requests, ensure_ascii=False))
        temp.replace(self.journal)

    def execute(self, action, body):
        ident = body.get("request_id")
        if not isinstance(ident, str) or not 8 <= len(ident) <= 100:
            raise ValueError("缺少有效操作编号")
        with self.lock:
            old = self.requests.get(ident)
            if old:
                if old["action"] != action:
                    raise ValueError("操作编号已用于另一操作")
                if "result" not in old:
                    raise RuntimeError("上次操作结果需核对，未重复执行")
                return old["result"]
            state = self.runtime.status()
            if state["mock"]:
                raise RuntimeError("工作台真实录制接口不接受模拟录制器")
            if action == "start":
                task = body.get("task")
                if not isinstance(task, str) or not task.strip() or len(task) > 100:
                    raise ValueError("请填写 1～100 字的任务名称")
                if state["mode"] != "idle":
                    raise RuntimeError("录制器未就绪：" + state["message"])
            elif action in ("stop", "cancel"):
                if state["mode"] != "recording" or state["episode"] != body.get("episode"):
                    raise RuntimeError("当前条目已变化，请核对状态，未重复结束录制")
            else:
                raise ValueError("未知操作")
            self.requests[ident] = {"action": action}
            self.persist()  # Reserve before mutation: restart never replays an uncertain operation.
            if action == "start":
                self.runtime.record(task.strip())
            else:
                if action == "cancel":
                    self.runtime.stop_recording(cancel=True)
                else:
                    self.runtime.stop_recording()
            result = self.runtime.status()
            self.requests[ident]["result"] = result
            self.persist()
            return result
