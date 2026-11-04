"""Small capture API. One worker owns device polling and home motion."""
import copy
import json
from pathlib import Path
import threading
import time

import numpy as np

from .cameras import Cameras
from .core import ARMS, CAMERAS, History
from .synchronization import Synchronizer
from .recording import Recorder
from .robot import MockPiperX, PiperX

SHARED_CAN = "leader_follower_shared_can"
HOME_UNAVAILABLE = ""


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    output = Path(cfg["output"])
    cfg["output"] = str(output if output.is_absolute() else path.parent / output)
    if cfg.get("topology", "independent") not in ("independent", SHARED_CAN):
        raise ValueError("Unknown robot topology")
    if set(cfg["arms"]) != set(ARMS) or set(cfg["cameras"]) != set(CAMERAS):
        raise ValueError("Configure two arms and three cameras")
    if not 1 <= cfg["record_hz"] <= 60 or not 30 <= cfg["poll_hz"] <= 200:
        raise ValueError("Invalid recording/poll frequency")
    if not 0 < cfg.get("camera_recovery_s", 2.0) <= 10:
        raise ValueError("camera_recovery_s must be in (0, 10]")
    sync = cfg["sync"]
    if min(sync.values()) <= 0 or sync["delay_ms"] < max(
            sync["camera_span_ms"], 1000/cfg["record_hz"] + sync["state_tolerance_ms"]):
        raise ValueError("Sync delay must cover RGB matching and next-step action tolerance")
    for arm in cfg["arms"].values():
        if not 1 <= arm["home_speed_percent"] <= 20:
            raise ValueError("Home speed must be between 1 and 20 percent")
    return cfg


class Capture:
    def __init__(self, cfg, *, mock=False, home_path="home.json", cameras=None, robots=None):
        self.cfg, self.mock = copy.deepcopy(cfg), mock
        self.home_path = Path(home_path)
        self.cameras = cameras if cameras is not None else Cameras(cfg, mock)
        self.robots = robots if robots is not None else {}
        self.states = {s: History() for s in ARMS}
        self.recorder = Recorder(cfg["output"], self.cfg, mock)
        self.action_rate = History()
        self.lock, self.done = threading.RLock(), threading.Event()
        self.thread = None
        self.mode, self.message = "warming", "Waiting for cameras and robot feedback"
        self.end_time = None
        self.home_job = None
        self.commanded = set()
        self.camera_waiting = False
        self.camera_stall_since = None
        self.camera_recovered_since = None
        self.recover_camera_fault = False
        self.reconnect_thread = None
        self.closing = False

    @property
    def shared_can(self):
        return self.cfg.get("topology") == SHARED_CAN

    @property
    def home_available(self):
        return self.mock or not self.shared_can

    def open(self):
        try:
            factory = MockPiperX if self.mock else PiperX
            for side in ARMS:
                if side not in self.robots:
                    self.robots[side] = factory(self.cfg["arms"][side])
            self.cameras.start()
            self.started = time.perf_counter()
            self.thread = threading.Thread(target=self._run, daemon=True, name="piper-capture")
            self.thread.start()
            return self
        except Exception:
            self._close_devices()
            raise

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()

    def _fresh_states(self, now):
        result = {}
        for side in ARMS:
            item = self.states[side].latest()
            if item is None or any(not -.05 <= now - t <= self.cfg["feedback_timeout_s"]
                                   for t in item.value["source_timestamps"].values()):
                return None
            result[side] = item.value
        return result

    def _cameras_ready(self, now):
        return not self.cameras.errors and all(
            (item := self.cameras.histories[n].latest()) is not None
            and -.05 <= now - item.t <= self.cfg["camera_timeout_s"] for n in CAMERAS)

    def _camera_diagnostics(self, now):
        result = {}
        for name in CAMERAS:
            item = self.cameras.histories[name].latest()
            result[name] = {
                "frame_age_s": now-item.t if item else None,
                "receive_age_s": now-item.value.get("receive_s", item.t) if item else None,
                "frame_number": item.value.get("frame_number") if item else None,
                "error": self.cameras.errors.get(name),
            }
        return result

    def _event(self, event, reason):
        # Keep diagnostics even when no episode is recording; logging must
        # never interrupt acquisition (e.g. when the disk itself is full).
        try:
            path = Path(self.cfg["output"]) / "capture_events.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size > 1_000_000:
                path.replace(path.with_suffix(".previous.jsonl"))
            row = {"time_unix_s": time.time(), "event": event, "reason": reason,
                   "mode": self.mode, "episode": str(self.recorder.path),
                   "cameras": self._camera_diagnostics(time.perf_counter())}
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _require_idle(self):
        if self.mode != "idle" or self.recorder.busy:
            raise RuntimeError(f"Not ready: {self.mode}")
        states = self._fresh_states(time.perf_counter())
        if states is None:
            raise RuntimeError("Robot feedback is stale")
        return states

    def record(self, *, task=None, capture_index=None):
        with self.lock:
            self._require_idle()
            task = self.cfg["task"] if task is None else task
            if not isinstance(task, str) or not task.strip():
                raise ValueError("请填写任务名")
            if not self._cameras_ready(time.perf_counter()):
                raise RuntimeError("Cameras are not ready")
            task = task.strip()
            self.sync = Synchronizer(self.cameras.histories, self.states, self.cfg["sync"],
                                     1 / self.cfg["record_hz"])
            self.action_rate = History()
            self.start_time, self.grid, self.end_time = time.perf_counter(), 0, None
            self.reference_first_number = self.reference_last_number = None
            self.recorder.start(self.start_time, task=task, capture_index=capture_index)
            self.cfg["task"] = task
            self.mode, self.message = "recording", "Recording manual demonstration"
            return self.status()

    def stop(self):
        with self.lock:
            if self.mode == "recording":
                self.end_time = time.perf_counter()
                self.mode, self.message = "saving", "Aligning final frames and saving"
            return self.status()

    def _home_identity(self):
        return {s: {k: self.cfg["arms"][s][k] for k in ("interface", "channel", "firmware", "tcp_offset")}
                for s in ARMS}

    def set_home(self):
        with self.lock:
            states = self._require_idle()
            arms = {s: {"joints": states[s]["joints"], "gripper_m": states[s]["vector"][7]} for s in ARMS}
            for s in ARMS:
                self.robots[s].validate_home(arms[s]["joints"], arms[s]["gripper_m"])
            saved = {"simulated": self.mock, "devices": self._home_identity(), "arms": arms,
                     "scope": "followers_only" if self.shared_can else "two_independent_arms"}
            self.home_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.home_path.with_suffix(".tmp")
            temp.write_text(json.dumps(saved, indent=2), encoding="utf-8")
            temp.replace(self.home_path)
            self.message = ("Follower reference poses saved; leader poses are not included" if self.shared_can
                            else "Initial joint poses and gripper widths saved")
            return self.status()

    def go_home(self):
        with self.lock:
            if not self.home_available:
                raise RuntimeError(HOME_UNAVAILABLE)
            self._require_idle()
            home = json.loads(self.home_path.read_text(encoding="utf-8"))
            if home["simulated"] != self.mock or home["devices"] != self._home_identity():
                raise ValueError("Home belongs to different devices or simulation mode; save it again")
            for s in ARMS:
                self.robots[s].validate_home(home["arms"][s]["joints"], home["arms"][s]["gripper_m"])
            self.home_targets = home["arms"]
            self.home_queue, self.home_job = list(ARMS), None
            self.commanded.clear()
            self.mode, self.message = "returning", "Returning left then right; C cancels"
            return self.status()

    def cancel_home(self):
        with self.lock:
            if self.mode == "returning":
                states = self._fresh_states(time.perf_counter())
                try:
                    if states is None:
                        raise RuntimeError("Cannot hold: feedback is stale")
                    for side in self.commanded:
                        self.robots[side].hold(states[side])
                    self.commanded.clear()
                    self.home_job = None
                    self.mode, self.message = "idle", "Return cancelled; arms hold their positions"
                except Exception as error:
                    self._fault(str(error))
            return self.status()

    def _home_tick(self, states, now):
        if self.home_job is None:
            if not self.home_queue:
                self.commanded.clear()
                self.mode, self.message = "idle", "At home; use the robot's teaching control before dragging"
                return
            side = self.home_queue.pop(0)
            self.commanded.add(side)
            self.home_job = {"side": side, "stage": "enabling", "start": now, "settled": None}
            self.robots[side].hold(states[side])  # Replace old targets before enabling motors.
        job = self.home_job
        side, robot, target = job["side"], self.robots[job["side"]], self.home_targets[job["side"]]
        if now - job["start"] > self.cfg["home_timeout_s"]:
            raise TimeoutError(f"{side}: home motion timed out")
        if job["stage"] == "enabling":
            if robot.enable():
                robot.move_home(target)
                job["stage"] = "moving"
            elif now - job["start"] > 5:
                raise TimeoutError(f"{side}: enable timed out")
            return
        reached = (np.max(np.abs(np.array(states[side]["joints"]) - target["joints"])) < .03
                   and abs(states[side]["vector"][7] - target["gripper_m"]) < .003)
        job["settled"] = (job["settled"] or now) if reached else None
        if job["settled"] is not None and now - job["settled"] >= .3:
            self.home_job = None

    def _record_tick(self, now):
        limit = now - self.cfg["sync"]["delay_ms"] / 1000
        if self.end_time is not None:
            limit = min(limit, self.end_time)
        # Crop to the streams' common time interval and use the reference
        # camera's actual timestamps. Clicking at a different clock phase must
        # not change which otherwise valid RGB groups can be recorded.
        histories = {n: self.cameras.histories[n].snapshot() for n in CAMERAS}
        if not all(histories.values()):
            return
        overlap_start = max(self.start_time, max(items[0].t for items in histories.values()))
        overlap_end = min(limit, min(items[-1].t for items in histories.values()))
        for frame in histories[CAMERAS[0]]:
            t, number = frame.t, frame.value["frame_number"]
            if t < overlap_start or t > overlap_end:
                continue
            if self.reference_last_number is not None and number <= self.reference_last_number:
                continue
            if self.reference_first_number is None:
                self.reference_first_number = number
            self.reference_last_number = number
            index = int((number-self.reference_first_number) * self.cfg["record_hz"]
                        / self.cfg["cameras"][CAMERAS[0]]["fps"])
            if index < self.grid:
                continue  # Intentional downsampling when record_hz is lower.
            if index > self.grid:
                self.recorder.drop("reference_frame_missing", index-self.grid)
            self.grid = index
            if self.end_time is not None and t + 1/self.cfg["record_hz"] > self.end_time:
                self.recorder.drop("episode_end")
                self.grid += 1
                continue
            sample = self.sync.sample(t)
            if sample is None:
                self.recorder.drop(self.sync.rejection)
            else:
                self.recorder.add(sample, self.grid)
                self.action_rate.add(t, self.grid)
            self.grid += 1
        if self.end_time is not None and now >= self.end_time + self.cfg["sync"]["delay_ms"] / 1000:
            self.recorder.stop("operator_stop", end_time=self.end_time)

    def _fault(self, reason, *, recover_camera=False):
        self.recover_camera_fault = recover_camera
        self.camera_recovered_since = None
        self.mode, self.message = "fault", reason
        self._event("fault", reason)
        self.recorder.stop(reason, complete=False, end_time=time.perf_counter())
        for side in list(self.commanded):
            try:
                self.robots[side].stop()
            except Exception as error:
                self.message += f"; {side} stop failed: {error}"
        self.commanded.clear()

    def _tick(self, now):
        for side, robot in self.robots.items():
            sample = robot.read()
            if sample is not None:
                t, value = sample
                if (np.asarray(value["vector"]).shape != (8,) or not np.isfinite(value["vector"]).all()
                        or np.asarray(value["joints"]).shape != (6,) or not np.isfinite(value["joints"]).all()):
                    raise ValueError(f"{side}: invalid robot feedback")
                self.states[side].add(t, value)
        # Polling or OS scheduling can take time. Do not compare newly received
        # frames with the time captured before polling and call them "future".
        now = time.perf_counter()
        states = self._fresh_states(now)
        camera_ready = self._cameras_ready(now)
        if self.mode == "fault":
            # Only a passive camera watchdog fault is recoverable here. Never
            # resume an interrupted episode or auto-resume commanded motion.
            if states is not None and camera_ready and not self.recorder.busy and not self.recorder.error:
                if self.camera_recovered_since is None:
                    self.camera_recovered_since = now
                if now-self.camera_recovered_since >= .5:
                    self.camera_waiting = self.recover_camera_fault = False
                    self.camera_stall_since = None
                    self.mode, self.message = "idle", "相机已恢复；中断片段保留，请开始新一段采集"
                    self._event("camera_recovered", self.message)
            else:
                self.camera_recovered_since = None
            return
        if self.mode == "warming":
            if states is not None and camera_ready:
                self.mode, self.message = "idle", "Ready; manual teaching mode is controlled on the robot"
            else:
                waiting = [f"{s}: {getattr(self.robots[s], 'feedback_issue', '') or 'waiting for fresh feedback'}"
                           for s in ARMS if (item := self.states[s].latest()) is None or
                           any(not -.05 <= now-t <= self.cfg["feedback_timeout_s"]
                               for t in item.value["source_timestamps"].values())]
                if not camera_ready:
                    waiting.append(f"cameras: {self.cameras.errors or 'warming up'}")
                self.message = "Waiting | " + " | ".join(waiting)
                if now - self.started > 12:
                    raise TimeoutError(self.message)
            return
        if states is None:
            detail = "; ".join(f"{s}: {getattr(r, 'feedback_issue', '')}" for s, r in self.robots.items()
                               if getattr(r, "feedback_issue", ""))
            raise TimeoutError("Robot feedback timed out" + (f"; {detail}" if detail else ""))
        if self.recorder.error:
            raise RuntimeError(self.recorder.error)
        if self.cameras.errors or (not camera_ready and self.mode == "returning"):
            raise TimeoutError(f"Camera feedback timed out: {self._camera_diagnostics(now)}")
        if not camera_ready:
            if self.camera_stall_since is None:
                self.camera_stall_since = now
                self._event("camera_stall", "Waiting for fresh camera frames")
            self.camera_waiting = True
            self.message = f"相机暂时缺帧，等待恢复：{self._camera_diagnostics(now)}"
            if now-self.camera_stall_since >= self.cfg.get("camera_recovery_s", 2.0):
                self._fault(f"Camera feedback timed out: {self._camera_diagnostics(now)}",
                            recover_camera=True)
                return
        elif self.camera_waiting:
            self.camera_waiting = False
            self.camera_stall_since = None
            self.message = "相机已恢复，继续按原始时间戳采集"
            self._event("camera_resumed", self.message)
        if self.mode == "returning":
            self._home_tick(states, now)
        elif self.mode in ("recording", "saving"):
            if self.recorder.active:
                self._record_tick(now)
                # Saving must finish even if a camera never produced a frame.
                if self.end_time is not None and now >= self.end_time + self.cfg["sync"]["delay_ms"] / 1000:
                    self.recorder.stop("operator_stop", end_time=self.end_time)
            if self.mode == "saving" and not self.recorder.busy:
                self.mode = "idle"
                self.message = f"{self.recorder.result}: {self.recorder.path}"

    def _run(self):
        while not self.done.is_set():
            start = time.perf_counter()
            with self.lock:
                if self.mode != "fault" or self.recover_camera_fault:
                    try:
                        self._tick(start)
                    except Exception as error:
                        self._fault(f"{type(error).__name__}: {error}")
            self.done.wait(max(0, 1 / self.cfg["poll_hz"] - (time.perf_counter() - start)))

    def status(self):
        with self.lock:
            now = time.perf_counter()
            return {"mode": self.mode, "message": self.message,
                    "home_available": self.home_available,
                    "home_unavailable_reason": "" if self.home_available else HOME_UNAVAILABLE,
                    "recording": self.mode == "recording", "saving": self.mode == "saving",
                    "camera_waiting": self.camera_waiting,
                    "frames": self.recorder.count, "dropped": self.recorder.dropped,
                    "drop_reasons": dict(self.recorder.drop_reasons),
                    "episode": str(self.recorder.path) if self.recorder.path else None,
                    "fps": {"cameras": {n: self.cameras.histories[n].fps(now) for n in CAMERAS},
                            "feedback": {s: self.states[s].fps(now) for s in ARMS},
                            "action": self.action_rate.fps(now), "recording": self.recorder.writes.fps(now)}}

    def reconnect(self):
        with self.lock:
            if self.closing or (self.reconnect_thread is not None and self.reconnect_thread.is_alive()):
                raise RuntimeError("正在关闭或重新连接设备，请稍候")
            if self.mode not in ("idle", "fault") or self.recorder.busy or self.commanded:
                raise RuntimeError("请先结束录制并等待保存完成，再重新连接设备")
            self._event("reconnect_requested", self.message)
            self.mode, self.message = "reconnecting", "正在重新连接相机和机械臂反馈，请稍候"
            self.done.set()
            self.reconnect_thread = threading.Thread(target=self._reconnect, daemon=True,
                                                     name="capture-reconnect")
            self.reconnect_thread.start()
            return self.status()

    def _reconnect(self):
        try:
            self._close_devices()
            with self.lock:
                if self.closing:
                    return
                self.cameras = Cameras(self.cfg, self.mock)
                self.robots = {}
                self.states = {side: History() for side in ARMS}
                self.recorder = Recorder(self.cfg["output"], self.cfg, self.mock)
                self.action_rate = History()
                self.sync = None
                self.done = threading.Event()
                self.thread = None
                self.end_time = None
                self.camera_waiting = self.recover_camera_fault = False
                self.camera_stall_since = self.camera_recovered_since = None
                self.mode, self.message = "warming", "重新连接后等待新鲜相机和机械臂反馈"
            self.open()
            self._event("reconnected", "Devices reopened; waiting for fresh feedback")
        except Exception as error:
            with self.lock:
                self._fault(f"重新连接失败：{type(error).__name__}: {error}")

    def _close_devices(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join()
        with self.lock:
            if self.commanded:
                self._fault("Closed during home motion")
            if self.recorder.active:
                self.recorder.stop("closed_before_save", complete=False, end_time=time.perf_counter())
        # Device shutdown may block. Keep status/GUI reads responsive while
        # the reconnect worker waits, and attempt every device even on error.
        self.recorder.wait()
        errors = []
        for device in (self.cameras, *self.robots.values()):
            try:
                device.close()
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
        if errors:
            raise RuntimeError("Device shutdown failed: " + "; ".join(errors))

    def close(self):
        self.closing = True
        reconnect = self.reconnect_thread
        if reconnect is not None and reconnect is not threading.current_thread():
            reconnect.join()
        self._close_devices()
        self.mode = "closed"
