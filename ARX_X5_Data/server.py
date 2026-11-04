#!/usr/bin/env python3
"""Local-only collection console. No third-party dependencies."""

import argparse
import copy
import fcntl
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from episode_browser import EpisodeBrowser
from take_recycle import MARKER, recycle, restore

ROOT = Path(__file__).resolve().parent


class WorkflowError(Exception):
    pass


class DemoAdapter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.teleop = False
        self.recording = False
        self.ident = None

    def call(self, action, operation_id, trial=False):
        if action == "teleop_start":
            self.teleop = True
        elif action == "teleop_stop":
            self.teleop = False
        elif action == "record_start":
            self.ident = "demo-" + uuid.uuid4().hex[:10]
            self.recording = True
        elif action in ("record_stop", "record_discard"):
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / (self.ident + ".json")
            with path.open("x") as handle:
                json.dump(
                    {
                        "demo": True,
                        "id": self.ident,
                        "trial": trial,
                        "note": "演示文件，不包含设备数据",
                    },
                    handle,
                    ensure_ascii=False,
                )
            self.recording = False
            return {
                "teleop_running": self.teleop,
                "recording": False,
                ("discarded" if action == "record_discard" else "saved"): {
                    "id": self.ident,
                    "path": str(path),
                },
            }
        return {
            "teleop_running": self.teleop,
            "recording": self.recording,
            "recording_id": self.ident,
        }


class CommandAdapter:
    def __init__(self, config):
        self.config = config

    def call(self, action, operation_id, trial=False):
        argv = self.config["commands"].get(action)
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise WorkflowError("操作尚未配置：" + action)
        env = dict(
            os.environ,
            ARX_OPERATION_ID=operation_id,
            ARX_TRIAL="1" if trial else "0",
            ARX_SAVE_DIR=self.config["save_dir"],
        )
        if self.config.get("teleop_script"):
            env["ARX_TELEOP_SCRIPT"] = self.config["teleop_script"]
        if self.config.get("recorder_url"):
            env["ARX_RECORDER_URL"] = self.config["recorder_url"]
        try:
            result = subprocess.run(
                argv,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.get("command_timeout_seconds", 30),
            )
        except subprocess.TimeoutExpired as exc:
            raise WorkflowError(
                "操作超时，结果未知。请检查现有程序，再点击“重新核对状态”；不要重复启动。"
            ) from exc
        except OSError as exc:
            raise WorkflowError("无法执行已配置程序：" + str(exc)) from exc
        if result.returncode:
            raise WorkflowError(
                "现有程序返回失败：" + (result.stderr.strip()[-800:] or str(result.returncode))
            )
        try:
            return json.loads(result.stdout)
        except ValueError as exc:
            raise WorkflowError("程序没有返回有效 JSON，请联系接入人员检查适配脚本。") from exc


class Workbench:
    def __init__(self, config, runtime, adapter=None, demo=False):
        self.config = config
        self.runtime = Path(runtime)
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.file = self.runtime / "state.json"
        self.lock = threading.RLock()
        self.demo = demo
        self.adapter = adapter or CommandAdapter(config)
        self.state = dict(
            phase="setup",
            teleop_running=None,
            recording=None,
            switches=False,
            channels=[False] * 4,
            trial_verified=False,
            trial=True,
            recording_id=None,
            started_at=None,
            saved=[],
            error=None,
            busy=False,
            requests=[],
            updated_at=None,
        )
        if self.file.exists():
            self.state.update(json.loads(self.file.read_text()))
            # A process restart never replays a hardware command or trusts old CAN checks.
            self.state.update(
                switches=False,
                channels=[False] * 4,
                teleop_running=None,
                recording=None,
                busy=False,
            )
            if self.state["phase"] not in ("setup", "ready", "reset", "stopped"):
                self.state.update(
                    phase="error", error="服务曾重启，操作结果需重新核对；不会自动重发设备命令。"
                )
        self.persist()

    @property
    def configured(self):
        return self.demo or (
            bool(self.config.get("save_dir"))
            and all(
                self.config.get("commands", {}).get(k)
                for k in ("status", "teleop_start", "teleop_stop", "record_start", "record_stop")
            )
        )

    @property
    def teleop_configured(self):
        return self.demo or all(
            self.config.get("commands", {}).get(k) for k in ("status", "teleop_start")
        )

    def persist(self):
        self.state["updated_at"] = time.time()
        temp = self.file.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2))
        temp.replace(self.file)

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.state)
            for item in result["saved"]:
                item.update(self.processing_state(item))
            counts = {}
            for item in result["saved"]:
                key = item["processing"]
                counts[key] = counts.get(key, 0) + 1
            result["processing_counts"] = counts
            result["validated_count"] = sum(
                not item["trial"] and item["validation"] == "passed" for item in result["saved"]
            )
            result.update(
                demo=self.demo,
                configured=self.configured,
                teleop_configured=self.teleop_configured,
                config=self.config,
                saved_count=sum(
                    not item["trial"] and not item.get("deleted") for item in result["saved"]
                ),
                can_doctor_embed_url=self.config.get("can_doctor_url") or None,
                can_doctor_demo=False,
                discard_configured=self.demo
                or bool(self.config.get("commands", {}).get("record_discard")),
                teleop_stop_configured=self.demo
                or bool(self.config.get("commands", {}).get("teleop_stop")),
            )
            return result

    def processing_state(self, item):
        if item.get("deleted") or (
            Path(item["path"]).is_dir() and (Path(item["path"]) / MARKER).exists()
        ):
            return dict(
                processing="deleted",
                validation="deleted",
                deleted=True,
                processing_error=None,
                can_restore=not item.get("discarded"),
            )
        if self.demo:
            return dict(processing="demo", validation="demo", processing_error=None)
        path = Path(item["path"])
        try:
            manifest = json.loads((path / "manifest.json").read_text())
            job_path = path / "job.json"
            job = json.loads(job_path.read_text()) if job_path.exists() else {}
            status = manifest.get("status")
            if status in ("incomplete", "cancelled"):
                return dict(
                    processing="failed",
                    validation="failed",
                    processing_error=manifest.get("error") or status,
                )
            validation = (
                "passed"
                if status == "complete"
                else "pending"
                if status == "pending_validation"
                else "unknown"
            )
            processing = job.get("state") or (
                "queued"
                if validation == "pending"
                else "validated"
                if validation == "passed"
                else "unknown"
            )
            if processing not in (
                "queued",
                "validating",
                "converting",
                "done",
                "failed",
                "validated",
            ):
                processing = "unknown"
            return dict(
                processing=processing,
                validation=validation,
                processing_error=job.get("error") if processing == "failed" else None,
            )
        except (OSError, ValueError, TypeError) as error:
            return dict(
                processing="unknown",
                validation="unknown",
                processing_error="无法读取后台处理状态：" + str(error),
            )

    def open_can(self):
        url = self.config.get("can_doctor_url")
        if not url:
            raise WorkflowError("CAN Doctor 地址尚未配置，请联系现场负责人。")

        def healthy():
            try:
                with urlopen(url, timeout=1) as response:
                    return response.status == 200
            except Exception:
                return False

        # Serialize launch requests; opening the page never invokes CAN control APIs.
        with self.lock:
            if healthy():
                return {"url": url}
            argv = self.config.get("can_doctor_command")
            parsed = urlparse(url)
            if not argv or parsed.hostname not in ("127.0.0.1", "localhost"):
                raise WorkflowError("CAN Doctor 网页暂不可访问，请检查其网页服务。")
            # The vendor launcher terminates an existing port owner.
            # Never invoke it on an occupied port.
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", parsed.port or 80)) == 0:
                    raise WorkflowError(
                        "CAN Doctor 端口已有服务，但网页检查失败；未重复启动，请联系现场负责人。"
                    )
            with (self.runtime / "can-doctor.log").open("a") as log:
                subprocess.Popen(
                    argv,
                    cwd=self.config.get("can_doctor_cwd") or ROOT,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if healthy():
                    return {"url": url}
                time.sleep(0.2)
            raise WorkflowError(
                "CAN Doctor 网页尚未就绪，请查看本地 can-doctor.log 或联系现场负责人。"
            )

    def validate_status(self, response):
        if (
            not isinstance(response, dict)
            or type(response.get("teleop_running")) is not bool
            or type(response.get("recording")) is not bool
        ):
            raise WorkflowError("设备状态返回不完整，必须明确给出遥操和录制的真实状态。")
        return response

    def saved_item(self, response):
        artifact = response.get("saved")
        if not isinstance(artifact, dict) or not artifact.get("id") or not artifact.get("path"):
            raise WorkflowError("未取得保存完成凭据：需要本条编号与实际路径。")
        base = Path(self.config["save_dir"]).resolve()
        path = Path(artifact["path"])
        if not path.is_absolute():
            raise WorkflowError("保存结果必须返回绝对路径。")
        path = path.resolve()
        if not path.is_relative_to(base) or path == base:
            raise WorkflowError("返回的数据路径不在已配置保存目录内。")
        nonempty = path.is_file() and path.stat().st_size > 0
        if path.is_dir():
            nonempty = any(p.is_file() and p.stat().st_size > 0 for p in path.rglob("*"))
        if not nonempty:
            raise WorkflowError("保存路径不存在或内容为空，请保留现场并检查录制程序。")
        if self.state["recording_id"] and artifact["id"] != self.state["recording_id"]:
            raise WorkflowError("保存编号与当前录制不一致，请人工核对。")
        return dict(id=artifact["id"], path=str(path), trial=self.state["trial"], time=time.time())

    def act(self, action, payload, request_id):
        with self.lock:
            if not isinstance(request_id, str) or not 8 <= len(request_id) <= 100:
                raise WorkflowError("请求缺少有效操作编号，请刷新页面。")
            if request_id in self.state["requests"]:
                return self.snapshot()
            if self.state["busy"]:
                raise WorkflowError("上一项操作尚未结束，请等待。")
            if action in ("take_delete", "take_restore"):
                item = next(
                    (item for item in self.state["saved"] if item["id"] == payload.get("id")), None
                )
                if item is None:
                    raise WorkflowError("找不到这条记录，未操作其他文件。")
                path = Path(item["path"]).resolve()
                if (
                    not path.is_relative_to(Path(self.config["save_dir"]).resolve())
                    or path == Path(self.config["save_dir"]).resolve()
                ):
                    raise WorkflowError("条目路径不属于本地数据目录。")
                if self.state["recording"] and item["id"] == self.state["recording_id"]:
                    raise WorkflowError("本条正在录制，请使用“删除本条”结束并作废。")
                if action == "take_delete":
                    if not self.demo:
                        recycle(path)
                    item["deleted"] = True
                else:
                    if item.get("discarded"):
                        raise WorkflowError("录制中作废的条目不能恢复为有效数据。")
                    if not self.demo:
                        try:
                            restore(path)
                        except ValueError as error:
                            raise WorkflowError(str(error)) from error
                    item["deleted"] = False
                if not any(x["trial"] and not x.get("deleted") for x in self.state["saved"]):
                    self.state["trial_verified"] = False
                self.state["requests"] = (self.state["requests"] + [request_id])[-300:]
                self.state["notice"] = (
                    "本条已删除（原始文件保留，可在采集记录恢复）。"
                    if action == "take_delete"
                    else "本条已恢复。"
                )
                self.persist()
                return self.snapshot()
            if action == "prepare":
                checks = payload.get("channels")
                if payload.get("switches") is not True or checks != [True] * 4:
                    raise WorkflowError("请确认设备开关和四路 CAN 状态。")
                self.state.update(switches=True, channels=checks)
                self.persist()
                return self.snapshot()
            if action == "verify_trial":
                if not any(x["trial"] and not x.get("deleted") for x in self.state["saved"]):
                    raise WorkflowError("请先完成试采并核对本地文件。")
                self.state["trial_verified"] = True
                self.persist()
                return self.snapshot()
            if action == "reset":
                if self.state["phase"] != "reset" or self.state["recording"] is not False:
                    raise WorkflowError("必须先结束录制并确认保存完成。")
                # Verify live status before arming the next take.
            elif action not in (
                "status",
                "teleop_start",
                "teleop_stop",
                "record_start",
                "record_stop",
                "record_discard",
            ):
                raise WorkflowError("不支持的操作。")
            if not (
                (self.demo or bool(self.config.get("commands", {}).get("teleop_stop")))
                if action == "teleop_stop"
                else self.teleop_configured
                if action in ("status", "teleop_start")
                else self.configured
            ):
                raise WorkflowError("设备接入尚未配置，请先完成 config.local.json。")
            phase = self.state["phase"]
            if action == "teleop_stop" and (
                phase in ("recording", "saving", "starting") or self.state["recording"] is not False
            ):
                raise WorkflowError("请先结束并保存当前录制，再结束遥操。")
            if action in ("teleop_start", "record_start"):
                if not self.state["switches"] or not all(self.state["channels"]):
                    raise WorkflowError("请先完成开工确认。")
            if action == "teleop_start" and (
                phase not in ("setup", "stopped")
                or self.state["teleop_running"] is not False
                or self.state["recording"] is not False
            ):
                raise WorkflowError("请先核对设备状态；遥操已运行时无需重复启动。")
            if action == "record_start" and (
                phase != "ready"
                or self.state["recording"] is not False
                or self.state["teleop_running"] is not True
            ):
                raise WorkflowError("设备当前不满足开始录制的条件。")
            if action in ("record_stop", "record_discard") and phase != "recording":
                raise WorkflowError("当前没有可结束的录制。")
            self.state["requests"] = (self.state["requests"] + [request_id])[-300:]
            self.state.update(busy=True, error=None)
            if action == "record_start":
                self.state.update(phase="starting", trial=not self.state["trial_verified"])
            elif action in ("record_stop", "record_discard"):
                self.state["phase"] = "saving"
            self.persist()
        try:
            # Starting/stopping operations are preceded by a real status check.
            if action in (
                "teleop_start",
                "teleop_stop",
                "record_start",
                "record_stop",
                "record_discard",
                "reset",
            ):
                before = self.validate_status(
                    self.adapter.call("status", request_id, self.state["trial"])
                )
                if action == "teleop_stop" and before["recording"]:
                    raise WorkflowError("真实录制器仍在录制或保存，请先完成保存。")
                if action in ("record_start", "reset") and (
                    not before["teleop_running"] or before["recording"]
                ):
                    raise WorkflowError("真实状态不满足操作条件：请核对遥操与录制程序。")
                if action == "teleop_start" and (before["teleop_running"] or before["recording"]):
                    raise WorkflowError("程序状态已变化，请重新核对，避免重复启动。")
            if action == "record_discard" and not before["recording"]:
                raise WorkflowError("当前录制已停止，请先核对保存结果，再从采集记录删除对应条目。")
            if action == "record_stop" and not before["recording"]:
                if not before.get("saved"):
                    raise WorkflowError("录制已停止但缺少保存凭据；未重复执行结束命令。")
                response = before
            else:
                response = (
                    before
                    if action == "reset"
                    else self.validate_status(
                        self.adapter.call(action, request_id, self.state["trial"])
                    )
                )
            with self.lock:
                self.state.update(
                    teleop_running=response["teleop_running"],
                    recording=response["recording"],
                    teleop_detail=response.get("teleop_detail"),
                    teleop_process_running=response.get(
                        "teleop_process_running", response["teleop_running"]
                    ),
                    background=response.get("background", {}),
                    worker_alive=response.get("worker_alive"),
                )
                if action == "teleop_stop":
                    if response.get("stop_pending") or response.get(
                        "teleop_process_running", response["teleop_running"]
                    ):
                        self.state.update(
                            error=response.get("teleop_detail")
                            or "遥操尚未退出，请手动回正后再次点击结束遥操。"
                        )
                    else:
                        self.state.update(
                            phase="stopped",
                            resume_phase=phase
                            if phase != "stopped"
                            else self.state.get("resume_phase", "setup"),
                        )
                elif action == "record_start":
                    if (
                        not response["teleop_running"]
                        or not response["recording"]
                        or not response.get("recording_id")
                    ):
                        raise WorkflowError("未确认录制已启动，请核对现有程序。")
                    self.state.update(
                        phase="recording",
                        recording_id=response["recording_id"],
                        started_at=time.time(),
                    )
                elif action == "record_discard":
                    if response["recording"] or not response.get("discarded"):
                        raise WorkflowError("尚未确认本条已作废，请核对录制器。")
                    item = self.saved_item({"saved": response["discarded"]})
                    item.update(deleted=True, discarded=True)
                    self.state["saved"].append(item)
                    self.state.update(
                        phase="reset",
                        started_at=None,
                        notice="本条录制已结束并作废，请复位后采集下一条。",
                    )
                elif action == "record_stop" or (
                    action == "status" and phase == "error" and response.get("saved")
                ):
                    if response["recording"]:
                        raise WorkflowError("录制仍在运行，尚不能进入复位。")
                    item = self.saved_item(response)
                    if not any(x["id"] == item["id"] for x in self.state["saved"]):
                        self.state["saved"].append(item)
                    self.state.update(phase="reset", started_at=None)
                elif action == "reset":
                    self.state.update(phase="ready", recording_id=None, started_at=None)
                elif action in ("status", "teleop_start"):
                    if response["recording"]:
                        if not response["teleop_running"]:
                            raise WorkflowError(
                                "录制期间遥操停止，请检查设备并在原程序处理当前录制。"
                            )
                        if not response.get("recording_id"):
                            raise WorkflowError("正在录制但缺少条目编号，请核对现有程序。")
                        if phase not in ("recording", "error") or (
                            self.state["recording_id"]
                            and response.get("recording_id") != self.state["recording_id"]
                        ):
                            raise WorkflowError("发现未匹配的录制，请在原程序核对并结束后返回。")
                        self.state.update(
                            phase="recording",
                            recording_id=response.get("recording_id"),
                            started_at=self.state["started_at"] or time.time(),
                        )
                    elif phase == "recording":
                        raise WorkflowError("录制意外中断；未收到保存凭据，不计入完成条数。")
                    elif phase == "error" and any(
                        item["id"] == self.state["recording_id"] for item in self.state["saved"]
                    ):
                        self.state.update(
                            phase="reset" if response["teleop_running"] else "stopped",
                            resume_phase="reset",
                        )
                    elif phase == "error":
                        raise WorkflowError(
                            "录制已停止，但缺少保存凭据。请核对文件，"
                            "并让状态接口返回本条 saved 信息。"
                        )
                    elif not response["teleop_running"]:
                        if phase in ("reset", "ready"):
                            self.state.update(phase="stopped", resume_phase=phase)
                        elif phase != "stopped":
                            self.state["phase"] = "setup"
                    elif phase in ("setup", "stopped"):
                        self.state["phase"] = (
                            self.state.get("resume_phase", "ready")
                            if phase == "stopped"
                            else "ready"
                        )
                        if self.state["phase"] not in ("ready", "reset"):
                            self.state["phase"] = "ready"
                if action == "teleop_start" and not response["teleop_running"]:
                    raise WorkflowError("遥操程序未报告运行成功。")
        except Exception as exc:
            with self.lock:
                # Unknown outcomes require reconciliation, never blind replay.
                self.state.update(
                    phase=phase
                    if action == "teleop_stop"
                    else "error"
                    if action in ("record_start", "record_stop", "record_discard", "reset")
                    or phase in ("recording", "error", "reset")
                    else "setup",
                    error=str(exc),
                    teleop_running=None,
                    recording=None,
                )
        finally:
            with self.lock:
                self.state["busy"] = False
                self.persist()
        return self.snapshot()


def load_config(path, demo=False):
    config = json.loads(Path(path).read_text())
    if len(config.get("channels", [])) != 4:
        raise ValueError("配置必须包含四路 CAN。")
    if config.get("save_dir") and not Path(config["save_dir"]).is_absolute():
        raise ValueError("save_dir 必须是绝对路径。")
    url = config.get("can_doctor_url", "")
    if url:
        can_origin(url)
    return config


def can_origin(url):
    parsed = urlparse(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or any(c.isspace() for c in url)
        or any(c in url for c in (";", "'", '"', "<", ">", "\\"))
    ):
        raise ValueError("CAN Doctor 地址必须是完整的 http/https 地址，不能包含账号密码。")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("CAN Doctor 端口无效。") from exc
    host = parsed.hostname
    if ":" in host:
        host = "[" + host + "]"
    return f"{parsed.scheme}://{host}" + (f":{port}" if port else "")


def make_handler(workbench):
    browser = EpisodeBrowser(
        workbench.config.get("save_dir"),
        workbench.runtime,
        workbench.config.get("visualization_python")
        or str(ROOT / ".runtime/lerobot-main-venv/bin/python"),
    )

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, mime="application/json; charset=utf-8"):
            raw = (
                json.dumps(body, ensure_ascii=False).encode()
                if mime.startswith("application/json")
                else body
            )
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            frame_ancestors = "'none'"
            frame_origin = (
                can_origin(workbench.config["can_doctor_url"])
                if workbench.config.get("can_doctor_url")
                else ""
            )
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self'; style-src 'self'; "
                "script-src 'self'; connect-src 'self'; "
                f"frame-src 'self' {frame_origin}; frame-ancestors {frame_ancestors}",
            )
            self.end_headers()
            self.wfile.write(raw)

        def media(self, target, mime):
            size = target.stat().st_size
            start, end, code = 0, size - 1, 200
            value = self.headers.get("Range")
            if value:
                import re

                match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
                try:
                    if not match or not any(match.groups()):
                        raise ValueError()
                    a, b = match.groups()
                    start = int(a) if a else max(0, size - int(b))
                    end = min(int(b), size - 1) if a and b else size - 1
                    if start > end or start >= size or (not a and int(b) == 0):
                        raise ValueError()
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                code = 206
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with target.open("rb") as file:
                    file.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = file.read(min(262144, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def allowed(self):
            expected = "127.0.0.1:" + str(self.server.server_port)
            return (
                self.headers.get("Host") == expected
                and self.headers.get("Origin", "http://" + expected) == "http://" + expected
            )

        def do_GET(self):
            if not self.allowed():
                return self.send(403, {"error": "仅接受本机同源访问。"})
            path = urlparse(self.path).path
            if path.startswith("/api/episodes"):
                query = parse_qs(urlparse(self.path).query)
                ident = query.get("id", [""])[0]
                try:
                    if path == "/api/episodes":
                        return self.send(200, browser.list())
                    if path == "/api/episodes/detail":
                        return self.send(200, browser.detail(ident))
                    if path == "/api/episodes/visualization":
                        return self.send(200, browser.visualization(ident))
                    if path == "/api/episodes/video":
                        return self.media(
                            browser.video(ident, query.get("camera", [""])[0]), "video/mp4"
                        )
                    if path == "/api/episodes/rrd":
                        if browser.visualization(ident)["state"] != "ready":
                            raise ValueError("可视化文件尚未生成")
                        return self.media(
                            browser.viz_dir(ident) / "local_episode_episode_0.rrd",
                            "application/octet-stream",
                        )
                    return self.send(404, {"error": "接口不存在"})
                except (ValueError, OSError, TypeError) as exc:
                    return self.send(409, {"error": str(exc)})
            if path == "/api/state":
                return self.send(200, workbench.snapshot())
            files = {
                "/browse": ("web/browse.html", "text/html; charset=utf-8"),
                "/browse.js": ("web/browse.js", "text/javascript; charset=utf-8"),
                "/browse.css": ("web/browse.css", "text/css; charset=utf-8"),
                "/": ("web/index.html", "text/html; charset=utf-8"),
                "/app.js": ("web/app.js", "text/javascript; charset=utf-8"),
                "/style.css": ("web/style.css", "text/css; charset=utf-8"),
                "/guide": ("web/guide.html", "text/html; charset=utf-8"),
            }
            if path.startswith("/assets/"):
                target = (ROOT / "web" / path.lstrip("/")).resolve()
                if (
                    target.is_relative_to(ROOT / "web" / "assets")
                    and target.is_file()
                    and target.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                ):
                    return self.send(
                        200,
                        target.read_bytes(),
                        "image/"
                        + ("jpeg" if target.suffix in (".jpg", ".jpeg") else target.suffix[1:]),
                    )
            if path not in files:
                return self.send(404, {"error": "页面不存在。"})
            file, mime = files[path]
            self.send(200, (ROOT / file).read_bytes(), mime)

        def do_POST(self):
            if (
                not self.allowed()
                or self.headers.get("Content-Type", "").split(";")[0] != "application/json"
            ):
                return self.send(403, {"error": "仅接受本机同源 JSON 请求。"})
            if self.path not in (
                "/api/action",
                "/api/can/open",
                "/api/episodes/review",
                "/api/episodes/export",
                "/api/episodes/open-rerun",
            ):
                return self.send(404, {"error": "接口不存在。"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 8192:
                    raise WorkflowError("请求大小无效。")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise WorkflowError("请求必须是对象。")
                if self.path.startswith("/api/episodes/"):
                    handlers = {
                        "/api/episodes/review": lambda: browser.review(body),
                        "/api/episodes/export": lambda: browser.export(body.get("id")),
                        "/api/episodes/open-rerun": lambda: browser.open_rerun(body.get("id")),
                    }
                    return self.send(200, handlers[self.path]())
                result = (
                    workbench.open_can()
                    if self.path == "/api/can/open"
                    else workbench.act(body.get("action"), body, body.get("request_id"))
                )
                self.send(200, result)
            except (WorkflowError, ValueError, TypeError, OSError) as exc:
                self.send(409, {"error": str(exc)})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--config", default=str(ROOT / "config.local.json"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    path = Path(args.config)
    if not path.exists():
        path = ROOT / "config.example.json"
    config = load_config(path)
    if args.demo:
        config["save_dir"] = str(ROOT / ".demo-data")
    runtime = ROOT / ".runtime" / ("demo" if args.demo else "device")
    runtime.mkdir(parents=True, exist_ok=True)
    lease = (runtime / "server.lock").open("w")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("同一模式已有服务在运行，请使用现有窗口。")
    # Demo adapter is ephemeral; clear only its workflow state on explicit demo startup.
    if args.demo:
        (runtime / "state.json").unlink(missing_ok=True)
    workbench = Workbench(
        config, runtime, DemoAdapter(config["save_dir"]) if args.demo else None, args.demo
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(workbench))
    print(
        f"ARX 数采工作台: http://127.0.0.1:{args.port}  模式: {'演示' if args.demo else '设备'}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
