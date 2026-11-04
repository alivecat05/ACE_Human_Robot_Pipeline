#!/usr/bin/env python3
"""Adapter for the existing local teleop; status never opens CAN."""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from take_recycle import MARKER, recycle

ROOT = Path(__file__).resolve().parent
_local_path = ROOT / "config.local.json"
_local = json.loads(_local_path.read_text()) if _local_path.is_file() else {}
_script = os.environ.get("ARX_TELEOP_SCRIPT") or _local.get("teleop_script")
SCRIPT = Path(_script).resolve() if _script else None
RECORDER = (
    os.environ.get("ARX_RECORDER_URL") or _local.get("recorder_url") or "http://127.0.0.1:8080"
).rstrip("/") + "/status"
DATA_ROOT = Path(os.environ.get("ARX_SAVE_DIR") or _local.get("save_dir") or ROOT / "data")
RUNTIME = ROOT / ".runtime" / "device"


def processes():
    if SCRIPT is None:
        return []
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = path.read_bytes().split(b"\0")
            if len(argv) < 2 or b"python" not in Path(os.fsdecode(argv[0])).name.encode():
                continue
            for arg in argv[1:3]:
                if os.fsdecode(arg) in (str(SCRIPT), SCRIPT.name):
                    cwd = (path.parent / "cwd").resolve()
                    if (cwd / os.fsdecode(arg)).resolve() == SCRIPT:
                        found.append(int(path.parent.name))
        except (OSError, ValueError):
            continue
    return found


def receipt(recorder):
    if recorder.get("mode") not in ("idle", "warming") or not recorder.get("episode"):
        return None
    path = Path(recorder["episode"]).resolve()
    if not path.is_relative_to(DATA_ROOT.resolve()):
        raise RuntimeError("录制路径不属于已配置数据目录")
    if (path / MARKER).exists():
        return None
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        manifest.get("status") not in ("pending_validation", "complete")
        or manifest.get("mock") is not False
        or manifest.get("frames", 0) < 2
    ):
        return None
    return {
        "id": str(path),
        "path": str(path),
        "validation": "passed" if manifest["status"] == "complete" else "pending",
    }


def record(action):
    operation_id = os.environ.get("ARX_OPERATION_ID")
    if not operation_id:
        raise RuntimeError("缺少操作编号")
    with urlopen(RECORDER, timeout=3) as response:
        current = json.load(response)
    body = {"request_id": operation_id}
    if action == "start":
        task = os.environ.get("ARX_TASK", "").strip() or "数采任务"
        body["task"] = ("试采_" if os.environ.get("ARX_TRIAL") == "1" else "") + task
    else:
        body["episode"] = current.get("episode")
    request = Request(
        RECORDER.removesuffix("/status") + "/control/record/" + action,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            result = json.load(response)
    except HTTPError as error:
        raise RuntimeError("录制服务拒绝操作：" + error.read().decode()[:500]) from error
    if action == "start":
        if result.get("mode") != "recording" or not result.get("episode"):
            raise RuntimeError("未确认录制已启动")
        return status()
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        if action == "cancel":
            with urlopen(RECORDER, timeout=3) as response:
                raw = json.load(response)
            if raw.get("episode") != body["episode"]:
                raise RuntimeError("当前录制已变化，未作废其他条目")
            if raw["mode"] not in ("recording", "saving"):
                path = Path(body["episode"]).resolve()
                if not path.is_relative_to(DATA_ROOT.resolve()):
                    raise RuntimeError("录制路径超出数据目录")
                manifest = json.loads((path / "manifest.json").read_text())
                if manifest.get("status") != "cancelled":
                    raise RuntimeError("尚未确认本条已作废，请核对原录制器")
                recycle(path)
                result = status()
                result["discarded"] = {"id": str(path), "path": str(path)}
                return result
            time.sleep(0.2)
            continue
        result = status()
        if result.get("saved"):
            if result["saved"]["id"] != body["episode"]:
                raise RuntimeError("保存结果编号不一致")
            return result
        with urlopen(RECORDER, timeout=3) as response:
            raw = json.load(response)
        if raw["mode"] == "fault":
            raise RuntimeError("本条录制失败：" + raw.get("message", ""))
        episode = raw.get("episode")
        if episode and (Path(episode) / "manifest.json").exists():
            manifest = json.loads((Path(episode) / "manifest.json").read_text())
            if manifest.get("status") in ("incomplete", "cancelled"):
                raise RuntimeError("本条数据未通过保存检查：" + manifest.get("status", ""))
        time.sleep(0.3)
    raise RuntimeError("本条文件尚未完成写盘，请核对状态；不会重复结束录制")


def status():
    pids = processes()
    with urlopen(RECORDER, timeout=2) as response:
        recorder = json.load(response)
    if recorder.get("mock") is not False:
        raise RuntimeError("现有数采服务不是明确的真实设备模式，无法核对遥操反馈。")
    ready = bool(pids) and recorder.get("inputs", {}).get("telemetry") is True
    result = dict(
        teleop_running=ready,
        recording=recorder["mode"] in ("recording", "saving"),
        teleop_process_running=bool(pids),
        background=recorder.get("background", {}),
        worker_alive=recorder.get("worker_alive"),
        teleop_detail=(
            "已收到真实遥操数据"
            if ready
            else "遥操进程存在，尚未收到新鲜反馈"
            if pids
            else "真实遥操进程未运行"
        ),
    )
    if result["recording"]:
        result["recording_id"] = str(recorder.get("episode") or "")
    saved = receipt(recorder)
    if saved:
        result["saved"] = saved
    return result


def start():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "teleop-launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = status()
        if before["recording"]:
            raise RuntimeError("现有数采程序正在录制或保存，请先处理当前条目。")
        if before["teleop_running"]:
            return before
        child = None
        if not processes():
            if SCRIPT is None or not SCRIPT.is_file():
                raise RuntimeError(
                    "请在 config.local.json 配置 teleop_script，"
                    "或设置 ARX_TELEOP_SCRIPT 为现有遥操脚本的绝对路径。"
                )
            env = dict(os.environ, ARX_RECORD_SOCKET="/tmp/arx-record.sock", PYTHONUNBUFFERED="1")
            with (RUNTIME / "teleop.log").open("a") as log:
                child = subprocess.Popen(
                    ["/usr/bin/python3", str(SCRIPT)],
                    cwd=SCRIPT.parent,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
        deadline = time.monotonic() + 22
        while time.monotonic() < deadline:
            current = status()
            if current["teleop_running"]:
                return current
            if child and child.poll() is not None:
                raise RuntimeError(
                    "遥操进程已退出，请检查 .runtime/device/teleop.log；未重复启动。"
                )
            time.sleep(0.3)
        raise RuntimeError(
            "未确认遥操反馈就绪。进程可能仍在启动；请先核对状态，"
            "日志在 .runtime/device/teleop.log。"
        )


def stop():
    before = status()
    if before["recording"]:
        raise RuntimeError("请先结束并保存当前录制，再结束遥操。")
    pids = processes()
    if len(pids) > 1:
        raise RuntimeError("发现多个遥操进程，请联系现场负责人核对，未发送退出请求。")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while processes() and time.monotonic() < deadline:
        time.sleep(0.2)
    result = status()
    result["stop_pending"] = bool(processes())
    if result["stop_pending"]:
        result["teleop_detail"] = (
            "退出未完成：请用示教臂手动回正双臂，稳定后再次点击“结束遥操”；当前遥操可能仍在跟随。"
        )
        log = RUNTIME / "teleop.log"
        if log.exists():
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 4000))
                lines = handle.read().decode(errors="replace").splitlines()
            reasons = [line for line in lines if "暂不退出" in line]
            if reasons:
                result["teleop_detail"] += " " + reasons[-1][-500:]
    else:
        result["teleop_running"] = False
        result["teleop_process_running"] = False
        result["teleop_detail"] = "遥操进程已退出。重新开始需重新启动遥操。"
    return result


def main():
    try:
        if len(sys.argv) != 2 or sys.argv[1] not in (
            "status",
            "teleop_start",
            "teleop_stop",
            "record_start",
            "record_stop",
            "record_discard",
        ):
            raise RuntimeError("不支持的适配操作")
        handlers = {
            "status": status,
            "teleop_start": start,
            "teleop_stop": stop,
            "record_start": lambda: record("start"),
            "record_stop": lambda: record("stop"),
            "record_discard": lambda: record("cancel"),
        }
        print(json.dumps(handlers[sys.argv[1]](), ensure_ascii=False))
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
