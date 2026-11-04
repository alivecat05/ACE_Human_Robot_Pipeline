"""Recoverable per-episode deletion; original files are retained."""

import fcntl
import json
import time
from pathlib import Path

MARKER = ".workbench-deleted.json"
LOCK = ".workbench-processing.lock"


def atomic(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False))
    temp.replace(path)


def recycle(path):
    path = Path(path)
    marker = path / MARKER
    if marker.exists():
        return
    atomic(marker, {"deleted_at": time.time(), "reason": "operator_delete"})


def restore(path):
    path = Path(path)
    with (path / LOCK).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("后台正在收尾，请稍后点击恢复。") from exc
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("status") in ("cancelled", "incomplete", "recording"):
            raise ValueError("本条录制已作废或不完整，原始文件保留，但不能恢复为有效数据。")
        # Publish queued work before removing deletion marker, so a crash remains recoverable.
        if (path / "job.json").exists():
            job = json.loads((path / "job.json").read_text())
            if job.get("state") == "cancelled":
                job.update(state="queued", error=None)
                atomic(path / "job.json", job)
        (path / MARKER).unlink(missing_ok=True)
