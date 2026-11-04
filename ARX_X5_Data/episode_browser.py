"""Local episode catalogue, independent human review, and official LeRobot visualization."""

import hashlib
import json
import math
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from take_recycle import MARKER

CAMERAS = ("head", "left_wrist", "right_wrist")


class EpisodeBrowser:
    def __init__(self, root, runtime, python):
        self.root = Path(root).resolve() if root else None
        self.runtime = Path(runtime) / "visualization"
        self.python = Path(python)
        self.lock = threading.RLock()
        self.jobs = {}
        self.viewer = None

    def episode(self, ident):
        if not self.root or not isinstance(ident, str):
            raise ValueError("数据目录尚未配置")
        parts = Path(ident).parts
        if (
            len(parts) != 2
            or parts[0] in (".", "..", "lerobot", "mock")
            or not parts[1].startswith("episode_")
        ):
            raise ValueError("无效 Episode 编号")
        path = (self.root / ident).resolve()
        if (
            not path.is_relative_to(self.root)
            or path.parent.parent != self.root
            or not (path / "manifest.json").is_file()
        ):
            raise ValueError("Episode 不存在或超出数据目录")
        return path

    @staticmethod
    def read(path, fallback=None):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return fallback

    def converted(self, ident):
        path = (self.root / "lerobot" / ident).resolve()
        return (
            path if path.is_relative_to(self.root) and (path / "meta/info.json").is_file() else None
        )

    def summary(self, ident):
        p = self.episode(ident)
        m = self.read(p / "manifest.json", {})
        frames, fps = m.get("frames", 0), m.get("fps", 30)
        review = self.read(p / ".workbench-review.json", {})
        job = self.read(p / "job.json", {})
        return dict(
            id=ident,
            task=p.parent.name,
            name=p.name,
            frames=frames,
            fps=fps,
            duration=frames / fps if fps else 0,
            status=m.get("status", "unknown"),
            deleted=(p / MARKER).exists(),
            trial=p.parent.name.startswith("试采_"),
            drops=m.get("dropped_grid_ticks", 0),
            drop_reasons=m.get("drop_reasons", {}),
            error=m.get("error"),
            processing=job.get("state", "unknown"),
            processing_error=job.get("error"),
            converted=bool(self.converted(ident)),
            review=review,
            recorded_at=(m.get("start_monotonic_s", 0) + m.get("monotonic_to_unix_offset_s", 0)),
            cameras={
                c: "/api/episodes/video?" + urlencode(dict(id=ident, camera=c))
                for c in CAMERAS
                if (p / (c + ".mp4")).is_file()
            },
        )

    def list(self):
        items, errors = [], []
        if self.root and self.root.is_dir():
            for m in self.root.glob("*/episode_*/manifest.json"):
                ident = m.parent.relative_to(self.root).as_posix()
                if ident.split("/")[0] in ("mock", "lerobot"):
                    continue
                try:
                    if self.read(m, {}).get("mock") is True:
                        continue
                    items.append(self.summary(ident))
                except (ValueError, OSError, TypeError) as exc:
                    errors.append(dict(id=ident, error=str(exc)))
        return dict(
            items=sorted(items, key=lambda x: (x["recorded_at"], x["id"]), reverse=True),
            errors=errors,
        )

    def detail(self, ident):
        item = self.summary(ident)
        p = self.episode(ident)
        m = self.read(p / "manifest.json", {})
        qc = dict(
            samples=0,
            invalid_vectors=0,
            internal_missing=0,
            max_gap_ms=0,
            max_camera_span_ms=0,
            max_camera_offset_ms=0,
            error=None,
        )
        previous = None
        try:
            with (p / "samples.jsonl").open() as file:
                for line in file:
                    s = json.loads(line)
                    qc["samples"] += 1
                    if any(
                        len(s.get(k, [])) != 14
                        or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in s[k])
                        for k in ("observation.state", "action")
                    ):
                        qc["invalid_vectors"] += 1
                    grid = s["grid_index"]
                    if previous is not None:
                        qc["internal_missing"] += max(0, grid - previous - 1)
                        qc["max_gap_ms"] = max(
                            qc["max_gap_ms"], (grid - previous) / item["fps"] * 1000
                        )
                    previous = grid
                    stamps = [v["timestamp_monotonic_s"] for v in s["cameras"].values()]
                    qc["max_camera_span_ms"] = max(
                        qc["max_camera_span_ms"], (max(stamps) - min(stamps)) * 1000
                    )
                    qc["max_camera_offset_ms"] = max(
                        qc["max_camera_offset_ms"],
                        max(abs(t - s["timestamp_monotonic_s"]) for t in stamps) * 1000,
                    )
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            qc["error"] = str(exc)
        item.update(
            qc=qc,
            camera_config=m.get("config", {}).get("cameras", {}),
            vector_names=m.get("vector_names", []),
            path=str(p),
            visualization=self.visualization(ident),
        )
        return item

    def review(self, body):
        ident = body.get("id")
        with self.lock:
            p = self.episode(ident)
            item = self.summary(ident)
            status = body.get("status")
            if status not in ("pending", "approved", "rejected", "needs_review"):
                raise ValueError("无效审核状态")
            if status == "approved" and (item["status"] != "complete" or item["deleted"]):
                raise ValueError("已作废、删除或尚未完成校验的条目不能标记为通过")
            old = item["review"]
            if body.get("revision", 0) != old.get("revision", 0):
                raise ValueError("审核已在其他窗口更新，请刷新后重试")
            note, reviewer = body.get("note", ""), body.get("reviewer", "")
            if (
                not isinstance(note, str)
                or len(note) > 2000
                or not isinstance(reviewer, str)
                or not 1 <= len(reviewer.strip()) <= 80
            ):
                raise ValueError("请填写审核人，备注最多 2000 字")
            if status in ("rejected", "needs_review") and not note.strip():
                raise ValueError("不通过或需复核时请填写原因")
            entry = dict(
                status=status,
                note=note.strip(),
                reviewer=reviewer.strip(),
                updated_at=time.time(),
                revision=old.get("revision", 0) + 1,
            )
            result = dict(entry, history=old.get("history", []) + [entry])
            target = p / ".workbench-review.json"
            temp = target.with_suffix(".tmp")
            temp.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            temp.replace(target)
            return result

    def video(self, ident, camera):
        if camera not in CAMERAS:
            raise ValueError("无效相机")
        p = self.episode(ident)
        path = (p / (camera + ".mp4")).resolve()
        if path.parent != p or not path.is_file():
            raise ValueError("视频不存在或路径无效")
        return path

    def viz_dir(self, ident):
        self.episode(ident)
        return self.runtime / hashlib.sha256(ident.encode()).hexdigest()[:24]

    def source_signature(self, ident):
        source = self.converted(ident)
        if source is None:
            return None
        stats = [
            (str(p.relative_to(source)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(source.rglob("*"))
            if p.is_file()
        ]
        return hashlib.sha256(json.dumps(stats).encode()).hexdigest()

    def visualization(self, ident):
        with self.lock:
            job = self.jobs.get(ident, {})
            if job.get("state") in ("running", "failed"):
                return dict(job)
            directory = self.viz_dir(ident)
            marker = self.read(directory / "ready.json", {})
            if (
                (directory / "local_episode_episode_0.rrd").is_file()
                and marker.get("source")
                and marker["source"] == self.source_signature(ident)
            ):
                return dict(
                    state="ready", download="/api/episodes/rrd?" + urlencode(dict(id=ident))
                )
            return dict(state="idle")

    def export(self, ident):
        with self.lock:
            item = self.summary(ident)
            if item["deleted"] or item["status"] != "complete" or not self.converted(ident):
                raise ValueError("请等待有效条目的 LeRobot 转换完成")
            state = self.visualization(ident)
            if state["state"] in ("running", "ready"):
                return state
            if any(job.get("state") == "running" for job in self.jobs.values()):
                raise ValueError("已有 Episode 正在生成可视化，请完成后再开始另一条")
            self.jobs[ident] = dict(state="running")
            threading.Thread(target=self._export, args=(ident,), daemon=True).start()
            return dict(state="running")

    def _export(self, ident):
        directory = self.viz_dir(ident)
        directory.mkdir(parents=True, exist_ok=True)
        try:
            env = dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2")
            with (directory / "export.log").open("w") as log:
                result = subprocess.run(
                    [
                        str(self.python),
                        "-m",
                        "lerobot.scripts.lerobot_dataset_viz",
                        "--repo-id",
                        "local/episode",
                        "--root",
                        str(self.converted(ident)),
                        "--episode-index",
                        "0",
                        "--num-workers",
                        "0",
                        "--batch-size",
                        "8",
                        "--save",
                        "1",
                        "--display-compressed-images",
                        "--output-dir",
                        str(directory),
                    ],
                    stdout=log,
                    stderr=log,
                    env=env,
                    timeout=600,
                )
            if result.returncode or not (directory / "local_episode_episode_0.rrd").exists():
                raise ValueError((directory / "export.log").read_text()[-1600:])
            marker = directory / "ready.tmp"
            marker.write_text(json.dumps({"source": self.source_signature(ident)}))
            marker.replace(directory / "ready.json")
            with self.lock:
                self.jobs[ident] = dict(state="ready")
        except Exception as exc:
            with self.lock:
                self.jobs[ident] = dict(state="failed", error=str(exc))

    def open_rerun(self, ident):
        with self.lock:
            if self.visualization(ident)["state"] != "ready":
                raise ValueError("请先生成 Rerun 可视化")
            if self.viewer and self.viewer["process"].poll() is None:
                if self.viewer["id"] == ident:
                    return dict(url=self.viewer["url"])
                self.viewer["process"].terminate()
                try:
                    self.viewer["process"].wait(timeout=5)
                except subprocess.TimeoutExpired as exc:
                    raise ValueError("上一条可视化正在关闭，请稍后重试") from exc
            # Allocate local-only ports without touching any existing viewer or hardware service.
            with socket.socket() as web, socket.socket() as grpc:
                web.bind(("127.0.0.1", 0))
                grpc.bind(("127.0.0.1", 0))
                web_port, grpc_port = web.getsockname()[1], grpc.getsockname()[1]
            url = f"http://127.0.0.1:{web_port}/?" + urlencode(
                {"url": f"rerun+http://127.0.0.1:{grpc_port}/proxy"}
            )
            directory = self.viz_dir(ident)
            with (directory / "viewer.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        str(self.python.parent / "rerun"),
                        "--serve-web",
                        "--bind",
                        "127.0.0.1",
                        "--web-viewer-port",
                        str(web_port),
                        "--port",
                        str(grpc_port),
                        "--server-memory-limit",
                        "1GB",
                        "--threads",
                        "2",
                        str(directory / "local_episode_episode_0.rrd"),
                    ],
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            self.viewer = dict(id=ident, process=process, url=url)
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise ValueError(
                        "Rerun 启动失败：" + (directory / "viewer.log").read_text()[-1200:]
                    )
                try:
                    with urlopen(f"http://127.0.0.1:{web_port}/", timeout=0.3) as response:
                        if response.status == 200:
                            return dict(url=url)
                except OSError:
                    pass
                time.sleep(0.1)
            process.terminate()
            raise ValueError("Rerun 网页尚未就绪，请稍后重试")
