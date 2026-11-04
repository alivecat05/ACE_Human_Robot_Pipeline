"""Durable, single-process low-priority validation/export queue."""

import argparse
import fcntl
import json
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from .recording import atomic_json


def jobs(root):
    return sorted(root.glob("*/episode_*/job.json"))


def summary(root):
    counts = {}
    errors = []
    for path in jobs(root):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        state = data["state"]
        counts[state] = counts.get(state, 0) + 1
        if state == "failed":
            errors.append({"episode": str(path.parent), "error": data.get("error", "")})
    try:
        worker = json.loads((root / ".worker-status.json").read_text())
        worker["stale"] = time.time() - worker["updated_at_s"] > 5
    except (OSError, ValueError, KeyError):
        worker = {}
    return {"counts": counts, "failures": errors, "queue": worker}


def retry_failed(root):
    for path in jobs(root):
        data = json.loads(path.read_text())
        if data["state"] == "failed":
            data.update(state="queued", error=None)
            atomic_json(path, data)


class BackgroundJobs:
    def __init__(self, root):
        self.root = root
        self.process = None

    def start(self):
        self.root.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if env.get("ARX_WORKER_PYTHONPATH"):
            env["PYTHONPATH"] = env["ARX_WORKER_PYTHONPATH"]
        python = env.get("ARX_WORKER_PYTHON", sys.executable)
        self.log = (self.root / ".worker.log").open("a")
        self.process = subprocess.Popen(
            [
                python,
                "-m",
                "arx_x5_data.jobs",
                str(self.root),
                "--workers",
                env.get("ARX_EXPORT_WORKERS", "2"),
                "--cpu-budget",
                env.get("ARX_EXPORT_CPU_BUDGET", "4"),
            ],
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if hasattr(self, "log"):
            self.log.close()


def process_alive(pid):
    try:
        os.kill(pid, 0)
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state not in {"Z", "X"}
    except (ProcessLookupError, FileNotFoundError):
        return False


def run_one(path):
    with (path.parent / ".workbench-processing.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _run_one(path)


def _run_one(path):
    from .export import convert
    from .validation import episode_fingerprint, validate_episode

    job = json.loads(path.read_text())
    if (path.parent / ".workbench-deleted.json").exists():
        job.update(state="cancelled", error=None)
        atomic_json(path, job)
        return
    started_job = time.perf_counter()
    job.setdefault("enqueued_at_s", path.stat().st_mtime)
    job.update(
        state="validating",
        attempts=job.get("attempts", 0) + 1,
        error=None,
        started_at_s=time.time(),
    )
    atomic_json(path, job)
    episode = path.parent
    try:
        manifest = json.loads((episode / "manifest.json").read_text())
        started = time.perf_counter()
        verified = validate_episode(
            episode,
            allow_mock=manifest["mock"],
            _pending=True,
            _collect_stats=os.environ.get("ARX_H264_BACKEND") == "vaapi",
        )
        manifest["status"] = "complete"
        manifest.setdefault("timings", {})["background_validation_s"] = (
            time.perf_counter() - started
        )
        atomic_json(episode / "manifest.json", manifest)
        fingerprint = episode_fingerprint(episode.resolve(), manifest)
        if verified.fingerprint[1:] != fingerprint[1:]:
            raise ValueError("Source changed between validation and conversion")
        verified = replace(verified, manifest=manifest, fingerprint=fingerprint)
        if (episode / ".workbench-deleted.json").exists():
            job.update(state="cancelled", error=None)
            atomic_json(path, job)
            return
        job.update(state="converting")
        atomic_json(path, job)
        target = episode.parents[1] / "lerobot" / episode.parent.name / episode.name
        lock = target.with_name(f".{target.name}.export.lock")
        if lock.exists():
            pid = int(lock.read_text())
            if not process_alive(pid):
                lock.unlink()
                for stale in target.parent.glob(f".{target.name}.export-*"):
                    shutil.rmtree(stale)
            else:
                raise RuntimeError("Export lock is held by a live process")
        if not target.exists():
            result = convert(
                episode,
                target,
                "local/arx_x5_2023",
                allow_mock=manifest["mock"],
                mode=os.environ.get("ARX_EXPORT_MODE", "reencode_h264"),
                _validated_episode=verified,
                h264_backend=os.environ.get("ARX_H264_BACKEND", "software"),
            )
            job["metrics"] = result["metrics"]
        else:
            # Atomic export publishes only after official reload; no partial target is visible.
            provenance = json.loads((target / "meta/arx_sources.json").read_text())["episodes"]
            if len({e["source"] for e in provenance}) != 1 or provenance[0]["source"] != str(
                episode.resolve()
            ):
                raise ValueError("Existing export belongs to another source")
        job.update(state="done", output=str(target))
    except Exception as error:
        job.update(state="failed", error=f"{type(error).__name__}: {error}")
    if (path.parent / ".workbench-deleted.json").exists():
        job.update(state="cancelled", error=None)
    job.update(finished_at_s=time.time(), total_job_s=time.perf_counter() - started_job)
    atomic_json(path, job)


def physical_cpu_groups(allowed, workers, budget, topology=None):
    """Select distinct physical cores, reserving two control cores when possible."""
    if topology is None:
        topology = []
        for cpu in sorted(allowed):
            folder = Path(f"/sys/devices/system/cpu/cpu{cpu}")
            try:
                package = int((folder / "topology/physical_package_id").read_text())
                core = int((folder / "topology/core_id").read_text())
            except OSError:
                package, core = 0, cpu
            try:
                speed = int((folder / "cpufreq/cpuinfo_max_freq").read_text())
            except OSError:
                speed = 0
            topology.append((cpu, package, core, speed))
    distinct = {}
    for cpu, package, core, speed in sorted(topology):
        if cpu in allowed:
            distinct.setdefault((package, core), (cpu, speed))
    control_first = sorted(distinct.values())
    reserve = 2 if len(control_first) >= budget + 2 else 0
    ranked = sorted(control_first[reserve:], key=lambda item: (-item[1], item[0]))
    cpus = [cpu for cpu, _ in ranked[:budget]]
    count = min(workers, len(cpus))
    if count < 1:
        raise ValueError("No CPU available for background export")
    return [cpus[i::count] for i in range(count)]


def memory_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def queue_paths(root, active=()):
    candidates = []
    for path in jobs(root):
        if path in active:
            continue
        data = json.loads(path.read_text())
        if data["state"] == "queued":
            candidates.append((data.get("enqueued_at_s", path.stat().st_mtime), str(path), path))
    return [path for _, _, path in sorted(candidates)]


def initialize_worker(groups):
    # Each warm worker gets disjoint physical cores; the parent owns the queue lock.
    identity = multiprocessing.current_process()._identity[-1]
    group = groups[(identity - 1) % len(groups)]
    os.sched_setaffinity(0, group)
    if shutil.which("ionice"):
        # Background reads/writes yield to the foreground recorder's disk I/O.
        subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=True)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if sys.platform == "linux":
        import ctypes

        # A forced parent shutdown must not leave orphan encoders running.
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    from .export import dataset_class

    dataset_class()  # Import Torch/official codecs once, reused across episodes.


def recover(root):
    for path in jobs(root):
        job = json.loads(path.read_text())
        if job["state"] in {"validating", "converting"}:
            job["state"] = "queued"
            atomic_json(path, job)
    for manifest in root.glob("*/episode_*/manifest.json"):
        path = manifest.parent / "job.json"
        if not path.exists() and json.loads(manifest.read_text())["status"] == "pending_validation":
            atomic_json(
                path, {"state": "queued", "attempts": 0, "enqueued_at_s": manifest.stat().st_mtime}
            )


def run_queue(root, workers, cpu_budget):
    groups = physical_cpu_groups(os.sched_getaffinity(0), workers, cpu_budget)
    count = len(groups)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    executor = ProcessPoolExecutor(
        max_workers=count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(groups,),
    )
    active = {}
    try:
        while True:
            for path, future in list(active.items()):
                if future.done():
                    try:
                        future.result()
                    except FileNotFoundError:
                        if path.exists():
                            raise
                    del active[path]
            queued = queue_paths(root, active)
            available = memory_available()
            capacity = count if available >= 3 * 1024**3 else min(count, 1)
            if available < 1536 * 1024**2:
                capacity = 0
            for path in queued[: max(0, capacity - len(active))]:
                active[path] = executor.submit(run_one, path)
            pending = queue_paths(root, active)
            oldest = 0.0
            if pending:
                first = json.loads(pending[0].read_text())
                oldest = max(
                    0, time.time() - first.get("enqueued_at_s", pending[0].stat().st_mtime)
                )
            atomic_json(
                root / ".worker-status.json",
                {
                    "updated_at_s": time.time(),
                    "workers": count,
                    "physical_cpu_groups": groups,
                    "active": [str(p.parent) for p in active],
                    "waiting": len(pending),
                    "oldest_wait_s": oldest,
                    "memory_available_bytes": available,
                    "waiting_for_memory": bool(pending and capacity < count),
                },
            )
            time.sleep(0.5)
    finally:
        for process in multiprocessing.active_children():
            process.terminate()
        executor.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--cpu-budget", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / ".worker.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return

        def terminate(signum, frame):
            raise SystemExit(0)

        signal.signal(signal.SIGTERM, terminate)
        os.setpriority(os.PRIO_PROCESS, 0, 10)
        recover(args.root)
        run_queue(args.root, args.workers, args.cpu_budget)


if __name__ == "__main__":
    main()
