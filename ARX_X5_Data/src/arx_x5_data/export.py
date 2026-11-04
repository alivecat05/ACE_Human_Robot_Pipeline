"""Offline, transactional conversion to the locked official LeRobot main v3 writer."""

import contextlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .media import decoded_frames
from .model import CAMERAS, VECTOR_NAMES
from .recording import atomic_json
from .validation import discover_episodes, rows, video_path

LEROBOT_COMMIT = "ba83e4cf43737e2665be0aafcb5f513b83c7d0f3"


def dataset_class():
    if importlib.metadata.version("lerobot") != "0.6.2":
        raise RuntimeError("Use the locked official LeRobot main environment (0.6.2)")
    import lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_root = Path(lerobot.__file__).resolve().parents[2]
    if (source_root / ".git").exists():
        actual = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
    else:
        direct_url = json.loads(
            importlib.metadata.distribution("lerobot").read_text("direct_url.json") or "{}"
        )
        actual = direct_url.get("vcs_info", {}).get("commit_id")
    if actual != LEROBOT_COMMIT:
        raise RuntimeError("LeRobot source does not match lerobot-main.lock.json")
    return LeRobotDataset


def _save_profiled(dataset, metrics):
    # Profile only this thread and one official function; no global monkey patch.
    from lerobot.datasets.compute_stats import compute_episode_stats
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.dataset_writer import DatasetWriter

    codes = {
        compute_episode_stats.__code__: "statistics_s",
        DatasetWriter._save_episode_data.__code__: "metadata_s",
        LeRobotDatasetMetadata.save_episode.__code__: "metadata_s",
    }
    starts = {}
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if frame.f_code in codes:
            if event == "call":
                starts[id(frame)] = time.perf_counter()
            elif event == "return":
                metrics[codes[frame.f_code]] += time.perf_counter() - starts.pop(id(frame))

    sys.setprofile(profile)
    try:
        dataset.save_episode(parallel_encoding=False)
    finally:
        sys.setprofile(previous)


def _signature(meta: dict) -> str:
    config = meta["config"]
    return json.dumps(
        {
            "fps": meta["fps"],
            "mock": meta["mock"],
            "cameras": config["cameras"],
            "action_semantics": meta["action_semantics"],
        },
        sort_keys=True,
    )


def convert(
    source: Path,
    target: Path,
    repo_id: str,
    *,
    allow_mock: bool = False,
    factory=None,
    mode: str = "reencode_h264",
    h264_gop: int = 8,
    encoder_threads: int = 2,
    h264_backend: str = "software",
    _validated_episode=None,
) -> dict:
    if mode not in {"reuse_mp4", "reencode_av1", "reencode_h264"}:
        raise ValueError("Unknown export mode")
    if mode == "reencode_h264" and (h264_gop not in (2, 8, 15) or not 1 <= encoder_threads <= 6):
        raise ValueError("H.264 supports GOP2/8/15 and 1..6 encoder threads")
    if h264_backend not in {"software", "vaapi"}:
        raise ValueError("Unknown H.264 backend")
    started = time.perf_counter()
    metrics = dict.fromkeys(("copy_s", "statistics_s", "metadata_s", "encoding_s"), 0.0)
    target = target.absolute()
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f".{target.name}.export.lock")
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(descriptor, str(os.getpid()).encode())
    os.fsync(descriptor)
    staging = None
    try:
        phase = time.perf_counter()
        if _validated_episode is None:
            if h264_backend == "vaapi" and (source / "manifest.json").is_file():
                from .validation import validate_episode

                episodes = [validate_episode(source, allow_mock=allow_mock, _collect_stats=True)]
            else:
                episodes = discover_episodes(source, allow_mock=allow_mock)
        else:
            if (
                _validated_episode.path != source.resolve()
                or _validated_episode.manifest["status"] != "complete"
                or (_validated_episode.manifest["mock"] and not allow_mock)
            ):
                raise ValueError("Prevalidated source does not match export request")
            _validated_episode.assert_current()
            episodes = [_validated_episode]
            metrics["validation_reused"] = True
        metrics["validation_s"] = time.perf_counter() - phase
        if len({_signature(episode.manifest) for episode in episodes}) != 1:
            raise ValueError(
                "Cannot combine different camera configurations, fps or mock/real data"
            )
        cls = factory or dataset_class()
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.export-", dir=target.parent))
        temporary_target = staging / "dataset"
        meta = episodes[0].manifest
        features = {
            key: {"dtype": "float32", "shape": (14,), "names": list(VECTOR_NAMES)}
            for key in ("observation.state", "action")
        }
        features["observation.capture_offset_s"] = {
            "dtype": "float32",
            "shape": (4,),
            "names": [*CAMERAS, "telemetry"],
        }
        for name in CAMERAS:
            camera = meta["config"]["cameras"][name]
            features[f"observation.images.{name}"] = {
                "dtype": "video",
                "shape": (camera["height"], camera["width"], 3),
                "names": ["height", "width", "channels"],
            }
        dataset = cls.create(
            repo_id=repo_id,
            root=temporary_target,
            fps=meta["fps"],
            robot_type="arx_x5_2023_bimanual",
            features=features,
            use_videos=True,
            video_backend="pyav",
            image_writer_threads=3 if mode == "reencode_av1" else 0,
        )
        if factory is None and mode == "reencode_av1":
            encode = dataset.writer._encode_temporary_episode_video

            def measured_encode(*args, **kwargs):
                phase = time.perf_counter()
                try:
                    return encode(*args, **kwargs)
                finally:
                    metrics["encoding_s"] += time.perf_counter() - phase

            dataset.writer._encode_temporary_episode_video = measured_encode
        provenance = []
        frame_count = 0
        try:
            with (temporary_target / "meta" / "source_timing.jsonl").open(
                "w", encoding="utf-8"
            ) as timing:
                if mode in {"reuse_mp4", "reencode_h264"}:
                    from .lerobot_reuse import write_reused

                    provenance, frame_count = write_reused(
                        dataset,
                        episodes,
                        timing,
                        metrics,
                        transcode=mode == "reencode_h264",
                        gop=h264_gop,
                        encoder_threads=encoder_threads,
                        h264_backend=h264_backend,
                    )
                for episode in episodes if mode == "reencode_av1" else []:
                    manifest = episode.manifest
                    starts = {a: b for a, b in episode.segments}
                    selected_end = -1
                    with ExitStack() as stack:
                        videos = {}
                        for name in CAMERAS:
                            camera = manifest["config"]["cameras"][name]
                            video = decoded_frames(
                                video_path(episode.path, manifest["videos"][name]),
                                camera["width"],
                                camera["height"],
                                manifest["fps"],
                            )
                            stack.callback(video.close)
                            videos[name] = video
                        for index, row in enumerate(rows(episode.path)):
                            phase = time.perf_counter()
                            images = {name: next(video) for name, video in videos.items()}
                            metrics["rgb_decode_s"] = (
                                metrics.get("rgb_decode_s", 0) + time.perf_counter() - phase
                            )
                            if index in starts:
                                selected_end = starts[index]
                                provenance.append(
                                    {
                                        "episode_index": len(provenance),
                                        "source": str(episode.path),
                                        "saved_frame_range": [index, selected_end],
                                        "manifest": manifest,
                                    }
                                )
                            if index >= selected_end:
                                continue
                            frame = {
                                key: np.asarray(row[key], dtype=np.float32)
                                for key in ("observation.state", "action")
                            }
                            stamp = row["timestamp_monotonic_s"]
                            frame["observation.capture_offset_s"] = np.array(
                                [
                                    *(
                                        row["cameras"][n]["timestamp_monotonic_s"] - stamp
                                        for n in CAMERAS
                                    ),
                                    row["telemetry_timestamp_monotonic_s"] - stamp,
                                ],
                                dtype=np.float32,
                            )
                            frame.update(
                                {
                                    f"observation.images.{name}": image
                                    for name, image in images.items()
                                }
                            )
                            frame["task"] = manifest["task"]
                            phase = time.perf_counter()
                            dataset.add_frame(frame)
                            metrics["add_frame_s"] = (
                                metrics.get("add_frame_s", 0) + time.perf_counter() - phase
                            )
                            timing.write(
                                json.dumps(
                                    {"episode_index": len(provenance) - 1, "source_row": row},
                                    allow_nan=False,
                                )
                                + "\n"
                            )
                            frame_count += 1
                            if index + 1 == selected_end:
                                # The pinned library otherwise forks encoders from
                                # a process with active image-writer threads.
                                phase = time.perf_counter()
                                if factory is None:
                                    _save_profiled(dataset, metrics)
                                else:
                                    dataset.save_episode(parallel_encoding=False)
                                metrics["save_episode_including_stats_s"] = (
                                    metrics.get("save_episode_including_stats_s", 0)
                                    + time.perf_counter()
                                    - phase
                                )
            dataset.finalize()
            if hasattr(dataset, "stop_image_writer"):
                dataset.stop_image_writer()
        except BaseException:
            with contextlib.suppress(Exception):
                dataset.finalize()
            if hasattr(dataset, "stop_image_writer"):
                with contextlib.suppress(Exception):
                    dataset.stop_image_writer()
            raise
        for episode in episodes:
            episode.assert_current()
        atomic_json(temporary_target / "meta" / "arx_sources.json", {"episodes": provenance})
        if factory is None:
            from .lerobot_reuse import verify_official

            phase = time.perf_counter()
            verify_official(temporary_target, repo_id, provenance)
            metrics["official_reload_s"] = time.perf_counter() - phase
        metrics["total_s"] = time.perf_counter() - started
        metrics["mode"] = mode
        metrics["lerobot_commit"] = LEROBOT_COMMIT
        metrics["lerobot_version"] = (
            importlib.metadata.version("lerobot") if factory is None else "test"
        )
        if mode == "reencode_h264":
            metrics["encoder"] = {
                "codec": "h264",
                "gop": h264_gop,
                "requested_backend": h264_backend,
                "per_video": [
                    {k: v for k, v in video.items() if k not in {"source", "camera", "encoding_s"}}
                    for video in metrics.get("videos", [])
                ],
                "threads": encoder_threads,
                "pix_fmt": "yuv420p",
            }
        atomic_json(temporary_target / "meta" / "conversion_metrics.json", metrics)
        for episode in episodes:
            episode.assert_current()
        if target.exists():
            raise FileExistsError(f"Output appeared during conversion: {target}")
        temporary_target.rename(target)
        return {
            "root": str(target),
            "episodes": len(provenance),
            "frames": frame_count,
            "metrics": metrics,
        }
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
        if staging is not None:
            shutil.rmtree(staging)
