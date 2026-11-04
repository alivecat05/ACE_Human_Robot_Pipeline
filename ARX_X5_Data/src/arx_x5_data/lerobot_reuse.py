"""Commit-locked main adapter: MP4 copy or direct H.264 conversion without PNG."""

import shutil
import subprocess
import time

import av
import numpy as np

from .media import decoded_frames
from .model import CAMERAS
from .validation import rows, video_path


def write_reused(
    dataset,
    episodes,
    timing,
    metrics,
    *,
    transcode=False,
    gop=8,
    encoder_threads=2,
    h264_backend="software",
):
    from lerobot.datasets.compute_stats import (
        auto_downsample_height_width,
        compute_episode_stats,
        get_feature_stats,
        sample_indices,
    )

    provenance = []
    for source_index, episode in enumerate(episodes):
        meta = episode.manifest
        fps = meta["fps"]
        records = list(rows(episode.path))
        started = time.perf_counter()
        for name in CAMERAS:
            key = f"observation.images.{name}"
            dest = dataset.root / dataset.meta.video_path.format(
                video_key=key, chunk_index=source_index // 1000, file_index=source_index % 1000
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = video_path(episode.path, meta["videos"][name])
            if transcode:
                from .h264_encoder import encode_h264

                phase = time.perf_counter()
                encoder = encode_h264(
                    source, dest, gop=gop, threads=encoder_threads, backend=h264_backend
                )
                elapsed = time.perf_counter() - phase
                metrics["encoding_s"] += elapsed
                metrics.setdefault("videos", []).append(
                    {"source": str(source), "camera": name, "encoding_s": elapsed, **encoder}
                )
                # Verify every output PTS/frame before publishing without RGB materialization.
                phase = time.perf_counter()
                with av.open(str(dest)) as container:
                    stream = container.streams.video[0]
                    stream.thread_count = encoder_threads
                    camera = meta["config"]["cameras"][name]
                    if (
                        (stream.width, stream.height) != (camera["width"], camera["height"])
                        or stream.codec_context.name != "h264"
                        or stream.pix_fmt != "yuv420p"
                    ):
                        raise ValueError("H.264 output format/dimensions changed")
                    hardware_verified = False
                    if encoder["backend"] == "vaapi":
                        from .h264_encoder import verify_hardware_decode

                        try:
                            verify_hardware_decode(
                                dest,
                                frames=meta["frames"],
                                fps=fps,
                                width=camera["width"],
                                height=camera["height"],
                                threads=encoder_threads,
                            )
                            hardware_verified = True
                        except (OSError, subprocess.SubprocessError):
                            # Fall back to full software decode; never skip verification.
                            pass
                    metrics.setdefault("output_verifiers", []).append(
                        "vaapi" if hardware_verified else "software"
                    )
                    if not hardware_verified:
                        count = 0
                        for count, frame in enumerate(container.decode(video=0), start=1):
                            if (
                                frame.pts is None
                                or abs(float(frame.pts * frame.time_base) - (count - 1) / fps)
                                > 0.002
                            ):
                                raise ValueError("H.264 output timestamp changed")
                        if count != meta["frames"]:
                            raise ValueError("H.264 output frame count changed")
                metrics["output_validation_s"] = (
                    metrics.get("output_validation_s", 0) + time.perf_counter() - phase
                )
            else:
                shutil.copyfile(source, dest)
        if not transcode:
            metrics["copy_s"] += time.perf_counter() - started
        started = time.perf_counter()
        if episode.camera_stats is not None:
            camera_stats = episode.camera_stats
            metrics["statistics_reused"] = True
        else:
            camera_stats = [{} for _ in episode.segments]
            metrics.setdefault("statistics_rgb_frames", 0)
            for name in CAMERAS:
                camera = meta["config"]["cameras"][name]
                frames = decoded_frames(
                    video_path(episode.path, meta["videos"][name]),
                    camera["width"],
                    camera["height"],
                    fps,
                    rgb=False,
                )
                cursor = 0
                try:
                    for segment_index, (begin, end) in enumerate(episode.segments):
                        wanted = set(begin + i for i in sample_indices(end - begin))
                        images = []
                        while cursor < end:
                            frame = next(frames)
                            if cursor in wanted:
                                image = frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
                                images.append(auto_downsample_height_width(image).copy())
                                metrics["statistics_rgb_frames"] += 1
                            cursor += 1
                        values = get_feature_stats(np.stack(images), axis=(0, 2, 3), keepdims=True)
                        camera_stats[segment_index][f"observation.images.{name}"] = {
                            k: v if k == "count" else np.squeeze(v / 255.0, axis=0)
                            for k, v in values.items()
                        }
                finally:
                    frames.close()
            metrics["statistics_s"] += time.perf_counter() - started
        for segment_index, (begin, end) in enumerate(episode.segments):
            started = time.perf_counter()
            index = dataset.meta.total_episodes
            selected = records[begin:end]
            n = len(selected)
            dataset.meta.save_episode_tasks([meta["task"]])
            buffer = {
                "index": np.arange(dataset.meta.total_frames, dataset.meta.total_frames + n),
                "episode_index": np.full(n, index),
                "frame_index": np.arange(n),
                "timestamp": np.arange(n, dtype=np.float32) / fps,
                "task_index": np.full(n, dataset.meta.get_task_index(meta["task"])),
            }
            for key in ("observation.state", "action"):
                buffer[key] = np.asarray([r[key] for r in selected], dtype=np.float32)
            buffer["observation.capture_offset_s"] = np.asarray(
                [
                    [
                        *(
                            r["cameras"][c]["timestamp_monotonic_s"] - r["timestamp_monotonic_s"]
                            for c in CAMERAS
                        ),
                        r["telemetry_timestamp_monotonic_s"] - r["timestamp_monotonic_s"],
                    ]
                    for r in selected
                ],
                dtype=np.float32,
            )
            stats = compute_episode_stats(buffer, dataset.features)
            stats.update(camera_stats[segment_index])
            metrics["statistics_s"] += time.perf_counter() - started
            started = time.perf_counter()
            metadata = dataset.writer._save_episode_data(buffer)
            for name in CAMERAS:
                key = f"observation.images.{name}"
                metadata.update(
                    {
                        f"videos/{key}/chunk_index": source_index // 1000,
                        f"videos/{key}/file_index": source_index % 1000,
                        f"videos/{key}/from_timestamp": begin / fps,
                        f"videos/{key}/to_timestamp": end / fps,
                    }
                )
            if transcode and all(v["backend"] == "software" for v in metrics["videos"]):
                from lerobot.configs.video import RGBEncoderConfig

                dataset.meta.update_video_info(
                    video_encoder=RGBEncoderConfig(vcodec="h264", g=gop, crf=20, preset="veryfast")
                )
            else:
                dataset.meta.update_video_info()
            dataset.meta.save_episode(index, n, [meta["task"]], stats, metadata)
            metrics["metadata_s"] += time.perf_counter() - started
            provenance.append(
                {
                    "episode_index": index,
                    "source": str(episode.path),
                    "saved_frame_range": [begin, end],
                    "manifest": meta,
                }
            )
            for row in selected:
                import json

                timing.write(
                    json.dumps({"episode_index": index, "source_row": row}, allow_nan=False) + "\n"
                )
    return provenance, dataset.meta.total_frames


def verify_official(root, repo_id, provenance):
    """Read numerical arrays in full and decode segment edges using the official loader."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    cursor = 0
    for item in provenance:
        begin, end = item["saved_frame_range"]
        expected = list(rows(__import__("pathlib").Path(item["source"])))[begin:end]
        for key in ("observation.state", "action"):
            actual = np.asarray(dataset.hf_dataset[cursor : cursor + len(expected)][key])
            if not np.allclose(
                actual, np.asarray([r[key] for r in expected], dtype=np.float32), rtol=0, atol=1e-7
            ):
                raise ValueError(f"Official reload differs: {key}")
        for offset in {0, len(expected) - 1}:
            sample = dataset[cursor + offset]
            for name in CAMERAS:
                if not np.isfinite(sample[f"observation.images.{name}"].numpy()).all():
                    raise ValueError("Official video decode returned invalid pixels")
        cursor += len(expected)
    if dataset.num_frames != cursor or dataset.num_episodes != len(provenance):
        raise ValueError("Official reload frame/episode count mismatch")
