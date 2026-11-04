"""Bounded Intel hardware encode with the existing official software fallback."""

import os
import re
import subprocess
import sys
from fractions import Fraction


def run_ffmpeg(args):
    return subprocess.run(
        [sys.executable, "-m", "arx_x5_data.h264_encoder", str(os.getpid()), *args],
        check=True,
        capture_output=True,
        timeout=600,
    )


def encode_h264(source, dest, *, gop, threads, backend="software"):
    if backend not in {"software", "vaapi"}:
        raise ValueError("Unknown H.264 backend")
    if dest.exists():
        raise FileExistsError(dest)
    fallback = None
    if backend == "vaapi":
        try:
            # Inherits the queue worker's CPU affinity/nice. No GPU driver changes.
            run_ffmpeg(
                [
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-vaapi_device",
                    "/dev/dri/renderD128",
                    "-threads",
                    str(threads),
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-vf",
                    "format=nv12,hwupload",
                    "-filter_threads",
                    "1",
                    "-c:v",
                    "h264_vaapi",
                    "-qp",
                    "20",
                    "-g",
                    str(gop),
                    "-bf",
                    "0",
                    "-movflags",
                    "+faststart",
                    "-n",
                    str(dest),
                ],
            )
            return {"backend": "vaapi", "rate_control": "CQP", "qp": 20}
        except (OSError, subprocess.SubprocessError) as error:
            dest.unlink(missing_ok=True)
            fallback = str(error)
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.video_utils import reencode_video

    reencode_video(
        source,
        dest,
        video_encoder=RGBEncoderConfig(vcodec="h264", g=gop, crf=20, preset="veryfast"),
        encoder_threads=threads,
    )
    result = {"backend": "software", "rate_control": "CRF", "crf": 20, "preset": "veryfast"}
    if fallback:
        result["fallback_reason"] = fallback
    return result


def verify_decode_log(log, *, frames, fps, width, height):
    time_base = re.search(r"config in time_base:\s*(\d+/\d+)", log)
    if time_base is None:
        raise ValueError("Missing hardware decoder time base")
    time_base = Fraction(time_base[1])
    entries = re.findall(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+).*?\bs:(\d+)x(\d+)\b", log)
    if len(entries) != frames:
        raise ValueError("H.264 output frame count changed")
    for expected, (index, pts, w, h) in enumerate(entries):
        if int(index) != expected or abs(float(int(pts) * time_base) - expected / fps) > 0.002:
            raise ValueError("H.264 output timestamp changed")
        if (int(w), int(h)) != (width, height):
            raise ValueError("H.264 output dimensions changed")


def verify_hardware_decode(path, *, frames, fps, width, height, threads):
    result = run_ffmpeg(
        [
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "info",
            "-xerror",
            "-vaapi_device",
            "/dev/dri/renderD128",
            "-hwaccel",
            "vaapi",
            "-hwaccel_output_format",
            "vaapi",
            "-threads",
            str(threads),
            "-err_detect",
            "explode",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "showinfo=checksum=0",
            "-filter_threads",
            "1",
            "-f",
            "null",
            "-",
        ]
    )
    verify_decode_log(
        result.stderr.decode("utf-8", errors="replace"),
        frames=frames,
        fps=fps,
        width=width,
        height=height,
    )


if __name__ == "__main__":
    # Set parent-death handling after exec (safe with multi-threaded parent workers).
    # A killed worker cannot leave an unbounded orphan ffmpeg encoder behind.
    import ctypes
    import signal

    expected_parent = int(sys.argv[1])
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0:
        raise RuntimeError("Cannot set encoder parent-death signal")
    if os.getppid() != expected_parent:
        sys.exit(1)
    os.execv("/usr/bin/ffmpeg", ["/usr/bin/ffmpeg", *sys.argv[2:]])
