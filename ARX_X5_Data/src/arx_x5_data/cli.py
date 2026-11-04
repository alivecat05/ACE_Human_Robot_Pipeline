"""Command-line tools for camera inspection, independent recording and conversion."""

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from .config import load_config
from .export import convert
from .model import CAMERAS
from .runtime import Runtime
from .sources import Sources
from .validation import discover_episodes


def _record(args) -> None:
    if args.seconds is not None and args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    config = load_config(args.config)
    with Runtime(config, mock=args.mock) as runtime:
        print(f"Waiting for cameras and telemetry at {config.socket}", flush=True)
        while runtime.status()["mode"] == "warming":
            time.sleep(0.05)
        if runtime.status()["mode"] != "idle":
            raise RuntimeError(runtime.status()["message"])
        runtime.record(args.task)
        print(f"RECORDING: {runtime.status()['episode']}\nCtrl+C saves this episode.", flush=True)
        deadline = time.monotonic() + args.seconds if args.seconds else float("inf")
        try:
            while time.monotonic() < deadline and runtime.status()["mode"] == "recording":
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        runtime.stop_recording()
        while runtime.status()["mode"] == "saving":
            time.sleep(0.05)
        status = runtime.status()
        if runtime.writer:
            runtime.writer.wait()
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
        if status["mode"] == "fault":
            raise RuntimeError(status["message"])


def _camera_check(args) -> None:
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = Sources(config, mock=args.mock, cameras_only=True)
    try:
        sources.start()
        sources.wait_ready()
        time.sleep(1)  # Discard the initial exposure/stream startup frames.
        sources.check()
        report = {}
        for name in CAMERAS:
            frame = sources.cameras[name].latest()
            Image.fromarray(frame.rgb).save(args.output / f"{name}.jpg", quality=95)
            camera = config.cameras[name]
            report[name] = {
                "device": camera.device,
                "shape": list(frame.rgb.shape),
                "clock": frame.clock,
                "frame_age_ms": (time.monotonic() - frame.timestamp) * 1000,
                "discarded_invalid_packets": frame.discarded_invalid_packets,
            }
        (args.output / "cameras.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        sources.close()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("record", "ui", "camera-check"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("config.local.toml"))
        command.add_argument(
            "--mock", action="store_true", help="Synthetic inputs; never opens devices"
        )
        if name == "record":
            command.add_argument("--task", required=True)
            command.add_argument("--seconds", type=float)
        elif name == "camera-check":
            command.add_argument("-o", "--output", type=Path, required=True)
        else:
            command.add_argument("--port", type=int, default=8080)
    validate = commands.add_parser("validate")
    validate.add_argument("source", type=Path)
    validate.add_argument("--allow-mock", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("-i", "--input", type=Path, required=True)
    export.add_argument("-o", "--output", type=Path, required=True)
    export.add_argument("--repo-id", default="local/arx_x5_2023")
    export.add_argument("--allow-mock", action="store_true")
    export.add_argument(
        "--mode", choices=("reuse_mp4", "reencode_av1", "reencode_h264"), default="reencode_h264"
    )
    export.add_argument("--h264-gop", type=int, choices=(2, 8, 15), default=8)
    export.add_argument("--encoder-threads", type=int, choices=range(1, 7), default=2)
    commands.add_parser("devices", help="List stable V4L2 device paths without opening streams")
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "record":
            _record(args)
        elif args.command == "camera-check":
            _camera_check(args)
        elif args.command == "ui":
            from .dashboard import run

            run(load_config(args.config), mock=args.mock, port=args.port)
        elif args.command == "validate":
            episodes = discover_episodes(args.source, allow_mock=args.allow_mock)
            print(
                json.dumps(
                    [
                        {
                            "path": str(e.path),
                            "frames": e.manifest["frames"],
                            "segments": e.segments,
                        }
                        for e in episodes
                    ],
                    indent=2,
                )
            )
        elif args.command == "export":
            print(
                json.dumps(
                    convert(
                        args.input,
                        args.output,
                        args.repo_id,
                        allow_mock=args.allow_mock,
                        mode=args.mode,
                        h264_gop=args.h264_gop,
                        encoder_threads=args.encoder_threads,
                    ),
                    indent=2,
                )
            )
        elif args.command == "devices":
            for directory in ("by-id", "by-path"):
                for path in sorted((Path("/dev/v4l") / directory).glob("*video-index*")):
                    print(f"{path} -> {path.resolve()}")
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
