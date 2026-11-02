#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense multi-camera recording script.

Features:
1) Automatically discovers and starts all connected RealSense cameras
    (optionally limited by CLI arguments).
2) Records RGB and depth streams for each camera.
3) Saves per-frame timestamps to CSV.
4) For D435i devices, also saves IMU data (accelerometer and gyroscope) to CSV.

Output files:
- {serial}_rgb.mp4
- {serial}_depth.mp4
- {serial}_ts.csv
- {serial}_imu.csv (D435i only)
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, TextIO

import imageio
import numpy as np
import pyrealsense2 as rs

COLOR_WIDTH = 848
COLOR_HEIGHT = 480
STREAM_FPS = 15
WARMUP_TIMEOUT_SECONDS = 10
WARMUP_FRAME_TIMEOUT_MS = 3000
RECORD_FRAME_TIMEOUT_MS = 5000
PIPELINE_START_RETRY_COUNT = 5
PIPELINE_START_RETRY_INTERVAL_SECONDS = 1.5


@dataclass
class CameraSpec:
    """Runtime information for one RealSense camera."""

    serial_number: str
    model: str
    device: Any


class CameraWriters:
    """Manage per-camera output files for RGB, depth, timestamps and IMU."""

    def __init__(self, save_dir: str, serial_number: str, has_imu: bool) -> None:
        self._rgb_writer = imageio.get_writer(
            os.path.join(save_dir, f"{serial_number}_rgb.mp4"),
            fps=STREAM_FPS,
            mode="I",
            codec="libx264",
            format="FFMPEG",
            macro_block_size=1,
        )
        self._depth_writer = imageio.get_writer(
            os.path.join(save_dir, f"{serial_number}_depth.mp4"),
            fps=STREAM_FPS,
            mode="I",
            codec="libx264",
            format="FFMPEG",
            macro_block_size=1,
        )
        self._ts_file = open(
            os.path.join(save_dir, f"{serial_number}_ts.csv"),
            "w",
            encoding="utf-8",
        )

        self._imu_file: Optional[TextIO] = None
        if has_imu:
            self._imu_file = open(
                os.path.join(save_dir, f"{serial_number}_imu.csv"),
                "w",
                encoding="utf-8",
            )
            self._imu_file.write(
                "timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n"
            )

    def write_video_and_timestamp(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        timestamp_us: int,
    ) -> None:
        """Write RGB/depth frame and frame timestamp."""
        self._rgb_writer.append_data(rgb_image)
        self._depth_writer.append_data(depth_image)
        self._ts_file.write(f"{timestamp_us}\n")

    def write_imu(
        self,
        timestamp_us: int,
        accel_data: Any,
        gyro_data: Any,
    ) -> None:
        """Write one IMU row when IMU output is enabled."""
        if self._imu_file is None:
            return
        self._imu_file.write(
            f"{timestamp_us},{accel_data.x},{accel_data.y},{accel_data.z},"
            f"{gyro_data.x},{gyro_data.y},{gyro_data.z}\n"
        )

    def close(self) -> None:
        """Close all open file handles for the camera."""
        self._rgb_writer.close()
        self._depth_writer.close()
        self._ts_file.close()
        if self._imu_file is not None:
            self._imu_file.close()

    def flush(self) -> None:
        """Flush CSV buffers to disk."""
        self._ts_file.flush()
        if self._imu_file is not None:
            self._imu_file.flush()


class RealSenseRecorder:
    """Coordinate camera discovery, recording threads, and resource cleanup."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._stop_event = threading.Event()
        self._camera_specs: list[CameraSpec] = []
        self._pipelines: list[Any] = []
        self._writers: list[CameraWriters] = []
        self._threads: list[threading.Thread] = []
        self._save_dir: Optional[str] = None
        self._started = False
        self._colorizer = rs.colorizer()

        try:
            self._context = rs.context()
        except Exception as error:
            raise RuntimeError(
                f"Failed to create RealSense context: {error}"
            ) from error

    def run(self) -> int:
        """Run recorder lifecycle and return process exit code."""
        try:
            if not self.start():
                return 1

            for thread in self._threads:
                thread.join()
        finally:
            self.stop()

        return 0

    def start(self) -> bool:
        """Start camera discovery, pipelines, and recording threads."""
        if self._started:
            return True

        print("=== RealSense Recording Script ===")
        self._stop_event.clear()
        self._camera_specs = []
        self._pipelines = []
        self._writers = []
        self._threads = []

        if not self._discover_cameras():
            return False

        self._save_dir = self._build_save_directory()
        if self._save_dir is None:
            return False

        print(f"Saving mp4 files to: {self._save_dir}", flush=True)
        signal.signal(signal.SIGINT, self._handle_signal)

        try:
            self._start_pipelines()
            self._create_writers(self._save_dir)
            self._warm_up_cameras()
            self._start_recording_threads()
            self._started = True
            return True
        except Exception:
            self.stop()
            raise

    def save(self) -> Optional[str]:
        """Flush buffered metadata to disk and return output directory."""
        for writer in self._writers:
            try:
                writer.flush()
            except Exception as error:
                print(f"Warning: Failed to flush writer: {error}")
        return self._save_dir

    def stop(self) -> None:
        """Stop recording threads and release all resources safely."""
        if not self._started and not self._pipelines and not self._writers:
            return

        self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=2.0)

        self.save()
        self._cleanup()
        self._threads = []
        self._started = False

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle Ctrl+C to stop recording gracefully."""
        del signum, frame
        self._stop_event.set()
        print("\nCtrl+C pressed. Exiting...")

    def _discover_cameras(self) -> bool:
        """Discover all connected RealSense cameras."""
        print("=== Detecting RealSense Cameras ===\n")
        device_list = self._context.query_devices()

        raw_count = len(device_list)
        probe_count = raw_count

        if self._args.max_cameras > 0:
            probe_count = min(probe_count, self._args.max_cameras)
            print(f"Limiting to first {probe_count} camera(s) by --max_cameras")
        else:
            usb_hint_count = self._get_realsense_usb_hint_count()
            if usb_hint_count > 0:
                probe_count = min(probe_count, usb_hint_count)

        if raw_count > probe_count:
            print(
                "Warning: RealSense SDK reported more device entries than probed "
                f"({raw_count} -> {probe_count}) to avoid unstable entries"
            )

        devices = [device_list[i] for i in range(probe_count)]

        if len(devices) == 0:
            self._print_no_camera_hint("No RealSense devices found in system")
            return False

        print(f"Found {len(devices)} RealSense device(s)\n")
        for device in devices:
            camera_spec = self._build_camera_spec(device)
            if camera_spec is None:
                continue
            self._camera_specs.append(camera_spec)
            index = len(self._camera_specs)
            print(
                f"  Camera {index}: {camera_spec.model} "
                f"(SN: {camera_spec.serial_number})"
            )

        if not self._camera_specs:
            self._print_no_camera_hint("Error: No usable RealSense cameras detected!")
            return False

        print(
            f"\n=== Successfully detected {len(self._camera_specs)} "
            "RealSense Camera(s) ==="
        )
        for index, spec in enumerate(self._camera_specs, start=1):
            print(f"  {index}. {spec.model} - SN: {spec.serial_number}")
        return True

    def _get_realsense_usb_hint_count(self) -> int:
        """Best-effort hint for number of physical Intel RealSense USB devices."""
        try:
            result = subprocess.check_output(
                "lsusb | grep -i '8086' | grep -vi 'Hub' | wc -l",
                shell=True,
                text=True,
            )
            return max(0, int(result.strip() or "0"))
        except (subprocess.SubprocessError, OSError, ValueError):
            return 0

    def _build_camera_spec(self, device: Any) -> Optional[CameraSpec]:
        """Create CameraSpec from device metadata."""
        try:
            if not device.supports(rs.camera_info.serial_number):
                print("  Skipping device without serial number")
                return None

            serial_number = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name).upper()
            model = "D405" if "D405" in name else "D435i"
            return CameraSpec(
                serial_number=serial_number,
                model=model,
                device=device,
            )
        except Exception as error:
            print(f"  Error reading device info: {error}")
            return None

    def _print_no_camera_hint(self, title: str) -> None:
        """Print standard troubleshooting hints for camera detection failure."""
        print(f"\n{title}")
        print("Please check:")
        print("  1. Cameras are properly connected via USB 3.0")
        print("  2. No other process is using the cameras")
        print("  3. USB permissions are correct")

    def _build_save_directory(self) -> Optional[str]:
        """Build and create recording directory."""
        if not self._args.record_name:
            print("Error: missing --record_name")
            return None

        date_str = datetime.now().strftime("%Y%m%d")
        save_dir = os.path.join(
            self._args.save_path,
            date_str,
            self._args.record_name,
            "realsense",
        )
        if os.path.exists(save_dir):
            print("Record already exists!!")
            return None

        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _start_pipelines(self) -> None:
        """Start a RealSense pipeline for each detected camera."""
        for spec in self._camera_specs:
            pipeline = rs.pipeline(self._context)
            config = rs.config()
            config.enable_device(spec.serial_number)

            config.enable_stream(
                rs.stream.color,
                COLOR_WIDTH,
                COLOR_HEIGHT,
                rs.format.rgb8,
                STREAM_FPS,
            )
            config.enable_stream(
                rs.stream.depth,
                COLOR_WIDTH,
                COLOR_HEIGHT,
                rs.format.z16,
                STREAM_FPS,
            )

            self._try_enable_imu(spec, config)
            self._start_pipeline_with_retry(pipeline, config, spec.serial_number)
            self._configure_exposure(pipeline)

            self._pipelines.append(pipeline)
            print(f"Started camera {spec.serial_number} ({spec.model})", flush=True)

    @staticmethod
    def _is_device_busy_error(error: Exception) -> bool:
        """Returns True when error indicates camera device is busy."""
        text = str(error).lower()
        busy_markers = [
            "resource busy",
            "errno=16",
            "vidioc_s_fmt",
            "device or resource busy",
        ]
        return any(marker in text for marker in busy_markers)

    def _start_pipeline_with_retry(
        self, pipeline: Any, config: Any, serial_number: str
    ) -> None:
        """Starts one pipeline with retries when UVC device is temporarily busy."""
        last_error: Optional[Exception] = None
        for attempt in range(1, PIPELINE_START_RETRY_COUNT + 1):
            try:
                pipeline.start(config)
                return
            except Exception as error:
                last_error = error
                if not self._is_device_busy_error(error):
                    raise

                print(
                    f"  Camera {serial_number} busy on start "
                    f"(attempt {attempt}/{PIPELINE_START_RETRY_COUNT}): {error}"
                )

                if attempt == 1:
                    self._hardware_reset_camera(serial_number)

                if attempt < PIPELINE_START_RETRY_COUNT:
                    time.sleep(PIPELINE_START_RETRY_INTERVAL_SECONDS)

        raise RuntimeError(
            f"Camera {serial_number} start failed after retries: {last_error}"
        )

    def _hardware_reset_camera(self, serial_number: str) -> None:
        """Best-effort hardware reset for one busy camera."""
        try:
            for device in self._context.query_devices():
                if not device.supports(rs.camera_info.serial_number):
                    continue
                if device.get_info(rs.camera_info.serial_number) != serial_number:
                    continue
                print(f"  Camera {serial_number} hardware reset ...")
                device.hardware_reset()
                # 等待设备重新枚举
                time.sleep(2.0)
                return
        except Exception as error:
            print(f"  Camera {serial_number} hardware reset failed: {error}")

    def _try_enable_imu(self, spec: CameraSpec, config: Any) -> None:
        """Enable IMU streams for D435i when matching profiles are available."""
        if spec.model != "D435i":
            return

        accel_profile, gyro_profile = self._lock_imu_profiles(
            spec.device, spec.serial_number
        )
        if accel_profile is None or gyro_profile is None:
            print(f"  {spec.serial_number} no usable IMU profile, skipped")
            return

        config.enable_stream(
            accel_profile.stream_type(),
            accel_profile.format(),
            accel_profile.fps(),
        )
        config.enable_stream(
            gyro_profile.stream_type(),
            gyro_profile.format(),
            gyro_profile.fps(),
        )
        print(
            f"  {spec.serial_number} IMU locked "
            f"{accel_profile.fps()}/{gyro_profile.fps()} Hz"
        )

    def _lock_imu_profiles(
        self,
        device: Any,
        serial_number: str,
    ) -> tuple[Optional[Any], Optional[Any]]:
        """Return matched accel/gyro profiles or (None, None)."""
        try:
            motion_sensor = None
            for sensor in device.query_sensors():
                sensor_name = sensor.get_info(rs.camera_info.name).lower()
                if "motion" in sensor_name:
                    motion_sensor = sensor
                    break

            if motion_sensor is None:
                return None, None

            accel_profiles = {
                profile
                for profile in motion_sensor.get_stream_profiles()
                if profile.stream_type() == rs.stream.accel
            }
            gyro_profiles = {
                profile
                for profile in motion_sensor.get_stream_profiles()
                if profile.stream_type() == rs.stream.gyro
            }

            if serial_number == "934222070042":
                accel = next((p for p in accel_profiles if p.fps() == 63), None)
                gyro = next((p for p in gyro_profiles if p.fps() == 200), None)
                return accel, gyro

            for accel_fps, gyro_fps in [(200, 200), (63, 200)]:
                accel = next((p for p in accel_profiles if p.fps() == accel_fps), None)
                gyro = next((p for p in gyro_profiles if p.fps() == gyro_fps), None)
                if accel is not None and gyro is not None:
                    return accel, gyro

            return None, None
        except Exception as error:
            print(f"Warning: Failed to configure IMU for {serial_number}: {error}")
            return None, None

    def _configure_exposure(self, pipeline: Any) -> None:
        """Configure color sensor exposure policy based on CLI arguments."""
        device = pipeline.get_active_profile().get_device()
        for sensor in device.query_sensors():
            if sensor.is_color_sensor():
                try:
                    if self._args.exposure_mode == "manual":
                        sensor.set_option(rs.option.enable_auto_exposure, 0)
                        if self._args.manual_exposure is not None:
                            sensor.set_option(
                                rs.option.exposure, float(self._args.manual_exposure)
                            )
                        if self._args.manual_gain is not None:
                            sensor.set_option(
                                rs.option.gain, float(self._args.manual_gain)
                            )
                        print(
                            "  Exposure mode: manual"
                            f" (exposure={self._args.manual_exposure}, gain={self._args.manual_gain})"
                        )
                    else:
                        sensor.set_option(rs.option.enable_auto_exposure, 1)
                        print("  Exposure mode: auto")

                    sensor.set_option(rs.option.enable_auto_white_balance, 1)
                except Exception as error:
                    print(f"Warning: Failed to set exposure options: {error}")

    def _create_writers(self, save_dir: str) -> None:
        """Create all file writers for current camera list."""
        self._writers = [
            CameraWriters(
                save_dir=save_dir,
                serial_number=spec.serial_number,
                has_imu=(spec.model == "D435i"),
            )
            for spec in self._camera_specs
        ]

    def _warm_up_cameras(self) -> None:
        """Warm up camera streams before starting recording threads."""
        print("\nWarming up cameras...")
        warmup_start = time.time()

        for index, pipeline in enumerate(self._pipelines):
            if time.time() - warmup_start > WARMUP_TIMEOUT_SECONDS:
                print("  Warmup timeout reached, skipping remaining cameras")
                break

            serial_number = self._camera_specs[index].serial_number
            try:
                pipeline.wait_for_frames(timeout_ms=WARMUP_FRAME_TIMEOUT_MS)
                print(f"  Camera {serial_number} ready")
            except RuntimeError:
                print(
                    f"  Camera {serial_number} warmup skipped "
                    "(will start in recording thread)"
                )

        print(f"Warmup completed in {time.time() - warmup_start:.1f}s")
        time.sleep(0.2)

    def _start_recording_threads(self) -> None:
        """Start one recording thread per camera."""
        print("\n=== Recording RGBD+IMU... Press Ctrl+C to stop ===")
        barrier = threading.Barrier(len(self._pipelines))
        self._threads = [
            threading.Thread(
                target=self._record_worker,
                args=(index, barrier),
                daemon=True,
            )
            for index in range(len(self._pipelines))
        ]

        for thread in self._threads:
            thread.start()

    def _record_worker(self, index: int, barrier: threading.Barrier) -> None:
        """Record frames for one camera until stop event is set."""
        pipeline = self._pipelines[index]
        writer = self._writers[index]
        spec = self._camera_specs[index]

        barrier.wait()
        while not self._stop_event.is_set():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=RECORD_FRAME_TIMEOUT_MS)
            except RuntimeError as error:
                print(f"  Warning: Camera {spec.serial_number} timeout: {error}")
                continue
            except Exception as error:
                print(f"  Error in camera {spec.serial_number}: {error}")
                break

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            rgb_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(
                self._colorizer.colorize(depth_frame).get_data()
            )
            timestamp_us = frames.get_frame_metadata(
                rs.frame_metadata_value.backend_timestamp
            )
            writer.write_video_and_timestamp(rgb_image, depth_image, timestamp_us)

            if spec.model == "D435i":
                self._write_imu_data(frames, writer, timestamp_us)

    def _write_imu_data(
        self, frames: Any, writer: CameraWriters, timestamp_us: int
    ) -> None:
        """Write IMU row if current frameset contains accel and gyro data."""
        accel_frame = frames.first_or_default(rs.stream.accel, rs.format.motion_xyz32f)
        gyro_frame = frames.first_or_default(rs.stream.gyro, rs.format.motion_xyz32f)
        if accel_frame and gyro_frame:
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            writer.write_imu(timestamp_us, accel_data, gyro_data)

    def _cleanup(self) -> None:
        """Stop pipelines and close all output files safely."""
        for writer in self._writers:
            try:
                writer.close()
            except Exception as error:
                print(f"Warning: Failed to close writer: {error}")

        for pipeline in self._pipelines:
            try:
                pipeline.stop()
            except Exception as error:
                print(f"Warning: Failed to stop pipeline: {error}")

        self._writers = []
        self._pipelines = []
        print(">>> All camera pipelines stopped.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Record multi-camera RealSense RGBD+IMU"
    )
    parser.add_argument(
        "--record_name",
        type=str,
        default="112233",
        help="Recording name, required for output directory naming.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=".",
        help="Base directory to save result files.",
    )
    parser.add_argument(
        "--max_cameras",
        type=int,
        default=2,
        help="Maximum number of cameras to record. 0 means all detected cameras.",
    )
    parser.add_argument(
        "--exposure_mode",
        type=str,
        choices=["auto", "manual"],
        default="auto",
        help="Exposure strategy for color sensor(s). Default keeps current auto mode.",
    )
    parser.add_argument(
        "--manual_exposure",
        type=float,
        default=None,
        help="Manual exposure value when --exposure_mode manual.",
    )
    parser.add_argument(
        "--manual_gain",
        type=float,
        default=None,
        help="Manual gain value when --exposure_mode manual.",
    )
    return parser.parse_args()


def main() -> int:
    """Program entry point."""
    args = parse_args()
    try:
        recorder = RealSenseRecorder(args)
        return recorder.run()
    except RuntimeError as error:
        print(error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
