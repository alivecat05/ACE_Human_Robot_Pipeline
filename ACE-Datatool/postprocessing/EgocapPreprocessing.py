"""Postprocessing module: EgocapPreprocessing.

Args:
    None.

Returns:
    None.

Raises:
    None.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import concurrent.futures
import csv
import json
import os
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from utils import prefillter
from ColorSync import ColorSyncConfig, ColorSyncPipeline


IMURow = Tuple[int, int, float, float, float, float, float, float]
FrameRow = Tuple[int, int, bytes]


@dataclass
class EgocapPreprocessingConfig:
    """Class EgocapPreprocessingConfig.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    truncate_list: Tuple[int, ...] = (120, 180)
    rs_default_latency_value: float = 110
    ego_default_latency_value: float = 190
    latency_diff_range: Tuple[float, float] = (80, 150)
    max_image_workers: int = 16
    truncate_time: int = 60


class EgocapPreprocessor:

    """Class EgocapPreprocessor.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    def __init__(
        self,
        folder_path: str,
    ) -> None:
        """__init__.

        Args:
            folder_path: Input parameter.

        Returns:
            None.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        self.root = os.path.abspath(folder_path)
        self.config = EgocapPreprocessingConfig
        self.color_sync_pipeline_cls = ColorSyncPipeline
        self.prefilter_func = prefillter
        self.ts_name = self._build_ts_name(self.root)

    def _run_color_sync(
        self,
        data_path: str,
        color_log_path: str = "",
        truncate_time: int = 80,
    ) -> bool:
        """_run_color_sync.

        Args:
            data_path: Input parameter.
            color_log_path: Input parameter.
            truncate_time: Input parameter.

        Returns:
            bool: Result of _run_color_sync.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        pipeline = self.color_sync_pipeline_cls(
            ColorSyncConfig(
                data_path=data_path,
                color_log_path=color_log_path,
                truncate_time=self.config.truncate_time,
            )
        )
        return pipeline.execute()

    @staticmethod
    def _build_ts_name(root_path: str) -> str:
        """_build_ts_name.

        Args:
            root_path: Input parameter.

        Returns:
            str: Result of _build_ts_name.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        root_name = os.path.basename(root_path)
        return root_name.replace("2026", "").split("-")[0]

    @staticmethod
    def _write_single_image(args: Tuple[str, bytes]) -> None:
        """_write_single_image.

        Args:
            args: Input parameter.

        Returns:
            None.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        out_path, jpeg_data = args
        with open(out_path, "wb") as f:
            f.write(jpeg_data)

    def load_imu_csv(self, path: str) -> List[IMURow]:
        """load_imu_csv.

        Args:
            path: Input parameter.

        Returns:
            List[IMURow]: Result of load_imu_csv.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        imu_data: List[IMURow] = []
        has_dual_timestamps = None

        with open(path, "rb") as f:
            for raw in f:
                if not raw:
                    break
                if b"\x00" in raw:
                    break

                line = raw.decode("ascii", errors="ignore").strip()
                if not line:
                    continue

                if line.startswith("device_timestamp_ns"):
                    has_dual_timestamps = True
                    continue
                if line.startswith("timestamp_ns"):
                    has_dual_timestamps = False
                    continue

                parts = line.split(",")

                if has_dual_timestamps is None:
                    has_dual_timestamps = len(parts) == 8

                if has_dual_timestamps:
                    if len(parts) != 8:
                        break
                    try:
                        device_ts = int(parts[0])
                        host_ts = int(parts[1])
                        gx, gy, gz = float(parts[2]), float(parts[3]), float(parts[4])
                        ax, ay, az = float(parts[5]), float(parts[6]), float(parts[7])
                    except ValueError:
                        break
                    imu_data.append((device_ts, host_ts, gx, gy, gz, ax, ay, az))
                else:
                    if len(parts) != 7:
                        break
                    try:
                        device_ts = int(parts[0])
                        host_ts = device_ts
                        gx, gy, gz = float(parts[1]), float(parts[2]), float(parts[3])
                        ax, ay, az = float(parts[4]), float(parts[5]), float(parts[6])
                    except ValueError:
                        break
                    imu_data.append((device_ts, host_ts, gx, gy, gz, ax, ay, az))

        return imu_data

    def load_mjpeg(self, path: str) -> List[FrameRow]:
        """load_mjpeg.

        Args:
            path: Input parameter.

        Returns:
            List[FrameRow]: Result of load_mjpeg.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        frames: List[FrameRow] = []
        with open(path, "rb") as f:
            magic = f.read(10)
            if len(magic) != 10 or not magic.startswith(b"ROBATA"):
                return frames

            version = magic[6]
            has_dual_timestamps = version >= 2

            while True:
                if has_dual_timestamps:
                    header = f.read(20)
                    if len(header) < 20:
                        break
                    device_ts_ns, host_ts_ns, size = struct.unpack("<QQI", header)
                else:
                    header = f.read(12)
                    if len(header) < 12:
                        break
                    device_ts_ns, size = struct.unpack("<QI", header)
                    host_ts_ns = device_ts_ns

                if size <= 0:
                    break

                data = f.read(size)
                if len(data) < size:
                    break

                frames.append((device_ts_ns, host_ts_ns, data))

        return frames

    def write_clean_imu(self, dir_path: str, imu_data: List[IMURow]) -> None:
        """write_clean_imu.

        Args:
            dir_path: Input parameter.
            imu_data: Input parameter.

        Returns:
            None.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        out_path = os.path.join(dir_path, "imu_clean.csv")
        if not imu_data:
            return

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "device_timestamp_ns",
                    "host_timestamp_ns",
                    "gyro_x",
                    "gyro_y",
                    "gyro_z",
                    "acc_x",
                    "acc_y",
                    "acc_z",
                ]
            )
            writer.writerows(imu_data)

    def write_frames(self, dir_path: str, frames: List[FrameRow]) -> None:
        """write_frames.

        Args:
            dir_path: Input parameter.
            frames: Input parameter.

        Returns:
            None.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        rgb_dir = os.path.join(dir_path, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)

        tasks = []
        for index, (device_ts_ns, host_ts_ns, jpeg) in enumerate(frames):
            filename = f"{index:06d}_{device_ts_ns}_{host_ts_ns}.jpg"
            out_path = os.path.join(rgb_dir, filename)
            tasks.append((out_path, jpeg))

        max_workers = min(
            self.config.max_image_workers,
            max(1, (os.cpu_count() or 1) * 4),
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(self._write_single_image, tasks))

        print(f"[INFO] wrote {len(frames)} images to: {rgb_dir}")

    @staticmethod
    def align_by_timestamp(
        imu_data: List[IMURow], frames: List[FrameRow]
    ) -> Tuple[List[IMURow], List[FrameRow]]:
        """align_by_timestamp.

        Args:
            imu_data: Input parameter.
            frames: Input parameter.

        Returns:
            Tuple[List[IMURow], List[FrameRow]]: Result of align_by_timestamp.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        if not imu_data or not frames:
            return [], []

        imu_ts = [row[0] for row in imu_data]
        frame_ts = [device_ts for device_ts, _, _ in frames]

        t_min = max(imu_ts[0], frame_ts[0])
        t_max = min(imu_ts[-1], frame_ts[-1])

        if t_min > t_max:
            return [], []

        imu_aligned = [row for row in imu_data if t_min <= row[0] <= t_max]
        frames_aligned = [
            (device_ts, host_ts, img)
            for device_ts, host_ts, img in frames
            if t_min <= device_ts <= t_max
        ]

        return imu_aligned, frames_aligned

    def process_one_recording(self, dir_path: str) -> None:
        """process_one_recording.

        Args:
            dir_path: Input parameter.

        Returns:
            None.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        imu_csv_path = os.path.join(dir_path, "imu.csv")
        mjpeg_path = os.path.join(dir_path, "mjpeg.bin")

        if not os.path.isfile(imu_csv_path) or not os.path.isfile(mjpeg_path):
            return

        print(f"[INFO] processing recording directory: {dir_path}")

        imu_data = self.load_imu_csv(imu_csv_path)
        frames = self.load_mjpeg(mjpeg_path)

        imu_aligned, frames_aligned = self.align_by_timestamp(imu_data, frames)

        if imu_aligned and frames_aligned:
            self.write_clean_imu(dir_path, imu_aligned)
            self.write_frames(dir_path, frames_aligned)

    @staticmethod
    def find_recording_dirs(root_path: str) -> List[str]:
        """find_recording_dirs.

        Args:
            root_path: Input parameter.

        Returns:
            List[str]: Result of find_recording_dirs.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        if os.path.isdir(root_path):
            if os.path.isfile(os.path.join(root_path, "imu.csv")) and os.path.isfile(
                os.path.join(root_path, "mjpeg.bin")
            ):
                return [root_path]

            rec_dirs = []
            for name in sorted(os.listdir(root_path)):
                p = os.path.join(root_path, name)
                if (
                    os.path.isdir(p)
                    and os.path.isfile(os.path.join(p, "imu.csv"))
                    and os.path.isfile(os.path.join(p, "mjpeg.bin"))
                ):
                    rec_dirs.append(p)
            return rec_dirs
        return []

    def sync_color_again(self, path: str, color_log_path: str = "") -> None:
        """sync_color_again.

        Args:
            path: Input parameter.
            color_log_path: Input parameter.

        Returns:
            None.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        is_check = False

        for truncate_time in self.config.truncate_list:
            print(
                f"\n[INFO] Re-checking synchronization with truncate_time={truncate_time} seconds..."
            )
            is_check = self._run_color_sync(
                data_path=path,
                color_log_path=color_log_path,
                truncate_time=truncate_time,
            )
            if is_check:
                print(
                    f"[WARNING] Synchronization check still indicates potential default latency issues with truncate_time={truncate_time} seconds. Please review the logs for {path} for details."
                )
            else:
                print(
                    f"[INFO] Synchronization check passed with truncate_time={truncate_time} seconds. Latency issues may be resolved or less severe."
                )

        if not is_check:
            return

        with open(
            os.path.join(path, "sync_color_results", "all_camera_latencies.json"), "r"
        ) as f:
            latencies = json.load(f)

        rs_default_value = self.config.rs_default_latency_value
        ego_default_value = self.config.ego_default_latency_value

        diff_range = self.config.latency_diff_range

        camera_ids = [cam_id for cam_id in latencies.keys() if cam_id != "egocap"]

        if len(camera_ids) == 2:
            cam1_id, cam2_id = camera_ids[0], camera_ids[1]
            cam1_latency = latencies[cam1_id]
            cam2_latency = latencies[cam2_id]

            if cam1_latency == rs_default_value and cam2_latency != rs_default_value:
                latencies[cam1_id] = cam2_latency
                print(
                    f"[INFO] Camera {cam1_id} has default value {rs_default_value}, replaced with {cam2_latency} from camera {cam2_id}"
                )
            elif cam2_latency == rs_default_value and cam1_latency != rs_default_value:
                latencies[cam2_id] = cam1_latency
                print(
                    f"[INFO] Camera {cam2_id} has default value {rs_default_value}, replaced with {cam1_latency} from camera {cam1_id}"
                )
            elif cam1_latency == rs_default_value and cam2_latency == rs_default_value:
                print(
                    f"[WARNING] Both cameras {cam1_id} and {cam2_id} have default value {rs_default_value}, cannot replace each other"
                )

        if "egocap" in latencies and latencies["egocap"] == ego_default_value:
            print(
                f"\n[INFO] Egocap has default value {ego_default_value}, searching neighboring folders..."
            )

            other_values = [
                latencies[cid]
                for cid in camera_ids
                if latencies[cid] != ego_default_value
            ]

            parent_dir = os.path.dirname(path)
            current_folder = os.path.basename(path)

            all_folders = sorted(
                [
                    f
                    for f in os.listdir(parent_dir)
                    if os.path.isdir(os.path.join(parent_dir, f))
                ]
            )

            current_idx = (
                all_folders.index(current_folder)
                if current_folder in all_folders
                else -1
            )

            replacement_found = False

            if other_values:
                avg_other = sum(other_values) / len(other_values)
                print(f"[INFO] Average of other cameras: {avg_other:.2f}")

                search_offsets = [-1, 1, -2, 2, -3, 3]

                for offset in search_offsets:
                    neighbor_idx = current_idx + offset
                    if 0 <= neighbor_idx < len(all_folders):
                        neighbor_folder = all_folders[neighbor_idx]
                        neighbor_path = os.path.join(
                            parent_dir,
                            neighbor_folder,
                            "sync_color_results",
                            "all_camera_latencies.json",
                        )

                        if os.path.exists(neighbor_path):
                            try:
                                with open(neighbor_path, "r") as f:
                                    neighbor_data = json.load(f)

                                if "egocap" in neighbor_data:
                                    neighbor_egocap = neighbor_data["egocap"]
                                    diff = abs(neighbor_egocap - avg_other)
                                    print(
                                        f"[INFO] Found egocap={neighbor_egocap:.2f} in {neighbor_folder}, diff={diff:.2f}"
                                    )

                                    if diff_range[0] <= diff <= diff_range[1]:
                                        latencies["egocap"] = neighbor_egocap
                                        print(
                                            f"[SUCCESS] Replaced egocap with {neighbor_egocap:.2f} from {neighbor_folder}"
                                        )
                                        print(
                                            f"         (Difference {diff:.2f} is within range {diff_range})"
                                        )
                                        replacement_found = True
                                        break
                                    else:
                                        print(
                                            f"[SKIP] Difference {diff:.2f} not in valid range {diff_range}"
                                        )
                            except (OSError, json.JSONDecodeError) as e:
                                print(f"[ERROR] Failed to read {neighbor_path}: {e}")

            if not replacement_found:
                if not other_values:
                    print(
                        "[WARNING] No valid other camera values to calculate difference"
                    )

                random_value = random.uniform(diff_range[0], diff_range[1])
                latencies["egocap"] = min(cam1_latency, cam2_latency) + random_value
                print(
                    f"[INFO] Generated random value {random_value:.2f} in range {diff_range} as egocap value"
                )
                print(
                    f"[INFO] Replaced default value {ego_default_value} with generated value{latencies['egocap']}"
                )

        output_path = os.path.join(
            path, "sync_color_results", "all_camera_latencies.json"
        )
        with open(output_path, "w") as f:
            json.dump(latencies, f, indent=4)

        print("\n[INFO] Processed latencies saved to all_camera_latencies.json")
        print(f"       Final values: {latencies}")

    def execute(self) -> None:
        """execute.

        Args:
            None.

        Returns:
            None.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        tobe_check: List[str] = []
        self.prefilter_func(self.root)

        data_dirs = [
            d
            for d in sorted(os.listdir(self.root))
            if os.path.isdir(os.path.join(self.root, d))
        ]

        for data in data_dirs:
            if data.startswith("ts"):
                continue

            path = os.path.join(self.root, data, "egocap")
            rgb_folder = list(Path(path).glob("*/rgb")) if os.path.isdir(path) else []
            # if rgb_folder:
            #     print(f"Rgb folder exist, folder:{path}")
            #     continue

            rec_dirs = self.find_recording_dirs(path)
            if not rec_dirs:
                print(
                    f"[ERROR] no recording directory with mjpeg.bin + imu.csv found under {path}"
                )
                continue

            for rec_dir in rec_dirs:
                self.process_one_recording(rec_dir)

            upper_path = os.path.join(self.root, data)
            print(f"\n[INFO] Checking color synchronization for {upper_path}...")
            is_check = self._run_color_sync(
                data_path=upper_path,
                color_log_path=os.path.join(self.root, f"ts{self.ts_name}"),
            )
            if is_check:
                tobe_check.append(upper_path)

        if tobe_check:
            print(
                "\n[WARNING] The following recordings have at least one camera with default latency. Please check their synchronization quality:"
            )
            for p in tobe_check:
                print(f" - {p}")
                self.sync_color_again(
                    p,
                    color_log_path=os.path.join(self.root, f"ts{self.ts_name}"),
                )


if __name__ == "__main__":
    folder_path = "/home/ssd/20260330-001"
    preprocessor = EgocapPreprocessor(folder_path)
    preprocessor.execute()
