"""Postprocessing module: TimestampSync.

Args:
    None.

Returns:
    None.

Raises:
    None.
"""

import numpy as np
from pathlib import Path
import os
import pandas as pd
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shutil

from dataclasses import dataclass


from utils import remove_rgb_folders, get_preferred_ffmpeg_encoder
from VideoSyncer import (
    ZEDVideoProcessor,
    EgocapVideoProcessor,
    RealSenseVideoProcessor,
    PathUtils,
)

ZED_LATENCY = 0
REALSENSE_LATENCY = 0
EGOCAP_LATENCY = 0

DEFAULT_VIDEO_CODEC, DEFAULT_VIDEO_CODEC_ARGS = get_preferred_ffmpeg_encoder()


class DataProcessor:
    """Class DataProcessor.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    def __init__(self):
        """__init__.

        Args:
            None.

        Returns:
            Any: Result of __init__.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        self.processed_base = None

    def _is_processed_folder_complete(self, processed_base):
        """_is_processed_folder_complete.

        Args:
            processed_base: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """检查已存在输出目录是否完整，避免把半成品当成已完成数据。"""
        ts_files = sorted(Path(processed_base).rglob("*_ts.csv"))
        if not ts_files:
            return False

        ts_lengths = []
        for ts_file in ts_files:
            try:
                with open(ts_file, "r") as f:
                    line_count = sum(1 for _ in f)
            except OSError:
                return False

            if line_count <= 0:
                return False
            ts_lengths.append(line_count)

        expected_frames = min(ts_lengths)
        if expected_frames <= 0:
            return False

        mp4_files = sorted(Path(processed_base).rglob("*.mp4"))
        if not mp4_files:
            return False

        min_required_frames = max(1, int(expected_frames * 0.9))
        for mp4_file in mp4_files:
            cap = cv2.VideoCapture(str(mp4_file))
            if not cap.isOpened():
                cap.release()
                return False
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count < min_required_frames:
                return False

        return True

    def _load_config_data(self, base_path, seq_no, date):
        """_load_config_data.

        Args:
            base_path: Input parameter.
            seq_no: Input parameter.
            date: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """加载配置文件"""
        original_config_path = os.path.join(base_path, "config.json")

        if os.path.exists(original_config_path):
            with open(original_config_path, "r", encoding="utf-8-sig") as f:
                config_data = json.load(f)
            config_data["no"] = seq_no
            config_data["date"] = date.split("-")[0]
            sn = config_data["device"]["ego"]["head"]["sn"]
        else:
            config_data = None
            sn = "default"
        self.sn = sn
        return config_data, sn, original_config_path

    def get_device_info_from_config(self, config_data):
        """get_device_info_from_config.

        Args:
            config_data: Input parameter.

        Returns:
            Any: Result of get_device_info_from_config.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        for device_key, device_info in (
            config_data.get("device", {}).get("ego", {}).items()
        ):
            if device_info.get("sn") == self.sn:
                return {
                    "name": device_key,
                    "location": device_info.get("location"),
                    "cam_type": device_info.get("cam_type"),
                    "sn": self.sn,
                }

        return None

    def auto_calculate_cut_time(self, input_folder):

        """auto_calculate_cut_time.

        Args:
            input_folder: Input parameter.

        Returns:
            Any: Result of auto_calculate_cut_time.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        try:
            input_path = Path(input_folder)

            task_file = input_path / "task_timestamps.csv"
            keypoint_file = input_path / "keypoint_timestamps.csv"
            print(f"正在尝试自动计算剪切时间，输入文件夹: {task_file}")
            first_timestamp = None

            if task_file.exists():
                df_task = pd.read_csv(task_file)

                if len(df_task) == 0:
                    print(f"✗ task_timestamps.csv 为空")
                elif "timestamp" not in df_task.columns:
                    print(f"✗ task_timestamps.csv 中未找到 timestamp 列")
                else:
                    first_timestamp = df_task["timestamp"].iloc[0]
                    last_timestamp = df_task["timestamp"].iloc[-1]
                    if not pd.api.types.is_number(first_timestamp):
                        print(f"✗ task_timestamps.csv 的 timestamp 不是数字")
                        first_timestamp = None
                    else:
                        print(
                            f"✓ task_timestamps.csv 第一行 timestamp: {first_timestamp}"
                        )

            if first_timestamp is None and keypoint_file.exists():
                df_keypoint = pd.read_csv(keypoint_file)

                if len(df_keypoint) == 0:
                    print(f"✗ keypoint_timestamps.csv 为空")
                elif "timestamp" not in df_keypoint.columns:
                    print(f"✗ keypoint_timestamps.csv 中未找到 timestamp 列")
                else:
                    first_timestamp = df_keypoint["timestamp"].iloc[0]
                    if not pd.api.types.is_number(first_timestamp):
                        print(f"✗ keypoint_timestamps.csv 的 timestamp 不是数字")
                        first_timestamp = None
                    else:
                        print(
                            f"✓ keypoint_timestamps.csv 第一行 timestamp: {first_timestamp}"
                        )

            if first_timestamp is None:
                print(
                    "✗ 未找到有效的 task_timestamps.csv 或 keypoint_timestamps.csv，无法自动计算裁剪时间"
                )
                return None

            return first_timestamp, last_timestamp

        except (
            OSError,
            pd.errors.ParserError,
            UnicodeDecodeError,
            ValueError,
            TypeError,
            KeyError,
        ) as e:
            print(f"✗ 自动计算裁剪时间失败: {e}")
            return None

    def _setup_processed_directory(
        self, output_dir, date, sn, seq_no, config_data, original_config_path, base_path
    ):
        """_setup_processed_directory.

        Args:
            output_dir: Input parameter.
            date: Input parameter.
            sn: Input parameter.
            seq_no: Input parameter.
            config_data: Input parameter.
            original_config_path: Input parameter.
            base_path: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """设置处理后的目录结构"""
        processed_base = os.path.join(output_dir, date.split("-")[0], sn, seq_no)

        if os.path.exists(processed_base):
            if self._is_processed_folder_complete(processed_base):
                print(
                    f"Processed folder already exists at {processed_base}, skipping video processing"
                )
                return None

            print(
                f"Processed folder exists but looks incomplete: {processed_base}. Rebuilding..."
            )
            shutil.rmtree(processed_base)

        os.makedirs(processed_base, exist_ok=True)

        self._copy_task_files(base_path, processed_base)
        self._save_config(config_data, original_config_path, processed_base, base_path)

        return processed_base

    def _copy_task_files(self, base_path, processed_base):
        """_copy_task_files.

        Args:
            base_path: Input parameter.
            processed_base: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """复制任务时间戳文件"""
        task_timestamps_path = os.path.join(base_path, "task_timestamps.csv")
        if os.path.exists(task_timestamps_path):
            shutil.copyfile(
                task_timestamps_path,
                os.path.join(processed_base, "task_timestamps.csv"),
            )

    def _save_config(
        self, config_data, original_config_path, processed_base, base_path
    ):
        """_save_config.

        Args:
            config_data: Input parameter.
            original_config_path: Input parameter.
            processed_base: Input parameter.
            base_path: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """保存配置文件"""
        dest_config = os.path.join(processed_base, "config.json")

        if config_data and os.path.exists(original_config_path):
            with open(dest_config, "w", encoding="utf-8-sig") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        elif config_data:
            config_path = os.path.join(os.path.dirname(base_path), "config.json")
            with open(config_path, "r", encoding="utf-8-sig") as f:
                config_data = json.load(f)
            with open(dest_config, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

    def _apply_camera_latency(self, all_timestamps, camera_latency):
        """_apply_camera_latency.

        Args:
            all_timestamps: Input parameter.
            camera_latency: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """应用相机延迟校正"""
        for key, val in all_timestamps.items():
            sn = self._extract_serial_number(key)
            ts_with_camera_latency = val - camera_latency.get(sn, 0)
            all_timestamps[key] = ts_with_camera_latency.astype(np.int64)

    def _extract_serial_number(self, key):
        """_extract_serial_number.

        Args:
            key: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """从路径中提取序列号"""
        sn = key.split("/")[-1].split("\\")[-1].split(".")[0].split("_")[0]
        if "egocap" in key:
            sn = "egocap"
        return sn

    def get_camera_latencies(self, base_path):
        """get_camera_latencies.

        Args:
            base_path: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """运行颜色同步脚本"""
        latency_file = os.path.join(
            base_path, "sync_color_results/all_camera_latencies.json"
        )
        if os.path.exists(latency_file):
            print("Camera latency file already exists, skipping sync_color.py")
        else:
            os.system(f"python ./sync_color.py {base_path}")

        return json.load(open(latency_file, "r"))


@dataclass
class TimestampSyncConfig:
    """Class TimestampSyncConfig.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    folder_path: str
    output_dir: str = "processed"
    target_fps: int = 15
    max_cache_size: int = 1000


class TimestampSyncProcessor:
    """Class TimestampSyncProcessor.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    def __init__(self, config: TimestampSyncConfig):
        """__init__.

        Args:
            config: Input parameter.

        Returns:
            Any: Result of __init__.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        self.config = config
        self.target_fps = self.config.target_fps
        self.data_processor = DataProcessor()
        self.pathutils = PathUtils()
        self.realsense_processor = RealSenseVideoProcessor(
            target_fps=self.target_fps,
            data_processor=self.data_processor,
        )
        self.egocap_processor = EgocapVideoProcessor(
            pathutils=self.pathutils,
            target_fps=self.target_fps,
        )
        self.zed_processor = ZEDVideoProcessor(target_fps=self.target_fps)

    def _collect_timestamps(self, rs_data_path, egocap_data_path, zed_data_path):
        """_collect_timestamps.

        Args:
            rs_data_path: Input parameter.
            egocap_data_path: Input parameter.
            zed_data_path: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """收集所有时间戳数据"""
        all_timestamps = {}

        if rs_data_path:
            self.realsense_processor._collect_realsense_timestamps(
                rs_data_path, all_timestamps
            )

        if egocap_data_path:
            self.egocap_processor._collect_egocap_timestamps(
                egocap_data_path, all_timestamps
            )
        if zed_data_path:
            if self.zed_processor.sl is not None:
                self.zed_processor._collect_zed_timestamps(
                    zed_data_path, all_timestamps
                )
        return all_timestamps

    # ==================== 核心同步算法 ====================
    def process_data(
        self,
        rs_data_path=None,
        egocap_data_path=None,
        zed_data_path=None,
        output_dir="processed",
    ):
        """process_data.

        Args:
            rs_data_path: Input parameter.
            egocap_data_path: Input parameter.
            zed_data_path: Input parameter.
            output_dir: Input parameter.

        Returns:
            Any: Result of process_data.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        rs_data_path = rs_data_path if os.path.exists(rs_data_path or "") else None
        egocap_data_path = (
            egocap_data_path if os.path.exists(egocap_data_path or "") else None
        )
        zed_data_path = zed_data_path if os.path.exists(zed_data_path or "") else None

        base_path = os.path.dirname(rs_data_path or egocap_data_path or zed_data_path)

        camera_latency = self.data_processor.get_camera_latencies(base_path)
        print("Camera latencies loaded:", camera_latency)

        seq_no = os.path.basename(base_path)
        date = os.path.basename(os.path.dirname(base_path))

        config_data, sn, original_config_path = self.data_processor._load_config_data(
            base_path, seq_no, date
        )
        processed_base = self.data_processor._setup_processed_directory(
            output_dir, date, sn, seq_no, config_data, original_config_path, base_path
        )
        if processed_base is None:
            return

        self.processed_base = processed_base
        all_timestamps = self._collect_timestamps(
            rs_data_path, egocap_data_path, zed_data_path
        )
        self.data_processor._apply_camera_latency(all_timestamps, camera_latency)
        cut_timestamp, last_task_timestamp = (
            self.data_processor.auto_calculate_cut_time(base_path)
        )
        sync_index, anchor_timestamps = self.sync_timestamps(
            all_timestamps,
            fps=self.target_fps,
            cut_timestamp=cut_timestamp,
            last_task_timestamp=last_task_timestamp,
        )

        self._execute_parallel_processing(
            sync_index,
            anchor_timestamps,
            config_data,
            self.config.max_cache_size,
            processed_base,
        )

    def _execute_parallel_processing(
        self, sync_index, anchor_timestamps, config_data, max_cache_size, processed_base
    ):
        """_execute_parallel_processing.

        Args:
            sync_index: Input parameter.
            anchor_timestamps: Input parameter.
            config_data: Input parameter.
            max_cache_size: Input parameter.
            processed_base: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """执行并行视频处理"""
        max_workers = (
            min(len(sync_index), os.cpu_count() * 2)
            if os.cpu_count()
            else len(sync_index)
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_filepath = {
                executor.submit(
                    self.save_sync_video,
                    filepath,
                    ts_index,
                    anchor_timestamps,
                    config_data,
                    max_cache_size,
                    processed_base,
                ): filepath
                for filepath, ts_index in sync_index.items()
            }
            for future in as_completed(future_to_filepath):
                filepath = future_to_filepath[future]
                try:
                    future.result()
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                    print(f"✗ Error processing {filepath}: {exc}")

    def sync_timestamps(
        self, timestamps_dict, fps=15, cut_timestamp=None, last_task_timestamp=None
    ):

        """sync_timestamps.

        Args:
            timestamps_dict: Input parameter.
            fps: Input parameter.
            cut_timestamp: Input parameter.
            last_task_timestamp: Input parameter.

        Returns:
            Any: Result of sync_timestamps.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        def _calculate_common_range(timestamps_dict, interval_ms, cut_timestamp):
            """_calculate_common_range.

            Args:
                timestamps_dict: Input parameter.
                interval_ms: Input parameter.
                cut_timestamp: Input parameter.

            Returns:
                Any: Function result.

            Raises:
                Exception: If an unexpected runtime error occurs.
            """
            """计算所有时间戳的共同范围"""
            min_common = max(
                max(np.min(v[1:]) for k, v in timestamps_dict.items())
                - interval_ms // 4,
                cut_timestamp if cut_timestamp else -1e2,
            )
            max_common = (
                min(np.max(v[1:]) for k, v in timestamps_dict.items())
                + interval_ms // 4
            )
            return min_common, max_common

        ts_index = {}
        for key, val in timestamps_dict.items():
            ts_index[key] = []

        interval_ms = int(1000.0 / fps)
        min_common, max_common = _calculate_common_range(
            timestamps_dict, interval_ms, cut_timestamp
        )

        anchor_timestamps = np.arange(
            min_common, max_common, 1000.0 / fps, dtype=np.float64
        ).astype(np.int64)
        if len(anchor_timestamps) == 0:
            print(
                "Anchor Timestamps is empty, try task timestamp calculation or check timestamp files."
            )
            anchor_timestamps = np.arange(
                cut_timestamp, last_task_timestamp, 1000.0 / fps, dtype=np.float64
            ).astype(np.int64)

        print("Initial anchor timestamps: ", anchor_timestamps)
        best_bias = self._find_optimal_bias(
            timestamps_dict, anchor_timestamps, interval_ms
        )
        anchor_timestamps += best_bias
        print("anchor timestamps: ", anchor_timestamps)

        for k, v in timestamps_dict.items():
            for a in anchor_timestamps:
                ts_index[k].append(int(np.argmin(np.abs(v - a))))

        return ts_index, anchor_timestamps

    def _find_optimal_bias(self, timestamps_dict, anchor_timestamps, interval_ms):

        """_find_optimal_bias.

        Args:
            timestamps_dict: Input parameter.
            anchor_timestamps: Input parameter.
            interval_ms: Input parameter.

        Returns:
            Any: Result of _find_optimal_bias.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        def get_std_timestamps(timestamps_dict):

            """get_std_timestamps.

            Args:
                timestamps_dict: Input parameter.

            Returns:
                Any: Result of get_std_timestamps.

            Raises:
                Exception: If an unexpected runtime error occurs.
            """
            zed_ts_list = [v for k, v in timestamps_dict.items() if k.endswith("svo2")]
            if not zed_ts_list:
                zed_ts_list = [
                    v for k, v in timestamps_dict.items() if k.endswith("_ts.csv")
                ]
                print("No ZED data found, using RealSense timestamps as anchor")
            return zed_ts_list

        def _calculate_bias_loss(zed_ts_list, anchor_timestamps, bias):

            """_calculate_bias_loss.

            Args:
                zed_ts_list: Input parameter.
                anchor_timestamps: Input parameter.
                bias: Input parameter.

            Returns:
                Any: Result of _calculate_bias_loss.

            Raises:
                Exception: If an unexpected runtime error occurs.
            """
            total_loss = 0.0
            adjust_timestamps = anchor_timestamps + bias
            for ts in adjust_timestamps:
                for zed_ts in zed_ts_list:
                    total_loss += np.min(np.abs(zed_ts - ts))
            return total_loss

        best_loss = float("inf")
        best_bias = 0

        std_ts_list = get_std_timestamps(timestamps_dict)

        for bias in range(interval_ms):
            total_loss = _calculate_bias_loss(std_ts_list, anchor_timestamps, bias)
            if total_loss < best_loss:
                best_loss = total_loss
                best_bias = bias

        return best_bias

    # ==================== 主保存接口 ====================

    def save_sync_video(
        self,
        filepath,
        ts_index,
        anchor_timestamps,
        config_data=None,
        max_cache_size=5000,
        base_save_path=None,
    ):
        """save_sync_video.

        Args:
            filepath: Input parameter.
            ts_index: Input parameter.
            anchor_timestamps: Input parameter.
            config_data: Input parameter.
            max_cache_size: Input parameter.
            base_save_path: Input parameter.

        Returns:
            Any: Result of save_sync_video.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        data_root = self.pathutils.resolve_data_root(filepath)

        if base_save_path is None:
            base_save_path, _ = self.pathutils.get_base_save_path(
                data_root, config_data
            )

        synced_timestamps = anchor_timestamps

        if filepath.endswith("_ts.csv"):
            self.realsense_processor._process_realsense_data(
                filepath, base_save_path, config_data, ts_index, synced_timestamps
            )
        elif filepath.endswith("egocap.mp4"):
            self.egocap_processor._process_egocap_data(
                filepath, base_save_path, config_data, ts_index, synced_timestamps
            )
        elif filepath.endswith("imu.csv") and "egocap" in filepath:
            self.egocap_processor._sync_egocap_imu(
                filepath, base_save_path, synced_timestamps
            )
        elif filepath.endswith("svo2"):
            if self.zed_processor.sl is not None:
                self.zed_processor._process_zed_data(
                    filepath, base_save_path, config_data, ts_index, synced_timestamps
                )

    def setup_realsense_paths(self, filepath, base_save_path, config_data):

        """setup_realsense_paths.

        Args:
            filepath: Input parameter.
            base_save_path: Input parameter.
            config_data: Input parameter.

        Returns:
            Any: Result of setup_realsense_paths.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        camera_name = os.path.basename(filepath).split("_ts")[0]
        device_info = (
            self.data_processor.get_device_info_from_config(
                config_data,
                # camera_name
            )
            if config_data
            else None
        )

        if device_info:
            device_name = device_info["name"]
            rs_folder = os.path.join(
                base_save_path,
                "realsense",
                device_info["cam_type"].replace("realsense_", ""),
            )
            os.makedirs(rs_folder, exist_ok=True)
            ts_file = os.path.join(rs_folder, f"{device_name}_{camera_name}_ts.csv")
            return device_info, rs_folder, ts_file, camera_name, device_name
        return None, None, None, camera_name, None

    # ==================== 时间戳读取工具 ====================

    def get_rs_ts_array(self, ts_path):
        """get_rs_ts_array.

        Args:
            ts_path: Input parameter.

        Returns:
            Any: Result of get_rs_ts_array.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        timestamps_list = []
        with open(ts_path, "r") as f:
            for line in f:
                timestamps_list.append(int(line.strip()))
        return np.asarray(timestamps_list, dtype=np.int64) - REALSENSE_LATENCY

    def get_egocap_ts_array(self, egocap_path):
        """get_egocap_ts_array.

        Args:
            egocap_path: Input parameter.

        Returns:
            Any: Result of get_egocap_ts_array.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        timestamps_list = []
        rgb_dir = os.path.join(os.path.dirname(egocap_path), "rgb")
        rgb_files = [item.path for item in os.scandir(rgb_dir) if item.is_file()]
        rgb_files.sort()

        for rgb_file in rgb_files:
            timestamps_list.append(int(rgb_file.split("_")[-1].split(".")[0]))

        return np.asarray(timestamps_list, dtype=np.int64) // 1000000 - EGOCAP_LATENCY

    def get_imu_ts_array(self, imu_path, camera_type="egocap"):
        """get_imu_ts_array.

        Args:
            imu_path: Input parameter.
            camera_type: Input parameter.

        Returns:
            Any: Result of get_imu_ts_array.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        timestamps_list = []
        with open(imu_path, "r") as f:
            # header = f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                try:
                    timestamp = self._parse_imu_timestamp(parts, camera_type)
                    timestamps_list.append(timestamp)
                except (ValueError, IndexError):
                    continue

        timestamps_array = np.asarray(timestamps_list, dtype=np.int64)
        return self._apply_latency_offset(timestamps_array, camera_type)

    def _parse_imu_timestamp(self, parts, camera_type):
        """_parse_imu_timestamp.

        Args:
            parts: Input parameter.
            camera_type: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """解析IMU时间戳"""
        if camera_type == "egocap":
            return int(parts[1]) // 1000000
        elif camera_type == "zed":
            return int(parts[0]) // 1000000
        elif camera_type == "realsense":
            return int(parts[0])
        raise ValueError(f"Unknown camera type: {camera_type}")

    def _apply_latency_offset(self, timestamps_array, camera_type):
        """_apply_latency_offset.

        Args:
            timestamps_array: Input parameter.
            camera_type: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """应用延迟偏移"""
        latency_map = {
            "egocap": EGOCAP_LATENCY,
            "realsense": REALSENSE_LATENCY,
            "zed": ZED_LATENCY,
        }
        return timestamps_array - latency_map.get(camera_type, 0)

    # ==================== 执行接口 ====================

    def execute(self):
        """execute.

        Args:
            None.

        Returns:
            Any: Result of execute.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        folder_path = self.config.folder_path
        output_dir = self.config.output_dir

        for data in sorted(os.listdir(folder_path)):
            try:
                _data = os.path.join(folder_path, data)
                rs_data_path = (
                    os.path.join(_data, "realsense")
                    if os.path.exists(os.path.join(_data, "realsense"))
                    else None
                )
                egocap_data_path = (
                    os.path.join(_data, "egocap")
                    if os.path.exists(os.path.join(_data, "egocap"))
                    else None
                )
                zed_data_path = (
                    os.path.join(_data, "zed")
                    if os.path.exists(os.path.join(_data, "zed"))
                    else None
                )
                if not rs_data_path and not egocap_data_path and not zed_data_path:
                    continue
                self.process_data(
                    rs_data_path=rs_data_path,
                    egocap_data_path=egocap_data_path,
                    zed_data_path=zed_data_path,
                    output_dir=output_dir,
                )
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
                print(f"Error processing folder {_data}: {e}")
                continue

        remove_rgb_folders(folder_path)
