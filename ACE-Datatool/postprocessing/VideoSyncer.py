"""Postprocessing module: VideoSyncer.

Args:
    None.

Returns:
    None.

Raises:
    None.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

import imageio
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import queue
import threading

from utils import progress_bar, get_preferred_ffmpeg_encoder

ZED_LATENCY = 0
REALSENSE_LATENCY = 0
EGOCAP_LATENCY = 0

DEFAULT_VIDEO_CODEC, DEFAULT_VIDEO_CODEC_ARGS = get_preferred_ffmpeg_encoder()


class TimestampReaderMixin:
    """Class TimestampReaderMixin.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
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
            Any: Result of _parse_imu_timestamp.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        if camera_type == "egocap":
            return int(parts[1]) // 1000000
        if camera_type == "zed":
            return int(parts[0]) // 1000000
        if camera_type == "realsense":
            return int(parts[0])
        raise ValueError(f"Unknown camera type: {camera_type}")

    def _apply_latency_offset(self, timestamps_array, camera_type):
        """_apply_latency_offset.

        Args:
            timestamps_array: Input parameter.
            camera_type: Input parameter.

        Returns:
            Any: Result of _apply_latency_offset.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        latency_map = {
            "egocap": EGOCAP_LATENCY,
            "realsense": REALSENSE_LATENCY,
            "zed": ZED_LATENCY,
        }
        return timestamps_array - latency_map.get(camera_type, 0)


class RealSenseVideoProcessor(TimestampReaderMixin):
    """Class RealSenseVideoProcessor.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    def __init__(self, target_fps=15, data_processor=None):
        """__init__.

        Args:
            target_fps: Input parameter.
            data_processor: Input parameter.

        Returns:
            Any: Result of __init__.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        self.target_fps = target_fps
        self.data_processor = data_processor

    def get_rs_ts_array(self, ts_path):
        """get_rs_ts_array.

        Args:
            ts_path: Input parameter.

        Returns:
            Any: Result of get_rs_ts_array.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        return super().get_rs_ts_array(ts_path)

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
        return super().get_imu_ts_array(imu_path, camera_type)

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
            self.data_processor.get_device_info_from_config(config_data)
            if self.data_processor and config_data
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

    def _collect_realsense_timestamps(self, rs_data_path, all_timestamps):
        """_collect_realsense_timestamps.

        Args:
            rs_data_path: Input parameter.
            all_timestamps: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """收集RealSense时间戳"""
        base_dir = Path(rs_data_path)
        if not base_dir.is_dir():
            return

        ts_files = sorted(base_dir.rglob("*_ts.csv"))
        for ts_path in ts_files:
            ts_path = ts_path.resolve()
            ts_array = self.get_rs_ts_array(str(ts_path))
            all_timestamps[str(ts_path)] = ts_array

            imu_path = str(ts_path).replace("_ts.csv", "_imu.csv")
            if os.path.exists(imu_path):
                imu_ts_array = self.get_imu_ts_array(imu_path, "realsense")
                all_timestamps[imu_path] = imu_ts_array

    def _process_realsense_data(
        self, filepath, base_save_path, config_data, ts_index, synced_timestamps
    ):
        """_process_realsense_data.

        Args:
            filepath: Input parameter.
            base_save_path: Input parameter.
            config_data: Input parameter.
            ts_index: Input parameter.
            synced_timestamps: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """处理RealSense数据"""
        fps_target = self.target_fps
        device_info, rs_folder, ts_file, camera_name, device_name = (
            self.setup_realsense_paths(filepath, base_save_path, config_data)
        )

        if ts_file:
            self._write_realsense_timestamps(ts_file, synced_timestamps)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    self._process_realsense_video_type,
                    vtype,
                    filepath,
                    rs_folder,
                    device_info,
                    camera_name,
                    device_name,
                    base_save_path,
                    ts_index,
                    fps_target,
                )
                for vtype in ["rgb", "depth"]
            ]
            for future in as_completed(futures):
                future.result()

        self._sync_realsense_imu(
            filepath,
            rs_folder,
            device_info,
            camera_name,
            base_save_path,
            ts_index,
            synced_timestamps,
        )
        sys.stdout.write("Syncing and saving video complete.\n")

    def _write_realsense_timestamps(self, ts_file, synced_timestamps):
        """_write_realsense_timestamps.

        Args:
            ts_file: Input parameter.
            synced_timestamps: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """写入RealSense时间戳文件"""
        with open(ts_file, "w") as f:
            for ts in synced_timestamps:
                f.write(f"{ts}\n")
        print(f"Saved RealSense timestamps to: {ts_file}")

    def _create_video_writer(self, video_file, fps_target):
        """_create_video_writer.

        Args:
            video_file: Input parameter.
            fps_target: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """创建视频写入器"""
        codec = DEFAULT_VIDEO_CODEC
        output_params = list(DEFAULT_VIDEO_CODEC_ARGS)

        return imageio.get_writer(
            video_file,
            fps=fps_target,
            mode="I",
            codec=codec,
            format="FFMPEG",
            macro_block_size=1,
            pixelformat="yuv420p",
            output_params=output_params,
        )

    def _process_realsense_video_type(
        self,
        video_type,
        filepath,
        rs_folder,
        device_info,
        camera_name,
        device_name,
        base_save_path,
        ts_index,
        fps_target,
    ):
        """_process_realsense_video_type.

        Args:
            video_type: Input parameter.
            filepath: Input parameter.
            rs_folder: Input parameter.
            device_info: Input parameter.
            camera_name: Input parameter.
            device_name: Input parameter.
            base_save_path: Input parameter.
            ts_index: Input parameter.
            fps_target: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """处理单个RealSense视频类型（rgb或depth）"""
        video_file, output_folder = self._get_video_output_path(
            video_type, rs_folder, device_info, camera_name, device_name, base_save_path
        )

        video_writer = self._create_video_writer(video_file, fps_target)
        filepath_ = filepath.replace("_ts.csv", f"_{video_type}.mp4")
        video_capture = cv2.VideoCapture(filepath_)

        if not video_capture.isOpened():
            raise ValueError(f"Error opening {video_type} video file.")

        written_frames = self._process_video_frames(
            video_capture,
            video_writer,
            ts_index,
            stream_name=f"realsense-{video_type}",
        )
        expected_frames = len(ts_index)
        if written_frames != expected_frames:
            raise RuntimeError(
                f"{video_type} stream frame mismatch: expected {expected_frames}, got {written_frames}"
            )

        video_capture.release()
        video_writer.close()

    def _get_video_output_path(
        self,
        video_type,
        rs_folder,
        device_info,
        camera_name,
        device_name,
        base_save_path,
    ):
        """_get_video_output_path.

        Args:
            video_type: Input parameter.
            rs_folder: Input parameter.
            device_info: Input parameter.
            camera_name: Input parameter.
            device_name: Input parameter.
            base_save_path: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """获取视频输出路径"""
        if device_info:
            video_file = os.path.join(
                rs_folder, f"{device_name}_{camera_name}_{video_type}.mp4"
            )
            return video_file, None
        else:
            output_folder = os.path.join(base_save_path, video_type)
            os.makedirs(output_folder, exist_ok=True)
            video_file = os.path.join(output_folder, f"{camera_name}.mp4")
            return video_file, output_folder

    def _process_video_frames(
        self, video_capture, video_writer, ts_index, stream_name="realsense"
    ):
        """_process_video_frames.

        Args:
            video_capture: Input parameter.
            video_writer: Input parameter.
            ts_index: Input parameter.
            stream_name: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """处理视频帧"""
        frame_count = 0
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counts = Counter(ts_index)
        expected_frames = len(ts_index)
        written_frames = 0
        last_frame_rgb = None

        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                count = frame_counts.get(frame_count, 0)
                if count > 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    last_frame_rgb = frame_rgb
                    for _ in range(count):
                        video_writer.append_data(frame_rgb)
                    written_frames += count

                frame_count += 1
                pbar.update(1)

        if written_frames < expected_frames:
            if last_frame_rgb is None:
                raise RuntimeError(
                    f"{stream_name} has no writable frame, expected {expected_frames}"
                )
            missing_frames = expected_frames - written_frames
            print(
                f"Warning: {stream_name} stopped early, padding {missing_frames} frames with last valid frame"
            )
            for _ in range(missing_frames):
                video_writer.append_data(last_frame_rgb)
            written_frames = expected_frames

        return written_frames

    def _sync_realsense_imu(
        self,
        filepath,
        rs_folder,
        device_info,
        camera_name,
        base_save_path,
        ts_index,
        anchor_timestamps,
    ):
        """_sync_realsense_imu.

        Args:
            filepath: Input parameter.
            rs_folder: Input parameter.
            device_info: Input parameter.
            camera_name: Input parameter.
            base_save_path: Input parameter.
            ts_index: Input parameter.
            anchor_timestamps: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """同步RealSense IMU数据"""
        imu_csv_path = filepath.replace("_ts.csv", "_imu.csv")
        if not os.path.exists(imu_csv_path):
            return

        imu_output_file = self._get_imu_output_path(
            imu_csv_path, rs_folder, device_info, camera_name, base_save_path
        )

        self._write_synced_imu_data(
            imu_csv_path,
            imu_output_file,
            ts_index,
            anchor_timestamps,
            REALSENSE_LATENCY,
        )
        print(f"Saved synchronized RealSense IMU data to: {imu_output_file}")

    def _get_imu_output_path(
        self, imu_csv_path, rs_folder, device_info, camera_name, base_save_path
    ):
        """_get_imu_output_path.

        Args:
            imu_csv_path: Input parameter.
            rs_folder: Input parameter.
            device_info: Input parameter.
            camera_name: Input parameter.
            base_save_path: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """获取IMU输出路径"""
        if device_info:
            return os.path.join(
                rs_folder, f"{device_info['name']}_{camera_name}_imu.csv"
            )
        else:
            imu_folder = os.path.join(base_save_path, "imu")
            os.makedirs(imu_folder, exist_ok=True)
            return os.path.join(imu_folder, f"{camera_name}_imu_synced.csv")

    def _write_synced_imu_data(
        self, input_path, output_path, ts_index, anchor_timestamps, latency
    ):
        """_write_synced_imu_data.

        Args:
            input_path: Input parameter.
            output_path: Input parameter.
            ts_index: Input parameter.
            anchor_timestamps: Input parameter.
            latency: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """写入同步后的IMU数据"""
        with open(input_path, "r") as f_in, open(output_path, "w", newline="") as f_out:
            header = f_in.readline()
            f_out.write(header)
            all_lines = f_in.readlines()
            for i, idx in enumerate(ts_index):
                if idx < len(all_lines):
                    parts = all_lines[idx].strip().split(",")
                    synced_ts_ms = anchor_timestamps[i] + latency
                    parts[0] = str(synced_ts_ms)
                    f_out.write(",".join(parts) + "\n")


class EgocapVideoProcessor(TimestampReaderMixin):
    """Class EgocapVideoProcessor.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    def __init__(self, pathutils=None, target_fps=15):
        """__init__.

        Args:
            pathutils: Input parameter.
            target_fps: Input parameter.

        Returns:
            Any: Result of __init__.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        self.pathutils = pathutils or PathUtils()
        self.target_fps = target_fps

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
        return super().get_imu_ts_array(imu_path, camera_type)

    def get_egocap_ts_array(self, egocap_path):
        """get_egocap_ts_array.

        Args:
            egocap_path: Input parameter.

        Returns:
            Any: Result of get_egocap_ts_array.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        return super().get_egocap_ts_array(egocap_path)

    def _write_timestamps(self, ts_file, synced_timestamps):
        """_write_timestamps.

        Args:
            ts_file: Input parameter.
            synced_timestamps: Input parameter.

        Returns:
            Any: Result of _write_timestamps.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        with open(ts_file, "w") as f:
            for ts in synced_timestamps:
                f.write(f"{ts}\n")


    def _collect_egocap_timestamps(self, egocap_data_path, all_timestamps):
        """_collect_egocap_timestamps.

        Args:
            egocap_data_path: Input parameter.
            all_timestamps: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """收集Egocap时间戳"""
        base_dir = Path(egocap_data_path)
        if not base_dir.is_dir():
            return

        ts_files = sorted(base_dir.rglob("imu.csv"))
        for ts_path in ts_files:
            ts_path = ts_path.resolve()
            ts_array = self.get_imu_ts_array(str(ts_path), "egocap")
            all_timestamps[str(ts_path)] = ts_array

            ts_rgb_path = str(ts_path).replace("imu.csv", "egocap.mp4")
            ts_rgb_array = self.get_egocap_ts_array(str(ts_path))
            all_timestamps[ts_rgb_path] = ts_rgb_array

    def _process_egocap_data(
        self, filepath, base_save_path, config_data, ts_index, synced_timestamps
    ):
        """_process_egocap_data.

        Args:
            filepath: Input parameter.
            base_save_path: Input parameter.
            config_data: Input parameter.
            ts_index: Input parameter.
            synced_timestamps: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """处理Egocap数据"""
        egocap_folder, _ = self.pathutils.setup_egocap_paths(
            filepath, base_save_path, config_data
        )

        ts_file = os.path.join(egocap_folder, "egocap_ts.csv")
        self._write_timestamps(ts_file, synced_timestamps)

        rgb_dir = os.path.join(os.path.dirname(filepath), "rgb")
        if not os.path.exists(rgb_dir):
            print(f"Error: Egocap rgb directory not found at {rgb_dir}")
            return

        rgb_files = sorted(
            [item.path for item in os.scandir(rgb_dir) if item.is_file()]
        )
        writers = self._setup_egocap_writers(egocap_folder, self.target_fps)
        write_pool = ThreadPoolExecutor(max_workers=16)

        write_pool, written_frames = self._process_egocap_frames(
            ts_index, rgb_files, writers, write_pool
        )

        expected_frames = len(ts_index)
        if written_frames != expected_frames:
            raise RuntimeError(
                f"egocap stream frame mismatch: expected {expected_frames}, got {written_frames}"
            )

        write_pool.shutdown()
        for writer in writers.values():
            writer.close()

        sys.stdout.write("Egocap video processing complete.\n")

    def _create_video_writer(self, video_file, fps_target):
        """_create_video_writer.

        Args:
            video_file: Input parameter.
            fps_target: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """创建视频写入器"""
        codec = DEFAULT_VIDEO_CODEC
        output_params = list(DEFAULT_VIDEO_CODEC_ARGS)

        return imageio.get_writer(
            video_file,
            fps=fps_target,
            mode="I",
            codec=codec,
            format="FFMPEG",
            macro_block_size=1,
            pixelformat="yuv420p",
            output_params=output_params,
        )

    def _setup_egocap_writers(self, egocap_folder, fps_target):
        """_setup_egocap_writers.

        Args:
            egocap_folder: Input parameter.
            fps_target: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """设置Egocap视频写入器"""
        writers = {}
        for cam_id in range(4):
            writers[f"cam_{cam_id}"] = self._create_video_writer(
                os.path.join(egocap_folder, f"egocap_cam_{cam_id}.mp4"), fps_target
            )
        return writers

    def _extract_unique_indices(self, ts_index, rgb_files):
        """_extract_unique_indices.

        Args:
            ts_index: Input parameter.
            rgb_files: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """提取唯一索引列表用于顺序读取"""
        unique_indices = []
        for idx in ts_index:
            if idx >= len(rgb_files):
                continue
            if not unique_indices or unique_indices[-1] != idx:
                unique_indices.append(idx)
        return unique_indices

    def _start_frame_producer(self, unique_indices, rgb_files, frame_queue):
        """_start_frame_producer.

        Args:
            unique_indices: Input parameter.
            rgb_files: Input parameter.
            frame_queue: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """启动帧生产者线程"""

        def frame_producer():
            """frame_producer.

            Args:
                None.

            Returns:
                Any: Result of frame_producer.

            Raises:
                Exception: If an unexpected runtime error occurs.
            """
            for idx in unique_indices:
                frame_bgr = cv2.imread(rgb_files[idx])
                if frame_bgr is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_queue.put((idx, frame_rgb))
                else:
                    frame_queue.put((idx, None))
            frame_queue.put((None, None))  # EOF

        producer_thread = threading.Thread(target=frame_producer)
        producer_thread.start()
        return producer_thread

    def _write_cam_slice(self, cam_id, frame_data, writers, height_step):
        """_write_cam_slice.

        Args:
            cam_id: Input parameter.
            frame_data: Input parameter.
            writers: Input parameter.
            height_step: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """写入单个相机切片的帧"""
        frame_cam = frame_data[cam_id * height_step : (cam_id + 1) * height_step, :]
        writers[f"cam_{cam_id}"].append_data(frame_cam)

    def _process_egocap_frames(self, ts_index, rgb_files, writers, write_pool):
        """_process_egocap_frames.

        Args:
            ts_index: Input parameter.
            rgb_files: Input parameter.
            writers: Input parameter.
            write_pool: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """处理Egocap帧"""
        frame_queue = queue.Queue(maxsize=100)
        unique_indices = self._extract_unique_indices(ts_index, rgb_files)
        producer_thread = self._start_frame_producer(
            unique_indices, rgb_files, frame_queue
        )

        last_idx = -1
        last_frame = None
        written_frames = 0
        expected_frames = len(ts_index)

        for idx in tqdm(ts_index, desc="Processing Egocap (Max Speed)"):
            if idx >= len(rgb_files):
                continue

            if idx != last_idx:
                q_idx, frame_rgb = frame_queue.get()
                if q_idx == idx and frame_rgb is not None:
                    last_frame = frame_rgb
                last_idx = idx

            if last_frame is None:
                continue

            self._write_frame_to_all_cams(last_frame, writers, write_pool)
            written_frames += 1

        if written_frames < expected_frames and last_frame is not None:
            missing_frames = expected_frames - written_frames
            print(
                f"Warning: egocap missing {missing_frames} frames, padding with last valid frame"
            )
            for _ in range(missing_frames):
                self._write_frame_to_all_cams(last_frame, writers, write_pool)
            written_frames = expected_frames

        if written_frames < expected_frames and last_frame is None:
            raise RuntimeError(
                f"egocap has no writable frame, expected {expected_frames}"
            )

        producer_thread.join()
        return write_pool, written_frames

    def _write_frame_to_all_cams(self, frame, writers, write_pool):
        """_write_frame_to_all_cams.

        Args:
            frame: Input parameter.
            writers: Input parameter.
            write_pool: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """将一帧写入所有4个相机"""
        frame_shape = frame.shape
        height_step = frame_shape[0] // 4

        futures = [
            write_pool.submit(
                self._write_cam_slice, cam_id, frame, writers, height_step
            )
            for cam_id in range(4)
        ]
        for f in futures:
            f.result()

    def _sync_egocap_imu(self, filepath, base_save_path, anchor_timestamps):
        """同步Egocap IMU数据"""
        egocap_folder = os.path.join(base_save_path, "egocap", "x1")
        os.makedirs(egocap_folder, exist_ok=True)
        
        imu_output_file = os.path.join(egocap_folder, "egocap_imu.csv")
        imu_ts_output_file = os.path.join(egocap_folder, "egocap_ts.csv")
        
        # 获取锚点时间戳范围
        first_timestamp = anchor_timestamps[0] if len(anchor_timestamps) > 0 else 0
        last_timestamp = anchor_timestamps[-1] if len(anchor_timestamps) > 0 else 0
        
        sync_first_timestamp = first_timestamp + EGOCAP_LATENCY
        sync_last_timestamp = last_timestamp + EGOCAP_LATENCY
        
        print(f"Syncing Egocap IMU data with anchor timestamps range: {first_timestamp} -> {last_timestamp}")
        
        # 读取原始数据
        df = pd.read_csv(filepath)

        # 假设 host_timestamp_ns 是纳秒，anchor_timestamps 是毫秒，所以这里 * 1000000
        timestamps = df['host_timestamp_ns'].values
        start_idx = np.searchsorted(timestamps, sync_first_timestamp * 1000000, side='left')
        end_idx = np.searchsorted(timestamps, sync_last_timestamp * 1000000 , side='right')
        
        # 提取裁剪后的时间戳列表
        cropped_timestamps = (timestamps[start_idx:end_idx] //1000000).tolist() # 转换回毫秒
        
        # 1. 保存裁剪后的完整数据 (保留原始表头)
        cropped_df = df.iloc[start_idx:end_idx]
        cropped_df.to_csv(imu_output_file, index=False)
        
        # 2. 保存裁剪后的时间戳数据 (修复了 unhashable type 错误，并设置 header=False)
        # 直接将列表传入 DataFrame，并在 to_csv 时指定 header=False
        cropped_timestamps_df = pd.DataFrame(cropped_timestamps)
        cropped_timestamps_df.to_csv(imu_ts_output_file, index=False, header=False)
        
        print(f"Saved synchronized Egocap IMU data to: {imu_output_file}")
        print(f"Saved timestamps (no header) to: {imu_ts_output_file}")



class PathUtils:

    """Class PathUtils.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    def resolve_data_root(self, filepath):
        """resolve_data_root.

        Args:
            filepath: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """根据文件路径解析数据根目录"""
        if filepath.endswith("_ts.csv") or (
            filepath.endswith("_imu.csv") and "egocap" not in filepath
        ):
            camera_dir = os.path.dirname(filepath)
            data_root = os.path.dirname(camera_dir)
            return data_root
        elif filepath.endswith("imu.csv") and "egocap" in filepath:
            egocap_timestamp_dir = os.path.dirname(filepath)
            egocap_dir = os.path.dirname(egocap_timestamp_dir)
            return egocap_dir
        elif filepath.endswith("egocap.mp4"):
            egocap_timestamp_dir = os.path.dirname(filepath)
            egocap_dir = os.path.dirname(egocap_timestamp_dir)
            return egocap_dir
        elif filepath.endswith("svo2"):
            camera_dir = os.path.dirname(filepath)
            data_root = os.path.dirname(camera_dir)
            return data_root
        else:
            return os.path.abspath(filepath + "../../..")

    def get_base_save_path(self, data_root, config_data):
        """get_base_save_path.

        Args:
            data_root: Input parameter.
            config_data: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """获取基础保存路径"""
        seq_no = (
            config_data.get("no", os.path.basename(data_root))
            if config_data
            else os.path.basename(data_root)
        )
        if not seq_no:
            seq_no = os.path.basename(data_root)

        base_save_path = os.path.join(os.path.dirname(data_root), "processed", seq_no)
        os.makedirs(base_save_path, exist_ok=True)
        return base_save_path, seq_no

    def setup_egocap_paths(self, filepath, base_save_path, config_data):
        """setup_egocap_paths.

        Args:
            filepath: Input parameter.
            base_save_path: Input parameter.
            config_data: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """设置Egocap相关路径"""
        device_type = config_data["device"]["ego"][
            list(config_data["device"]["ego"].keys())[0]
        ]["cam_type"].replace("egocap_", "")
        egocap_folder = os.path.join(base_save_path, "egocap", device_type)
        os.makedirs(egocap_folder, exist_ok=True)
        return egocap_folder, device_type


class ZEDVideoProcessor:
    """Class ZEDVideoProcessor.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    def __init__(self, target_fps=15):
        """__init__.

        Args:
            target_fps: Input parameter.

        Returns:
            Any: Result of __init__.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        try:
            import pyzed.sl as sl

            self.sl = sl
        except ImportError:
            print(
                "Error: ZED SDK not found. Please install the ZED SDK and its Python API."
            )
            self.sl = None
        self.target_fps = target_fps

    def _collect_zed_timestamps(self, zed_data_path, all_timestamps):
        """_collect_zed_timestamps.

        Args:
            zed_data_path: Input parameter.
            all_timestamps: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """收集ZED时间戳"""
        base_dir = Path(zed_data_path)
        if not base_dir.is_dir():
            return

        ts_files = sorted(base_dir.rglob("*.svo2"))
        for ts_path in ts_files:
            ts_path = ts_path.resolve()
            ts_array = self.get_zed_ts_array(str(ts_path))
            all_timestamps[str(ts_path)] = ts_array

    def _create_video_writer(self, video_file, fps_target):
        """_create_video_writer.

        Args:
            video_file: Input parameter.
            fps_target: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """创建ZED视频写入器"""
        codec = DEFAULT_VIDEO_CODEC
        output_params = list(DEFAULT_VIDEO_CODEC_ARGS)

        # ZED视频通常分辨率较高，额外增加编码线程
        threads_for_zed = max(6, os.cpu_count() // 2) if os.cpu_count() else 8
        output_params.extend(["-threads", str(threads_for_zed), "-preset", "veryfast"])

        return imageio.get_writer(
            video_file,
            fps=fps_target,
            mode="I",
            codec=codec,
            format="FFMPEG",
            macro_block_size=1,
            pixelformat="yuv420p",
            output_params=output_params,
        )

    def _get_device_info_from_config(self, config_data, sn):
        """_get_device_info_from_config.

        Args:
            config_data: Input parameter.
            sn: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """根据SN号从config中获取设备信息"""
        if not config_data:
            return None

        for device_key, device_info in (
            config_data.get("device", {}).get("ego", {}).items()
        ):
            if device_info.get("sn") == sn:
                return {
                    "name": device_key,
                    "location": device_info.get("location"),
                    "cam_type": device_info.get("cam_type", "zed_unknown"),
                    "sn": sn,
                }
        return None

    def _build_output_paths(self, filepath, base_save_path, config_data):
        """_build_output_paths.

        Args:
            filepath: Input parameter.
            base_save_path: Input parameter.
            config_data: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """构建ZED输出路径"""
        zed_serial = os.path.splitext(os.path.basename(filepath))[0]
        device_info = self._get_device_info_from_config(config_data, zed_serial)

        if device_info:
            device_name = device_info["name"]
            zed_folder = os.path.join(
                base_save_path,
                "zed",
                device_info["cam_type"].replace("zed_", ""),
            )
            os.makedirs(zed_folder, exist_ok=True)
            video_file = os.path.join(zed_folder, f"{device_name}_{zed_serial}_rgb.mp4")
            ts_file = os.path.join(zed_folder, f"{device_name}_{zed_serial}_ts.csv")
            imu_output_file = os.path.join(
                zed_folder, f"{device_name}_{zed_serial}_imu.csv"
            )
            return zed_serial, device_info, video_file, ts_file, imu_output_file

        # fallback
        rgb_folder = os.path.join(base_save_path, "rgb")
        imu_folder = os.path.join(base_save_path, "imu")
        os.makedirs(rgb_folder, exist_ok=True)
        os.makedirs(imu_folder, exist_ok=True)
        video_file = os.path.join(rgb_folder, f"{zed_serial}_rgb.mp4")
        ts_file = None
        imu_output_file = os.path.join(imu_folder, f"{zed_serial}_imu_synced.csv")
        return zed_serial, None, video_file, ts_file, imu_output_file

    def _write_timestamps(self, ts_file, synced_timestamps):
        """_write_timestamps.

        Args:
            ts_file: Input parameter.
            synced_timestamps: Input parameter.

        Returns:
            Any: Result of _write_timestamps.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        if not ts_file:
            return
        with open(ts_file, "w") as f:
            for ts in synced_timestamps:
                f.write(f"{ts}\n")
        print(f"Saved ZED timestamps to: {ts_file}")

    def _write_synced_zed_imu(
        self, imu_csv_path, imu_output_file, ts_index, synced_timestamps
    ):
        """_write_synced_zed_imu.

        Args:
            imu_csv_path: Input parameter.
            imu_output_file: Input parameter.
            ts_index: Input parameter.
            synced_timestamps: Input parameter.

        Returns:
            Any: Result of _write_synced_zed_imu.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        if not os.path.exists(imu_csv_path):
            return

        with open(imu_csv_path, "r") as f_in, open(
            imu_output_file, "w", newline=""
        ) as f_out:
            header = f_in.readline()
            f_out.write(header)
            all_lines = f_in.readlines()
            for i, idx in enumerate(ts_index):
                if i >= len(synced_timestamps) or idx >= len(all_lines):
                    continue
                parts = all_lines[idx].strip().split(",")
                synced_ts_ns = (synced_timestamps[i] + ZED_LATENCY) * 1000000
                parts[0] = str(synced_ts_ns)
                f_out.write(",".join(parts) + "\n")

        print(f"Saved synchronized ZED IMU data to: {imu_output_file}")
    @staticmethod
    def _to_xyz(v):
        if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
            return float(v.x), float(v.y), float(v.z)
        return float(v[0]), float(v[1]), float(v[2])

    @staticmethod
    def _imu_timestamp_us(imu_data):
        ts = getattr(imu_data, "timestamp", None)
        if ts is None:
            return ""
        if hasattr(ts, "get_microseconds"):
            return int(ts.get_microseconds() * 1000)
        if hasattr(ts, "get_nanoseconds"):
            return int(ts.get_nanoseconds())
        return ""

    @staticmethod
    def _append_imu_row(imu_rows, imu_data):
        try:
            if hasattr(imu_data, "get_linear_acceleration"):
                linear_acc = imu_data.get_linear_acceleration()
            else:
                linear_acc = imu_data.linear_acceleration

            if hasattr(imu_data, "get_angular_velocity"):
                angular_vel = imu_data.get_angular_velocity()
            else:
                angular_vel = imu_data.angular_velocity

            acc_x, acc_y, acc_z = ZEDVideoProcessor._to_xyz(linear_acc)
            gyro_x, gyro_y, gyro_z = ZEDVideoProcessor._to_xyz(angular_vel)
            imu_rows.append(
                [
                    ZEDVideoProcessor._imu_timestamp_us(imu_data),
                    acc_x,
                    acc_y,
                    acc_z,
                    gyro_x,
                    gyro_y,
                    gyro_z,
                ]
            )
        except Exception:
            pass

    def _process_zed_data(
        self, filepath, base_save_path, config_data, ts_index, synced_timestamps
    ):
        """_process_zed_data.

        Args:
            filepath: Input parameter.
            base_save_path: Input parameter.
            config_data: Input parameter.
            ts_index: Input parameter.
            synced_timestamps: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """处理单个ZED SVO视频及其IMU同步"""
        if not filepath.endswith("svo2"):
            return

        (
            zed_serial,
            _,
            video_file,
            ts_file,
            imu_output_file,
        ) = self._build_output_paths(filepath, base_save_path, config_data)

        self._write_timestamps(ts_file, synced_timestamps)

        frame_counts = Counter(ts_index)
        with reserve_encoder_slots(1):
            video_writer = self._create_video_writer(video_file, self.target_fps)

            init_params = self.sl.InitParametersOne()
            init_params.set_from_svo_file(filepath)
            init_params.svo_real_time_mode = False
            init_params.coordinate_units = self.sl.UNIT.MILLIMETER
            zed = self.sl.CameraOne()
            err = zed.open(init_params)
            if err != self.sl.ERROR_CODE.SUCCESS:
                video_writer.close()
                raise RuntimeError(f"Failed to open ZED SVO: {filepath}, error: {err}")
            
            single_image = self.sl.Mat()
            frame_count = 0
            nb_frames = zed.get_svo_number_of_frames()
            imu_rows = []
            sensors_data = self.sl.SensorsData()
            try:
                while True:
                    err = zed.grab()
                    if err == self.sl.ERROR_CODE.SUCCESS:
                        count = frame_counts.get(frame_count, 0)
                        if count > 0:
                            zed.retrieve_image(single_image, self.sl.VIEW.LEFT)
                            svo_image_rgb = cv2.cvtColor(
                                np.asanyarray(single_image.get_data()), cv2.COLOR_BGR2RGB
                            )
                            for _ in range(count):
                                video_writer.append_data(svo_image_rgb)

                        got_batch_imu = False
                        try:
                            all_sensors_data = []
                            batch_err = zed.get_sensors_data_batch(all_sensors_data)
                            if (
                                batch_err == self.sl.ERROR_CODE.SUCCESS
                                or batch_err is None
                            ):
                                for sensor_packet in all_sensors_data:
                                    imu_data = None
                                    if hasattr(sensor_packet, "imu"):
                                        imu_candidate = sensor_packet.imu
                                        if getattr(imu_candidate, "is_available", True):
                                            imu_data = imu_candidate
                                    elif hasattr(sensor_packet, "get_imu_data"):
                                        imu_data = sensor_packet.get_imu_data()

                                    if imu_data is not None:
                                        ZEDVideoProcessor._append_imu_row(
                                            imu_rows, imu_data
                                        )
                                        got_batch_imu = True
                        except Exception:
                            pass

                        if not got_batch_imu:
                            try:
                                sensors_err = zed.get_sensors_data(
                                    sensors_data, self.sl.TIME_REFERENCE.IMAGE
                                )
                                if sensors_err == self.sl.ERROR_CODE.SUCCESS:
                                    imu_data = sensors_data.get_imu_data()
                                    ZEDVideoProcessor._append_imu_row(
                                        imu_rows, imu_data
                                    )
                            except Exception:
                                pass

                        progress_bar((frame_count + 1) / nb_frames * 100, 30)
                        frame_count += 1
                    elif err == self.sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                        progress_bar(100, 30)
                        sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                        break
                    else:
                        raise RuntimeError(f"Error while reading SVO frame: {err}")
            finally:
                video_writer.close()
                zed.close()
        if imu_rows:
            with open(imu_output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "accel_x",
                        "accel_y",
                        "accel_z",
                        "gyro_x",
                        "gyro_y",
                        "gyro_z",
                    ]
                )
                writer.writerows(imu_rows)
        if len(synced_timestamps) > 0 and imu_rows:
            start_ts_ns = int(synced_timestamps[0]) * 1000000
            end_ts_ns = int(synced_timestamps[-1]) * 1000000
            self._crop_imu_csv_to_range(imu_output_file, start_ts_ns, end_ts_ns)

    @staticmethod
    def _crop_imu_csv_to_range(csv_path, start_ts_ns, end_ts_ns):
        if start_ts_ns > end_ts_ns:
            start_ts_ns, end_ts_ns = end_ts_ns, start_ts_ns

        dir_name = os.path.dirname(csv_path) or "."
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", delete=False, dir=dir_name, prefix="imu_crop_"
        ) as tmp_file:
            writer = csv.writer(tmp_file)
            with open(csv_path, "r") as f_in:
                reader = csv.reader(f_in)
                header = next(reader, None)
                if header:
                    writer.writerow(header)
                for row in reader:
                    if not row or not row[0]:
                        continue
                    try:
                        ts_ns = int(row[0])
                    except ValueError:
                        continue
                    if start_ts_ns <= ts_ns <= end_ts_ns:
                        writer.writerow(row)

        os.replace(tmp_file.name, csv_path)


    def get_zed_ts_array(self, svo_path):
        """get_zed_ts_array.

        Args:
            svo_path: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """读取SVO中每帧图像时间戳（毫秒）"""
        init_params = self.sl.InitParametersOne()
        init_params.set_from_svo_file(svo_path)
        init_params.svo_real_time_mode = False
        init_params.coordinate_units = self.sl.UNIT.MILLIMETER

        zed = self.sl.CameraOne()
        err = zed.open(init_params)
        if err != self.sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(
                f"Failed to open SVO for timestamp reading: {svo_path}, error: {err}"
            )

        nb_frames = zed.get_svo_number_of_frames()
        timestamps_list = []

        try:
            while True:
                err = zed.grab()
                if err == self.sl.ERROR_CODE.SUCCESS:
                    svo_position = zed.get_svo_position()
                    ts_image = zed.get_timestamp(
                        self.sl.TIME_REFERENCE.IMAGE
                    ).get_microseconds()
                    timestamps_list.append(ts_image)
                    progress_bar((svo_position + 1) / nb_frames * 100, 30)
                elif err == self.sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    progress_bar(100, 30)
                    sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                    break
                else:
                    raise RuntimeError(f"Error while reading SVO timestamps: {err}")
        finally:
            zed.close()

        return np.asarray(timestamps_list, dtype=np.int64) // 1000 - ZED_LATENCY
