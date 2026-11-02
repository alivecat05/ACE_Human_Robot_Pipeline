"""Postprocessing module: ColorSync.

Args:
    None.

Returns:
    None.

Raises:
    None.
"""

import sys
import numpy as np

import pyzed.sl as sl  # ZED disabled

import cv2
from pathlib import Path
import json
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import concurrent.futures
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

CALIBRATION_CAM_CHAIN = {
    "cam_0": {
        "cam_overlaps": [1, 2, 3],
        "camera_model": "omni",
        "distortion_coeffs": [
            -0.15720000545983642,
            1.3874605001372953,
            -0.000518729735182317,
            -0.0008002800802234733,
        ],
        "distortion_model": "radtan",
        "intrinsics": [
            3.3134622357764707,
            1718.2847912245609,
            1717.521366523875,
            552.6843901702701,
            691.5552737426997,
        ],
        "resolution": [1088, 1280],
    },
    "cam_1": {
        "T_cn_cnm1": [
            [
                0.0005523548439568815,
                -0.4849229737734174,
                0.8745566902213205,
                0.12307473919476013,
            ],
            [
                0.4921160545423723,
                0.7614593800485349,
                0.4219021230069229,
                0.058965810249917365,
            ],
            [
                -0.8705294272830609,
                0.4301503481840337,
                0.23905897638737483,
                -0.10512324252979721,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "cam_overlaps": [0, 2, 3],
        "camera_model": "omni",
        "distortion_coeffs": [
            -0.1599097889355886,
            0.8184769660283362,
            -0.0005081633265232919,
            0.002869689668312163,
        ],
        "distortion_model": "radtan",
        "intrinsics": [
            3.2744881462893085,
            1697.4782002534716,
            1696.6177412539128,
            561.2061962844216,
            624.7688349557004,
        ],
        "resolution": [1088, 1280],
    },
    "cam_2": {
        "T_cn_cnm1": [
            [
                -0.002540670008769119,
                -0.474114754021505,
                0.8804593943022281,
                0.07891400437567692,
            ],
            [
                0.49187451275654287,
                0.7659953339189014,
                0.41389686168778356,
                0.0665302376316608,
            ],
            [
                -0.87066239650994,
                0.43412711091754047,
                0.2312588222532697,
                -0.12141252732082745,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "cam_overlaps": [0, 1, 3],
        "camera_model": "omni",
        "distortion_coeffs": [
            -0.13124658509609222,
            0.9173786974068344,
            0.0032363136088274526,
            0.004540879695094806,
        ],
        "distortion_model": "radtan",
        "intrinsics": [
            3.268428085574126,
            1686.5544088396566,
            1686.431907560401,
            541.7854549752411,
            618.0161391513167,
        ],
        "resolution": [1088, 1280],
    },
    "cam_3": {
        "T_cn_cnm1": [
            [
                -0.011408627652438164,
                -0.49508314140592563,
                0.8687706983495268,
                0.17977223400785347,
            ],
            [
                0.5181734794729961,
                0.7401417949427601,
                0.4285864773295512,
                0.09072807633620515,
            ],
            [
                -0.8551994436305094,
                0.45506351916468285,
                0.24809495186222108,
                -0.15867733405165946,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "cam_overlaps": [0, 1, 2],
        "camera_model": "omni",
        "distortion_coeffs": [
            -0.1597118104058307,
            0.8769535367883307,
            0.0022925047359325657,
            0.00046093254346684634,
        ],
        "distortion_model": "radtan",
        "intrinsics": [
            3.2018007455997837,
            1660.153874816637,
            1659.9711947364515,
            550.9579762819737,
            641.5074469410732,
        ],
        "resolution": [1088, 1280],
    },
}

CAMERA_SETTINGS = {
    "zed": {"default_latency": 80, "min_latency": 40, "max_latency": 1000, "fps": 30},
    "realsense": {
        "default_latency": 110,
        "min_latency": 40,
        "max_latency": 1500,
        "fps": 15,
    },
    "egocap": {
        "default_latency": 190,
        "min_latency": 100,
        "max_latency": 1500,
        "fps": 20,
    },
}

COLOR_HSV_RANGES = {
    "zed": {
        "Blue": [([100, 160, 220], [120, 255, 255])],
        "Green": [([35, 60, 160], [60, 255, 255])],
        "Red": [([160, 160, 200], [180, 255, 255]), ([0, 160, 200], [20, 255, 255])],
        "Magenta": [([130, 60, 160], [155, 255, 255])],
        "Yellow": [([20, 160, 200], [35, 255, 255])],
    },
    "realsense": {
        "Blue": [([100, 160, 160], [120, 255, 255])],
        "Green": [([35, 60, 160], [60, 255, 255])],
        "Red": [([160, 160, 160], [180, 255, 255]), ([0, 160, 160], [20, 255, 255])],
        "Magenta": [([130, 60, 160], [155, 255, 255])],
        "Yellow": [([20, 60, 160], [35, 220, 255])],
    },
    "egocap": {
        "Blue": [([100, 160, 160], [120, 255, 255])],
        "Green": [([35, 60, 160], [60, 255, 255])],
        "Red": [([160, 160, 160], [180, 255, 255]), ([0, 160, 160], [20, 255, 255])],
        "Magenta": [([130, 60, 160], [155, 255, 255])],
        "Yellow": [([20, 60, 160], [35, 255, 255])],
    },
}

COLOR_NAMES = ["Blue", "Green", "Red", "Magenta", "Yellow"]
COLOR_IDX = {"Blue": 0, "Green": 1, "Red": 2, "Magenta": 3, "Yellow": 4}
COLOR_TRANS = {
    "Blue": "Green",
    "Green": "Red",
    "Red": "Magenta",
    "Magenta": "Yellow",
    "Yellow": "Blue",
}

SYNC_CONFIGS = {
    "color_interval": 0.4,  # time of one color display (seconds)
    "sequence_interval": 2.0,  # time of one sequence display (seconds)
    "MAX_COLOR_GAP": 3,  # maximum color gap allowed in transition
    "PEAK_LOSS_TOL": 0.3,  # maximum peak loss tolerance
}


class Analysis:

    """Class Analysis.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    def progress_bar(self, percent_done, bar_length=50):
        """progress_bar.

        Args:
            percent_done: Input parameter.
            bar_length: Input parameter.

        Returns:
            Any: Result of progress_bar.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        done_length = int(bar_length * percent_done / 100)
        bar = "=" * done_length + "-" * (bar_length - done_length)
        sys.stdout.write("[%s] %i%s\r" % (bar, percent_done, "%"))
        sys.stdout.flush()

    def plot_all_color(self, all_colors, result_path, camera_transition=None):
        """plot_all_color.

        Args:
            all_colors: Input parameter.
            result_path: Input parameter.
            camera_transition: Input parameter.

        Returns:
            Any: Result of plot_all_color.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        for video_name, video_color in all_colors.items():
            video_name_parts = list(Path(video_name).parts)
            frames = [item["frame"] for item in video_color]
            plt.figure(figsize=(12, 6))
            for color_name in COLOR_NAMES:
                color_sizes = [item["colors"][color_name] for item in video_color]
                plt.plot(
                    np.array(frames),
                    color_sizes,
                    label=color_name,
                    color=color_name.lower(),
                )
            if camera_transition is not None:
                for peak in camera_transition[video_name]["color_transition"]:
                    plt.axvline(
                        x=peak["idx"], color=peak["to_color"], linestyle="--", alpha=0.5
                    )
            plt.xlabel("Frame Index")
            plt.ylabel("Color Size Ratio")
            plt.title("Color Size Ratios Over Time")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    result_path,
                    f"{video_name_parts[-2]}_{video_name_parts[-1].split('.')[0]}.png",
                )
            )
            plt.close()


@dataclass
class ColorSyncConfig:
    """Class ColorSyncConfig.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    data_path: str = ""
    color_log_path: str = ""
    truncate_time: int = 80


class ColorSyncPipeline:

    """Class ColorSyncPipeline.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """
    def __init__(
        self,
        config: Optional[ColorSyncConfig] = None,
        input_func: Callable[[str], str] = input,
        exit_func: Callable[[int], None] = sys.exit,
    ) -> None:
        """__init__.

        Args:
            config: Input parameter.
            input_func: Input parameter.
            exit_func: Input parameter.

        Returns:
            None.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        self.config = config
        self.input_func = input_func
        self.exit_func = exit_func
        self.analysis = Analysis()
        self.data_path = self.config.data_path
        print(
            f"\n[INFO] Initialized ColorSyncPipeline with data path: {self.data_path}"
        )
        self.color_log_path = self.config.color_log_path
        self.truncate_time = self.config.truncate_time
        self.result_path = os.path.join(self.data_path, "sync_color_results")

    def get_color_size(self, hsv, camera):
        """get_color_size.

        Args:
            hsv: Input parameter.
            camera: Input parameter.

        Returns:
            Any: Result of get_color_size.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        color_size = {}
        for name, ranges in COLOR_HSV_RANGES[camera].items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for rge in ranges:
                mask |= cv2.inRange(hsv, np.array(rge[0]), np.array(rge[1]))
            area = cv2.countNonZero(mask)
            color_size[name] = int(area) / (hsv.shape[0] * hsv.shape[1])
        return color_size

    def get_zed_color_frames(self, svo_path, truncate_time=30, fps=30):
        # Create ZED One camera object
        """get_zed_color_frames.

        Args:
            svo_path: Input parameter.
            truncate_time: Input parameter.
            fps: Input parameter.

        Returns:
            Any: Result of get_zed_color_frames.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        try:
            init_params = sl.InitParametersOne()
            init_params.set_from_svo_file(svo_path)
            init_params.svo_real_time_mode = False
            init_params.coordinate_units = sl.UNIT.MILLIMETER
            zed = sl.CameraOne()
        except AttributeError:
            sys.stdout.write(
                "ZED One classes not available in this SDK version. Trying standard camera...\n"
            )
            is_zed_one = False
        err = zed.open(init_params)

        single_image = sl.Mat()
        nb_frames = zed.get_svo_number_of_frames()

        video_colors = []
        frame_idx = 0
        stop_frame = int(truncate_time * fps)
        camera = "zed"

        while True:
            if frame_idx >= stop_frame:
                break
            err = zed.grab()
            if err == sl.ERROR_CODE.SUCCESS:
                svo_position = zed.get_svo_position()
                zed.retrieve_image(single_image, sl.VIEW.LEFT)
                hsv = cv2.cvtColor(
                    np.asanyarray(single_image.get_data()), cv2.COLOR_BGR2HSV
                )
                ts_image = (
                    zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_microseconds()
                    // 1000
                )

                color_sizes = self.get_color_size(hsv, camera)

                video_colors.append(
                    {
                        "camera": camera,
                        "frame": frame_idx,
                        "timestamp": ts_image,
                        "colors": color_sizes,
                    }
                )
                frame_idx += 1

                # Display progress
                self.analysis.progress_bar((svo_position + 1) / nb_frames * 100, 30)

            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                self.analysis.progress_bar(100, 30)
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break

        zed.close()

        return video_colors

    def get_rs_color_frames(self, mp4_path, ts_path, truncate_time=30, fps=15):
        """get_rs_color_frames.

        Args:
            mp4_path: Input parameter.
            ts_path: Input parameter.
            truncate_time: Input parameter.
            fps: Input parameter.

        Returns:
            Any: Result of get_rs_color_frames.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        timestamps_list = []
        with open(ts_path, "r") as f:
            for line in f:
                timestamps_list.append(int(line.strip()))

        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {mp4_path}")

        video_colors = []
        frame_idx = 0
        stop_frame = int(truncate_time * fps)
        camera = "realsense"

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx >= stop_frame:
                break

            # 【优化】缩小画面，提升处理速度
            frame = cv2.resize(
                frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR
            )

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            ts = timestamps_list[frame_idx]

            color_sizes = self.get_color_size(hsv, camera)

            video_colors.append(
                {
                    "camera": camera,
                    "frame": frame_idx,
                    "timestamp": ts,
                    "colors": color_sizes,
                }
            )
            frame_idx += 1

        cap.release()
        return video_colors

    def _process_single_egocap_frame(self, args):
        """_process_single_egocap_frame.

        Args:
            args: Input parameter.

        Returns:
            Any: Function result.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        """用于多线程处理单帧 Egocap 图像的辅助函数"""
        rgb_file, frame_idx, mapx, mapy, camera = args
        img = cv2.imread(rgb_file)
        if img is None:
            return None

        img_shape = img.shape
        frame_cam = img[img_shape[0] // 4 : img_shape[0] // 2, :]
        persp = cv2.remap(frame_cam, mapx, mapy, interpolation=cv2.INTER_LINEAR)

        hsv = cv2.cvtColor(persp, cv2.COLOR_BGR2HSV)
        color_sizes = self.get_color_size(hsv, camera)

        timestamp = int(Path(rgb_file).stem.split("_")[-1]) // 1000000

        return {
            "camera": camera,
            "frame": frame_idx,
            "timestamp": timestamp,
            "colors": color_sizes,
        }

    def get_egocap_color_frames(self, rgb_path, truncate_time=30, fps=20):
        """get_egocap_color_frames.

        Args:
            rgb_path: Input parameter.
            truncate_time: Input parameter.
            fps: Input parameter.

        Returns:
            Any: Result of get_egocap_color_frames.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        rgb_files = [item.path for item in os.scandir(rgb_path) if item.is_file()]
        rgb_files.sort()
        cam_params = {}
        for c in ["cam_0", "cam_1", "cam_2", "cam_3"]:
            cam_params[c] = {}
            xi, fx, fy, cx, cy = CALIBRATION_CAM_CHAIN[c]["intrinsics"]
            cam_params[c]["K"] = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
            )
            cam_params[c]["xi"] = np.array([xi], dtype=np.float64)
            cam_params[c]["dist"] = np.array(
                CALIBRATION_CAM_CHAIN[c]["distortion_coeffs"], dtype=np.float64
            )

        cam1_out = np.array(CALIBRATION_CAM_CHAIN["cam_1"]["T_cn_cnm1"])
        key_rots = R.from_matrix(np.stack([np.eye(3), cam1_out[:3, :3]]))
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        R_center = slerp(0.5)
        R_center_matrix = R_center.as_matrix()
        R_cam0_center = R_center_matrix.T
        look_down_angle = np.radians(10)
        R_look_down = np.array(
            [
                [1, 0, 0],
                [0, np.cos(look_down_angle), -np.sin(look_down_angle)],
                [0, np.sin(look_down_angle), np.cos(look_down_angle)],
            ]
        )
        R_c = [
            R_look_down @ (R_cam0_center @ cam1_out[:3, :3]),
            R_look_down @ R_cam0_center,
        ]

        fov = np.deg2rad(90)
        # 【优化】降低 remap 分辨率至 128x128
        out_size = (128, 128)
        fx = fy = out_size[0] / (2 * np.tan(fov / 2))
        cx = out_size[0] / 2
        cy = out_size[1] / 2
        Kp = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # mapping matrix
        target_cam_id = 1
        mapx, mapy = cv2.omnidir.initUndistortRectifyMap(
            cam_params[f"cam_{target_cam_id}"]["K"],
            cam_params[f"cam_{target_cam_id}"]["dist"],
            cam_params[f"cam_{target_cam_id}"]["xi"],
            R_c[target_cam_id],
            Kp,
            out_size,
            cv2.CV_32F,
            cv2.omnidir.RECTIFY_PERSPECTIVE,
        )

        camera = "egocap"
        stop_frame = int(truncate_time * fps)

        # 截取需要的处理范围
        files_to_process = rgb_files[:stop_frame]

        # 构造任务参数
        tasks = [
            (files_to_process[i], i, mapx, mapy, camera)
            for i in range(len(files_to_process))
        ]

        # 【优化】使用多线程池加速 I/O 阻塞和处理
        max_workers = min(32, (os.cpu_count() or 1) * 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(self._process_single_egocap_frame, tasks),
                    total=len(tasks),
                )
            )

        # 过滤可能处理失败的空帧
        video_colors = [r for r in results if r is not None]

        return video_colors

    def check_color_log_overlap(self, color_readout, color_log):
        """check_color_log_overlap.

        Args:
            color_readout: Input parameter.
            color_log: Input parameter.

        Returns:
            Any: Result of check_color_log_overlap.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        non_overlap_cameras = []
        color_log_start = color_log[0]["timestamp"]
        color_log_end = color_log[-1]["timestamp"]
        for video_name, video_color in color_readout.items():
            video_name_parts = list(Path(video_name).parts)
            video_color_start = video_color[0]["timestamp"] / 1000
            video_color_end = video_color[-1]["timestamp"] / 1000
            if color_log_end < video_color_start or color_log_start > video_color_end:
                non_overlap_cameras.append(
                    f"{video_name_parts[-2]}_{video_name_parts[-1].split('.')[0]}"
                )
        return list(set(non_overlap_cameras))

    def find_color_transitions(
        self, video_name, video_color, result_path, color_log=None
    ):
        """find_color_transitions.

        Args:
            video_name: Input parameter.
            video_color: Input parameter.
            result_path: Input parameter.
            color_log: Input parameter.

        Returns:
            Any: Result of find_color_transitions.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        color_transitions = {c: [] for c in COLOR_NAMES}
        if len(video_color) == 0:
            return color_transitions

        camera = video_color[0]["camera"]
        fps = CAMERA_SETTINGS[camera]["fps"]
        color_interval = SYNC_CONFIGS["color_interval"]
        vnp = list(Path(video_name).parts)

        # sliding window filter
        interval_frames = int(fps * color_interval - 1)
        trans_from_filter = np.array(
            [1] * interval_frames + [0] + [-1] * interval_frames
        )
        trans_to_filter = np.array([-1] * interval_frames + [0] + [1] * interval_frames)
        plain_filter = np.array([1] * (2 * interval_frames + 1))

        # for each color transition pair, pass through filter
        all_video_colors = {
            color: np.array([item["colors"][color] for item in video_color])
            for color in COLOR_NAMES
        }
        all_video_ts = np.array([item["timestamp"] for item in video_color])
        all_transition_signals = {"from": {}, "to": {}}
        for color in COLOR_NAMES:
            from_conv = np.convolve(
                all_video_colors[color], trans_from_filter, mode="same"
            )
            to_conv = np.convolve(all_video_colors[color], trans_to_filter, mode="same")
            assert from_conv.shape == all_video_ts.shape
            assert to_conv.shape == all_video_ts.shape
            from_conv[from_conv < 0.0] = 0.0
            to_conv[to_conv < 0.0] = 0.0
            all_transition_signals["from"][color] = from_conv
            all_transition_signals["to"][color] = to_conv

        # pair from and to signals with COLOR_TRANS
        plt.figure(figsize=(12, 6))
        all_peaks = []
        for f_c, t_c in COLOR_TRANS.items():
            from_signal = all_transition_signals["to"][f_c]
            to_signal = all_transition_signals["from"][t_c]

            combined_signal = from_signal + to_signal

            # find all peaks
            peaks, properties = find_peaks(
                combined_signal, distance=interval_frames, prominence=0.02
            )
            for p in peaks:
                all_peaks.append(
                    {
                        "idx": p,
                        "from_color": f_c,
                        "to_color": t_c,
                        "timestamp": all_video_ts[p],
                        "prominence": combined_signal[p],
                    }
                )

        plt.close()

        # match color transition pattern with all peaks
        all_peaks = sorted(all_peaks, key=lambda x: x["idx"])
        # build a queue of pattern matched candidates
        pattern_candidates = []
        for peak in all_peaks:
            matched = False
            # if a peak matches former candidates, add to them
            for pat_cand in pattern_candidates:
                last_peak = pat_cand["peaks"][-1]
                last_peak_idx = COLOR_IDX[last_peak["to_color"]]
                current_peak_idx = COLOR_IDX[peak["to_color"]]
                color_gap = current_peak_idx - last_peak_idx
                if last_peak_idx > current_peak_idx:
                    color_gap += len(COLOR_IDX)
                if color_gap > SYNC_CONFIGS["MAX_COLOR_GAP"]:
                    continue
                # check time difference
                expected_time = (
                    last_peak["timestamp"] / 1000
                    + SYNC_CONFIGS["color_interval"] * color_gap
                )
                loss = (
                    abs(peak["timestamp"] / 1000 - expected_time)
                    / SYNC_CONFIGS["color_interval"]
                )
                if loss < SYNC_CONFIGS["PEAK_LOSS_TOL"]:  # within 30% tolerance
                    pat_cand["peaks"].append(peak)
                    pat_cand["gap"].append(color_gap)
                    pat_cand["len"] += 1
                    pat_cand["loss"] += loss * color_gap
                    matched = True
            # if not, start a new candidate
            if not matched:
                pattern_candidates.append(
                    {"peaks": [peak], "gap": [], "len": 1, "loss": 0.0}
                )

        # get the longest one with least loss
        best_candidate = None
        best_loss = float("inf")
        for pat_cand in pattern_candidates:
            if pat_cand["len"] > 1:
                if best_candidate is None:
                    best_candidate = pat_cand
                    best_loss = pat_cand["loss"] / (pat_cand["len"] - 1)
                else:
                    if pat_cand["len"] > best_candidate["len"]:
                        best_candidate = pat_cand
                        best_loss = pat_cand["loss"] / (pat_cand["len"] - 1)
                    elif (
                        pat_cand["len"] == best_candidate["len"]
                        and pat_cand["loss"] / (pat_cand["len"] - 1) < best_loss
                    ):
                        best_candidate = pat_cand
                        best_loss = pat_cand["loss"] / (pat_cand["len"] - 1)

        if best_candidate is None:
            return []

        if color_log is not None:
            peak_ts = [p["timestamp"] / 1000 for p in best_candidate["peaks"]]
            peak_color = [p["from_color"] for p in best_candidate["peaks"]]
            standardized_peaks = []
            for i in range(len(peak_ts)):
                # align to the cloest log ts before the peak ts
                peak_time = peak_ts[i]
                closest_log = None
                for log in color_log:
                    if log["color_name"] != peak_color[i]:
                        continue
                    log_time = log["timestamp"]
                    if log_time <= peak_time:
                        closest_log = log_time
                    else:
                        break
                standardized_peaks.append(closest_log)
            for i in range(len(best_candidate["peaks"])):
                best_candidate["peaks"][i]["std_ts"] = standardized_peaks[i]
        else:
            # align peaks to standardized pattern
            standardized_peaks = [best_candidate["peaks"][0]["timestamp"] / 1000]
            peak_ts = [p["timestamp"] / 1000 for p in best_candidate["peaks"]]
            prominences = [p["prominence"] for p in best_candidate["peaks"]]
            all_gap = 0
            for p in best_candidate["gap"]:
                all_gap += p
                last_ts = standardized_peaks[-1]
                standardized_peaks.append(last_ts + SYNC_CONFIGS["color_interval"] * p)
            std = np.array(standardized_peaks)
            raw = np.array(peak_ts)
            pro = np.array(prominences)
            weighted_diff = pro * (raw - std)
            bias_w = np.sum(weighted_diff) / np.sum(pro)
            bias = np.mean(raw - std)
            aligned_std = [round(float(x), 3) for x in (std + bias).tolist()]
            aligned_std_w = [round(float(x), 3) for x in (std + bias_w).tolist()]
            for i in range(len(best_candidate["peaks"])):
                best_candidate["peaks"][i]["std_ts"] = aligned_std[i]

        return best_candidate["peaks"]

    def aggregate_transitions(self, camera_transitions):
        """aggregate_transitions.

        Args:
            camera_transitions: Input parameter.

        Returns:
            Any: Result of aggregate_transitions.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        camera_latencies = {}
        for video_name, video_transition in camera_transitions.items():
            vnp = list(Path(video_name).parts)
            color_transition = video_transition["color_transition"]
            camera = video_transition["camera"]

            if len(color_transition) == 0:
                print(vnp[-2], vnp[-1], f"WARNING: No color transitions detected!")
                camera_latencies[video_name] = CAMERA_SETTINGS[camera][
                    "default_latency"
                ]
                continue

            valid_transitions = [
                vid_trans
                for vid_trans in color_transition
                if vid_trans.get("std_ts") is not None
            ]
            skipped_none = len(color_transition) - len(valid_transitions)
            if skipped_none > 0:
                print(
                    vnp[-2],
                    vnp[-1],
                    f"WARNING: Skipping {skipped_none} transition(s) with std_ts=None",
                )
            if len(valid_transitions) == 0:
                print(
                    vnp[-2],
                    vnp[-1],
                    f"WARNING: No valid transitions with std_ts! Using default latency.",
                )
                camera_latencies[video_name] = CAMERA_SETTINGS[camera][
                    "default_latency"
                ]
                continue

            latencies = np.array(
                [
                    vid_trans["timestamp"] - vid_trans["std_ts"] * 1000
                    for vid_trans in valid_transitions
                ]
            )
            latencies = latencies[latencies > CAMERA_SETTINGS[camera]["min_latency"]]
            latencies = latencies[latencies < CAMERA_SETTINGS[camera]["max_latency"]]

            if latencies.size == 0:
                print(
                    vnp[-2],
                    vnp[-1],
                    f"WARNING: No valid latencies after filtering! Using default latency.",
                )
                camera_latencies[video_name] = CAMERA_SETTINGS[camera][
                    "default_latency"
                ]
            else:
                camera_latencies[video_name] = np.mean(latencies)
                print(
                    vnp[-2],
                    vnp[-1],
                    f"Latency: {np.mean(latencies):.2f} (std {np.std(latencies):.2f}) ms over {latencies.size} transitions",
                )

        return camera_latencies

    def _process_realsense(self, color_readout: Dict[str, List[dict]]) -> None:
        """_process_realsense.

        Args:
            color_readout: Input parameter.

        Returns:
            None.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        rs_data_path = os.path.join(self.data_path, "realsense")

        print(
            f"\n[INFO] Processing Realsense data under:    {self.data_path}                 {rs_data_path}..."
        )
        if not os.path.exists(rs_data_path):
            print(
                f"Warning: Realsense data path does not exist: {rs_data_path}. Skipping Realsense processing."
            )
            return

        mp4_files = sorted(Path(rs_data_path).rglob("*_rgb.mp4"))
        ts_files = sorted(Path(rs_data_path).rglob("*_ts.csv"))
        if len(mp4_files) == 0:
            print(f"Warning: No Realsense mp4 files found under: {rs_data_path}")
            return
        if len(mp4_files) != len(ts_files):
            print(
                f"Warning: Number of Realsense mp4 ({len(mp4_files)}) and ts files ({len(ts_files)}) do not match! Skipping Realsense processing."
            )
            return

        for rs_idx in range(len(ts_files)):
            print(f"Processing Realsense file {os.path.basename(mp4_files[rs_idx])}")
            mp4_file = mp4_files[rs_idx]
            ts_file = ts_files[rs_idx]
            color_dict = self.get_rs_color_frames(
                str(mp4_file),
                str(ts_file),
                truncate_time=self.truncate_time,
            )
            color_readout[str(mp4_file)] = color_dict

    def _process_egocap(self, color_readout: Dict[str, List[dict]]) -> None:
        """_process_egocap.

        Args:
            color_readout: Input parameter.

        Returns:
            None.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        egocap_data_path = os.path.join(self.data_path, "egocap")
        if not os.path.exists(egocap_data_path):
            print(
                f"Warning: Egocap data path does not exist: {egocap_data_path}. Skipping Egocap processing."
            )
            return

        egocap_subdirs = [
            item.path for item in os.scandir(egocap_data_path) if item.is_dir()
        ]
        if len(egocap_subdirs) == 0:
            print(
                f"Warning: No subdirectories found in Egocap data path: {egocap_data_path}"
            )
            return

        print("Processing Egocap")
        egocap_data_dir = egocap_subdirs[0]
        rgb_path = os.path.join(egocap_data_dir, "rgb")
        if not os.path.exists(rgb_path):
            print(f"Warning: RGB path does not exist: {rgb_path}")
            return

        color_dict = self.get_egocap_color_frames(
            rgb_path,
            truncate_time=self.truncate_time,
        )
        color_readout[str(egocap_data_dir)] = color_dict

    def _ask_log_continue_or_exit(self, color_log: Optional[List[dict]]):
        """_ask_log_continue_or_exit.

        Args:
            color_log: Input parameter.

        Returns:
            Any: Result of _ask_log_continue_or_exit.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        while True:
            user_input = self.input_func(
                "Do you want to: (1) Continue with this log? (2) Continue with no log (3) Exit?\n"
            )
            if user_input == "1":
                print(
                    "Continuing with provided color log, which may not work for some cameras."
                )
                return color_log
            if user_input == "2":
                print("Continuing with relative synchronization.")
                return None
            if user_input == "3":
                print("Exiting program.")
                self.exit_func(0)
                return None
            print("Invalid input. Please enter '1', '2', or '3'.")

    def _load_color_log(
        self, color_readout: Dict[str, List[dict]]
    ) -> Optional[List[dict]]:
        """_load_color_log.

        Args:
            color_readout: Input parameter.

        Returns:
            Optional[List[dict]]: Result of _load_color_log.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        if not os.path.exists(self.color_log_path):
            print(
                "Color log file not provided! We can only infer relative synchronization."
            )
            return None

        if self.color_log_path.endswith("json"):
            print("Color log file provided!")
            color_log = json.load(open(self.color_log_path, "r"))
        else:
            print("Color log directory provided!")
            color_log = []
            log_files = sorted(Path(self.color_log_path).rglob("*.json"))
            for log_file in log_files:
                log_data = json.load(open(log_file, "r"))
                color_log.extend(log_data)

        non_overlap = self.check_color_log_overlap(color_readout, color_log)
        if len(non_overlap) == 0:
            print(
                "All cameras have overlap with log! We can infer absolute camera latency for real synchronization."
            )
            return color_log

        print(
            "Warning: The following cameras have no overlapping colors between readout and log:"
        )
        for camera in non_overlap:
            print(f" - {camera}")
        return self._ask_log_continue_or_exit(color_log)

    def _process_single_svo(self, args):
        """_process_single_svo.

        Args:
            args: Input parameter.

        Returns:
            Any: Result of _process_single_svo.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        svo_path, truncate_time = args
        print(f"Processing ZED file: {svo_path}")
        color_dict = self.get_zed_color_frames(
            str(svo_path), truncate_time=truncate_time
        )
        return (str(svo_path), color_dict)

    def _process_zed(self, color_readout: Dict[str, List[dict]]) -> None:
        """_process_zed.

        Args:
            color_readout: Input parameter.

        Returns:
            None.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        zed_data_path = os.path.join(self.data_path, "zed")
        svo_files = sorted(Path(zed_data_path).rglob("*.svo2"))
        if not svo_files:
            print(f"No .svo2 files found under: {zed_data_path}")
            sys.exit(0)
        print(f"Found {len(svo_files)} .svo2 files.")
        # for svo_path in svo_files:
        #     print(f"Processing ZED file: {svo_path}")
        #     color_dict = self.get_zed_color_frames(str(svo_path), truncate_time=self.truncate_time)
        #     color_readout[str(svo_path)] = color_dict

        tasks = [(str(svo_path), self.truncate_time) for svo_path in svo_files]
        max_workers = min(32, (os.cpu_count() or 1) * 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(self._process_single_svo, tasks),
                    total=len(tasks),
                )
            )
        for video_name, color_dict in results:
            color_readout[video_name] = color_dict

    def execute(self) -> bool:
        """execute.

        Args:
            None.

        Returns:
            bool: Result of execute.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        os.makedirs(self.result_path, exist_ok=True)
        print("color log path:", self.color_log_path)
        color_readout: Dict[str, List[dict]] = {}

        self._process_realsense(color_readout)
        self._process_egocap(color_readout)
        self._process_zed(color_readout)  # ZED disabled
        if len(color_readout) == 0:
            print("Error: No camera data was successfully processed. Exiting.")
            self.exit_func(1)
            return True

        color_log = self._load_color_log(color_readout)

        all_camera_transition = {}
        for video_name, video_color in color_readout.items():
            if len(video_color) > 0:
                color_transition = self.find_color_transitions(
                    video_name,
                    video_color,
                    self.result_path,
                    color_log,
                )
                all_camera_transition[video_name] = {
                    "color_transition": color_transition,
                    "camera": video_color[0]["camera"],
                }
            else:
                print(f"Warning: No color data for {video_name}. Skipping.")

        if len(all_camera_transition) > 0:
            self.analysis.plot_all_color(
                color_readout, self.result_path, all_camera_transition
            )
        else:
            print("Warning: No camera transitions detected. Skipping plotting.")

        all_camera_latencies = self.aggregate_transitions(all_camera_transition)
        fix_zed_latency = 80  # ZED disabled
        all_camera_latencies_sn = {}
        for video_path, latency in all_camera_latencies.items():
            sn_path = (
                video_path.split("/")[-1].split("\\")[-1].split(".")[0].split("_")[0]
            )
            if (
                 len(sn_path) == 9
                 and latency != CAMERA_SETTINGS["zed"]["default_latency"]
             ):
                 fix_zed_latency = latency
            if "egocap" in video_path:
                sn_path = "egocap"
            all_camera_latencies_sn[sn_path] = latency
        all_camera_latencies_sn = {
             k: (v if len(k) != 9 else fix_zed_latency)
             for k, v in all_camera_latencies_sn.items()
        }  # ZED disabled
        json.dump(
            all_camera_latencies_sn,
            open(os.path.join(self.result_path, "all_camera_latencies.json"), "w"),
            indent=4,
        )

        for k, v in all_camera_latencies_sn.items():
            if (
                v == CAMERA_SETTINGS["realsense"]["default_latency"]
                or v == CAMERA_SETTINGS["egocap"]["default_latency"]
                or v == CAMERA_SETTINGS["zed"]["default_latency"]  # ZED disabled
            ):
                print(f"Camera {k} latency is set to default value {v} ms")
                return True

        return False
