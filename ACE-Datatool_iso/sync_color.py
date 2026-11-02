import sys
import pyzed.sl as sl
import numpy as np
import cv2
from pathlib import Path
import enum
import json
import yaml
import argparse
import os
import time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


CAMERA_SETTINGS = {
    "zed": {
        "default_latency": 80,
        "min_latency": 40,
        "max_latency": 300,
        "fps": 15
    },
    "realsense": {
        "default_latency": 110,
        "min_latency": 40,
        "max_latency": 750,
        "fps": 15
    },
    "egocap": {
        "default_latency": 190,
        "min_latency": 100,
        "max_latency": 800,
        "fps": 20
    }
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
    }
}

COLOR_NAMES = ["Blue", "Green", "Red", "Magenta", "Yellow"]
COLOR_IDX = {"Blue":0, "Green":1, "Red":2, "Magenta":3, "Yellow":4}
COLOR_TRANS = {
    "Blue": "Green",
    "Green": "Red",
    "Red": "Magenta",
    "Magenta": "Yellow",
    "Yellow": "Blue"
}

SYNC_CONFIGS = {
    "color_interval": 0.4,      # time of one color display (seconds)
    "sequence_interval": 2.0,   # time of one sequence display (seconds)
    "MAX_COLOR_GAP": 3,         # maximum color gap allowed in transition
    "PEAK_LOSS_TOL": 0.3        # maximum peak loss tolerance
}


def progress_bar(percent_done, bar_length=50):
    # Display a progress bar
    done_length = int(bar_length * percent_done / 100)
    bar = "=" * done_length + "-" * (bar_length - done_length)
    sys.stdout.write("[%s] %i%s\r" % (bar, percent_done, "%"))
    sys.stdout.flush()
    

def get_color_size(hsv, camera) :
    color_size = {}
    for name, ranges in COLOR_HSV_RANGES[camera].items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for rge in ranges:
            mask |= cv2.inRange(
                hsv, 
                np.array(rge[0]), 
                np.array(rge[1])
            )
        area = cv2.countNonZero(mask)
        color_size[name] = int(area) / (hsv.shape[0] * hsv.shape[1])
    return color_size


def get_zed_color_frames(svo_path, truncate_time=30, fps=15):
    # Create ZED One camera object
    try:
        init_params = sl.InitParametersOne()
        init_params.set_from_svo_file(svo_path)
        init_params.svo_real_time_mode = False
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        zed = sl.CameraOne()
    except AttributeError:
        sys.stdout.write("ZED One classes not available in this SDK version. Trying standard camera...\n")
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
            hsv = cv2.cvtColor(np.asanyarray(single_image.get_data()), cv2.COLOR_BGR2HSV)
            ts_image = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_microseconds() // 1000
            
            color_sizes = get_color_size(hsv, camera)

            video_colors.append({
                "camera": camera,
                "frame": frame_idx,
                "timestamp": ts_image,
                "colors": color_sizes,
            })
            frame_idx += 1

            # Display progress
            progress_bar((svo_position + 1) / nb_frames * 100, 30)

        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            progress_bar(100, 30)
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            break
    
    zed.close()
    
    return video_colors


def get_rs_color_frames(mp4_path, ts_path, truncate_time=30, fps=15):
    timestamps_list = []
    with open(ts_path, 'r') as f :
        for line in f :
            timestamps_list.append(int(line.strip()))

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {mp4_path}")

    video_colors = []
    frame_idx = 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stop_frame = int(truncate_time * fps)
    camera = "realsense"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= stop_frame:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ts = timestamps_list[frame_idx]
            
        color_sizes = get_color_size(hsv, camera)

        video_colors.append({
            "camera": camera,
            "frame": frame_idx,
            "timestamp": ts,
            "colors": color_sizes,
        })
        frame_idx += 1

    cap.release()
    return video_colors


def get_egocap_color_frames(rgb_path, truncate_time=30, fps=20):
    rgb_files = [item.path for item in os.scandir(rgb_path) if item.is_file()]
    rgb_files.sort()
    
    yaml_file = "./egocap/calibration2-camchain-4cam.yaml"
    assert os.path.exists(yaml_file), "Egocap Config not found!"
    with open(yaml_file, 'r') as f:
        yaml_ = yaml.safe_load(f)
    
    cam_params = {}
    for c in ["cam_0", "cam_1", "cam_2", "cam_3"] :
        cam_params[c] = {}
        xi, fx, fy, cx, cy = yaml_[c]["intrinsics"]
        cam_params[c]["K"] = np.array(
            [[fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]], dtype=np.float64)
        cam_params[c]["xi"] = np.array([xi], dtype=np.float64)
        cam_params[c]["dist"] = np.array(yaml_[c]["distortion_coeffs"], dtype=np.float64)
    
    cam1_out = np.array(yaml_["cam_1"]["T_cn_cnm1"])
    key_rots = R.from_matrix(np.stack([np.eye(3), cam1_out[:3, :3]]))
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    R_center = slerp(0.5)
    R_center_matrix = R_center.as_matrix()
    R_cam0_center = R_center_matrix.T
    look_down_angle = np.radians(10)
    R_look_down = np.array([
        [1, 0, 0],
        [0, np.cos(look_down_angle), -np.sin(look_down_angle)],
        [0, np.sin(look_down_angle), np.cos(look_down_angle)]
    ])
    R_c = [R_look_down @ (R_cam0_center @ cam1_out[:3, :3]), R_look_down @ R_cam0_center]

    fov = np.deg2rad(90)
    out_size = (512, 512)
    fx = fy = out_size[0] / (2 * np.tan(fov / 2))
    cx = out_size[0] / 2
    cy = out_size[1] / 2
    Kp = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]], dtype=np.float64)
    
    # mapping matrix
    target_cam_id = 1
    mapx, mapy = cv2.omnidir.initUndistortRectifyMap(
        cam_params[f"cam_{target_cam_id}"]['K'],
        cam_params[f"cam_{target_cam_id}"]['dist'],
        cam_params[f"cam_{target_cam_id}"]['xi'],
        R_c[target_cam_id],
        Kp,
        out_size,
        cv2.CV_32F,
        cv2.omnidir.RECTIFY_PERSPECTIVE
    )

    video_colors = []
    frame_idx = 0
    stop_frame = int(truncate_time * fps)
    camera = "egocap"
    
    for rgb_file in tqdm(rgb_files):
        if frame_idx >= stop_frame:
            break
        img = cv2.imread(rgb_file)
        img_shape = img.shape
        frame_cam = img[img_shape[0]//4: img_shape[0]//2, :]
        persp = cv2.remap(frame_cam, mapx, mapy, interpolation=cv2.INTER_LINEAR)

        hsv = cv2.cvtColor(persp, cv2.COLOR_BGR2HSV)
        color_sizes = get_color_size(hsv, camera)
        video_colors.append({
            "camera": camera,
            "frame": frame_idx,
            "timestamp": int(rgb_file.split("_")[-1].split(".")[0]) // 1000000,
            "colors": color_sizes,
        })
        frame_idx += 1
    
    return video_colors


def check_color_log_overlap(color_readout, color_log):
    non_overlap_cameras = []
    color_log_start = color_log[0]["timestamp"]
    color_log_end = color_log[-1]["timestamp"]
    for video_name, video_color in color_readout.items():
        video_name_parts = list(Path(video_name).parts)
        video_color_start = video_color[0]["timestamp"] / 1000
        video_color_end = video_color[-1]["timestamp"] / 1000
        if color_log_end < video_color_start or color_log_start > video_color_end:
            non_overlap_cameras.append(f"{video_name_parts[-2]}_{video_name_parts[-1].split('.')[0]}")
    return list(set(non_overlap_cameras))


def plot_all_color(all_colors, result_path, camera_transition=None):
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
                color=color_name.lower()
            )
        if camera_transition is not None :
            for peak in camera_transition[video_name]["color_transition"]:
                plt.axvline(x=peak['idx'], color=peak['to_color'], linestyle='--', alpha=0.5)
        plt.xlabel("Frame Index")
        plt.ylabel("Color Size Ratio")
        plt.title("Color Size Ratios Over Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                result_path, 
                f"{video_name_parts[-2]}_{video_name_parts[-1].split('.')[0]}.png"
            )
        )
        plt.close()


def find_color_transitions(video_name, video_color, result_path, color_log=None):
    color_transitions = {c: [] for c in COLOR_NAMES}
    if len(video_color) == 0:
        return color_transitions
    
    camera = video_color[0]["camera"]
    fps = CAMERA_SETTINGS[camera]["fps"]
    color_interval = SYNC_CONFIGS["color_interval"]
    vnp = list(Path(video_name).parts)
    
    # sliding window filter
    interval_frames = int(fps * color_interval - 1)
    trans_from_filter = np.array([1] * interval_frames + [0] + [-1] * interval_frames)
    trans_to_filter = np.array([-1] * interval_frames + [0] + [1] * interval_frames)
    plain_filter = np.array([1] * (2 * interval_frames + 1))
    
    # for each color transition pair, pass through filter
    all_video_colors = {color: np.array([item["colors"][color] for item in video_color]) for color in COLOR_NAMES}
    all_video_ts = np.array([item["timestamp"] for item in video_color])
    all_transition_signals = {"from": {}, "to": {}}
    for color in COLOR_NAMES:
        # plain_conv = np.convolve(all_video_colors[color], plain_filter, mode='same') + 1e-4
        from_conv = np.convolve(all_video_colors[color], trans_from_filter, mode='same')
        to_conv = np.convolve(all_video_colors[color], trans_to_filter, mode='same')
        assert from_conv.shape == all_video_ts.shape
        assert to_conv.shape == all_video_ts.shape
        from_conv[from_conv < 0.0] = 0.0
        to_conv[to_conv < 0.0] = 0.0
        all_transition_signals["from"][color] = from_conv
        all_transition_signals["to"][color] = to_conv
        # # make a plt plot and save
        # plt.figure(figsize=(12, 6))
        # plt.plot(from_conv, label=f'{color} From Filter', color='blue')
        # plt.plot(to_conv, label=f'{color} To Filter', color='red')
        # plt.title(f'Color Transition Filters for {color}')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(result_path, f'{vnp[-2]}_{vnp[-1]}_{color}.png'))
        # plt.close()
    
    # pair from and to signals with COLOR_TRANS
    plt.figure(figsize=(12, 6))
    all_peaks = []
    for f_c, t_c in COLOR_TRANS.items():
        from_signal = all_transition_signals["to"][f_c]
        to_signal = all_transition_signals["from"][t_c]
        
        combined_signal = from_signal + to_signal
        # plt.plot(combined_signal, label=f'{f_c} to {t_c} Combined Filter', color=t_c)
        
        # find all peaks
        peaks, properties = find_peaks(
            combined_signal, 
            distance=interval_frames, 
            prominence=0.02
        )
        for p in peaks :
            all_peaks.append({
                "idx": p,
                "from_color": f_c,
                "to_color": t_c,
                "timestamp": all_video_ts[p],
                "prominence": combined_signal[p]
            })
        
    #     # Plot peaks as markers on top of the signal
    #     if len(peaks) > 0:
    #         plt.scatter(peaks, combined_signal[peaks], 
    #                 color=t_c, s=50, zorder=5, 
    #                 label=f'{f_c} to {t_c} Peaks (n={len(peaks)})')
    
    # plt.title(f'Color Transition Filters')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(result_path, f'{vnp[-2]}_{vnp[-1]}_trans.png'))
    # plt.close()
    
    # match color transition pattern with all peaks
    all_peaks = sorted(all_peaks, key=lambda x: x["idx"])
    # build a queue of pattern matched candidates
    pattern_candidates = []
    for peak in all_peaks :
        matched = False
        # if a peak matches former candidates, add to them
        for pat_cand in pattern_candidates :
            last_peak = pat_cand["peaks"][-1]
            last_peak_idx = COLOR_IDX[last_peak["to_color"]]
            current_peak_idx = COLOR_IDX[peak["to_color"]]
            color_gap = current_peak_idx - last_peak_idx
            if last_peak_idx > current_peak_idx :
                color_gap += len(COLOR_IDX)
            if color_gap > SYNC_CONFIGS["MAX_COLOR_GAP"] :
                continue
            # check time difference
            expected_time = last_peak["timestamp"] / 1000 + SYNC_CONFIGS["color_interval"] * color_gap
            loss = abs(peak["timestamp"] / 1000 - expected_time) / SYNC_CONFIGS["color_interval"]
            if loss < SYNC_CONFIGS["PEAK_LOSS_TOL"] :   # within 30% tolerance
                pat_cand["peaks"].append(peak)
                pat_cand["gap"].append(color_gap)
                pat_cand["len"] += 1
                pat_cand["loss"] += loss * color_gap
                matched = True
        # if not, start a new candidate
        if not matched :
            pattern_candidates.append({
                "peaks": [peak],
                "gap": [],
                "len": 1, 
                "loss": 0.0
            })
    
    # print(video_name)
    # for p in pattern_candidates :
    #     if p["len"] > 1 :
    #         print(p)
    
    # get the longest one with least loss
    best_candidate = None
    best_loss = float('inf')
    for pat_cand in pattern_candidates :
        if pat_cand["len"] > 1 :
            if best_candidate is None :
                best_candidate = pat_cand
                best_loss = pat_cand["loss"] / (pat_cand["len"]-1)
            else :
                if pat_cand["len"] > best_candidate["len"] :
                    best_candidate = pat_cand
                    best_loss = pat_cand["loss"] / (pat_cand["len"]-1)
                elif pat_cand["len"] == best_candidate["len"] and pat_cand["loss"] / (pat_cand["len"]-1) < best_loss :
                    best_candidate = pat_cand
                    best_loss = pat_cand["loss"] / (pat_cand["len"]-1)
    
    if best_candidate is None :
        return []

    if color_log is not None :
        peak_ts = [p["timestamp"] / 1000 for p in best_candidate["peaks"]]
        peak_color = [p["from_color"] for p in best_candidate["peaks"]]
        standardized_peaks = []
        for i in range(len(peak_ts)) :
            # align to the cloest log ts before the peak ts
            peak_time = peak_ts[i]
            closest_log = None
            for log in color_log :
                if log["color_name"] != peak_color[i] :
                    continue
                log_time = log["timestamp"]
                if log_time <= peak_time :
                    closest_log = log_time
                else :
                    break
            standardized_peaks.append(closest_log)
        # print(vnp[-2], vnp[-1])
        # print("color", peak_color)
        # print("raw", peak_ts)
        # print("std", standardized_peaks)
        for i in range(len(best_candidate["peaks"])) :
            best_candidate["peaks"][i]["std_ts"] = standardized_peaks[i]
    else :
        # align peaks to standardized pattern
        standardized_peaks = [best_candidate["peaks"][0]["timestamp"] / 1000]
        peak_ts = [p["timestamp"] / 1000 for p in best_candidate["peaks"]]
        prominences = [p["prominence"] for p in best_candidate["peaks"]]
        all_gap = 0
        for p in best_candidate["gap"] :
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
        for i in range(len(best_candidate["peaks"])) :
            best_candidate["peaks"][i]["std_ts"] = aligned_std[i]
        # print(vnp[-2], vnp[-1])
        # print("real interval", round((peak_ts[-1] - peak_ts[0]) / all_gap, 3))
        # print("raw", peak_ts)
        # print("std", standardized_peaks)
        # print("alnw", aligned_std)
        # print("aln", aligned_std_w)
        # print()
    return best_candidate['peaks']
    

def aggregate_transitions(camera_transitions):
    camera_latencies = {}
    for video_name, video_transition in camera_transitions.items():
        vnp = list(Path(video_name).parts)
        color_transition = video_transition["color_transition"]
        camera = video_transition["camera"]
        
        if len(color_transition) == 0:
            print(vnp[-2], vnp[-1], f"WARNING: No color transitions detected!")
            camera_latencies[video_name] = CAMERA_SETTINGS[camera]["default_latency"]
            continue
        
        #  [{'idx': 21, 'from_color': 'Magenta', 'to_color': 'Yellow', 'timestamp': 1767860337834, 'prominence': 0.25771177279874213}, ...]
        latencies = np.array([
            vid_trans["timestamp"] - vid_trans["std_ts"] * 1000
            for vid_trans in color_transition
        ])
        latencies = latencies[latencies > CAMERA_SETTINGS[camera]["min_latency"]]
        latencies = latencies[latencies < CAMERA_SETTINGS[camera]["max_latency"]]
        
        if latencies.size == 0:
            print(vnp[-2], vnp[-1], f"WARNING: No valid latencies after filtering! Using default latency.")
            camera_latencies[video_name] = CAMERA_SETTINGS[camera]["default_latency"]
        else:
            camera_latencies[video_name] = np.mean(latencies)
            print(vnp[-2], vnp[-1], f"Latency: {np.mean(latencies):.2f} (std {np.std(latencies):.2f}) ms over {latencies.size} transitions")
        # print(latencies)
    
    return camera_latencies


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "data_path", 
        help="Path to the file"
    )
    parser.add_argument(
        "--color_log_path", 
        default="", 
        help="Path to the file"
    )
    
    opt = parser.parse_args()
    zed_data_path = os.path.join(opt.data_path, "zed")
    rs_data_path = os.path.join(opt.data_path, "realsense")
    egocap_data_path = os.path.join(opt.data_path, "egocap")
    result_path = os.path.join(opt.data_path, "sync_color_results")
    os.makedirs(result_path, exist_ok=True)
    
    color_readout = {}
    all_timestamps = {}
    
    # ZED data analyze
    if os.path.exists(zed_data_path):
        svo_files = sorted(Path(zed_data_path).rglob("*.svo2"))
        if not svo_files:
            print(f"Warning: No .svo2 files found under: {zed_data_path}")
        else:
            print(f"Found {len(svo_files)} .svo2 files.")
            for svo_path in svo_files:
                print(f"Processing ZED file: {svo_path}")
                color_dict = get_zed_color_frames(str(svo_path), truncate_time=30)
                color_readout[str(svo_path)] = color_dict
    else:
        print(f"Warning: ZED data path does not exist: {zed_data_path}. Skipping ZED processing.")

    # Realsense data analyze
    if os.path.exists(rs_data_path):
        mp4_files = sorted(Path(rs_data_path).rglob("*_rgb.mp4"))
        ts_files = sorted(Path(rs_data_path).rglob("*_ts.csv"))
        if len(mp4_files) == 0:
            print(f"Warning: No Realsense mp4 files found under: {rs_data_path}")
        elif len(mp4_files) != len(ts_files):
            print(f"Warning: Number of Realsense mp4 ({len(mp4_files)}) and ts files ({len(ts_files)}) do not match! Skipping Realsense processing.")
        else:
            for rs_idx in range(len(ts_files)):
                print(f"Processing Realsense file {os.path.basename(mp4_files[rs_idx])}")
                mp4_file = mp4_files[rs_idx]
                ts_file = ts_files[rs_idx]
                color_dict = get_rs_color_frames(str(mp4_file), str(ts_file), truncate_time=30)
                color_readout[str(mp4_file)] = color_dict
    else:
        print(f"Warning: Realsense data path does not exist: {rs_data_path}. Skipping Realsense processing.")
    
    # Egocap data analyze
    if os.path.exists(egocap_data_path):
        egocap_subdirs = [item.path for item in os.scandir(egocap_data_path) if item.is_dir()]
        if len(egocap_subdirs) == 0:
            print(f"Warning: No subdirectories found in Egocap data path: {egocap_data_path}")
        else:
            print("Processing Egocap")
            egocap_data_path = egocap_subdirs[0]
            rgb_path = os.path.join(egocap_data_path, "rgb")
            if os.path.exists(rgb_path):
                color_dict = get_egocap_color_frames(rgb_path, truncate_time=30)
                color_readout[str(egocap_data_path)] = color_dict
            else:
                print(f"Warning: RGB path does not exist: {rgb_path}")
    else:
        print(f"Warning: Egocap data path does not exist: {egocap_data_path}. Skipping Egocap processing.")
    
    # Check if any camera data was processed
    if len(color_readout) == 0:
        print("Error: No camera data was successfully processed. Exiting.")
        sys.exit(1)
    
    # Color log processing
    if not os.path.exists(opt.color_log_path):
        print("Color log file not provided! We can only infer relative synchronization.")
        color_log = None
    elif opt.color_log_path.endswith("json"):
        print("Color log file provided!")
        color_log = json.load(open(opt.color_log_path, 'r'))
        non_overlap = check_color_log_overlap(color_readout, color_log)
        if len(non_overlap) > 0:
            print("Warning: The following cameras have no overlapping colors between readout and log:")
            for camera in non_overlap:
                print(f" - {camera}")
            while True:
                user_input = input("Do you want to: (1) Continue with this log? (2) Continue with no log (3) Exit?\n")
                if user_input == '1':
                    print("Continuing with provided color log, which may not work for some cameras.")
                    break
                elif user_input == '2':
                    print("Continuing with relative synchronization.")
                    color_log = None
                    break
                elif user_input == '3':
                    print("Exiting program.")
                    sys.exit(0)
                else:
                    print("Invalid input. Please enter '1', '2', or '3'.")
        else:
            print("All cameras have overlap with log! We can infer absolute camera latency for real synchronization.")
    else :
        print("Color log directory provided!")
        color_log = []
        log_files = sorted(Path(opt.color_log_path).rglob("*.json"))
        for log_file in log_files :
            log_data = json.load(open(log_file, 'r'))
            color_log.extend(log_data)
        non_overlap = check_color_log_overlap(color_readout, color_log)
        if len(non_overlap) > 0:
            print("Warning: The following cameras have no overlapping colors between readout and log:")
            for camera in non_overlap:
                print(f" - {camera}")
            while True:
                user_input = input("Do you want to: (1) Continue with this log? (2) Continue with no log (3) Exit?\n")
                if user_input == '1':
                    print("Continuing with provided color log, which may not work for some cameras.")
                    break
                elif user_input == '2':
                    print("Continuing with relative synchronization.")
                    color_log = None
                    break
                elif user_input == '3':
                    print("Exiting program.")
                    sys.exit(0)
                else:
                    print("Invalid input. Please enter '1', '2', or '3'.")
    
    all_camera_transition = {}
    for video_name, video_color in color_readout.items():
        if len(video_color) > 0:
            color_transition = find_color_transitions(video_name, video_color, result_path, color_log)
            all_camera_transition[video_name] = {"color_transition": color_transition, "camera": video_color[0]["camera"]}
        else:
            print(f"Warning: No color data for {video_name}. Skipping.")
    
    if len(all_camera_transition) > 0:
        plot_all_color(color_readout, result_path, all_camera_transition)
    else:
        print("Warning: No camera transitions detected. Skipping plotting.")
    
    all_camera_latencies = aggregate_transitions(all_camera_transition)
    
    # Convert relative paths to absolute paths for JSON keys
    all_camera_latencies_sn = {}
    # import ipdb; ipdb.set_trace()
    for video_path, latency in all_camera_latencies.items():
        sn_path = video_path.split('/')[-1].split('\\')[-1].split('.')[0].split('_')[0]
        if 'egocap' in video_path:
            sn_path = 'egocap'
        all_camera_latencies_sn[sn_path] = latency
    
    json.dump(
        all_camera_latencies_sn, 
        open(os.path.join(result_path, "all_camera_latencies.json"), 'w'), 
        indent=4
    )


if __name__ == "__main__":
    main()
