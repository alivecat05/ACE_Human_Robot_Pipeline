import sys
import pyzed.sl as sl
import numpy as np
import cv2
from pathlib import Path
import enum
import imageio
import json
import argparse
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


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
    
    yaml_file = "./calibration2-camchain-4cam.yaml"
    assert os.path.exists(yaml_file), "Egocap Config not found!"
    with open(yaml_file, 'r') as f:
        yaml_ = yaml.safe_load(f)
    
    cam_params = {}
    for c in ["cam0", "cam1", "cam2", "cam3"] :
        cam_params[c] = {}
        xi, fx, fy, cx, cy = yaml_[c]["intrinsics"]
        cam_params[c]["K"] = np.array(
            [[fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]], dtype=np.float64)
        cam_params[c]["xi"] = np.array([xi], dtype=np.float64)
        cam_params[c]["dist"] = np.array(yaml_[c]["distortion_coeffs"], dtype=np.float64)
    
    cam1_out = np.array(yaml_["cam1"]["T_cn_cnm1"])
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

    fov = np.deg2rad(60)
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
        cam_params[f"cam{target_cam_id}"]['K'],
        cam_params[f"cam{target_cam_id}"]['dist'],
        cam_params[f"cam{target_cam_id}"]['xi'],
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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "data_path", 
        help="Path to the file"
    )
    opt = parser.parse_args()
    
    zed_data_path = os.path.join(opt.data_path, "zed")
    rs_data_path = os.path.join(opt.data_path, "realsense")
    egocap_data_path = os.path.join(opt.data_path, "egocap")
    result_path = os.path.join(opt.data_path, "color_readout")
    os.makedirs(result_path, exist_ok=True)

    all_colors = {}
    
    # ZED data analyze
    svo_files = sorted(Path(zed_data_path).rglob("*.svo2"))
    if not svo_files:
        print(f"No .svo2 files found under: {zed_data_path}")
        sys.exit(0)
    print(f"Found {len(svo_files)} .svo2 files.")
    for svo_path in svo_files:
        print(f"Processing ZED file: {svo_path}")
        svo_path = svo_path.resolve()
        color_dict = get_zed_color_frames(str(svo_path), truncate_time=30)
        all_colors[str(svo_path)] = color_dict

    # Realsense data analyze
    mp4_files = sorted(Path(rs_data_path).rglob("*_rgb.mp4"))
    ts_files = sorted(Path(rs_data_path).rglob("*_ts.csv"))
    assert len(mp4_files) == len(ts_files), "Number of Realsense mp4 and ts files do not match!"
    assert len(mp4_files) > 0, "No Realsense mp4 files found!"
    for rs_idx in range(len(ts_files)):
        print(f"Processing Realsense file {os.path.basename(mp4_files[rs_idx])}")
        mp4_file = mp4_files[rs_idx].resolve()
        ts_file = ts_files[rs_idx].resolve()
        color_dict = get_rs_color_frames(str(mp4_file), str(ts_file), truncate_time=30)
        all_colors[str(mp4_file)] = color_dict
    
    # Egocap data analyze
    print("Processing Egocap")
    egocap_data_path = [item.path for item in os.scandir(egocap_data_path) if item.is_dir()][0]
    rgb_path = os.path.join(egocap_data_path, "rgb")
    color_dict = get_egocap_color_frames(rgb_path, truncate_time=30)
    all_colors[str(egocap_data_path)] = color_dict
    
    json.dump(
        all_colors, 
        open(os.path.join(result_path, "color_readout_all.json"), 'w'), 
        indent=4
    )


if __name__ == "__main__":
    main()
