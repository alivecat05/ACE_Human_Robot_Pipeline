import sys
import pyzed.sl as sl
import numpy as np
import cv2
from pathlib import Path
import enum
import imageio
import argparse
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

ZED_LATENCY = 80
REALSENSE_LATENCY = 110
EGOCAP_LATENCY = 180


def progress_bar(percent_done, bar_length=50):
    # Display a progress bar
    done_length = int(bar_length * percent_done / 100)
    bar = "=" * done_length + "-" * (bar_length - done_length)
    sys.stdout.write("[%s] %i%s\r" % (bar, percent_done, "%"))
    sys.stdout.flush()


def get_zed_ts_array(svo_path):
    # Create ZED One camera object
    init_params = sl.InitParametersOne()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    zed = sl.CameraOne()
    err = zed.open(init_params)

    nb_frames = zed.get_svo_number_of_frames()
    timestamps_list = []

    while True:
        err = zed.grab()
        if err == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Get the timestamp and add it to our list for analysis (microseconds)
            ts_image = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_microseconds()
            timestamps_list.append(ts_image)
            
            # Display progress
            progress_bar((svo_position + 1) / nb_frames * 100, 30)

        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            progress_bar(100, 30)
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            break
    
    zed.close()

    timestamps_ms_array = np.asarray(timestamps_list, dtype=np.int64) // 1000 - ZED_LATENCY
    return timestamps_ms_array


def get_rs_ts_array(ts_path):
    timestamps_list = []
    with open(ts_path, 'r') as f :
        for line in f :
            timestamps_list.append(int(line.strip()))

    return np.asarray(timestamps_list, dtype=np.int64) - REALSENSE_LATENCY


def get_egocap_ts_array(egocap_path):
    timestamps_list = []

    rgb_dir = os.path.join(os.path.dirname(egocap_path), "rgb")
    rgb_files = [item.path for item in os.scandir(rgb_dir) if item.is_file()]
    rgb_files.sort()

    for rgb_file in rgb_files:
        timestamps_list.append(int(rgb_file.split("_")[-1].split(".")[0]))

    return np.asarray(timestamps_list, dtype=np.int64) // 1000000 - EGOCAP_LATENCY


def sync_timestamps(timestamps_dict, fps=15):
    ts_index = {}
    for key, val in timestamps_dict.items():
        ts_index[key] = []
    
    # common recording window, drop the first frame
    interval_ms = int(1000.0 / fps)
    min_common = max(np.min(v[1:]) for k, v in timestamps_dict.items()) - interval_ms // 4
    max_common = min(np.max(v[1:]) for k, v in timestamps_dict.items()) + interval_ms // 4
    anchor_timestamps = np.arange(min_common, max_common, 1000.0 / fps, dtype=np.float64).astype(np.int64)

    # use ZED timestamps to build anchor as they don't drop frames now
    best_loss = float('inf')
    best_bias = 0

    zed_ts_list = [v for k, v in timestamps_dict.items() if k.endswith("svo2")]
    for bias in range(interval_ms) :
        total_loss = 0.0
        adjust_timestamps = anchor_timestamps + bias
        for ts in adjust_timestamps:
            for zed_ts in zed_ts_list :
                total_loss += np.min(np.abs(zed_ts - ts))
        if total_loss < best_loss :
            best_loss = total_loss
            best_bias = bias
    anchor_timestamps += best_bias
    print("anchor timestamps: ", anchor_timestamps)

    # use anchor to retrieve from all timestamps and build index
    for k, v in timestamps_dict.items():
        if "csv" in k :
            anchor_timestamps_ego = np.arange(min_common+best_bias, max_common+best_bias, 1000.0 / 20, dtype=np.int64)
            for a in anchor_timestamps_ego :
                ts_index[k].append(int(np.argmin(np.abs(v - a))))
        else :
            for a in anchor_timestamps :
                ts_index[k].append(int(np.argmin(np.abs(v - a))))

    return ts_index


def save_sync_video(filepath, ts_index):
    data_root = os.path.abspath(filepath + '../../..')
    save_path = os.path.join(os.path.dirname(data_root), "processed", os.path.basename(data_root))
    os.makedirs(save_path, exist_ok=True)
    
    if filepath.endswith("svo2"):
        fps_ = 15
        video_file = os.path.join(save_path, f"{os.path.splitext(os.path.basename(filepath))[0]}_rgb.mp4")
        video_writer = imageio.get_writer(
            video_file,
            fps=fps_, 
            mode='I', 
            codec='libx264', 
            format='FFMPEG', 
            macro_block_size=1
        )
        init_params = sl.InitParametersOne()
        init_params.set_from_svo_file(filepath)
        init_params.svo_real_time_mode = False
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        zed = sl.CameraOne()
        err = zed.open(init_params)

        single_image = sl.Mat()
        frame_count = 0
        nb_frames = zed.get_svo_number_of_frames()
        
        while True:
            err = zed.grab()
            if err == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(single_image, sl.VIEW.LEFT)

                if frame_count in ts_index :
                    count = ts_index.count(frame_count)
                    svo_image_rgb = cv2.cvtColor(np.asanyarray(single_image.get_data()), cv2.COLOR_BGR2RGB)
                    for c in range(count):
                        video_writer.append_data(svo_image_rgb)
                
                progress_bar((frame_count + 1) / nb_frames * 100, 30)
                frame_count += 1

            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                progress_bar(100, 30)
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break
        video_writer.close()
        zed.close()
    elif filepath.endswith("_ts.csv"):
        fps_ = 15
        video_file = os.path.join(save_path, f"{os.path.basename(filepath).split('_ts')[0]}.mp4")
        video_writer = imageio.get_writer(
            video_file.replace(".mp4", "_rgb.mp4"),
            fps=fps_, 
            mode='I', 
            codec='libx264', 
            format='FFMPEG', 
            macro_block_size=1
        )
        filepath_ = filepath.replace("_ts.csv", "_rgb.mp4")
        video_capture = cv2.VideoCapture(filepath_)
        
        if not video_capture.isOpened():
            raise ValueError("Error opening video file.")

        # Get video properties
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read and save the required frames
        with tqdm(total=total_frames, desc="Processing Video") as pbar:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break  # End of video
                
                count = ts_index.count(frame_count)
                # If the current frame index is in ts_index, write the frame to the new video
                for c in range(count):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_writer.append_data(frame_rgb)

                frame_count += 1
                pbar.update(1)

        # Release the video capture object
        video_capture.release()
        video_writer.close()
        sys.stdout.write("Syncing and saving video complete.\n")


        # DEPTH
        video_writer = imageio.get_writer(
            video_file.replace(".mp4", "_depth.mp4"),
            fps=fps_, 
            mode='I', 
            codec='libx264', 
            format='FFMPEG', 
            macro_block_size=1
        )
        filepath_ = filepath.replace("_ts.csv", "_depth.mp4")
        video_capture = cv2.VideoCapture(filepath_)
        
        if not video_capture.isOpened():
            raise ValueError("Error opening video file.")

        # Get video properties
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read and save the required frames
        with tqdm(total=total_frames, desc="Processing Video") as pbar:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break  # End of video
                
                count = ts_index.count(frame_count)
                # If the current frame index is in ts_index, write the frame to the new video
                for c in range(count):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_writer.append_data(frame_rgb)

                frame_count += 1
                pbar.update(1)

        # Release the video capture object
        video_capture.release()
        video_writer.close()
        sys.stdout.write("Syncing and saving video complete.\n")
    
    elif filepath.endswith("csv") :
        yaml_file = "egocap/calibration2-camchain-4cam.yaml"
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

        fps_ = 20
        video_writer = {}
        for c in ["cam0", "cam1", "cam2", "cam3", "stereo0", "stereo1"] :
            video_writer[c] = imageio.get_writer(
                os.path.join(save_path, f"{c}.mp4"),
                fps=fps_, 
                mode='I', 
                codec='libx264', 
                format='FFMPEG', 
                macro_block_size=1
            )
        rgb_dir = os.path.join(os.path.dirname(filepath), "rgb")
        rgb_files = [item.path for item in os.scandir(rgb_dir) if item.is_file()]
        rgb_files.sort()

        # Read and save the required frames
        for i in tqdm(range(len(rgb_files))) :
            count = ts_index.count(i)
            # If the current frame index is in ts_index, write the frame to the new video
            for c in range(count):
                frame_rgb = cv2.imread(rgb_files[i])
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
                frame_shape = frame_rgb.shape
                for cam_id in range(4) :
                    frame_cam = frame_rgb[cam_id*frame_shape[0]//4: (cam_id+1)*frame_shape[0]//4, :]
                    video_writer[f"cam{cam_id}"].append_data(frame_cam)

                    fov = np.deg2rad(90)
                    out_size = (512, 512)
                    fx = fy = out_size[0] / (2 * np.tan(fov / 2))
                    cx = out_size[0] / 2
                    cy = out_size[1] / 2
                    Kp = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0,  0,  1]], dtype=np.float64)

                    if cam_id == 0 or cam_id == 1 :
                        fov = np.deg2rad(90)
                        out_size = (512, 512)
                        fx = fy = out_size[0] / (2 * np.tan(fov / 2))
                        cx = out_size[0] / 2
                        cy = out_size[1] / 2
                        Kp = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]], dtype=np.float64)

                        # mapping matrix
                        mapx, mapy = cv2.omnidir.initUndistortRectifyMap(
                            cam_params[f"cam{cam_id}"]['K'],
                            cam_params[f"cam{cam_id}"]['dist'],
                            cam_params[f"cam{cam_id}"]['xi'],
                            R_c[cam_id],
                            Kp,
                            out_size,
                            cv2.CV_32F,
                            cv2.omnidir.RECTIFY_PERSPECTIVE
                        )
                        persp = cv2.remap(frame_cam, mapx, mapy, interpolation=cv2.INTER_LINEAR)
                        video_writer[f"stereo{cam_id}"].append_data(persp)

        # Release the video capture object
        for c in ["cam0", "cam1", "cam2", "cam3", "stereo0", "stereo1"] :
            video_writer[c].close()
        sys.stdout.write("Syncing and saving video complete.\n")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--zed_data_path", 
        type=str, 
        default="", 
        help="Path to the .svo/.svo2 file"
    )
    parser.add_argument(
        "--rs_data_path",
        type=str,
        default="",
        help="Path to the ts.csv file"
    )
    parser.add_argument(
        "--egocap_data_path",
        type=str,
        default="",
        help="Path to the rgb file"
    )
    opt = parser.parse_args()

    all_timestamps = {}
    
    # ZED data analyze
    if opt.zed_data_path:
        base_dir = Path(opt.zed_data_path)
        if not base_dir.is_dir():
            print(f"--zed_data_path must be an existing directory: {opt.zed_data_path}")
            sys.exit(1)
        svo_files = sorted(base_dir.rglob("*.svo2"))
        if not svo_files:
            print(f"No .svo2 files found under: {opt.zed_data_path}")
            sys.exit(0)

        print(f"Found {len(svo_files)} .svo2 files.")
        for svo_path in svo_files:
            svo_path = svo_path.resolve()
            ts_array = get_zed_ts_array(str(svo_path))
            all_timestamps[str(svo_path)] = ts_array

    # Realsense data analyze
    if opt.rs_data_path:
        base_dir = Path(opt.rs_data_path)
        if not base_dir.is_dir():
            print(f"--rs_data_path must be an existing directory: {opt.rs_data_path}")
            sys.exit(1)
        ts_files = sorted(base_dir.rglob("*_ts.csv"))
        if not ts_files:
            print(f"No _ts.csv files found under: {opt.rs_data_path}")
            sys.exit(0)

        print(f"Found {len(ts_files)} _ts.csv files.")
        for ts_path in ts_files:
            ts_path = ts_path.resolve()
            ts_array = get_rs_ts_array(str(ts_path))
            all_timestamps[str(ts_path)] = ts_array
    
    # Egocap data analyze
    if opt.egocap_data_path:
        base_dir = Path(opt.egocap_data_path)
        if not base_dir.is_dir():
            print(f"--egocap_data_path must be an existing directory: {opt.egocap_data_path}")
            sys.exit(1)
        ts_files = sorted(base_dir.rglob("imu.csv"))
        if not ts_files:
            print(f"No .csv files found under: {opt.egocap_data_path}")
            sys.exit(0)

        print(f"Found {len(ts_files)} .csv files.")
        for ts_path in ts_files:
            ts_path = ts_path.resolve()
            ts_array = get_egocap_ts_array(str(ts_path))
            all_timestamps[str(ts_path)] = ts_array
    
    # check sizes
    for key, val in all_timestamps.items():
        print(key, val.size, val)
    
    # sync timestamp
    sync_index = sync_timestamps(all_timestamps, fps=15)

    # retrieval and output video
    for filepath, ts_index in sync_index.items():
        print(filepath, len(ts_index))
        save_sync_video(filepath, ts_index)


if __name__ == "__main__":
    main()
