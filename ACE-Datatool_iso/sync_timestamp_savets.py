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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import shutil
# before: python parse_databag.py G:\20260122\000001\egocap
#python sync_timestamp_savets.py --zed_data_path  G:\20260203_1\000004\zed  --rs_data_path G:\20260203_1\000004\realsense --egocap_data_path G:\20260203_1\000004\egocap
# dynamic latency values (in ms) can be set here if needed
ZED_LATENCY = 0 #80
REALSENSE_LATENCY = 0 #110
EGOCAP_LATENCY = 0 #180


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


def get_imu_ts_array(imu_path, camera_type="egocap"):
    """Read IMU timestamps from CSV file"""
    timestamps_list = []
    
    with open(imu_path, 'r') as f:
        # Skip header line
        header = f.readline()
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            
            try:
                if camera_type == "egocap":
                    # Egocap: device_timestamp_ns,host_timestamp_ns,gx,gy,gz,ax,ay,az
                    # Use host timestamp (second column), convert ns to ms
                    timestamp_ns = int(parts[1])
                    timestamps_list.append(timestamp_ns // 1000000)
                elif camera_type == "zed":
                    # ZED: timestamp_ns, ...
                    # Convert to ms
                    timestamp_ns = int(parts[0])
                    timestamps_list.append(timestamp_ns // 1000000)
                elif camera_type == "realsense":
                    # RealSense: timestamp_ms, ...
                    # Already in ms
                    timestamp_ms = int(parts[0])
                    timestamps_list.append(timestamp_ms)
            except (ValueError, IndexError):
                continue
    
    timestamps_array = np.asarray(timestamps_list, dtype=np.int64)
    
    # Apply latency correction
    if camera_type == "egocap":
        timestamps_array -= EGOCAP_LATENCY
    elif camera_type == "realsense":
        timestamps_array -= REALSENSE_LATENCY
    elif camera_type == "zed":
        timestamps_array -= ZED_LATENCY
    
    return timestamps_array


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
    # if no ZED data, use RealSense timestamps instead
    best_loss = float('inf')
    best_bias = 0

    zed_ts_list = [v for k, v in timestamps_dict.items() if k.endswith("svo2")]
    if not zed_ts_list:
        # No ZED data, use RealSense timestamps instead
        zed_ts_list = [v for k, v in timestamps_dict.items() if k.endswith("_ts.csv")]
        print("No ZED data found, using RealSense timestamps as anchor")
    
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
        # Unified sync rate (fps param) for all sensors
        for a in anchor_timestamps :
            ts_index[k].append(int(np.argmin(np.abs(v - a))))

    return ts_index, anchor_timestamps


def get_device_info_from_config(config_data, sn):
    """根据SN号从config中获取设备信息"""
    for device_key, device_info in config_data.get('device', {}).get('ego', {}).items():
        if device_info.get('sn') == sn:
            return {
                'name': device_key,
                'location': device_info.get('location'),
                'cam_type': device_info.get('cam_type'),
                'sn': sn
            }
    return None


def save_sync_video(filepath, ts_index, anchor_timestamps, config_data=None, max_cache_size=5000,base_save_path=None):
    # Calculate base_save_path based on file type
    if filepath.endswith("_ts.csv") or (filepath.endswith("_imu.csv") and "egocap" not in filepath):
        # For RealSense/ZED files: go up 2 levels to get record_name
        camera_dir = os.path.dirname(filepath)
        data_root = os.path.dirname(camera_dir)
    elif filepath.endswith("svo2"):
        # For svo2 files: go up 2 levels to get record_name
        camera_dir = os.path.dirname(filepath)
        data_root = os.path.dirname(camera_dir)
    elif filepath.endswith("imu.csv") and "egocap" in filepath:
        # For egocap imu.csv: go up 3 levels
        egocap_timestamp_dir = os.path.dirname(filepath)
        egocap_dir = os.path.dirname(egocap_timestamp_dir)
        data_root = os.path.dirname(egocap_dir)
    elif filepath.endswith("egocap.mp4"):
        # For egocap video task: go up 3 levels (same as imu.csv)
        egocap_timestamp_dir = os.path.dirname(filepath)
        egocap_dir = os.path.dirname(egocap_timestamp_dir)
        data_root = os.path.dirname(egocap_dir)
    else:
        # Fallback
        data_root = os.path.abspath(filepath + '../../..')
    
    # Get sequence number from config or use folder name
    seq_no = config_data.get('no', os.path.basename(data_root)) if config_data else os.path.basename(data_root)
    if not seq_no:
        seq_no = os.path.basename(data_root)
    if base_save_path is None:
        # Extract SN from config_data
        sn = config_data['device']['ego']['head']['sn'] if config_data else 'default'
        # Get date folder name
        date = os.path.basename(os.path.dirname(data_root))
        # Build path: processed/date/sn/no/
        processed_root = os.path.join(os.path.dirname(os.path.dirname(data_root)), "processed")
        base_save_path = os.path.join(processed_root, date, sn, seq_no)
        os.makedirs(base_save_path, exist_ok=True)
    
    # Use anchor timestamps directly (same for all sensors)
    synced_timestamps = anchor_timestamps
    
    if filepath.endswith("svo2"):
        fps_target = 15
        # Get device info from config
        zed_serial = os.path.splitext(os.path.basename(filepath))[0]
        device_info = get_device_info_from_config(config_data, zed_serial) if config_data else None
        
        if device_info:
            device_name = device_info['name']
            # Create zed/x_one_gs/ folder structure
            zed_folder = os.path.join(base_save_path, "zed", device_info['cam_type'].replace('zed_',''))
            os.makedirs(zed_folder, exist_ok=True)
            video_file = os.path.join(zed_folder, f"{device_name}_{zed_serial}_rgb.mp4")
            ts_file = os.path.join(zed_folder, f"{device_name}_{zed_serial}_ts.csv")
        else:
            # Fallback to old structure
            rgb_folder = os.path.join(base_save_path, "rgb")
            os.makedirs(rgb_folder, exist_ok=True)
            video_file = os.path.join(rgb_folder, f"{zed_serial}_rgb.mp4")
            ts_file = None
        
        # Save timestamp file first
        if ts_file:
            with open(ts_file, 'w') as f:
                for ts in synced_timestamps:
                    f.write(f"{ts}\n")
            print(f"Saved ZED timestamps to: {ts_file}")
        
        # Allocate more resources for ZED video encoding
        threads_for_zed = max(6, os.cpu_count() // 2) if os.cpu_count() else 8
        video_writer = imageio.get_writer(
            video_file,
            fps=fps_target, 
            mode='I', 
            codec='libx264', 
            format='FFMPEG', 
            macro_block_size=1,
            pixelformat='yuv420p',
            output_params=['-threads', str(threads_for_zed), '-preset', 'veryfast', '-crf', '23']
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
        
        # Save synchronized ZED IMU data if exists
        imu_csv_path = os.path.join(os.path.dirname(filepath), f"{zed_serial}_imu.csv")
        if os.path.exists(imu_csv_path):
            if device_info:
                imu_output_file = os.path.join(zed_folder, f"{device_name}_{zed_serial}_imu.csv")
            else:
                imu_folder = os.path.join(base_save_path, "imu")
                os.makedirs(imu_folder, exist_ok=True)
                imu_output_file = os.path.join(imu_folder, f"{zed_serial}_imu_synced.csv")
            
            # Use anchor timestamps for IMU as well
            imu_synced_ts = anchor_timestamps
            
            with open(imu_csv_path, 'r') as f_in, open(imu_output_file, 'w', newline='') as f_out:
                header = f_in.readline()
                f_out.write(header)
                all_lines = f_in.readlines()
                for i, idx in enumerate(ts_index):
                    if idx < len(all_lines):
                        parts = all_lines[idx].strip().split(',')
                        # Replace timestamp (first column) with synced timestamp
                        # Convert back: add latency and convert ms to ns
                        synced_ts_ns = (imu_synced_ts[i] + ZED_LATENCY) * 1000000
                        parts[0] = str(synced_ts_ns)
                        f_out.write(','.join(parts) + '\n')
            print(f"Saved synchronized ZED IMU data to: {imu_output_file}")
            
    elif filepath.endswith("_ts.csv"):
        fps_target = 15
        # Get device info from config
        camera_name = os.path.basename(filepath).split('_ts')[0]
        # import ipdb; ipdb.set_trace()
        device_info = get_device_info_from_config(config_data, camera_name) if config_data else None
        print(f"Device info for {camera_name}: {device_info}")
        assert device_info
        
        # Determine output folder structure based on device info
        if device_info:
            device_name = device_info['name']
            # Create realsense/d435i/ folder structure
            rs_folder = os.path.join(base_save_path, "realsense", device_info["cam_type"].replace('realsense_',''))
            os.makedirs(rs_folder, exist_ok=True)
            ts_file = os.path.join(rs_folder, f"{device_name}_{camera_name}_ts.csv")
        else:
            rs_folder = None
            ts_file = None
        
        # Save timestamp file first
        if ts_file:
            with open(ts_file, 'w') as f:
                for ts in synced_timestamps:
                    f.write(f"{ts}\n")
            print(f"Saved RealSense timestamps to: {ts_file}")
        
        def process_realsense_video(video_type):
            """Process RGB or Depth video in parallel"""
            if device_info:
                video_file = os.path.join(rs_folder, f"{device_name}_{camera_name}_{video_type}.mp4")
            else:
                output_folder = os.path.join(base_save_path, video_type)
                os.makedirs(output_folder, exist_ok=True)
                video_file = os.path.join(output_folder, f"{camera_name}.mp4")
            
            video_writer = imageio.get_writer(
                video_file,
                fps=fps_target, 
                mode='I', 
                codec='libx264', 
                format='FFMPEG', 
                macro_block_size=1,
                pixelformat='yuv420p',
                output_params=['-threads', '4', '-preset', 'faster']  # Use 4 threads per video encoder
            )
            filepath_ = filepath.replace("_ts.csv", f"_{video_type}.mp4")
            video_capture = cv2.VideoCapture(filepath_)
            
            if not video_capture.isOpened():
                raise ValueError(f"Error opening {video_type} video file.")

            frame_count = 0
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            # Read and save the required frames
            with tqdm(total=total_frames, desc=f"Processing {video_type}") as pbar:
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    
                    count = ts_index.count(frame_count)
                    for c in range(count):
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_writer.append_data(frame_rgb)

                    frame_count += 1
                    pbar.update(1)

            video_capture.release()
            video_writer.close()
        
        # Process RGB and Depth in parallel using threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(process_realsense_video, vtype) for vtype in ["rgb", "depth"]]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions
        
        # Save synchronized RealSense IMU data if exists
        imu_csv_path = filepath.replace("_ts.csv", "_imu.csv")
        if os.path.exists(imu_csv_path):
            if device_info:
                imu_output_file = os.path.join(rs_folder, f"{device_name}_{camera_name}_imu.csv")
            else:
                imu_folder = os.path.join(base_save_path, "imu")
                os.makedirs(imu_folder, exist_ok=True)
                imu_output_file = os.path.join(imu_folder, f"{camera_name}_imu_synced.csv")
            
            # Use anchor timestamps for IMU as well
            imu_synced_ts = anchor_timestamps
            
            with open(imu_csv_path, 'r') as f_in, open(imu_output_file, 'w', newline='') as f_out:
                header = f_in.readline()
                f_out.write(header)
                all_lines = f_in.readlines()
                for i, idx in enumerate(ts_index):
                    if idx < len(all_lines):
                        parts = all_lines[idx].strip().split(',')
                        # Replace timestamp (first column) with synced timestamp
                        # Add back latency (already in ms)
                        synced_ts_ms = imu_synced_ts[i] + REALSENSE_LATENCY
                        parts[0] = str(synced_ts_ms)
                        f_out.write(','.join(parts) + '\n')
            print(f"Saved synchronized RealSense IMU data to: {imu_output_file}")
        
        sys.stdout.write("Syncing and saving video complete.\n")
    
    elif filepath.endswith("egocap.mp4"):
        # Save Egocap videos to egocap/x1/ subfolder
        device_type = config_data['device']['ego'][list(config_data['device']['ego'].keys())[0]]['cam_type'].replace('egocap_','')
        egocap_folder = os.path.join(base_save_path, "egocap", device_type) 
        print ('egocap_folder:', egocap_folder)
        os.makedirs(egocap_folder, exist_ok=True)
        
        # Save timestamp file
        ts_file = os.path.join(egocap_folder, "egocap_ts.csv")
        with open(ts_file, 'w') as f:
            for ts in synced_timestamps:
                f.write(f"{ts}\n")
        print(f"Saved Egocap timestamps to: {ts_file}")
        
        fps_target = 15  # Use 15 FPS to match the sync anchor
        
        # Load Egocap configuration
        # Assuming we are in ACE-Datatool-main/, and yaml is in egocap/
        yaml_file = "egocap/calibration2-camchain-4cam.yaml"
        # If running from different CWD, might need adjustment, but assuming consistently run from root
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

        rgb_dir = os.path.join(os.path.dirname(filepath), "rgb")
        if not os.path.exists(rgb_dir):
             print(f"Error: Egocap rgb directory not found at {rgb_dir}")
             return

        rgb_files = [item.path for item in os.scandir(rgb_dir) if item.is_file()]
        rgb_files.sort()
        
        # Pre-compute undistortion maps for stereo cameras
        undistort_maps = {}
        for cam_id in [0, 1]:
            fov = np.deg2rad(90)
            out_size = (512, 512)
            fx = fy = out_size[0] / (2 * np.tan(fov / 2))
            cx = out_size[0] / 2
            cy = out_size[1] / 2
            Kp = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0,  0,  1]], dtype=np.float64)
            
            mapx, mapy = cv2.omnidir.initUndistortRectifyMap(
                cam_params[f"cam_{cam_id}"]['K'],
                cam_params[f"cam_{cam_id}"]['dist'],
                cam_params[f"cam_{cam_id}"]['xi'],
                R_c[cam_id],
                Kp,
                out_size,
                cv2.CV_32F,
                cv2.omnidir.RECTIFY_PERSPECTIVE
            )
            undistort_maps[cam_id] = (mapx, mapy)
        
        # Use thread lock for shared cache
        from threading import Lock
        frame_cache = {}
        cache_lock = Lock()
        # Cache size is passed as parameter and will be adjusted if video length mismatch occurs
        # Default: 5000 frames (~60GB for 2048x2048x3 RGB)
        # Automatically reduced if validation fails
        
        def load_frame(frame_idx):
            """Load frame with limited caching to prevent memory overflow"""
            with cache_lock:
                if frame_idx in frame_cache:
                    return frame_cache[frame_idx]
            
            # Load frame outside of lock
            frame_rgb = cv2.imread(rgb_files[frame_idx])
            if frame_rgb is None:
                return None
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            
            # Update cache with size limit
            with cache_lock:
                if len(frame_cache) >= max_cache_size:
                    # Remove oldest entry (first key)
                    frame_cache.pop(next(iter(frame_cache)))
                frame_cache[frame_idx] = frame_rgb
            
            return frame_rgb
        
        def process_camera_video(cam_name, is_stereo=False, stereo_id=None):
            """Process individual camera video"""
            # Allocate fewer threads per encoder to reduce memory pressure
            threads_per_encoder = 2
            video_writer = imageio.get_writer(
                os.path.join(egocap_folder, f"egocap_{cam_name}.mp4"),
                fps=fps_target,
                mode='I',
                codec='libx264',
                format='FFMPEG',
                macro_block_size=1,
                pixelformat='yuv420p',
                output_params=['-threads', str(threads_per_encoder), '-preset', 'veryfast', '-crf', '23']
            )
            
            # Iterate through ts_index in order to output frames in correct synchronized sequence
            # ts_index contains the frame indices that should be output (may have duplicates for frame repetition)
            
            for idx in tqdm(ts_index, desc=f"Processing {cam_name}", position=int(cam_name[-1]) if cam_name[-1].isdigit() else 0):
                # Load the frame at the specified index
                if idx >= len(rgb_files):
                    print(f"Warning: index {idx} exceeds rgb_files length {len(rgb_files)}, skipping")
                    continue
                    
                frame_rgb = load_frame(idx)
                if frame_rgb is None: 
                    print(f"Warning: failed to load frame at index {idx}")
                    continue

                frame_shape = frame_rgb.shape
                
                if is_stereo:
                    cam_id = stereo_id
                    frame_cam = frame_rgb[cam_id*frame_shape[0]//4: (cam_id+1)*frame_shape[0]//4, :]
                    mapx, mapy = undistort_maps[cam_id]
                    persp = cv2.remap(frame_cam, mapx, mapy, interpolation=cv2.INTER_LINEAR)
                    video_writer.append_data(persp)
                else:
                    cam_id = int(cam_name.replace("cam_", ""))
                    frame_cam = frame_rgb[cam_id*frame_shape[0]//4: (cam_id+1)*frame_shape[0]//4, :]
                    video_writer.append_data(frame_cam)
            
            video_writer.close()
        
        # Process all cameras in parallel using thread pool
        # Use more workers (12) to allow overlap between I/O and encoding operations
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = []
            # Regular cameras
            for cam_id in range(4):
                futures.append(executor.submit(process_camera_video, f"cam_{cam_id}", False))
            # Stereo cameras (only cam_0 and cam_1)
            for stereo_id in [0, 1]:
                futures.append(executor.submit(process_camera_video, f"stereo_{stereo_id}", True, stereo_id))
            
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        sys.stdout.write("Egocap video processing complete.\n")

    elif filepath.endswith("imu.csv") and "egocap" in filepath:
        # Save Egocap IMU to egocap/x1/ subfolder
        egocap_folder = os.path.join(base_save_path, "egocap", "x1")
        os.makedirs(egocap_folder, exist_ok=True)
        
        # Save synchronized Egocap IMU data
        imu_output_file = os.path.join(egocap_folder, "egocap_imu.csv")
        imu_synced_ts = synced_timestamps
        
        with open(filepath, 'r') as f_in, open(imu_output_file, 'w', newline='') as f_out:
            header = f_in.readline()
            f_out.write(header)
            all_lines = f_in.readlines()
            for i, idx in enumerate(ts_index):
                if idx < len(all_lines):
                    parts = all_lines[idx].strip().split(',')
                    # Replace host_timestamp_ns (second column) with synced timestamp
                    # Convert back: add latency and convert ms to ns
                    synced_ts_ns = (imu_synced_ts[i] + EGOCAP_LATENCY) * 1000000
                    parts[1] = str(synced_ts_ns)
                    f_out.write(','.join(parts) + '\n')
        
        print(f"Saved synchronized Egocap IMU data to: {imu_output_file}")


def check_video_lengths(processed_path, expected_frame_count):
    """Check if all output videos have the same length"""
    video_files = []
    for root, dirs, files in os.walk(processed_path):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        return True, "No video files found"
    
    mismatched_videos = []
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if frame_count != expected_frame_count:
            mismatched_videos.append((video_file, frame_count))
    
    if mismatched_videos:
        error_msg = f"Video length mismatch detected! Expected: {expected_frame_count} frames\n"
        for video, count in mismatched_videos:
            error_msg += f"  - {os.path.basename(video)}: {count} frames\n"
        return False, error_msg
    
    return True, f"All {len(video_files)} videos have correct length: {expected_frame_count} frames"


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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="F:\\processed",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--color_log_path",
        type=str,
        default=None,
        help="Directory to save processed data"
    )
    opt = parser.parse_args()

    if not os.path.exists(opt.zed_data_path):
        opt.zed_data_path = None
    if not os.path.exists(opt.rs_data_path):
        opt.rs_data_path = None
    if not os.path.exists(opt.egocap_data_path):    
        opt.egocap_data_path = None
    # import ipdb; ipdb.set_trace()
    all_timestamps = {}
    base_path = os.path.dirname(opt.zed_data_path or opt.rs_data_path or opt.egocap_data_path)
    print ("runing sync_color.py using:", base_path)
    
    date = opt.rs_data_path.split('\\')[1] #os.path.basename(os.path.dirname(base_path))
    # import ipdb; ipdb.set_trace()
    opt.color_log_path =  opt.rs_data_path.replace('\\realsense', f'\\color_sync_log_{date.split("_")[0]}.json')
    if '000' in opt.color_log_path:
        s = opt.color_log_path.split('\\')[-2]
        opt.color_log_path = opt.color_log_path.replace(s+'\\', '')
    print ("color_log_path:", opt.color_log_path)
    if not os.path.exists(opt.color_log_path):
        assert False, f"color_log_path not found: {opt.color_log_path}"
    if os.path.exists(os.path.join(base_path, "sync_color_results/all_camera_latencies.json")):
        print("Camera latency file already exists, skipping sync_color.py")
    else:
        os.system(f"python ./sync_color.py --data_path {base_path} --color_log_path {opt.color_log_path}")
    
    # Load camera latency file if it exists
    latency_file_path = os.path.join(base_path, "sync_color_results/all_camera_latencies.json")
    if os.path.exists(latency_file_path):
        camera_latency = json.load(open(latency_file_path, 'r'))
        print("Camera latencies loaded:", camera_latency)
    else:
        print(f"Warning: Camera latency file not found at {latency_file_path}")
        print("sync_color.py may have failed. Using default latency values (all zeros).")
        camera_latency = {}  # Empty dict, will need to handle this downstream

    # Update and save config.json with folder sequence number
    seq_no = os.path.basename(base_path)  # Use folder name (e.g., 000008) as sequence number
    # F:\\processed\\date\\no\\ to save processed data
    
    original_config_path = os.path.join(base_path, "config.json")
    
    
     # Determine SN from config.json
    if os.path.exists(original_config_path):
        with open(original_config_path, 'r', encoding='utf-8-sig') as f:
            config_data = json.load(f)
            print(f"Loaded config from: {original_config_path}")
        
        sn = config_data['device']['ego']['head']['sn']
    else:
        sn = 'default'
    processed_base = os.path.join(opt.output_dir, date, sn, seq_no)
    os.makedirs(processed_base, exist_ok=True)

    task_timestamps_path = os.path.join(base_path, "task_timestamps.csv")
    if os.path.exists(task_timestamps_path):
        shutil.copyfile(task_timestamps_path, os.path.join(processed_base, "task_timestamps.csv"))
        print(f"Copied task_timestamps.csv to processed folder.")
    
    # Copy egocap metadata.json if exists
    if opt.egocap_data_path:
        egocap_base_dir = Path(opt.egocap_data_path)
        # Find all subdirectories under egocap folder
        for subdir in egocap_base_dir.iterdir():
            if subdir.is_dir():
                metadata_path = subdir / "metadata.json"
                if metadata_path.exists():
                    # Copy to processed egocap folder
                    dest_metadata = os.path.join(processed_base, "egocap",config_data["device"]["ego"]["head"]["cam_type"].replace("egocap_", ""), "metadata.json")
                    os.makedirs(os.path.dirname(dest_metadata), exist_ok=True)
                    shutil.copyfile(str(metadata_path), dest_metadata)
                    print(f"Copied metadata.json from {subdir.name} to processed folder.")
                    break  # Only copy the first metadata.json found
    
    
    if os.path.exists(original_config_path):
        #directly copy config.json to processed/no/
        shutil.copyfile(original_config_path, os.path.join(processed_base, "config.json"))
        
    else:
        # Load config.json from base_path parent directory
        config_path = os.path.join(os.path.dirname(base_path), "config.json")
        config_data = None
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            config_data = json.load(f)
            print(f"Loaded config from: {config_path}")
            print(f"Sequence number: {config_data.get('no', 'N/A')}")
    
        # Update the "no" field with current folder sequence number from template
        config_data['no'] = seq_no
        
        dest_config = os.path.join(processed_base, "config.json")
        # Write updated config to destination
        with open(dest_config, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved config.json with no={seq_no} to: {dest_config}\n")

    
    
    
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
            
            # Also process ZED IMU file if exists
            zed_serial = svo_path.stem  # filename without extension
            imu_path = svo_path.parent / f"{zed_serial}_imu.csv"
            if imu_path.exists():
                print(f"Found ZED IMU file: {imu_path.name}")
                imu_ts_array = get_imu_ts_array(str(imu_path), "zed")
                all_timestamps[str(imu_path)] = imu_ts_array

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
            
            # Also process RealSense IMU file if exists
            imu_path = str(ts_path).replace("_ts.csv", "_imu.csv")
            if os.path.exists(imu_path):
                print(f"Found RealSense IMU file: {os.path.basename(imu_path)}")
                imu_ts_array = get_imu_ts_array(imu_path, "realsense")
                all_timestamps[imu_path] = imu_ts_array
    
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
            # Use get_imu_ts_array for egocap IMU data
            ts_array = get_imu_ts_array(str(ts_path), "egocap")
            all_timestamps[str(ts_path)] = ts_array
            
            # Use get_egocap_ts_array for egocap RGB data (VIDEO sync)
            # Create a pseudo-path to distinguish from IMU
            ts_rgb_path = str(ts_path).replace("imu.csv", "egocap.mp4")
            ts_rgb_array = get_egocap_ts_array(str(ts_path))
            all_timestamps[ts_rgb_path] = ts_rgb_array
    
    # check sizes
    for key, val in all_timestamps.items():
        print(key, val.size, "start:", val[0], "end:", val[-1])
    
    # add camera latency
    # import ipdb; ipdb.set_trace()
    for key, val in all_timestamps.items():
        sn = key.split('/')[-1].split('\\')[-1].split('.')[0].split('_')[0]
        if "egocap" in key:
            sn = "egocap"
        # Use camera latency if available, otherwise default to 0
        latency = camera_latency.get(sn, 0)
        if sn not in camera_latency:
            print(f"Warning: No camera latency found for SN {sn}, using default value 0")
        ts_with_camera_latency = val - latency
        all_timestamps[key] = ts_with_camera_latency.astype(np.int64)
    
    # sync timestamp
    sync_index, anchor_timestamps = sync_timestamps(all_timestamps, fps=15)
    # sync_index_val = sync_index['F:\\20251217\\000002\\realsense\\c: sync_index_val}
    # import ipdb; ipdb.set_trace()
    

    # Auto-retry mechanism with decreasing cache size if video lengths mismatch
    max_cache_size = 6000  # Default cache size
    min_cache_size = 50    # Minimum cache size before giving up
    expected_frames = len(anchor_timestamps)
    retry_count = 0
    max_retries = 5
    
    while retry_count <= max_retries:
        if retry_count > 0:
            print(f"\n{'='*60}")
            print(f"RETRY {retry_count}/{max_retries}: Reducing cache size to {max_cache_size}")
            print(f"{'='*60}\n")
            # Clean up previous attempt
            if os.path.exists(processed_base):
                for root, dirs, files in os.walk(processed_base):
                    for file in files:
                        if file.endswith('.mp4'):
                            os.remove(os.path.join(root, file))
                            print(f"Removed: {file}")
        
        # retrieval and output video (parallel processing)
        print(f"Starting parallel video processing for {len(sync_index)} files...")
        print(f"Cache size: {max_cache_size} frames (~{max_cache_size * 12 / 1024:.1f} GB)")
        # Use more workers than CPU count to maximize utilization (video processing has I/O waits)
        max_workers = min(len(sync_index), os.cpu_count() * 2) if os.cpu_count() else len(sync_index)
        print(f"Using {max_workers} parallel workers (CPU count: {os.cpu_count()})")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks with current cache size
            future_to_filepath = {executor.submit(save_sync_video, filepath, ts_index, anchor_timestamps, config_data, max_cache_size,processed_base): filepath 
                                 for filepath, ts_index in sync_index.items()}
            
            # Process completed tasks
            for future in as_completed(future_to_filepath):
                filepath = future_to_filepath[future]
                try:
                    future.result()
                    print(f"✓ Completed: {os.path.basename(filepath)}")
                except Exception as exc:
                    print(f"✗ Error processing {filepath}: {exc}")
        
        # Validate video lengths
        print(f"\n{'='*60}")
        print("Validating video lengths...")
        print(f"{'='*60}")
        is_valid, message = check_video_lengths(processed_base, expected_frames)
        print(message)
        
        if is_valid:
            print(f"\n✓ SUCCESS: All videos validated successfully!")
            break
        else:
            if max_cache_size <= min_cache_size:
                print(f"\n✗ FAILED: Cache size already at minimum ({min_cache_size}). Cannot retry.")
                print("Please check for other issues (disk space, memory, etc.)")
                break
            
            # Reduce cache size for next attempt
            max_cache_size = max(min_cache_size,max_cache_size // 2)#  max_cache_size-1000)# 
            retry_count += 1
            
            if retry_count > max_retries:
                print(f"\n✗ FAILED: Maximum retries ({max_retries}) reached.")
                break


if __name__ == "__main__":
    main()
