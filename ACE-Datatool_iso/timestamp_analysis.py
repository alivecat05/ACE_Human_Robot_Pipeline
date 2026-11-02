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


def progress_bar(percent_done, bar_length=50):
    # Display a progress bar
    done_length = int(bar_length * percent_done / 100)
    bar = "=" * done_length + "-" * (bar_length - done_length)
    sys.stdout.write("[%s] %i%s\r" % (bar, percent_done, "%"))
    sys.stdout.flush()


def calculate_average_fps(timestamps_us):
    """
    Calculates and prints the average frame rate from a list of timestamps.
    """

    # Convert timestamps to a numpy array for vectorized operations
    timestamps = timestamps_us[20:]

    # Convert microseconds → seconds
    time_start_sec = timestamps[0] / 1000
    time_end_sec   = timestamps[-1] / 1000

    # Duration in seconds
    duration_sec = time_end_sec - time_start_sec

    real_avg_fps = (timestamps.size - 20) / duration_sec
    print(f"Real Average FPS = {real_avg_fps:.2f}")

    # Calculate the difference between consecutive timestamps in milliseconds
    # Timestamps are in microseconds, so divide by 1,000 to get ms
    time_diffs_ms = np.diff(timestamps)

    # Calculate the average time difference
    avg_diff_ms = np.mean(time_diffs_ms)

    # Calculate FPS (1000 ms / average time difference per frame)
    if avg_diff_ms > 0:
        avg_fps = 1000.0 / avg_diff_ms
        print(f"Calculated Average FPS = {avg_fps:.2f}")
    else:
        print("Could not calculate average FPS (average time difference is zero).")
    
    return real_avg_fps


def plot_timestamp_differences(timestamps_array, output_path, video_filename=None):
    """
    Generates and saves a plot of the time differences between frames.
    """
    time_diffs_ms = np.diff(timestamps_array)

    # Frame indices for the plot (starts from frame 1)
    indices = np.arange(1, len(time_diffs_ms) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(indices, time_diffs_ms, marker="o", linestyle="-", markersize=4)
    plt.title("Timestamp Difference Between Consecutive Frames")
    plt.xlabel("Frame Index")
    plt.ylabel("Time Difference (ms)")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to the specified output directory
    os.makedirs(output_path, exist_ok=True)

    # Generate unique filename based on video name
    if video_filename:
        # Extract the base name without extension
        base_name = Path(video_filename).stem
        plot_filename = os.path.join(output_path, f"{base_name}.png")
    else:
        plot_filename = os.path.join(output_path, "timestamp_differences.png")

    plt.savefig(plot_filename)
    print(f"Saved timestamp difference plot to: {plot_filename}", flush=True)
    plt.close()


def analyze_zed(svo_path, save_path, save_name):
    # Create ZED One camera object
    init_params = sl.InitParametersOne()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    zed = sl.CameraOne()
    err = zed.open(init_params)

    zed_serial = 0
    if err == sl.ERROR_CODE.SUCCESS:
        try:
            zed_serial = zed.get_camera_information().serial_number
        except:
            pass

    nb_frames = zed.get_svo_number_of_frames()
    timestamps_list = []

    while True:
        err = zed.grab()
        if err == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Get the timestamp and add it to our list for analysis (microseconds)
            ts_image = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_microseconds()
            timestamps_list.append(ts_image)
            if ((svo_position + 1) % (nb_frames//5) == 0):
                # Display progress
                progress_bar((svo_position + 1) / nb_frames * 100, 30)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            progress_bar(100, 30)
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            break
    
    zed.close()

    # --- Analysis Section ---
    print("\n--- Generating Timestamp Difference Plot ---")
    if len(timestamps_list) < 2:
        print("Not enough timestamps to generate a plot.")
        return

    timestamps_us = np.asarray(timestamps_list, dtype=np.int64) / 1000
    analysis_output_path = os.path.join(save_path, "timestamp_analysis")

    # FPS and difference analyse
    real_fps = calculate_average_fps(timestamps_us)
    
    # Get recording name from parent directory of save_path (e.g. raw/000001/zed -> 000001)
    try:
        recording_name = Path(save_path).parent.name
    except:
        recording_name = save_name

    # Try to extract serial from filename if possible for better identification
    # ZED filename usually is the name provided, not detailed.
    
    camera_label = f"ZED ({zed_serial})" if zed_serial else "ZED"
    print(f"FPS_SUMMARY: {recording_name}, {camera_label}, {real_fps:.2f}", flush=True)
    
    print(f"timestamps_list length: {len(timestamps_list)}", flush=True)
    plot_timestamp_differences(timestamps_us, analysis_output_path, save_name)


def analyze_rs(ts_path, save_path, save_name):
    
    timestamps_list = []
    
    with open(ts_path, 'r') as f :
        for line in f :
            timestamps_list.append(int(line.strip()))

    # --- Analysis Section ---
    print("\n--- Generating Timestamp Difference Plot ---")
    if len(timestamps_list) < 2:
        print("Not enough timestamps to generate a plot.")
        return

    timestamps_us = np.asarray(timestamps_list, dtype=np.int64)
    analysis_output_path = os.path.join(save_path, "timestamp_analysis")

    # FPS and difference analyse
    real_fps = calculate_average_fps(timestamps_us)

    # Get recording name from parent directory
    try:
        recording_name = Path(save_path).parent.name
    except:
        recording_name = save_name

    # Extract serial number from filename if possible (e.g. ...841512070549_ts)
    # This helps distinguish multiple RealSense cameras
    camera_label = "RealSense"
    parts = save_name.split('_')
    for p in parts:
        if p.isdigit() and len(p) > 10: # Simple heuristic for serial number
             camera_label = f"RealSense ({p})"
             break

    print(f"FPS_SUMMARY: {recording_name}, {camera_label}, {real_fps:.2f}", flush=True)
    print(f"timestamps_list length: {len(timestamps_list)}", flush=True)
    plot_timestamp_differences(timestamps_us, analysis_output_path, save_name)


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
        help="Path to the _ts.csv file"
    )
    opt = parser.parse_args()
    
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
            save_path = os.path.dirname(svo_path)
            save_name = os.path.splitext(os.path.basename(svo_path))[0]
            analyze_zed(str(svo_path), save_path, save_name)
            sys.stdout.flush()

    # Realsense data analyze
    if opt.rs_data_path:
        base_dir = Path(opt.rs_data_path)
        if not base_dir.is_dir():
            print(f"--rs_data_path must be an existing directory: {opt.rs_data_path}")
            sys.exit(1)
        ts_files = sorted(base_dir.rglob("*_ts.csv"))
        if not ts_files:
            print(f"No .txt files found under: {opt.rs_data_path}")
            sys.exit(0)

        print(f"Found {len(ts_files)} ts.csv files.")
        for ts_path in ts_files:
            ts_path = ts_path.resolve()
            save_path = os.path.dirname(ts_path)
            save_name = os.path.splitext(os.path.basename(ts_path))[0]
            analyze_rs(ts_path, save_path, save_name)


if __name__ == "__main__":
    main()