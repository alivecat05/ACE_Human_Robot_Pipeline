# modified from https://github.com/stereolabs/zed-sdk/blob/43e634d5b9801d0fd12634743348f72baab6c4a9/recording/export/svo/python/svo_export.py
# command example: uv run convert.py --input_directory=/nfs/data/sz_45/20250909

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
import csv


def progress_bar(percent_done, bar_length=50):
    # Display a progress bar
    done_length = int(bar_length * percent_done / 100)
    bar = "=" * done_length + "-" * (bar_length - done_length)
    sys.stdout.write("[%s] %i%s\r" % (bar, percent_done, "%"))
    sys.stdout.flush()


def _to_xyz(v):
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
        return float(v.x), float(v.y), float(v.z)
    return float(v[0]), float(v[1]), float(v[2])


def _imu_timestamp_us(imu_data):
    ts = getattr(imu_data, "timestamp", None)
    if ts is None:
        return ""
    if hasattr(ts, "get_microseconds"):
        return int(ts.get_microseconds() * 1000)
    if hasattr(ts, "get_nanoseconds"):
        return int(ts.get_nanoseconds())
    return ""


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

        acc_x, acc_y, acc_z = _to_xyz(linear_acc)
        gyro_x, gyro_y, gyro_z = _to_xyz(angular_vel)
        imu_rows.append(
            [
                _imu_timestamp_us(imu_data),
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


def convert_one(svo_input_path, avi_output_path, png_output_path, imu_output_path, mode=0):
    """
    Convert a single SVO/SVO2 to AVI (mode 0/1) or image sequence.
    Returns the collected timestamps as a numpy array (microseconds).
    """
    # Check if exporting to AVI or SEQUENCE
    output_as_video = True if mode == 0 else False
    start_time = time.time()

    sys.stdout.write("Trying standard ZED camera initialization...\n")
    # Try standard camera first (best compatibility with sensor batch API)
    is_zed_one = False
    try:
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(svo_input_path)
        init_params.svo_real_time_mode = False
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        zed = sl.Camera()
    except AttributeError:
        sys.stdout.write("Standard camera classes not available. Trying ZED One camera...\n")
        init_params = sl.InitParametersOne()
        init_params.set_from_svo_file(svo_input_path)
        init_params.svo_real_time_mode = False
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        zed = sl.CameraOne()
        is_zed_one = True

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        if err == sl.ERROR_CODE.MODULE_NOT_COMPATIBLE_WITH_CAMERA:
            sys.stdout.write("Warning: MODULE NOT COMPATIBLE WITH CAMERA. Trying alternative approach...\n")
            if not is_zed_one:
                # Try with ZED One classes if standard camera failed
                try:
                    zed.close()
                    sys.stdout.write("Trying with ZED One classes...\n")
                    init_params = sl.InitParametersOne()
                    init_params.set_from_svo_file(svo_input_path)
                    init_params.svo_real_time_mode = False
                    init_params.coordinate_units = sl.UNIT.MILLIMETER
                    zed = sl.CameraOne()
                    err = zed.open(init_params)
                    if err == sl.ERROR_CODE.SUCCESS:
                        sys.stdout.write("Successfully opened with ZED One classes.\n")
                        is_zed_one = True
                    else:
                        sys.stdout.write(f"ZED One classes also failed: {repr(err)}\n")
                except AttributeError:
                    sys.stdout.write("ZED One classes not available in this SDK version.\n")

            if err != sl.ERROR_CODE.SUCCESS:
                sys.stdout.write(f"Error: Could not open SVO file: {repr(err)}\n")
                sys.stdout.write("This SVO file may require a different ZED SDK version or camera driver.\n")
                zed.close()
                return
        else:
            sys.stdout.write(repr(err))
            zed.close()
            return

    # ZED One cameras - use default values to avoid API issues
    # Most ZED One cameras use standard resolutions
    width = 1920  # Default width for ZED One
    height = 1080  # Default height for ZED One
    fps = 30  # Default fps
    sys.stdout.write(f"Using default ZED One parameters: {width}x{height}, fps: {fps}\n")
    print(f"image_size: {width}x{height}, fps: {fps}")

    # Prepare image container equivalent to CV_8UC4
    svo_image_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    # Prepare single image containers
    single_image = sl.Mat()

    video_writer = None
    if output_as_video:
        video_writer = imageio.get_writer(
            avi_output_path, 
            fps=max(fps, 15), 
            mode='I', 
            codec='libx264', 
            format='FFMPEG', 
            macro_block_size=1
        )
    else :
        os.makedirs(png_output_path, exist_ok=True)

    rt_param = sl.RuntimeParameters()

    # Start SVO conversion to AVI/SEQUENCE
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

    nb_frames = zed.get_svo_number_of_frames()

    # List to store timestamps for analysis
    timestamps_list = []
    imu_rows = []
    sensors_data = sl.SensorsData()

    while True:
        # Standard camera uses runtime params; ZED One grab() may not.
        err = zed.grab() if is_zed_one else zed.grab(rt_param)

        if err == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(single_image, sl.VIEW.LEFT)

            # Get the timestamp and add it to our list for analysis (microseconds)
            ts_image = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_microseconds()
            timestamps_list.append(ts_image)
            # print(f"svo_position: {svo_position}, ts_image: {int(ts_image / 1000)}")

            # Prefer batch retrieval to collect all IMU packets between two grabs.
            got_batch_imu = False
            try:
                all_sensors_data = []
                batch_err = zed.get_sensors_data_batch(all_sensors_data)
                if batch_err == sl.ERROR_CODE.SUCCESS or batch_err is None:
                    for sensor_packet in all_sensors_data:
                        imu_data = None
                        if hasattr(sensor_packet, "imu"):
                            imu_candidate = sensor_packet.imu
                            if getattr(imu_candidate, "is_available", True):
                                imu_data = imu_candidate
                        elif hasattr(sensor_packet, "get_imu_data"):
                            imu_data = sensor_packet.get_imu_data()

                        if imu_data is not None:
                            _append_imu_row(imu_rows, imu_data)
                            got_batch_imu = True
            except Exception:
                pass

            # Fallback for SDK variants without get_sensors_data_batch.
            if not got_batch_imu:
                try:
                    sensors_err = zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
                    if sensors_err == sl.ERROR_CODE.SUCCESS:
                        imu_data = sensors_data.get_imu_data()
                        _append_imu_row(imu_rows, imu_data)
                except Exception:
                    pass

            if output_as_video:
                # # Copy the left image to the left side of SBS image
                # svo_image_rgba[0:height, :, :] = single_image.get_data()
                # # Convert SVO image from RGBA to RGB
                # svo_image_rgba = cv2.cvtColor(svo_image_rgba, cv2.COLOR_RGBA2RGB)
                # Write the RGB image in the video
                svo_image_rgb = cv2.cvtColor(np.asanyarray(single_image.get_data()), cv2.COLOR_BGR2RGB)
                video_writer.append_data(svo_image_rgb)
            else:
                # Generate file names
                filename = os.path.join(png_output_path, f"{str(ts_image).zfill(6)}.png")
                # Save images
                cv2.imwrite(str(filename), single_image.get_data())

            # Display progress
            progress_bar((svo_position + 1) / nb_frames * 100, 30)

        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            progress_bar(100, 30)
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            break

    if output_as_video:
        # Close the video writer
        video_writer.close()

    with open(imu_output_path, "w", newline="") as f:
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
    print(f"IMU CSV saved: {imu_output_path} ({len(imu_rows)} rows)")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    zed.close()


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help=" Mode 0 is to export AVI. \n Mode 1 is to export image sequence.",
    )
    parser.add_argument(
        "--input_svo_file", 
        type=str, 
        default="", 
        help="Path to the .svo/.svo2 file"
    )
    parser.add_argument(
        "--input_directory",
        type=str,
        default="",
        help="If provided, batch convert all .svo2 files (recursively) in this directory to .avi and save timestamps to .npy",
    )
    opt = parser.parse_args()

    # export modefile format
    if opt.mode > 1 or opt.mode < 0:
        print(
            "Mode shoud be 0 or 1. \n Mode 0 is to export AVI. \n Mode 1 is to export image sequence."
        )
        sys.exit(1)
    
    # Batch mode: input_directory takes precedence
    if opt.input_directory:
        base_dir = Path(opt.input_directory)
        if not base_dir.is_dir():
            print(f"--input_directory must be an existing directory: {opt.input_directory}")
            sys.exit(1)

        svo_files = sorted(base_dir.rglob("*.svo2"))
        if not svo_files:
            print(f"No .svo2 files found under: {opt.input_directory}")
            sys.exit(0)

        print(f"Found {len(svo_files)} .svo2 files. Converting with mode={opt.mode}...")
        for svo_path in svo_files:
            svo_path = svo_path.resolve()
            avi_path = svo_path.with_suffix(".mp4")
            imu_path = svo_path.with_name(f"{svo_path.stem}_imu.csv")
            png_path = os.path.join(os.path.dirname(svo_path), os.path.splitext(os.path.basename(svo_path))[0])
            print(f"\n=== Converting: {svo_path} ===")
            print(f"Output AVI: {avi_path}")
            print(f"Output IMU CSV: {imu_path}")
            convert_one(str(svo_path), str(avi_path), str(png_path), str(imu_path), mode=opt.mode)
        return

    # Single-file mode (backward compatible)
    if not opt.input_svo_file.endswith((".svo", ".svo2")):
        print(
            "--input_svo_file parameter should be a .svo/.svo2 file but is not :", opt.input_svo_file, "Exit program."
        )
        sys.exit(1)
    if not os.path.isfile(opt.input_svo_file):
        print(
            "--input_svo_file parameter should be an existing file but is not :", opt.input_svo_file, "Exit program."
        )
        sys.exit(1)
    svo_path = Path(opt.input_svo_file).resolve()
    avi_path = svo_path.with_suffix(".avi")
    imu_path = svo_path.with_name(f"{svo_path.stem}_imu.csv")
    png_path = os.path.join(os.path.dirname(svo_path), os.path.splitext(os.path.basename(svo_path))[0])
    convert_one(str(svo_path), str(avi_path), str(png_path), str(imu_path), mode=opt.mode)


if __name__ == "__main__":
    main()
