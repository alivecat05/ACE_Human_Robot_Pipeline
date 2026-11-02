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


def progress_bar(percent_done, bar_length=50):
    # Display a progress bar
    done_length = int(bar_length * percent_done / 100)
    bar = "=" * done_length + "-" * (bar_length - done_length)
    sys.stdout.write("[%s] %i%s\r" % (bar, percent_done, "%"))
    sys.stdout.flush()


def convert_one(svo_input_path, avi_output_path, png_output_path, mode=0):
    """
    Convert a single SVO/SVO2 to AVI (mode 0/1) or image sequence.
    Returns the collected timestamps as a numpy array (microseconds).
    """
    # Check if exporting to AVI or SEQUENCE
    output_as_video = True if mode == 0 else False
    start_time = time.time()

    sys.stdout.write("Detected ZED One SVO file. Using CameraOne initialization...\n")
    # Create ZED One camera object
    try:
        init_params = sl.InitParametersOne()
        init_params.set_from_svo_file(svo_input_path)
        init_params.svo_real_time_mode = False
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        zed = sl.CameraOne()
    except AttributeError:
        sys.stdout.write("ZED One classes not available in this SDK version. Trying standard camera...\n")
        is_zed_one = False

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        if err == sl.ERROR_CODE.MODULE_NOT_COMPATIBLE_WITH_CAMERA:
            sys.stdout.write("Warning: MODULE NOT COMPATIBLE WITH CAMERA. Trying alternative approach...\n")
            if not is_zed_one:
                # Try with ZED One classes if not already tried
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
    fps = 15  # Default fps
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

    while True:
        # ZED One grab() doesn't take parameters
        err = zed.grab()

        if err == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(single_image, sl.VIEW.LEFT)

            # Get the timestamp and add it to our list for analysis (microseconds)
            ts_image = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_microseconds()
            timestamps_list.append(ts_image)
            # print(f"svo_position: {svo_position}, ts_image: {int(ts_image / 1000)}")

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
            png_path = os.path.join(os.path.dirname(svo_path), os.path.splitext(os.path.basename(svo_path))[0])
            print(f"\n=== Converting: {svo_path} ===")
            print(f"Output AVI: {avi_path}")
            convert_one(str(svo_path), str(avi_path), str(png_path), mode=opt.mode)
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
    svo_path = svo_path.resolve()
    avi_path = svo_path.with_suffix(".avi")
    png_path = os.path.join(os.path.dirname(svo_path), os.path.splitext(os.path.basename(svo_path))[0])
    convert_one(str(svo_path), str(avi_path), str(png_path), mode=opt.mode)


if __name__ == "__main__":
    main()
