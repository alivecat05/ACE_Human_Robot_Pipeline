import os
import cv2
from pathlib import Path
import argparse
import sys
import numpy as np
from pyzbar.pyzbar import decode

def readout(mp4_path, ts_path):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        print(f"Error opening video file: {mp4_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detected_count = 0
    print(f"Total frames in video: {frame_count}")

    with open(ts_path, 'w') as ts_file:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([30, 100, 60])
            upper_green = np.array([90, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            result = cv2.bitwise_not(mask)
            result = decode(result)

            if len(result) > 0:
                ts_file.write(f"{frame_idx}: {result[0].data.decode('utf-8')}\n")
                detected_count += 1

            frame_idx += 1

    cap.release()
    print(f"Timestamps written to: {ts_path}, total {detected_count} frames detected.")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--video_file", 
        type=str, 
        default="", 
        help="Path to the .mp4 file"
    )
    parser.add_argument(
        "--video_directory",
        type=str,
        default="",
        help="Path to the .mp4 directory"
    )
    opt = parser.parse_args()

    # Batch mode: input_directory takes precedence
    if opt.video_directory:
        base_dir = Path(opt.video_directory)
        if not base_dir.is_dir():
            print(f"--video_directory must be an existing directory: {opt.video_directory}")
            sys.exit(1)

        mp4_files = sorted(base_dir.rglob("*.mp4"))
        if not mp4_files:
            print(f"No .mp4 files found under: {opt.video_directory}")
            sys.exit(0)

        print(f"Found {len(mp4_files)} .mp4 files.")
        for mp4_path in mp4_files:
            mp4_path = mp4_path.resolve()
            ts_path = mp4_path.with_suffix(".txt")
            print(f"\n=== Converting: {mp4_path} ===")
            print(f"Output txt: {ts_path}")
            readout(str(mp4_path), str(ts_path))
        return

    # Single-file mode
    if not opt.video_file.endswith((".mp4")):
        print(
            "--video_file parameter should be a .mp4 file but is not :", opt.video_file, "Exit program."
        )
        sys.exit(1)
    if not os.path.isfile(opt.video_file):
        print(
            "--video_file parameter should be an existing file but is not :", opt.video_file, "Exit program."
        )
        sys.exit(1)
    mp4_path = Path(opt.video_file).resolve()
    ts_path = mp4_path.with_suffix(".txt")
    print(f"\n=== Converting: {mp4_path} ===")
    print(f"Output txt: {ts_path}")
    readout(str(mp4_path), str(ts_path))


if __name__ == "__main__":
    main()