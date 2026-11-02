import os
import imageio
import cv2
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--input_directory",
        type=str,
        default="",
    )
    opt = parser.parse_args()

    videos = [item.path for item in os.scandir(opt.input_directory) if item.is_file() and item.path.endswith("_rgb.mp4")]
    timestamps = [item.path for item in os.scandir(opt.input_directory) if item.is_file() and item.path.endswith("_ts.csv")]
    videos.sort()
    timestamps.sort()

    for i in range(len(videos)) :
        video_path = videos[i]
        ts_path = timestamps[i]
        item_name = os.path.basename(ts_path).split(".")[0]
        result_path = os.path.join(os.path.dirname(ts_path), item_name)
        os.makedirs(result_path, exist_ok=True)

        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise ValueError("Error opening video file.")
        
        timestamps_list = []
        with open(ts_path, 'r') as f :
            for line in f :
                timestamps_list.append(line.strip())

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
                
                # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(result_path, f"{timestamps_list[frame_count]}.png"), frame)

                frame_count += 1
                pbar.update(1)

        video_capture.release()


if __name__ == "__main__":
    main()
