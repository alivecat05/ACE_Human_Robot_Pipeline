#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import struct


def load_imu_csv(path):
    """
    Robustly load imu.csv.

    Supports both old format (1 timestamp) and new format (2 timestamps).
    - Open in binary mode so we can detect NUL bytes.
    - Stop reading when:
        * the line contains a NUL byte,
        * the number of comma-separated fields doesn't match expected format,
        * parsing to int/float fails.
    Returns:
        List of tuples: (device_ts_ns, host_ts_ns, gx, gy, gz, ax, ay, az)
        For old format files, host_ts_ns will be set to device_ts_ns.
    """
    imu_data = []
    has_dual_timestamps = None  # None means not yet detected
    
    with open(path, 'rb') as f:
        for raw in f:
            if not raw:
                break

            # If this line already contains a NUL, assume we've reached the corrupted tail.
            if b'\x00' in raw:
                break

            line = raw.decode('ascii', errors='ignore').strip()
            if not line:
                continue

            # Detect format from header
            if line.startswith("device_timestamp_ns"):
                has_dual_timestamps = True
                continue
            elif line.startswith("timestamp_ns"):
                has_dual_timestamps = False
                continue

            parts = line.split(',')
            
            # Auto-detect format from first data line if not yet detected
            if has_dual_timestamps is None:
                # New format: 2 timestamps + 6 values = 8 fields
                # Old format: 1 timestamp + 6 values = 7 fields
                has_dual_timestamps = (len(parts) == 8)
            
            # New format: 2 timestamps + 6 values = 8 fields
            # Old format: 1 timestamp + 6 values = 7 fields
            if has_dual_timestamps:
                if len(parts) != 8:
                    # Incomplete line => treat as corrupted tail, stop parsing
                    break
                try:
                    device_ts = int(parts[0])
                    host_ts = int(parts[1])
                    gx = float(parts[2])
                    gy = float(parts[3])
                    gz = float(parts[4])
                    ax = float(parts[5])
                    ay = float(parts[6])
                    az = float(parts[7])
                except ValueError:
                    # Parsing failed => treat as corrupted tail
                    break
                imu_data.append((device_ts, host_ts, gx, gy, gz, ax, ay, az))
            else:
                if len(parts) != 7:
                    # Incomplete line => treat as corrupted tail, stop parsing
                    break
                try:
                    device_ts = int(parts[0])
                    host_ts = device_ts  # Use device timestamp as host timestamp for backward compatibility
                    gx = float(parts[1])
                    gy = float(parts[2])
                    gz = float(parts[3])
                    ax = float(parts[4])
                    ay = float(parts[5])
                    az = float(parts[6])
                except ValueError:
                    # Parsing failed => treat as corrupted tail
                    break
                imu_data.append((device_ts, host_ts, gx, gy, gz, ax, ay, az))

    return imu_data


def load_mjpeg(path):
    """
    Load mjpeg.bin with the following format:

    - First 10 bytes: magic "ROBATA" + [version, 0, 0, 0]
      Version 1: [uint64 device_ts_ns][uint32 size][JPEG bytes]
      Version 2: [uint64 device_ts_ns][uint64 host_ts_ns][uint32 size][JPEG bytes]

    Only full frames are kept; a truncated last frame is discarded.

    Returns:
        List of (device_ts_ns, host_ts_ns, jpeg_bytes)
        For version 1 files, host_ts_ns will be set to device_ts_ns.
    """
    frames = []
    with open(path, 'rb') as f:
        magic = f.read(10)
        if len(magic) != 10:
            print(f"[WARN] {path}: magic length is incorrect, file may be corrupted")
            return frames

        if not magic.startswith(b'ROBATA'):
            print(f"[WARN] {path}: magic does not start with 'ROBATA', actual={magic[:6]!r}")
            return frames

        version = magic[6]
        has_dual_timestamps = (version >= 2)

        while True:
            if has_dual_timestamps:
                # Version 2: device_ts (8) + host_ts (8) + size (4) = 20 bytes
                header = f.read(8 + 8 + 4)
                if len(header) < 20:
                    # No full header left => done
                    break
                device_ts_ns, host_ts_ns, size = struct.unpack('<QQI', header)
            else:
                # Version 1: device_ts (8) + size (4) = 12 bytes
                header = f.read(8 + 4)
                if len(header) < 12:
                    # No full header left => done
                    break
                device_ts_ns, size = struct.unpack('<QI', header)
                host_ts_ns = device_ts_ns  # Use device timestamp as host timestamp for backward compatibility

            if size <= 0:
                # Invalid size => stop
                break

            data = f.read(size)
            if len(data) < size:
                # Truncated last frame => discard it and stop
                break

            frames.append((device_ts_ns, host_ts_ns, data))

    return frames


def write_clean_imu(dir_path, imu_data):
    """
    Write cleaned IMU data to dir_path/imu_clean.csv
    """
    out_path = os.path.join(dir_path, "imu_clean.csv")
    if not imu_data:
        print(f"[WARN] {dir_path}: no IMU data to write, skip imu_clean.csv")
        return

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["device_timestamp_ns", "host_timestamp_ns", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"])
        for row in imu_data:
            writer.writerow(row)

    print(f"[INFO] wrote cleaned IMU to: {out_path}")


def write_frames(dir_path, frames):
    """
    Write frames to dir_path/rgb as {index}_{device_ts_ns}_{host_ts_ns}.jpg
    """
    rgb_dir = os.path.join(dir_path, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    for index, (device_ts_ns, host_ts_ns, jpeg) in enumerate(frames):
        filename = f"{index:06d}_{device_ts_ns}_{host_ts_ns}.jpg"
        out_path = os.path.join(rgb_dir, filename)
        with open(out_path, 'wb') as f:
            f.write(jpeg)

    print(f"[INFO] wrote {len(frames)} images to: {rgb_dir}")


def align_by_timestamp(imu_data, frames):
    """
    Align IMU samples and frames by their device timestamp intersection:

    - t_min = max(first_imu_device_ts, first_frame_device_ts)
    - t_max = min(last_imu_device_ts, last_frame_device_ts)

    Returns:
        imu_aligned, frames_aligned
    """
    if not imu_data or not frames:
        return [], []

    # Use device timestamps for alignment (first element in each tuple)
    imu_ts = [row[0] for row in imu_data]
    frame_ts = [device_ts for device_ts, _, _ in frames]

    t_min = max(imu_ts[0], frame_ts[0])
    t_max = min(imu_ts[-1], frame_ts[-1])

    if t_min > t_max:
        print("[WARN] IMU and MJPEG have no overlapping time range, returning empty")
        return [], []

    imu_aligned = [row for row in imu_data if t_min <= row[0] <= t_max]
    frames_aligned = [(device_ts, host_ts, img) for device_ts, host_ts, img in frames if t_min <= device_ts <= t_max]

    print(f"[INFO] aligned time range: [{t_min}, {t_max}]")
    print(f"[INFO] aligned IMU samples: {len(imu_aligned)}, frames: {len(frames_aligned)}")
    return imu_aligned, frames_aligned


def process_one_recording(dir_path):
    """
    Process a single recording directory:

    - Read dir_path/imu.csv and dir_path/mjpeg.bin
    - Robust parsing + timestamp intersection
    - Write imu_clean.csv and rgb/*.jpg
    """
    imu_csv_path = os.path.join(dir_path, "imu.csv")
    mjpeg_path = os.path.join(dir_path, "mjpeg.bin")

    if not os.path.isfile(imu_csv_path):
        print(f"[WARN] {dir_path}: imu.csv not found, skipping")
        return
    if not os.path.isfile(mjpeg_path):
        print(f"[WARN] {dir_path}: mjpeg.bin not found, skipping")
        return

    print(f"[INFO] processing recording directory: {dir_path}")

    imu_data = load_imu_csv(imu_csv_path)
    frames = load_mjpeg(mjpeg_path)

    if not imu_data:
        print(f"[WARN] {dir_path}: parsed no IMU data")
    if not frames:
        print(f"[WARN] {dir_path}: parsed no MJPEG frames")

    imu_aligned, frames_aligned = align_by_timestamp(imu_data, frames)

    # If you prefer to always write cleaned IMU even without frames,
    # you can change this logic to use imu_data instead of imu_aligned.
    if imu_aligned and frames_aligned:
        write_clean_imu(dir_path, imu_aligned)
        write_frames(dir_path, frames_aligned)
    else:
        print(f"[WARN] {dir_path}: empty alignment result, nothing written")


def find_recording_dirs(root_path):
    """
    If root_path itself is a recording directory (contains mjpeg.bin & imu.csv),
    return [root_path].

    Otherwise, treat it as a root folder and return all subdirectories that
    contain both mjpeg.bin and imu.csv.
    """
    if os.path.isdir(root_path):
        imu_csv_path = os.path.join(root_path, "imu.csv")
        mjpeg_path = os.path.join(root_path, "mjpeg.bin")
        if os.path.isfile(imu_csv_path) and os.path.isfile(mjpeg_path):
            return [root_path]

        rec_dirs = []
        for name in sorted(os.listdir(root_path)):
            p = os.path.join(root_path, name)
            if not os.path.isdir(p):
                continue
            imu_csv_path = os.path.join(p, "imu.csv")
            mjpeg_path = os.path.join(p, "mjpeg.bin")
            if os.path.isfile(imu_csv_path) and os.path.isfile(mjpeg_path):
                rec_dirs.append(p)
        return rec_dirs
    else:
        return []


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parse and clean mjpeg.bin & imu.csv recordings, "
            "and export imu_clean.csv + rgb/{timestamp}.jpg"
        )
    )
    parser.add_argument(
        "path",
        help=(
            "Recording directory (containing mjpeg.bin and imu.csv), "
            "or a root directory that contains multiple such subdirectories "
            "(e.g. recordings/)"
        ),
    )
    args = parser.parse_args()

    root = os.path.abspath(args.path)
    rec_dirs = find_recording_dirs(root)

    if not rec_dirs:
        print(f"[ERROR] no recording directory with mjpeg.bin + imu.csv found under {root}")
        return

    for d in rec_dirs:
        process_one_recording(d)


if __name__ == "__main__":
    main()
