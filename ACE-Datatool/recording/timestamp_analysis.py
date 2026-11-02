#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pyzed.sl as sl

@dataclass(frozen=True)
class AnalysisTask:
    """One timestamp analysis task for a single file."""

    source_path: Path
    save_path: Path
    save_name: str


class TimestampAnalysisApp:
    """Analyze timestamp files from ZED SVO and RealSense CSV recordings."""

    def __init__(
        self,
        zed_data_path: str,
        rs_data_path: str,
        enable_plot: bool = False,
    ) -> None:
        self._zed_data_path = zed_data_path
        self._rs_data_path = rs_data_path
        self._enable_plot = enable_plot

    def run(self) -> int:
        """Run analysis tasks for requested data paths."""
        if self._zed_data_path:
            code = self._analyze_zed_directory(Path(self._zed_data_path))
            if code != 0:
                return code

        if self._rs_data_path:
            code = self._analyze_rs_directory(Path(self._rs_data_path))
            if code != 0:
                return code

        return 0

    @staticmethod
    def _progress_bar(percent_done: float, bar_length: int = 50) -> None:
        """Display a progress bar in terminal output."""
        done_length = int(bar_length * percent_done / 100)
        bar = "=" * done_length + "-" * (bar_length - done_length)
        sys.stdout.write("[%s] %i%s\r" % (bar, percent_done, "%"))
        sys.stdout.flush()

    @staticmethod
    def _calculate_average_fps(timestamps_us: np.ndarray) -> float:
        """Calculate and print average FPS from timestamp sequence."""
        warmup_frames = 20
        timestamps = timestamps_us[warmup_frames:]
        if timestamps.size < 2:
            timestamps = timestamps_us

        if timestamps.size < 2:
            print("Could not calculate average FPS (not enough timestamps).")
            return 0.0

        time_start_sec = timestamps[0] / 1_000_000.0
        time_end_sec = timestamps[-1] / 1_000_000.0
        duration_sec = time_end_sec - time_start_sec

        if duration_sec <= 0:
            print("Could not calculate average FPS (non-positive duration).")
            return 0.0

        real_avg_fps = (timestamps.size - 1) / duration_sec
        print(f"Real Average FPS = {real_avg_fps:.2f}")

        time_diffs_ms = np.diff(timestamps) / 1000.0
        avg_diff_ms = np.mean(time_diffs_ms)

        if avg_diff_ms > 0:
            avg_fps = 1000.0 / avg_diff_ms
            print(f"Calculated Average FPS = {avg_fps:.2f}")
        else:
            print("Could not calculate average FPS (average time difference is zero).")

        return real_avg_fps

    def _analyze_zed_directory(self, base_dir: Path) -> int:
        """Analyze all .svo2 files under given directory."""
        if not base_dir.is_dir():
            print(f"--zed_data_path must be an existing directory: {base_dir}")
            return 1

        svo_files = sorted(base_dir.rglob("*.svo2"))
        if not svo_files:
            print(f"No .svo2 files found under: {base_dir}")
            return 0

        print(f"Found {len(svo_files)} .svo2 files.")
        for svo_path in svo_files:
            task = self._build_task(svo_path)
            self._analyze_zed_file(task)
            sys.stdout.flush()
        return 0

    def _analyze_rs_directory(self, base_dir: Path) -> int:
        """Analyze all *_ts.csv files under given directory."""
        if not base_dir.is_dir():
            print(f"--rs_data_path must be an existing directory: {base_dir}")
            return 1

        ts_files = sorted(base_dir.rglob("*_ts.csv"))
        if not ts_files:
            print(f"No .txt files found under: {base_dir}")
            return 0

        print(f"Found {len(ts_files)} ts.csv files.")
        for ts_path in ts_files:
            task = self._build_task(ts_path)
            self._analyze_rs_file(task)
        return 0

    @staticmethod
    def _build_task(file_path: Path) -> AnalysisTask:
        """Build normalized analysis task from file path."""
        resolved = file_path.resolve()
        return AnalysisTask(
            source_path=resolved,
            save_path=resolved.parent,
            save_name=resolved.stem,
        )

    def _analyze_zed_file(self, task: AnalysisTask) -> None:
        """Analyze one ZED SVO file and print FPS summary."""
        init_params = sl.InitParametersOne()
        init_params.set_from_svo_file(str(task.source_path))
        init_params.svo_real_time_mode = False
        init_params.coordinate_units = sl.UNIT.MILLIMETER

        zed = sl.CameraOne()
        open_error = zed.open(init_params)
        if open_error != sl.ERROR_CODE.SUCCESS:
            print(f"Failed to open SVO file: {task.source_path}, error: {open_error}")
            return

        zed_serial = 0
        try:
            zed_serial = zed.get_camera_information().serial_number
        except Exception:
            pass

        frame_count = zed.get_svo_number_of_frames()
        timestamps_list: list[int] = []

        while True:
            grab_error = zed.grab()
            if grab_error == sl.ERROR_CODE.SUCCESS:
                svo_position = zed.get_svo_position()
                timestamp_us = zed.get_timestamp(
                    sl.TIME_REFERENCE.IMAGE
                ).get_microseconds()
                timestamps_list.append(timestamp_us)

                step = max(frame_count // 5, 1)
                if (svo_position + 1) % step == 0:
                    self._progress_bar((svo_position + 1) / frame_count * 100, 30)

            if grab_error == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                self._progress_bar(100, 30)
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break

        zed.close()
        self._finalize_analysis(
            timestamps_list=timestamps_list,
            save_path=task.save_path,
            save_name=task.save_name,
            default_camera_label="ZED",
            explicit_camera_label=(f"ZED ({zed_serial})" if zed_serial else None),
        )

    def _analyze_rs_file(self, task: AnalysisTask) -> None:
        """Analyze one RealSense timestamp CSV file and print FPS summary."""
        timestamps_list: list[int] = []
        with open(task.source_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    timestamps_list.append(int(line))

        self._finalize_analysis(
            timestamps_list=timestamps_list,
            save_path=task.save_path,
            save_name=task.save_name,
            default_camera_label="RealSense",
            explicit_camera_label=self._extract_rs_camera_label(task.save_name),
        )

    def _finalize_analysis(
        self,
        timestamps_list: list[int],
        save_path: Path,
        save_name: str,
        default_camera_label: str,
        explicit_camera_label: Optional[str],
    ) -> None:
        """Print standard analysis output and FPS summary line."""
        print("\n--- Generating Timestamp Difference Plot ---")
        if len(timestamps_list) < 2:
            print("Not enough timestamps to generate a plot.")
            return

        timestamps_us = np.asarray(timestamps_list, dtype=np.int64)
        real_fps = self._calculate_average_fps(timestamps_us)

        recording_name = self._extract_recording_name(save_path, save_name)
        camera_label = explicit_camera_label or default_camera_label

        print(
            f"FPS_SUMMARY: {recording_name}, {camera_label}, {real_fps:.2f}", flush=True
        )
        print(f"timestamps_list length: {len(timestamps_list)}", flush=True)

        if self._enable_plot:
            analysis_output_path = save_path / "timestamp_analysis"
            self._plot_timestamp_differences(
                timestamps_array=timestamps_us,
                output_path=analysis_output_path,
                video_filename=save_name,
            )

    @staticmethod
    def _extract_recording_name(save_path: Path, fallback_name: str) -> str:
        """Extract recording name from directory structure."""
        try:
            return save_path.parent.name
        except Exception:
            return fallback_name

    @staticmethod
    def _extract_rs_camera_label(save_name: str) -> Optional[str]:
        """Extract RealSense serial from filename when possible."""
        for part in save_name.split("_"):
            if part.isdigit() and len(part) > 10:
                return f"RealSense ({part})"
        return None

    def _plot_timestamp_differences(
        self,
        timestamps_array: np.ndarray,
        output_path: Path,
        video_filename: Optional[str] = None,
    ) -> None:
        """Generate and save a plot of timestamp differences."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as error:
            print(f"Plot skipped: failed to import matplotlib ({error})")
            return

        time_diffs_ms = np.diff(timestamps_array)
        indices = np.arange(1, len(time_diffs_ms) + 1)

        plt.figure(figsize=(12, 6))
        plt.plot(indices, time_diffs_ms, marker="o", linestyle="-", markersize=4)
        plt.title("Timestamp Difference Between Consecutive Frames")
        plt.xlabel("Frame Index")
        plt.ylabel("Time Difference (ms)")
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(output_path, exist_ok=True)
        if video_filename:
            base_name = Path(video_filename).stem
            plot_filename = output_path / f"{base_name}.png"
        else:
            plot_filename = output_path / "timestamp_differences.png"

        plt.savefig(str(plot_filename))
        print(f"Saved timestamp difference plot to: {plot_filename}", flush=True)
        plt.close()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--zed_data_path", type=str, default="", help="Path to the .svo/.svo2 file"
    )
    parser.add_argument(
        "--rs_data_path", type=str, default="", help="Path to the _ts.csv file"
    )
    parser.add_argument(
        "--enable_plot",
        action="store_true",
        help="Enable timestamp-difference visualization output.",
    )
    return parser.parse_args()


def main() -> int:
    """Program entry point."""
    args = parse_args()
    app = TimestampAnalysisApp(
        zed_data_path=args.zed_data_path,
        rs_data_path=args.rs_data_path,
        enable_plot=args.enable_plot,
    )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
