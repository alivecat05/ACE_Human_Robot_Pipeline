#!/usr/bin/env python3
import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple


def collect_task_csv_files(base_path: str) -> List[Tuple[str, str]]:
    task_csv_files: List[Tuple[str, str]] = []
    for root, _, files in os.walk(base_path):
        if "task_timestamps.csv" in files:
            csv_path = os.path.join(root, "task_timestamps.csv")
            folder_name = os.path.relpath(root, base_path)
            task_csv_files.append((folder_name, csv_path))
    task_csv_files.sort(key=lambda item: item[0])
    return task_csv_files


def calculate_total_video_duration(base_path: str) -> str:
    if not os.path.exists(base_path):
        return f"❌ 目录不存在: {base_path}"

    total_duration_seconds = 0.0
    total_task_count = 0
    folder_details: List[str] = []

    try:
        task_csv_files = collect_task_csv_files(base_path)
        if not task_csv_files:
            return "❌ 未找到任何 task_timestamps.csv"

        for folder_name, csv_path in task_csv_files:
            try:
                task_events: Dict[str, Dict[str, int]] = {}
                valid_timestamps: List[int] = []
                with open(csv_path, "r", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        task_id = str(row.get("task_id", "")).strip()
                        event = str(row.get("event", "")).strip().lower()
                        timestamp_raw = str(row.get("timestamp", "")).strip()

                        try:
                            timestamp_ms = int(timestamp_raw)
                        except (TypeError, ValueError):
                            continue

                        valid_timestamps.append(timestamp_ms)

                        if not task_id or event not in {"start", "end"}:
                            continue

                        if task_id not in task_events:
                            task_events[task_id] = {}
                        task_events[task_id][event] = timestamp_ms

                folder_task_count = 0
                for _, events in sorted(task_events.items()):
                    start_ms = events.get("start")
                    end_ms = events.get("end")

                    if start_ms is None or end_ms is None or end_ms < start_ms:
                        continue
                    folder_task_count += 1

                if len(valid_timestamps) < 2:
                    continue

                first_timestamp_ms = valid_timestamps[0]
                last_timestamp_ms = valid_timestamps[-1]
                if last_timestamp_ms < first_timestamp_ms:
                    continue

                folder_duration_seconds = (last_timestamp_ms - first_timestamp_ms) / 1000.0
                if folder_duration_seconds <= 0:
                    continue

                total_task_count += folder_task_count
                total_duration_seconds += folder_duration_seconds
                folder_details.append(
                    f"{folder_name}: 任务 {folder_task_count} 条, 时长 {folder_duration_seconds:.2f} 秒"
                )
            except Exception as error:
                print(f"Error processing task {folder_name}: {error}")

        total_minutes = int(total_duration_seconds // 60)
        total_seconds = int(total_duration_seconds % 60)
        total_hours = total_minutes // 60
        remaining_minutes = total_minutes % 60

        date_label = os.path.basename(os.path.normpath(base_path))
        result = f"📊 当日视频统计 ({date_label})\n"
        result += f"{'=' * 50}\n"
        result += f"统计目录: {base_path}\n"
        result += f"总任务数: {total_task_count}\n"
        if total_hours > 0:
            result += f"总时长: {total_hours}小时{remaining_minutes}分{total_seconds}秒\n"
        else:
            result += f"总时长: {total_minutes}分{total_seconds}秒\n"
        result += f"{'=' * 50}\n\n"
        result += "各任务详情:\n"
        result += "\n".join(folder_details) if folder_details else "无有效任务时间戳数据"
        return result
    except Exception as error:
        return f"❌ 统计失败: {error}"


def main() -> None:
    today = datetime.now().strftime("%Y%m%d")

    parser = argparse.ArgumentParser(description="统计日期目录内所有 task_timestamps.csv 的任务数和总时长")
    parser.add_argument(
        "path",
        nargs="?",
        default=f"/home/ssd/20260330-001",
        help="要统计的目录路径，例如 /home/ssd/20260414",
    )
    args = parser.parse_args()

    output = calculate_total_video_duration(args.path)
    print(output)


if __name__ == "__main__":
    main()