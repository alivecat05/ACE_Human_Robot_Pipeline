"""Postprocessing module: utils.

Args:
    None.

Returns:
    None.

Raises:
    None.
"""

import sys
import subprocess
import cv2
import os
from pathlib import Path
import shutil
import glob
from collections import defaultdict


def _should_process_record_dir(folder_name: str) -> bool:
    """_should_process_record_dir.

    Args:
        folder_name: Input parameter.

    Returns:
        bool: Result of _should_process_record_dir.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    return folder_name.startswith("0000")


def _has_nvidia_gpu():
    """_has_nvidia_gpu.

    Args:
        None.

    Returns:
        Any: Function result.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    """检查当前机器是否有可用 NVIDIA GPU。"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode != 0:
            return False
        return bool(result.stdout.strip())
    except (subprocess.SubprocessError, OSError, ValueError):
        return False


def check_dir_size(egocap_path):
    """check_dir_size.

    Args:
        egocap_path: Input parameter.

    Returns:
        Any: Result of check_dir_size.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(egocap_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def get_preferred_ffmpeg_encoder():
    """get_preferred_ffmpeg_encoder.

    Args:
        None.

    Returns:
        Any: Function result.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    """获取推荐的视频编码器及其参数。

    优先级:
    1) 环境变量 `ACE_VIDEO_CODEC` (可选)
    2) 检测到 NVIDIA GPU -> h264_nvenc
    3) 否则 -> libx264
    """
    codec_override = os.getenv("ACE_VIDEO_CODEC", "").strip()
    if codec_override:
        if codec_override == "h264_nvenc":
            return codec_override, ["-preset", "fast", "-cq", "23"]
        if codec_override == "libx264":
            return codec_override, ["-preset", "fast", "-crf", "23"]
        return codec_override, []

    if _has_nvidia_gpu():
        return "h264_nvenc", ["-preset", "fast", "-cq", "23"]

    return "libx264", ["-preset", "fast", "-crf", "23"]


def progress_bar(percent_done, bar_length=50):
    """progress_bar.

    Args:
        percent_done: Input parameter.
        bar_length: Input parameter.

    Returns:
        Any: Result of progress_bar.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    done_length = int(bar_length * percent_done / 100)
    bar = "=" * done_length + "-" * (bar_length - done_length)
    sys.stdout.write("[%s] %i%s\r" % (bar, percent_done, "%"))
    sys.stdout.flush()


def is_video_shorter_than(cap, k_minutes):

    # 获取视频的帧率和总帧数
    """is_video_shorter_than.

    Args:
        cap: Input parameter.
        k_minutes: Input parameter.

    Returns:
        Any: Result of is_video_shorter_than.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()  # 获取完信息后记得释放资源

    # 安全校验：防止因获取不到属性而导致除以 0 的报错
    if fps <= 0 or total_frames <= 0:
        print("错误：无法获取有效的帧率或总帧数，视频文件可能已损坏。")
        return False

    # 计算时长
    duration_seconds = total_frames / fps
    duration_minutes = duration_seconds / 60.0

    # print(f"视频时长: {duration_minutes:.2f} 分钟")

    # 判断是否不足 k 分钟
    return duration_minutes < k_minutes


def fillout_non_valid_dir(root_path, k_minutes=1):
    """fillout_non_valid_dir.

    Args:
        root_path: Input parameter.
        k_minutes: Input parameter.

    Returns:
        Any: Result of fillout_non_valid_dir.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    if not os.path.exists(root_path):
        print(f"根目录不存在：{root_path}")
        return
    should_delete = False
    delete_reason = ""
    for file in sorted(os.listdir(root_path)):
        file_path = os.path.join(root_path, file)

        if not _should_process_record_dir(file):
            continue

        # 确保是目录
        if not os.path.isdir(file_path):
            continue

        video_path = Path(os.path.join(file_path, "realsense"))

        if not video_path.exists():
            should_delete = True
            if should_delete:
                print(f"检测到无效数据：{delete_reason}，正在删除目录 {file_path}...")
                try:
                    shutil.rmtree(file_path)
                    print(f"目录 {file_path} 已成功删除。")
                except OSError as e:
                    print(f"删除目录 {file_path} 失败：{e}")
            continue
        # print(f"正在检查目录 {file_path} 中的视频文件...")
        print(f"目录 {file_path} 中的视频文件路径: {video_path}")

        if any(video_path.glob("*.mp4")):
            print(f"目录 {file_path} 中存在 mp4 视频...")
            for video in video_path.glob("*.mp4"):
                try:
                    # 检查视频时长，若无法打开也会抛出异常
                    cap = cv2.VideoCapture(video)

                    if not cap.isOpened():
                        print(f"错误：无法打开视频 {video_path}")
                        should_delete = True
                        break
                    else:
                        if is_video_shorter_than(cap, k_minutes):
                            should_delete = True
                            delete_reason = f"时长不足 {k_minutes} 分钟"
                            break
                except (cv2.error, OSError, ValueError) as e:
                    # 捕获 OpenCV 错误或其他异常（如文件损坏）
                    should_delete = True
                    delete_reason = f"文件损坏或无法读取 ({type(e).__name__})"
                    break
        else:
            should_delete = True
            delete_reason = "缺少 mp4 视频文件"
            print(f"目录 {file_path} 中未找到 mp4 视频文件...")
        if should_delete:
            print(f"检测到无效数据：{delete_reason}，正在删除目录 {file_path}...")
            try:
                shutil.rmtree(file_path)
                print(f"目录 {file_path} 已成功删除。")
            except OSError as e:
                print(f"删除目录 {file_path} 失败：{e}")


def check_standard_name(root_path):
    """check_standard_name.

    Args:
        root_path: Input parameter.

    Returns:
        Any: Result of check_standard_name.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    for file in sorted(os.listdir(root_path)):
        if not _should_process_record_dir(file):
            continue
        if "-" in file or "_" in file:
            new_file = file.split("-")[0]
            new_file = new_file.split("_")[0]
            new_name = os.path.join(root_path, new_file)
            os.rename(os.path.join(root_path, file), new_name)


def fillout_redundant_dir(root_path):
    """fillout_redundant_dir.

    Args:
        root_path: Input parameter.

    Returns:
        Any: Result of fillout_redundant_dir.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    for file in sorted(os.listdir(root_path)):
        if not _should_process_record_dir(file):
            continue
        file_path = os.path.join(root_path, file)
        egocap_path = os.path.join(file_path, "egocap")
        if os.path.isdir(egocap_path):
            if len(os.listdir(egocap_path)) > 1:
                sizes = []
                for subdir in os.listdir(egocap_path):
                    subdir_path = os.path.join(egocap_path, subdir)
                    if os.path.isdir(subdir_path):
                        size = check_dir_size(subdir_path)
                        sizes.append((subdir_path, size))
                if sizes:
                    ###find the largest one
                    largest_dir = max(sizes, key=lambda x: x[1])[0]
                    print(
                        f"目录 {egocap_path} 中存在多个子目录，正在保留 {largest_dir} 并删除其他目录..."
                    )
                    for subdir_path, size in sizes:
                        if subdir_path != largest_dir:
                            os.system(f"rm -rf {subdir_path}")
                            continue
            # print(f"目录 {egocap_path} 中只有一个子目录，保持不变。")


def keep_largest_egocap_data(base_dir, dry_run=True):
    """keep_largest_egocap_data.

    Args:
        base_dir: Input parameter.
        dry_run: Input parameter.

    Returns:
        Any: Result of keep_largest_egocap_data.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    if not os.path.exists(base_dir):
        print(f"错误: 目录 {base_dir} 不存在。")
        return

    # 内部辅助函数：获取目录大小
    def get_dir_size(folder_path):
        """get_dir_size.

        Args:
            folder_path: Input parameter.

        Returns:
            Any: Result of get_dir_size.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        total_size = 0
        for dirpath, _, filenames in os.walk(folder_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp) and os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        return total_size

    # print(f"开始扫描目录: {base_dir} | 测试模式(dry_run): {dry_run}\n")

    # 1. 自动扫描并对文件夹进行分组
    groups = defaultdict(list)
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)

        if not _should_process_record_dir(folder_name):
            continue

        if os.path.isdir(folder_path):
            base_prefix = folder_name.split("-")[0]
            if base_prefix.isdigit():
                groups[base_prefix].append(folder_path)

    if not groups:
        print("没有找到符合条件的文件夹。")
        return

    # 2. 遍历每一个分组进行对比和清理
    for base_prefix, candidate_folders in groups.items():
        # print(f"{'='*50}")
        # print(f"开始处理系列: {base_prefix} (共有 {len(candidate_folders)} 个候选文件夹)")

        final_target_path = os.path.join(base_dir, base_prefix)
        max_size = -1
        largest_folder = None

        # 计算该系列下，每个候选文件夹中 egocap/*/ 的大小
        for folder in candidate_folders:
            random_id_pattern = os.path.join(folder, "egocap", "*")
            random_id_folders = [
                f for f in glob.glob(random_id_pattern) if os.path.isdir(f)
            ]

            current_size = 0
            for rid_folder in random_id_folders:
                current_size += get_dir_size(rid_folder)

            # print(f"  - 检查: {folder} -> 容量: {current_size / (1024*1024):.2f} MB")

            if current_size > max_size:
                max_size = current_size
                largest_folder = folder

        # 3. 筛选出最大者后，进行删除和重命名
        if largest_folder:
            for folder in candidate_folders:
                if folder != largest_folder:
                    if dry_run:
                        print(f"    [Dry Run] 准备删除冗余文件夹: {folder}")
                    else:
                        print(f"    [执行] 正在删除: {folder}")
                        shutil.rmtree(folder)

            # 再将最大的文件夹重命名为纯净的编号
            if largest_folder != final_target_path:
                if dry_run:
                    print(
                        f"    [Dry Run] 准备重命名: {largest_folder} -> {final_target_path}"
                    )
                else:
                    # 增加一层安全校验，防止重命名时目标已存在导致崩溃
                    if os.path.exists(final_target_path):
                        print(
                            f"    [警告] 目标路径 {final_target_path} 已存在，跳过重命名。"
                        )
                    else:
                        print(
                            f"    [执行] 正在重命名: {largest_folder} -> {final_target_path}"
                        )
                        os.rename(largest_folder, final_target_path)


def prefillter(root):

    """prefillter.

    Args:
        root: Input parameter.

    Returns:
        Any: Result of prefillter.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    fillout_non_valid_dir(root, k_minutes=1)
    fillout_redundant_dir(root)
    keep_largest_egocap_data(root, dry_run=False)
    check_standard_name(root)


def remove_rgb_folders(folder_path):
    """remove_rgb_folders.

    Args:
        folder_path: Input parameter.

    Returns:
        Any: Result of remove_rgb_folders.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    for root, dirs, files in os.walk(folder_path):
        try:
            for dir_name in dirs:
                if dir_name == "rgb":
                    dir_path = os.path.join(root, dir_name)
                    # print(f"Removing directory: {dir_path}")
                    shutil.rmtree(dir_path)
        except OSError as e:
            print(f"Error removing directory {dir_path}: {e}")
            continue
    print("RGB folders removed successfully.")
