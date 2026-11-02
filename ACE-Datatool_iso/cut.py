#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频和传感器数据剪切工具
根据输入的时间（秒），将文件夹内的视频和CSV数据前a秒剪掉
"""
# python cut.py -i F:\processed\20260122\4653804456\000002 -o F:\upload\20260122\4653804456\000002 -t 10
import os
import sys
import pandas as pd
import subprocess
import shutil
from pathlib import Path
import argparse
import math
import json

# 设置标准输出编码为UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def get_video_duration(video_path):
    """获取视频时长（秒）"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"获取视频时长失败 {video_path}: {e}")
        return None


def cut_video(input_path, output_path, start_time, fps=15):
    """
    剪切视频，去掉前start_time秒
    尝试多种编码器以解决兼容性问题
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        start_time: 开始时间（秒）
        fps: 视频帧率，默认15fps
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 编码器优先级列表: (编码器名称, 参数列表)
        encoders_to_try = [
            ('h264_nvenc', ['-preset', 'fast']),           # NVIDIA GPU (common for ZED/Realsense)
            ('libx264', ['-preset', 'fast', '-crf', '23']), # Standard (may be missing)
            ('h264_amf', []),                              # AMD GPU
            ('mpeg4', ['-q:v', '5'])                       # Fallback (lowest quality but compatible)
        ]

        last_error = None

        for encoder, encoder_args in encoders_to_try:
            # 使用ffmpeg剪切视频
            # 将 -ss 放在 -i 后面以获得精确剪切
            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-ss', str(start_time),
                '-c:v', encoder
            ] + encoder_args + [
                '-c:a', 'copy',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ 视频剪切成功 (使用 {encoder}): {os.path.basename(output_path)}")
                return True
            
            last_error = result.stderr
            # 如果是未知编码器，直接尝试下一个
            if "Unknown encoder" in result.stderr:
                continue
            # 其他错误也继续尝试，保险起见
            
        print(f"✗ 视频剪切失败: {os.path.basename(output_path)}")
        if last_error:
            print(f"  错误信息 (最后尝试): {last_error[-500:]}")
        return False
        
    except Exception as e:
        print(f"✗ 视频剪切异常: {os.path.basename(output_path)}: {e}")
        return False


def cut_imu_csv(input_path, output_path, cut_time_seconds):
    """
    剪切IMU CSV文件，去掉前cut_time_seconds秒的数据
    支持多种时间戳列名：timestamp, device_timestamp_ns, host_timestamp_ns
    
    Args:
        input_path: 输入CSV路径
        output_path: 输出CSV路径
        cut_time_seconds: 要剪掉的时间（秒）
    """
    try:
        df = pd.read_csv(input_path)
        
        if len(df) == 0:
            print(f"✗ IMU CSV文件为空: {os.path.basename(input_path)}")
            return False
        
        # 确定时间戳列名
        timestamp_col = None
        for col in ['timestamp', 'device_timestamp_ns', 'host_timestamp_ns']:
            if col in df.columns:
                timestamp_col = col
                break
        
        if timestamp_col is None:
            print(f"✗ IMU CSV文件未找到时间戳列: {os.path.basename(input_path)}")
            return False
        
        # 获取起始时间戳（纳秒）
        first_timestamp = df[timestamp_col].iloc[0]
        
        # 计算剪切时间戳（纳秒）
        cut_timestamp = first_timestamp + (cut_time_seconds * 1e9)
        
        # 过滤数据
        df_cut = df[df[timestamp_col] >= cut_timestamp].copy()
        
        # 重置索引
        df_cut.reset_index(drop=True, inplace=True)
        
        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_cut.to_csv(output_path, index=False)
        
        print(f"✓ IMU CSV剪切成功: {os.path.basename(output_path)} (保留 {len(df_cut)}/{len(df)} 行)")
        return True
        
    except Exception as e:
        print(f"✗ IMU CSV剪切失败: {os.path.basename(input_path)}: {e}")
        return False


def cut_head_imu_csv(input_path, output_path, cut_time_seconds, fps=15):
    """
    剪切 head_imu.csv 文件，去掉前cut_time_seconds秒的数据
    特殊处理：首行始终保留，从第2行开始按帧数剪切（去掉 时间*fps 行）
    
    Args:
        input_path: 输入CSV路径
        output_path: 输出CSV路径
        cut_time_seconds: 要剪掉的时间（秒）
        fps: 帧率，默认15
    """
    try:
        df = pd.read_csv(input_path)
        
        if len(df) == 0:
            print(f"✗ head_imu CSV文件为空: {os.path.basename(input_path)}")
            return False
        
        # 保存第一行数据（始终保留）
        first_row = df.iloc[[0]].copy()
        
        # 计算需要剪掉的帧数
        frames_to_cut = int(cut_time_seconds * fps)
        
        # 从第2行开始剪切
        df_rest = df.iloc[1:].copy()
        
        # 如果剪切帧数超过剩余行数，只保留第一行
        if frames_to_cut >= len(df_rest):
            df_final = first_row
        else:
            # 剪掉前 frames_to_cut 行
            df_cut = df_rest.iloc[frames_to_cut:].copy()
            # 合并：第一行 + 剪切后的数据
            df_final = pd.concat([first_row, df_cut], ignore_index=True)
        
        # 保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, index=False)
        
        print(f"✓ head_imu CSV剪切成功: {os.path.basename(output_path)} (保留 {len(df_final)}/{len(df)} 行，首行始终保留)")
        return True
        
    except Exception as e:
        print(f"✗ head_imu CSV剪切失败: {os.path.basename(input_path)}: {e}")
        return False


def cut_ts_csv(input_path, output_path, cut_time_seconds, fps=15):
    """
    剪切时间戳CSV文件，去掉前cut_time_seconds秒的数据
    
    Args:
        input_path: 输入CSV路径
        output_path: 输出CSV路径
        cut_time_seconds: 要剪掉的时间（秒）
        fps: 视频帧率，默认15fps
    """
    try:
        # 读取时间戳文件（没有列名）
        df = pd.read_csv(input_path, header=None, names=['timestamp'])
        
        if len(df) == 0:
            print(f"✗ TS CSV文件为空: {os.path.basename(input_path)}")
            return False
        
        # 计算需要剪掉的帧数
        frames_to_cut = int(cut_time_seconds * fps)
        
        # 剪切数据
        df_cut = df.iloc[frames_to_cut:].copy()
        
        # 重置索引
        df_cut.reset_index(drop=True, inplace=True)
        
        # 保存（不带列名）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_cut.to_csv(output_path, index=False, header=False)
        
        print(f"✓ TS CSV剪切成功: {os.path.basename(output_path)} (保留 {len(df_cut)}/{len(df)} 行)")
        return True
        
    except Exception as e:
        print(f"✗ TS CSV剪切失败: {os.path.basename(input_path)}: {e}")
        return False


def cut_svo2(input_path, output_path, start_time):
    """
    剪切ZED SVO2文件，去掉前start_time秒
    
    Args:
        input_path: 输入SVO2路径
        output_path: 输出SVO2路径
        start_time: 开始时间（秒）
    """
    try:
        print(f"⚠ SVO2文件需要使用ZED SDK工具剪切: {os.path.basename(input_path)}")
        print(f"  建议使用ZED SDK的Python API或命令行工具处理")
        # 这里可以添加ZED SDK的调用代码
        # 暂时复制原文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(input_path, output_path)
        return True
    except Exception as e:
        print(f"✗ SVO2文件处理失败: {os.path.basename(input_path)}: {e}")
        return False


def process_folder(input_folder, output_folder, cut_time_seconds, fps=15, d435i_to_d405=False, sample_config=None):
    """
    处理整个文件夹，剪切所有视频和CSV文件
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        cut_time_seconds: 要剪掉的时间（秒）
        fps: 视频帧率，默认15fps
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    if not input_path.exists():
        print(f"✗ 输入文件夹不存在: {input_folder}")
        return
    
    print(f"\n{'='*60}")
    print(f"开始处理文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    print(f"剪切时间: {cut_time_seconds} 秒")
    print(f"视频帧率: {fps} fps")
    print(f"{'='*60}\n")
    
    success_count = 0
    fail_count = 0
    
    # 遍历所有文件
    for file_path in input_path.rglob('*'):
        if file_path.is_file():
            # 计算相对路径
            rel_path = file_path.relative_to(input_path)
            output_file = output_path / rel_path
            
            # 跳过指定文件：egocap_stereo_0.mp4, egocap_stereo_1.mp4
            if file_path.name in ['egocap_stereo_0.mp4', 'egocap_stereo_1.mp4']:
                print(f"=== 跳过文件: {rel_path}")
                continue
            
            # 跳过 zed 文件夹下的所有 _imu.csv 文件
            if 'zed' in file_path.parts and file_path.name.endswith('_imu.csv'):
                print(f"=== 跳过文件: {rel_path}")
                continue
            
            # 跳过 realsense 文件夹下的所有 _imu.csv 文件
            if 'realsense' in file_path.parts and file_path.name.endswith('_imu.csv'):
                print(f"=== 跳过文件: {rel_path}")
                continue
            file_ext = file_path.suffix.lower()
            
            # 处理MP4视频
            if file_ext == '.mp4':
                if cut_video(str(file_path), str(output_file), cut_time_seconds, fps):
                    success_count += 1
                else:
                    fail_count += 1
            
            # 处理SVO2文件
            elif file_ext == '.svo2':
                if cut_svo2(str(file_path), str(output_file), cut_time_seconds):
                    success_count += 1
                else:
                    fail_count += 1
            
            # 处理 head_imu.csv (特殊处理，保留首行)
            elif file_path.name == 'head_imu.csv' or file_path.name == 'egocap_imu.csv':
                if cut_head_imu_csv(str(file_path), str(output_file), cut_time_seconds, fps):
                    success_count += 1
                else:
                    fail_count += 1
            
            # 处理含有 hand 的 imu.csv (无表头，按帧数剪切，与TS文件方式相同)
            elif 'hand' in file_path.name.lower() and file_path.name.endswith('.csv'):
                if cut_ts_csv(str(file_path), str(output_file), cut_time_seconds, fps):
                    success_count += 1
                else:
                    fail_count += 1
            
            # 处理IMU CSV (包括 _imu.csv 和 egocap 的 imu.csv)
            elif (file_path.name.endswith('_imu.csv') or 
                  file_path.name == 'imu.csv'):
                if cut_imu_csv(str(file_path), str(output_file), cut_time_seconds):
                    success_count += 1
                else:
                    fail_count += 1
            
            # 处理TS CSV
            elif file_path.name.endswith('_ts.csv'):
                if cut_ts_csv(str(file_path), str(output_file), cut_time_seconds, fps):
                    success_count += 1
                else:
                    fail_count += 1
            
            # 其他文件直接复制
            else:
                try:
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    shutil.copy2(file_path, output_file)
                    print(f"→ 复制文件: {rel_path}")
                except Exception as e:
                    print(f"✗ 复制文件失败: {rel_path}: {e}")
                    fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"{'='*60}\n")
    
    # 处理 d435i_to_d405 转换
    if d435i_to_d405 and sample_config:
        print(f"\n{'='*60}")
        print(f"开始执行 d435i_to_d405 转换...")
        print(f"{'='*60}\n")
        
        # 1. 读取两个 config.json 文件
        input_config_path = input_path / 'config.json'
        sample_config_path = Path(sample_config)
        
        if not input_config_path.exists():
            print(f"✗ 输入config.json不存在: {input_config_path}")
            return
        
        if not sample_config_path.exists():
            print(f"✗ 样本config.json不存在: {sample_config_path}")
            return
        
        # 读取输入config
        with open(input_config_path, 'r', encoding='utf-8') as f:
            input_config = json.load(f)
        
        # 读取样本config
        with open(sample_config_path, 'r', encoding='utf-8') as f:
            sample_config_data = json.load(f)
        
        # 提取样本中的 hand2 和 hand3 配置
        if 'device' not in sample_config_data or 'ego' not in sample_config_data['device']:
            print(f"✗ 样本config.json格式不正确")
            return
        
        sample_ego = sample_config_data['device']['ego']
        
        # 获取 hand_2 和 hand_3 的新配置
        new_hand2 = sample_ego.get('hand_2')
        new_hand3 = sample_ego.get('hand_3')
        
        if not new_hand2 or not new_hand3:
            print(f"✗ 样本config.json中未找到hand_2或hand_3配置")
            return
        
        # 获取旧的 SN 值
        old_hand2_sn = input_config['device']['ego']['hand_2']['sn']
        old_hand3_sn = input_config['device']['ego']['hand_3']['sn']
        
        # 获取新的 SN 值
        new_hand2_sn = new_hand2['sn']
        new_hand3_sn = new_hand3['sn']
        
        print(f"✓ 读取配置成功")
        print(f"  hand_2: {old_hand2_sn} -> {new_hand2_sn}")
        print(f"  hand_3: {old_hand3_sn} -> {new_hand3_sn}")
        
        # 2. 替换输入config中的 hand_2 和 hand_3
        input_config['device']['ego']['hand_2'] = new_hand2
        input_config['device']['ego']['hand_3'] = new_hand3
        
        # 保存到输出文件夹
        output_config_path = output_path / 'config.json'
        with open(output_config_path, 'w', encoding='utf-8') as f:
            json.dump(input_config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ config.json已更新并保存到: {output_config_path}")
        
        # 3. 重命名 realsense/d435i 文件夹为 realsense/d405
        realsense_d435i_path = output_path / 'realsense' / 'd435i'
        realsense_d405_path = output_path / 'realsense' / 'd405'
        
        if realsense_d435i_path.exists():
            realsense_d435i_path.rename(realsense_d405_path)
            print(f"✓ 文件夹重命名: realsense/d435i -> realsense/d405")
        else:
            print(f"✗ 未找到文件夹: {realsense_d435i_path}")
            return
        
        # 4. 重命名文件
        # 文件命名规则示例: hand_2_011422070296_rgb.mp4 -> hand_2_344422070623_rgb.mp4
        # 同时支持 hand2_ 和 hand_2_ 两种格式
        rename_count = 0
        
        for file_path in realsense_d405_path.iterdir():
            if file_path.is_file():
                filename = file_path.name
                new_filename = filename
                
                # 替换 hand_2 或 hand2 的 SN
                if f'hand_2_{old_hand2_sn}' in filename:
                    new_filename = filename.replace(f'hand_2_{old_hand2_sn}', f'hand_2_{new_hand2_sn}')
                elif f'hand2_{old_hand2_sn}' in filename:
                    new_filename = filename.replace(f'hand2_{old_hand2_sn}', f'hand_2_{new_hand2_sn}')
                
                # 替换 hand_3 或 hand3 的 SN
                if f'hand_3_{old_hand3_sn}' in filename:
                    new_filename = filename.replace(f'hand_3_{old_hand3_sn}', f'hand_3_{new_hand3_sn}')
                elif f'hand3_{old_hand3_sn}' in filename:
                    new_filename = filename.replace(f'hand3_{old_hand3_sn}', f'hand_3_{new_hand3_sn}')
                
                # 如果文件名发生了变化，执行重命名
                if new_filename != filename:
                    new_file_path = file_path.parent / new_filename
                    file_path.rename(new_file_path)
                    print(f"✓ 文件重命名: {filename} -> {new_filename}")
                    rename_count += 1
        
        print(f"\n{'='*60}")
        print(f"d435i_to_d405 转换完成!")
        print(f"重命名文件数: {rename_count}")
        print(f"{'='*60}\n")
        
    


def auto_calculate_cut_time(input_folder):
    """
    自动计算要剪掉的时间（秒）
    
    Args:
        input_folder: 输入文件夹路径
        
    Returns:
        剪切时间（秒），如果无法自动计算则返回 None
    """
    try:
        input_path = Path(input_folder)
        
        # 优先检查 task_timestamps.csv 是否存在
        task_file = input_path / 'task_timestamps.csv'
        keypoint_file = input_path / 'keypoint_timestamps.csv'
        
        first_timestamp = None
        timestamp_source = None
        
        if task_file.exists():
            # 使用 task_timestamps.csv
            df_task = pd.read_csv(task_file)
            
            if len(df_task) == 0:
                print(f"✗ task_timestamps.csv 为空")
            elif 'timestamp' not in df_task.columns:
                print(f"✗ task_timestamps.csv 中未找到 timestamp 列")
            else:
                first_timestamp = df_task['timestamp'].iloc[0]
                if not pd.api.types.is_number(first_timestamp):
                    print(f"✗ task_timestamps.csv 的 timestamp 不是数字")
                    first_timestamp = None
                else:
                    timestamp_source = 'task_timestamps.csv'
                    print(f"✓ task_timestamps.csv 第一行 timestamp: {first_timestamp}")
        
        # 如果 task_timestamps.csv 不存在或无效，尝试 keypoint_timestamps.csv
        if first_timestamp is None and keypoint_file.exists():
            df_keypoint = pd.read_csv(keypoint_file)
            
            if len(df_keypoint) == 0:
                print(f"✗ keypoint_timestamps.csv 为空")
            elif 'timestamp' not in df_keypoint.columns:
                print(f"✗ keypoint_timestamps.csv 中未找到 timestamp 列")
            else:
                first_timestamp = df_keypoint['timestamp'].iloc[0]
                if not pd.api.types.is_number(first_timestamp):
                    print(f"✗ keypoint_timestamps.csv 的 timestamp 不是数字")
                    first_timestamp = None
                else:
                    timestamp_source = 'keypoint_timestamps.csv'
                    print(f"✓ keypoint_timestamps.csv 第一行 timestamp: {first_timestamp}")
        
        # 如果两个文件都不可用
        if first_timestamp is None:
            print(f"✗ 未找到有效的 task_timestamps.csv 或 keypoint_timestamps.csv，无法自动计算裁剪时间")
            return None
        
        # 读取 egocap/x1/egocap_ts.csv
        egocap_ts_file = input_path / 'egocap' / 'x1' / 'egocap_ts.csv'
        if not egocap_ts_file.exists():
            print(f"✗ 未找到 egocap/x1/egocap_ts.csv")
            return None
        
        # 读取 egocap_ts.csv（无表头，首行直接是数据）
        df_egocap = pd.read_csv(egocap_ts_file, header=None, names=['timestamp'])
        
        if len(df_egocap) == 0:
            print(f"✗ egocap_ts.csv 为空")
            return None
        
        egocap_first_timestamp = df_egocap['timestamp'].iloc[0]
        print(f"✓ egocap_ts.csv 第一行 timestamp: {egocap_first_timestamp}")
        
        # 计算时间差：(egocap_ts - keypoint_ts) 的绝对值 / 1000，向上取整
        time_diff = abs(egocap_first_timestamp - first_timestamp) / 1000.0
        cut_time = math.ceil(time_diff)
        
        print(f"✓ 自动计算的裁剪时间: {cut_time} 秒 (时间差: {time_diff:.3f} 秒)")
        
        return cut_time
        
    except Exception as e:
        print(f"✗ 自动计算裁剪时间失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='剪切视频和传感器数据，去掉前a秒',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 手动指定裁剪时间
  python cut.py -i F:\\BaiduNetdiskDownload\\raw1\\20260121\\000001 -o F:\\BaiduNetdiskDownload\\raw1\\20260121_cut\\000001 -t 5
  
  # 自动计算裁剪时间（需要 keypoint_timestamps.csv 和 egocap/x1/egocap_ts.csv）
  python cut.py -i F:\\BaiduNetdiskDownload\\raw1\\20260121\\000001 -o F:\\BaiduNetdiskDownload\\raw1\\20260121_cut\\000001
  
  # 指定帧率
  python cut.py -i F:\\BaiduNetdiskDownload\\raw1\\20260121 -o F:\\BaiduNetdiskDownload\\raw1\\20260121_cut -t 10 --fps 30
        """
    )
    
    parser.add_argument('-i', '--input', required=True, 
                        help='输入文件夹路径')
    parser.add_argument('-o', '--output', required=True,
                        help='输出文件夹路径')
    parser.add_argument('-t', '--time', type=float, required=False,
                        help='要剪掉的时间（秒），不指定则自动计算')
    parser.add_argument('--fps', type=int, default=15,
                        help='视频帧率（默认：15）')
    parser.add_argument('--d435i_to_d405', action='store_true',
                        help='是否将 d435i 视频格式转换为 d405 格式')
    
    parser.add_argument('--sample_config', default=None,
                        help='示例配置文件路径,用于 d435i_to_d405 转换')
    
    args = parser.parse_args()
    
    # 确定剪切时间
    cut_time = args.time
    
    if cut_time is None:
        # 尝试自动计算
        print("\n" + "="*60)
        print("未指定裁剪时间，尝试自动计算...")
        print("="*60 + "\n")
        
        cut_time = auto_calculate_cut_time(args.input)
        
        if cut_time is None:
            # 自动计算失败，提示用户手动输入
            print("\n" + "="*60)
            print("自动计算失败，请手动输入裁剪时间")
            print("="*60 + "\n")
            
            while True:
                try:
                    user_input = input("请输入要剪掉的时间（秒）: ").strip()
                    cut_time = float(user_input)
                    if cut_time < 0:
                        print("✗ 时间必须为非负数，请重新输入")
                        continue
                    break
                except ValueError:
                    print("✗ 无效输入，请输入数字")
                except KeyboardInterrupt:
                    print("\n用户取消操作")
                    return
    
    # 执行处理
    process_folder(args.input, args.output, cut_time, args.fps, args.d435i_to_d405, args.sample_config)


if __name__ == '__main__':
    main()
