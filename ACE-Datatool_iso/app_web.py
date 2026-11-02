import gradio as gr
import subprocess
import os
import signal
import time
import threading
import queue
import sys
from web_preview import (get_rgb_frame, get_depth_and_rgb, list_cameras, 
                         cleanup_device, stop_all_previews, get_preview_instance)
import json
from datetime import datetime
# 全局子进程对象
p_zed = None
p_realsense = None
p_egocap = None
p_tsa = None
exit_codes = {}

# Use maxsize to prevent unbounded memory growth, though typically small for this use case.
q_rs = queue.Queue(maxsize=1000)
q_zed = queue.Queue(maxsize=1000)
q_ego = queue.Queue(maxsize=1000)
q_terminal = queue.Queue(maxsize=1000)

# 任务时间戳记录
task_counter = 0
task_csv_path = None
current_record_name = None
task_file_writable = False
task_lock = threading.Lock()

def format_record_name(name):
    if not name:
        return str(time.time())
    
    parts = name.split('_', 1)
    if parts[0].isdigit():
        prefix = f"{int(parts[0]):06d}"
        if len(parts) > 1:
            return f"{prefix}_{parts[1]}"
        else:
            return prefix
    return name


def initialize_task_recording(record_name, auto_create_dir=False):
    """初始化任务时间戳记录,创建CSV文件"""
    global task_counter, task_csv_path, current_record_name, task_file_writable
    
    if not record_name:
        return "错误: 未指定录制名称"
    
    with task_lock:
        current_record_name = format_record_name(record_name)
        date_str = datetime.now().strftime("%Y%m%d")
        save_dir = os.path.join(date_str, current_record_name)
        
        # 如果是自动创建模式,先创建目录
        if auto_create_dir:
            os.makedirs(save_dir, exist_ok=True)
        elif not os.path.exists(save_dir):
            return f"错误: 录制目录不存在: {save_dir}"
        
        target_csv_path = os.path.join(save_dir, "task_timestamps.csv")

        # 幂等性检查：如果当前session已经为该路径初始化且可写，则直接返回成功
        if task_csv_path == target_csv_path and task_file_writable:
             return f"✅ 任务时间戳记录已初始化 (激活中): {target_csv_path}"
        
        # 如果已存在则不允许写入
        if os.path.exists(target_csv_path):
            task_csv_path = target_csv_path
            task_file_writable = False
            return f"❌ 时间戳文件已存在，将不会写入: {target_csv_path}"
        
        # 创建CSV文件,写入表头
        task_csv_path = target_csv_path
        task_counter = 0
        with open(task_csv_path, 'w') as f:
            f.write("task_id,event,timestamp\n")
        task_file_writable = True
        
        return f"✅ 任务时间戳记录已初始化: {task_csv_path}"


def record_task_start():
    """记录任务开始时间戳"""
    global task_counter, task_csv_path, task_file_writable
    
    with task_lock:
        if task_csv_path is None or not os.path.exists(os.path.dirname(task_csv_path)):
            return "错误: 任务时间戳记录未初始化。请先开始录制。", ""
        
        if not task_file_writable:
            return "错误: task_timestamps.csv 已存在。无法写入。", ""

        # 获取毫秒时间戳
        timestamp_ms = int(time.time() * 1000)
        
        # 记录任务开始，任务ID从1开始
        task_id = task_counter + 1
        
        try:
            with open(task_csv_path, 'a') as f:
                f.write(f"{task_id},start,{timestamp_ms}\n")
        except Exception as e:
            return f"写入 CSV 错误: {e}", ""
        
        status_msg = f"✅ 任务 #{task_id} 开始时间已记录: {timestamp_ms}"
        summary = f"当前任务: #{task_id}\nCSV: {task_csv_path}"
    
    return status_msg, summary


def record_task_end():
    """记录任务结束时间戳"""
    global task_counter, task_csv_path, task_file_writable
    
    with task_lock:
        if task_csv_path is None or not os.path.exists(os.path.dirname(task_csv_path)):
            return "错误: 任务时间戳记录未初始化。请先开始录制。", ""
        
        if not task_file_writable:
            return "错误: task_timestamps.csv 已存在。无法写入。", ""

        # 获取毫秒时间戳
        timestamp_ms = int(time.time() * 1000)
        
        # 使用当前计数器值作为任务ID
        task_id = task_counter + 1
        
        try:
            with open(task_csv_path, 'a') as f:
                f.write(f"{task_id},end,{timestamp_ms}\n")
        except Exception as e:
            return f"写入 CSV 错误: {e}", ""
        
        # 任务结束后递增计数器，准备下一个任务
        task_counter += 1
        
        status_msg = f"✅ 任务 #{task_id} 结束时间已记录: {timestamp_ms}"
        summary = f"已完成 {task_counter} 个任务\n下一个任务: #{task_counter + 1}\nCSV: {task_csv_path}"
    
    return status_msg, summary


camera_env = "PATH=/home/daxiao/.local/bin:/home/daxiao/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"

def stop_func():
    global p_zed, p_realsense, p_egocap, q_terminal, p_tsa

    # Function to gracefully terminate a process group
    def terminate_process(proc, name):
        if proc is not None:
            try:
                # Send SIGINT to the process group to terminate the child process and its descendants
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                proc.wait(timeout=5)  # Wait for termination
                print(f"{name} pipeline stopped.")
            except ProcessLookupError:
                print(f"{name} process already terminated.")
            except subprocess.TimeoutExpired:
                # If SIGINT didn't work, try SIGKILL
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()
                print(f"{name} pipeline killed after timeout.")
            except Exception as e:
                print(f"Error stopping {name}: {e}")
        return None

    p_zed = terminate_process(p_zed, "ZED")
    p_realsense = terminate_process(p_realsense, "RealSense")
    p_egocap = terminate_process(p_egocap, "EgoCap")
    p_tsa = terminate_process(p_tsa, "time analysis")

    print("All recording stopped!")

    # Clear the queue to prepare for the next run
    def clear_queue(q: queue.Queue):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
 
    clear_queue(q_rs)
    clear_queue(q_zed)
    clear_queue(q_ego)
 
    if isinstance(q_terminal, list):
        q_terminal.clear()
    else:                      
        clear_queue(q_terminal)

    return "All recording stopped! Ready for new start."


def run_pipeline(command, global_var_name, q):
    global p_zed, p_realsense, p_egocap
    print(f"Starting command: {command}")

    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        preexec_fn=os.setsid  # Create a new process group for easy termination
    )

    if global_var_name == "zed":
        p_zed = proc
    elif global_var_name == "realsense":
        p_realsense = proc
    elif global_var_name == "egocap":
        p_egocap = proc

    # Read output line by line and put into the queue
    for line in proc.stdout:
        # Check if the line ends with a newline, if not, add one for better formatting
        q.put(line.strip() + "\n")

    proc.wait()
    
    # Store exit code
    exit_codes[global_var_name] = proc.returncode

    # Once the process exits, reset the global pointer
    if global_var_name == "zed":
        p_zed = None
        q.put("ZED process finished.\n")
    elif global_var_name == "realsense":
        p_realsense = None
        q.put("RealSense process finished.\n")
    elif global_var_name == "egocap":
        p_egocap = None
        q.put("EgoCap process finished.\n")


def pipeline_generator(command, global_var_name, q):
    # Clear previous exit code
    exit_codes[global_var_name] = 0

    # Clear the queue before starting a new run
    with q.mutex:
        q.queue.clear()

    # Start the subprocess thread
    threading.Thread(target=run_pipeline, args=(command, global_var_name, q), daemon=True).start()

    output_text = f"{command}\n"
    while True:
        try:
            # Wait for content from the queue
            line = q.get(timeout=0.1)

            # Accumulate the output text
            output_text += line

            # Yield the full accumulated text to Gradio
            yield output_text, ""

            q.task_done()
        except queue.Empty:
            # If the process has ended and the queue is empty, break the loop
            is_process_running = (p_zed is not None and global_var_name == "zed") \
                or (p_realsense is not None and global_var_name == "realsense") \
                or (p_egocap is not None and global_var_name == "egocap") \

            if not is_process_running and q.empty():
                code = exit_codes.get(global_var_name, 0)
                if code is not None and code != 0:
                    # Ignore negative codes (signals) usually meaning manual stop or kill
                    if code > 0:
                         # Yield the error message to the hidden error box
                         yield output_text, f"❌ {global_var_name} Recording Failed!\n(Exit Code: {code})\nPlease check logs."
                break

            # Continue looping if the process is running or queue might get content soon
            time.sleep(0.1)


# ZED pipeline
def zed_pipeline(record_name):
    record_name = format_record_name(record_name)
    
    # 自动生成config.json
    auto_generate_config(record_name)
    
    # 自动初始化keypoint记录
    initialize_task_recording(record_name, auto_create_dir=True)
    
    # Use the unified generator
    yield from pipeline_generator(
        command=f"{camera_env} python zed_record.py --record_name {record_name}",
        global_var_name="zed",
        q=q_zed,
    )


# RealSense pipeline
def realsense_pipeline(record_name):
    record_name = format_record_name(record_name)
    date_str = datetime.now().strftime("%Y%m%d")
    
    # 自动生成config.json
    auto_generate_config(record_name)
    
    # 自动初始化keypoint记录
    initialize_task_recording(record_name, auto_create_dir=True)

    # Use the unified generator
    yield from pipeline_generator(
        command=f"{camera_env} python rs_record.py --record_name {date_str}/{record_name}",
        global_var_name="realsense",
        q=q_rs
    )


# EgoCap pipeline
def egocap_pipeline(record_name):
    record_name = format_record_name(record_name)
    date_str = datetime.now().strftime("%Y%m%d")
    
    # 自动生成config.json
    auto_generate_config(record_name)
    
    # 自动初始化keypoint记录
    initialize_task_recording(record_name, auto_create_dir=True)

    # Use the unified generator
    yield from pipeline_generator(
        command=f"cd egocap && ./egocap ../{date_str}/{record_name}/egocap",
        global_var_name="egocap",
        q=q_ego
    )


# RealSense pipeline
def run_timestamp_analysis(command, q):
    global p_tsa
    with q.mutex:
        q.queue.clear()
    p_tsa = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        preexec_fn=os.setsid  # Create a new process group for easy termination
    )

    # Read output line by line and put into the queue
    for line in p_tsa.stdout:
        # Check if the line ends with a newline, if not, add one for better formatting
        if line.rfind('\r'):
            q.put(line.strip()+"\r")
        else:
            q.put(line.strip() + "\n")

    p_tsa.wait()
    p_tsa = None
    q.put("timestamp analysis finished.\n")


def timestamp_analysis_pipe(record_name, camera_name):
    record_name = format_record_name(record_name)
    date_str = datetime.now().strftime("%Y%m%d")
    # Clear the queue before starting a new run
    q = q_terminal
    with q.mutex:
        q.queue.clear()
    print(camera_name)
    if camera_name == 'realsense':
        command = f'{camera_env} python timestamp_analysis.py --rs_data_path {date_str}/{record_name}/realsense'
    if camera_name == 'zed':
        command = f'{camera_env} python timestamp_analysis.py --zed_data_path {date_str}/{record_name}/zed'
    if camera_name == 'all':
        command = f'{camera_env} python timestamp_analysis.py --rs_data_path {date_str}/{record_name}/realsense --zed_data_path {date_str}/{record_name}/zed'
    # Start the subprocess thread
    threading.Thread(target=run_timestamp_analysis, args=(command, q), daemon=True).start()

    output_text = f"{command}\n"

    while True:
        try:
            # Wait for content from the queue
            line = q.get(timeout=5)

            # Accumulate the output text

            output_text += line 

            # Yield the full accumulated text to Gradio
            yield output_text

            q.task_done()
        except queue.Empty:
            # If the process has ended and the queue is empty, break the loop
            is_process_running = p_tsa is not None

            if not is_process_running and q.empty():
                break

            # Continue looping if the process is running or queue might get content soon
            time.sleep(0.1)


# Batch timestamp analysis pipeline
def batch_timestamp_analysis_pipe():
    # Use the existing bash script logic but implemented in python or call the script
    command = "bash batch_timestamp_analysis.sh"
    
    # Process the command similarly to other pipelines
    q = q_terminal
    with q.mutex:
        q.queue.clear()
        
    threading.Thread(target=run_timestamp_analysis, args=(command, q), daemon=True).start()

    output_text = f"{command}\n"
    summary_text = "File, Type, FPS\n--------------------------------------------------\n"

    while True:
        try:
            line = q.get(timeout=5)
            output_text += line 
            
            # Parse line for FPS summary
            if "FPS_SUMMARY:" in line:
                # Line format: ... FPS_SUMMARY: 000001, ZED, 30.01 ...
                # or just clean line: FPS_SUMMARY: 000001, ZED, 30.01
                clean_line = line.strip()
                if clean_line.startswith("FPS_SUMMARY:"):
                    # Extract the content after FPS_SUMMARY:
                    content = clean_line.split("FPS_SUMMARY:", 1)[1].strip()
                    summary_text += content + "\n"
            
            # Add separator logic if needed, but script prints lines.
            # The script output might contain color codes or other noise, but grep usually clean.
            # If the script prints "--------------------------------------------------", we can capture that too if we want.
            if "--------------------------------------------------" in line:
                 summary_text += "--------------------------------------------------\n"

            yield output_text, summary_text
            q.task_done()
        except queue.Empty:
            if p_tsa is None and q.empty():
                break
            time.sleep(0.1)



# Helper function to read the FPS summary file content
def get_fps_summary_content():
    summary_file = "fps_summary.txt"
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r") as f:
                content = f.read()
            return content
        except Exception as e:
            return f"Error reading summary file: {e}"
    else:
        return "Summary file not found (run batch analysis first)."

# Batch timestamp analysis pipeline wrapper
def batch_timestamp_analysis_pipe_wrapper():
    # Call the original generator
    # We yield the same output, but at the end we also update the summary box
    for output in batch_timestamp_analysis_pipe():
        yield output, gr.update() # Keep summary box as is during processing
    
    # After process finishes, read the file
    final_content = get_fps_summary_content()
    yield "Analysis Complete.\n", final_content


def restart_zed():
    passwd = "3456999"
    command = 'systemctl restart zed_x_daemon.service'
    res =  os.system('echo %s | sudo -S %s' % (passwd, command))
    if res == 0:
      output_text = "restart successed"
    else:
      output_text = "restart failed"
    return output_text


def reset_realsense():
    """重置 RealSense 相机权限和清理进程"""
    passwd = "3456999"
    output_text = "=== Resetting RealSense ===\n"
    
    # 1. 检查并杀死旧的 rs_record 进程
    try:
        check_cmd = "ps aux | grep '[r]s_record' | awk '{print $2}'"
        pids = os.popen(check_cmd).read().strip()
        if pids:
            output_text += f"Found existing processes: {pids}\n"
            kill_cmd = f"kill -9 {pids}"
            res = os.system('echo %s | sudo -S %s' % (passwd, kill_cmd))
            if res == 0:
                output_text += "✓ Killed old processes\n"
            else:
                output_text += "✗ Failed to kill processes\n"
        else:
            output_text += "✓ No old processes found\n"
    except Exception as e:
        output_text += f"✗ Process check failed: {e}\n"
    
    # 2. 设置 USB 权限
    try:
        chmod_cmd = 'chmod 666 /dev/video*'
        res = os.system('echo %s | sudo -S %s' % (passwd, chmod_cmd))
        if res == 0:
            output_text += "✓ USB permissions updated\n"
        else:
            output_text += "✗ Permission update failed\n"
    except Exception as e:
        output_text += f"✗ Permission update failed: {e}\n"
    
    # 3. 重置 USB 设备（可选）
    try:
        # 使用 usbreset 工具重置 RealSense 设备
        reset_cmd = 'bash -c "for dev in /sys/bus/usb/devices/*/idVendor; do if [ \$(cat \$dev) == \"8086\" ]; then echo \$(dirname \$dev) | xargs -I{} sh -c \"echo 0 > {}/authorized && echo 1 > {}/authorized\"; fi; done"'
        res = os.system('echo %s | sudo -S %s' % (passwd, reset_cmd))
        if res == 0:
            output_text += "✓ USB devices reset\n"
        else:
            output_text += "Note: USB reset command executed (may not support)\n"
    except Exception as e:
        output_text += f"Note: USB reset skipped: {e}\n"
    
    output_text += "\n=== Reset Complete ===\n"
    output_text += "You can now try recording again."
    return output_text


def shutdown_system():
    """关闭系统"""
    passwd = "3456999"
    output_text = "=== 正在关闭系统 ===\n"
    output_text += "⚠️ 系统将在 60 秒后关机...\n"
    output_text += "请确保所有数据已保存！\n"
    
    try:
        # 先停止所有录制
        stop_func()
        time.sleep(1)
        
        # 执行关机命令 (60秒后关机)
        command = 'shutdown -h 1 "System shutdown requested from web interface"'
        res = os.system('echo %s | sudo -S %s' % (passwd, command))
        if res == 0:
            output_text += "✓ 关机命令已发送\n"
            output_text += "系统将在 60 秒后关闭\n"
        else:
            output_text += "✗ 关机命令执行失败\n"
    except Exception as e:
        output_text += f"✗ 关机失败: {e}\n"
    
    return output_text


def get_recorded_folders():
    """获取当日目录下的录制任务文件夹列表，RealSense_Size 返回格式化大小列表"""
    date_str = datetime.now().strftime("%Y%m%d")
    base_path = date_str

    if not os.path.exists(base_path):
        return [[f"No recordings found for today ({date_str})", "", "", "", ""]]

    def _fmt_bytes(b):
        if b < 1024:
            return f"{b} B"
        for u in ["KB", "MB", "GB"]:
            b /= 1024.0
            if b < 1024:
                return f"{b:.2f} {u}"
        return f"{b:.2f} TB"

    def _sum_size_by_ext(folder, ext):
        if not os.path.isdir(folder):
            return 0
        total = 0
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(ext.lower()):
                    fp = os.path.join(root, f)
                    try:
                        total += os.path.getsize(fp)
                    except Exception:
                        pass
        return total

    rows = []
    try:
        for entry in sorted(os.listdir(base_path)):
            task_path = os.path.join(base_path, entry)
            if not os.path.isdir(task_path):
                continue

            # 1. 总大小
            total_size = 0
            for root, _, files in os.walk(task_path):
                for f in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, f))
                    except Exception:
                        pass

            # 2. RealSense：返回所有 *_rgb.mp4 的大小列表
            realsense_sizes = []
            realsense_dir = os.path.join(task_path, "realsense")
            if os.path.isdir(realsense_dir):
                for f in os.listdir(realsense_dir):
                    if f.lower().endswith("_rgb.mp4"):
                        fp = os.path.join(realsense_dir, f)
                        try:
                            size_bytes = os.path.getsize(fp)
                            realsense_sizes.append(_fmt_bytes(size_bytes))
                        except Exception:
                            pass

            # 3. EgoCap：egocap/ 下任意子目录里所有 *.bin
            egocap_bin_total = 0
            egocap_root = os.path.join(task_path, "egocap")
            if os.path.isdir(egocap_root):
                for sub in os.listdir(egocap_root):
                    sub_path = os.path.join(egocap_root, sub)
                    if os.path.isdir(sub_path):
                        egocap_bin_total += _sum_size_by_ext(sub_path, ".bin")

            # 4. 修改时间
            mtime = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(task_path))
            )

            rows.append([
                entry,
                mtime,
                _fmt_bytes(total_size),
                realsense_sizes,              # 列表: ["1.23 MB", "4.56 GB"]
                _fmt_bytes(egocap_bin_total),
            ])
        rows.sort(key=lambda x: x[1], reverse=True)
        
    except Exception as e:
        return [[f"Error: {e}", "", "", "", ""]]
    
    if not rows:
        return [[f"No recordings found in {date_str}/", "", "", "", ""]]
    return rows


def get_next_task_number():
    """获取下一个可用的任务序号"""
    # 使用当前日期作为base_path
    date_str = datetime.now().strftime("%Y%m%d")
    base_path = date_str
    
    if not os.path.exists(base_path):
        return "000001"
    
    max_num = 0
    try:
        # 遍历当日日期文件夹下的任务文件夹
        for task_folder in os.listdir(base_path):
            task_path = os.path.join(base_path, task_folder)
            if not os.path.isdir(task_path):
                continue
            
            # 提取数字部分
            parts = task_folder.split('_', 1)
            if parts[0].isdigit():
                num = int(parts[0])
                max_num = max(max_num, num)
    except Exception as e:
        print(f"Error getting next task number: {e}")
    
    return f"{max_num + 1:06d}"


def detect_all_cameras():
    """检测所有连接的相机并返回相机信息（排除 EgoCap 头部相机）"""
    all_cameras = list_cameras("all")
    
    camera_info = []
    for cam in all_cameras:
        cam_type = cam['type']
        sn = cam['sn']
        name = cam['name']
        
        # 跳过 EgoCap 头部相机（通常 SN 为 00000001）
        if cam_type == 'egocap' or sn == '00000001':
            continue
        
        if cam_type == 'realsense':
            camera_info.append({
                'type': 'realsense_d435i',
                'sn': sn,
                'name': name,
                'resolution': [848, 480],
                'fps': 15,
                'modal': ['rgb', 'depth']
            })
        elif cam_type == 'zed':
            camera_info.append({
                'type': 'zed_x_one_gs',
                'sn': sn,
                'name': name,
                'resolution': [960, 600],
                'fps': 15,
                'modal': ['rgb']
            })
    
    return camera_info


def generate_config_json(task_no, task_name, camera_locations, record_name):
    """基于 template_config.json 生成 config.json 文件"""
    try:
        # 获取当前日期
        date_str = datetime.now().strftime("%Y%m%d")
        
        # 读取模板 template_config.json
        template_path = "template_config.json"
        if not os.path.exists(template_path):
            return f"❌ 模板文件不存在: {template_path}"
        
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                content = f.read()
                # 移除注释
                lines = content.split('\n')
                clean_lines = []
                for line in lines:
                    if '#' in line:
                        line = line[:line.index('#')]
                    clean_lines.append(line)
                clean_content = '\n'.join(clean_lines)
                config = json.loads(clean_content)
        except Exception as e:
            return f"❌ 无法读取模板文件: {str(e)}"
        
        # 更新用户输入的信息
        config["no"] = task_no
        # 只有当有新输入时才更新task，否则保留模板中的原始task
        if task_name:
            config["task"] = [task_name]
        # 如果模板中没有task字段，则使用默认值
        elif "task" not in config:
            config["task"] = ["recording"]
        config["date"] = date_str
        
        # 检测所有相机
        cameras = detect_all_cameras()
        
        if not cameras:
            return "❌ 未检测到任何相机！"
        
        # 更新 location（仅当用户在 Gradio 中有新输入时更新，否则保留模板中的原始配置）
        updated_count = 0
        kept_count = 0
        for cam in cameras:
            sn = cam['sn']
            location = camera_locations.get(sn, "")
            
            # 如果没有新输入，跳过更新，保留模板中的原始location
            if not location:
                # 检查模板中是否已有此相机的配置
                for key, device in config.get('device', {}).get('ego', {}).items():
                    if device.get('sn') == sn:
                        kept_count += 1
                        break
                continue
            
            # 在模板的 ego 设备中查找匹配的相机并更新 location
            for key, device in config.get('device', {}).get('ego', {}).items():
                if device.get('sn') == sn:
                    device['location'] = location
                    updated_count += 1
                    break
        
        # 保存到文件
        record_name_formatted = format_record_name(record_name) if record_name else task_no
        # 生成路径格式: 日期/任务序号/config.json
        output_dir = f"{date_str}/{record_name_formatted}"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "config.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        status_msg = f"✅ 配置文件已生成: {output_path}\n检测到 {len(cameras)} 个相机"
        if updated_count > 0:
            status_msg += f", 已更新 {updated_count} 个相机的 location"
        if kept_count > 0:
            status_msg += f", 保留 {kept_count} 个相机的模板配置"
        return status_msg
    
    except Exception as e:
        return f"❌ 生成配置文件失败: {str(e)}"


def auto_generate_config(record_name, task_name=""):
    """录制时自动生成config.json,使用默认location"""
    try:
        # 检测所有相机
        cameras = detect_all_cameras()
        
        # 使用空的location字典(保持模板原有配置)
        camera_locations = {}
        
        result = generate_config_json(
            task_no=record_name,
            task_name=task_name if task_name else "recording",
            camera_locations=camera_locations,
            record_name=record_name
        )
        
        print(f"[Auto Config] {result}")
        return result
    except Exception as e:
        print(f"[Auto Config Error] {e}")
        return f"Config generation failed: {e}"


def update_camera_location_ui():
    """动态生成相机 location 选择界面"""
    cameras = detect_all_cameras()
    
    if not cameras:
        return [gr.update(visible=False)] * 10 + ["未检测到相机"]
    
    location_choices = [
        "chest_left", "chest_right",
        "hand_palm_left", "hand_palm_right",
        "hand_back_left", "hand_back_right",
        "waist_left", "waist_right"
    ]
    
    updates = []
    for i in range(10):
        if i < len(cameras):
            cam = cameras[i]
            label = f"{cam['type']} (SN: {cam['sn']})"
            updates.append(gr.update(visible=True, label=label, choices=location_choices, value=""))
        else:
            updates.append(gr.update(visible=False))
    
    status = f"检测到 {len(cameras)} 个相机"
    updates.append(status)
    
    return updates


def web_preview_update(device_index: int = 0, camera_type: str = "realsense"):
    """更新 Web 预览帧"""
    rgb = get_rgb_frame(device_index, camera_type)
    return rgb


def web_preview_dual_update(device_index: int = 0, camera_type: str = "realsense"):
    """更新 Web 预览（RGB + Depth）"""
    rgb, depth = get_depth_and_rgb(device_index, camera_type)
    return rgb, depth


def start_web_preview(device_index: int = 0, camera_type: str = "realsense"):
    """启动 Web 预览"""
    try:
        device_idx = int(device_index)
        
        # 先清理该设备的旧实例
        cleanup_device(device_idx, camera_type)
        time.sleep(0.3)
        
        # 创建新实例并启动
        preview = get_preview_instance(device_idx, show_depth=True, auto_start=True, camera_type=camera_type)
        
        if preview and preview.is_running:
            return f"{camera_type.upper()} preview for camera {device_idx} started!"
        else:
            return f"Failed to start {camera_type} preview for camera {device_idx}"
    except Exception as e:
        return f"Error starting {camera_type} preview: {e}"


def stop_web_preview(device_index: int = 0, camera_type: str = "realsense"):
    """停止 Web 预览"""
    try:
        cleanup_device(int(device_index), camera_type)
        return f"{camera_type.upper()} preview for camera {device_index} stopped!"
    except Exception as e:
        return f"Error stopping {camera_type} preview: {e}"


# 启动时清理所有旧的预览实例
try:
    stop_all_previews()
except:
    pass

# 获取可用的相机列表,包含序列号信息
available_cameras = list_cameras("all")  # 获取所有类型的相机
rs_cameras = [cam for cam in available_cameras if cam['type'] == 'realsense']
zed_cameras = [cam for cam in available_cameras if cam['type'] == 'zed']

# 生成包含序列号的选项
rs_camera_choices = [f"{cam['index']} (SN: {cam['sn']})" for cam in rs_cameras]
rs_camera_values = [f"{cam['index']}" for cam in rs_cameras]
zed_camera_choices = [f"{cam['index']} (SN: {cam['sn']})" for cam in zed_cameras]
zed_camera_values = [f"{cam['index']}" for cam in zed_cameras]

if not rs_camera_choices:
    rs_camera_choices = ["0 (No camera detected)"]
    rs_camera_values = ["0"]
if not zed_camera_choices:
    zed_camera_choices = []
    zed_camera_values = []

print(f"[app.py] Available RealSense cameras: {rs_camera_choices}")
print(f"[app.py] Available ZED cameras: {zed_camera_choices}")
print(f"[app.py] Available ZED cameras: {zed_camera_choices}")


# Gradio UI
with gr.Blocks(title="Jetson Orin AGX - Camera Control") as demo:
    gr.Markdown("# Jetson Orin AGX - Camera Control")
    gr.Markdown("Control the video recording process for ZED and RealSense cameras.")

    # Config Generator 区域 - 移到最前面
    with gr.Accordion("📝 Camera Configuration (Optional)", open=False):
        gr.Markdown("**Config.json will be auto-generated when recording starts.** Optionally configure camera locations here.")
        
        detect_cameras_btn = gr.Button("🔍 Detect Cameras", size="sm")
        config_status = gr.Textbox(
            label="Detected Cameras",
            value="Click 'Detect Cameras' to see connected cameras",
            interactive=False,
            lines=3
        )
        
        gr.Markdown("### Configure Camera Locations (Optional)")
        
        # 创建最多10个相机的 location 选择器（动态显示）
        camera_location_dropdowns = []
        with gr.Column():
            for i in range(10):
                dropdown = gr.Dropdown(
                    label=f"Camera {i}",
                    choices=[],
                    visible=False,
                    interactive=True
                )
                camera_location_dropdowns.append(dropdown)
        
        with gr.Row():
            manual_config_record_name = gr.Textbox(
                label="Recording Name",
                placeholder="Enter recording name to generate config",
                value=""
            )
            generate_config_btn = gr.Button("📄 Generate Config Manually", variant="secondary")
        
        config_output = gr.Textbox(
            label="Result",
            interactive=False,
            lines=3
        )
    
    # Web Preview 区域 - 移到Config下面
    with gr.Accordion("📱 Web Real-time Preview", open=False):
        gr.Markdown("## Live Camera Preview on Web")
        gr.Markdown("View camera streams in real-time (RGB + Depth). Works on phone/tablet/desktop!")
        
        with gr.Tab("RealSense Preview"):
            with gr.Row():
                with gr.Column(scale=3):
                    rs_web_camera_choice = gr.Dropdown(
                        choices=rs_camera_choices,
                        value=rs_camera_values[0] if rs_camera_values else "0",
                        label="Select RealSense Camera (with Serial Number)",
                        interactive=True
                    )
                with gr.Column(scale=1):
                    rs_refresh_cam_btn = gr.Button("🔄 Refresh")
            
            with gr.Row():
                rs_web_preview_btn = gr.Button("Start RealSense Preview", variant="primary")
                rs_web_stop_btn = gr.Button("Stop RealSense Preview", variant="secondary")
            
            with gr.Row():
                rs_web_rgb_output = gr.Image(label="RGB Stream", type="numpy")
                rs_web_depth_output = gr.Image(label="Depth Stream", type="numpy")
            
            rs_web_status = gr.Textbox(
                label="Status",
                interactive=False,
                value="Click 'Start RealSense Preview' to begin"
            )
            
            rs_timer = gr.Timer(value=0)
        
        if zed_camera_choices:  # 只有检测到ZED相机时才显示
            with gr.Tab("ZED Preview"):
                with gr.Row():
                    with gr.Column(scale=3):
                        zed_web_camera_choice = gr.Dropdown(
                            choices=zed_camera_choices,
                            value=zed_camera_values[0] if zed_camera_values else "0",
                            label="Select ZED Camera (with Serial Number)",
                            interactive=True
                        )
                    with gr.Column(scale=1):
                        zed_refresh_cam_btn = gr.Button("🔄 Refresh")
                
                with gr.Row():
                    zed_web_preview_btn = gr.Button("Start ZED Preview", variant="primary")
                    zed_web_stop_btn = gr.Button("Stop ZED Preview", variant="secondary")
                
                with gr.Row():
                    zed_web_rgb_output = gr.Image(label="RGB Stream", type="numpy")
                    zed_web_depth_output = gr.Image(label="Depth Stream", type="numpy")
                
                zed_web_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Click 'Start ZED Preview' to begin"
                )
                
                zed_timer = gr.Timer(value=0)

    with gr.Row():
        record_name_box = gr.Textbox(
            label="Record Item Name (Required)",
            placeholder="Enter recording name, e.g., 000001 or test_recording"
        )
    
    run_btn1 = gr.Button("Run RealSense Pipeline", elem_id="btn1_id")
    run_btn2 = gr.Button("Run ZED Pipeline", elem_id="btn2_id")
    run_btn3 = gr.Button("Run EgoCap Pipeline", elem_id="btn3_id")
    run_btn4 = gr.Button("Run all")
    run_btn4_5 = gr.Button("Run RealSense + EgoCap")
    run_btn5 = gr.Button("Stop all")
    with gr.Row():
        run_btn6 = gr.Button("timestamp analysis")
        btn6_choice = gr.Radio(
            show_label=False,
            choices=["realsense", "zed", "all"],
            value=["realsense"],
        )
    
    # Task Timestamp Recording Section
    with gr.Accordion("📍 任务时间戳记录", open=False):
        gr.Markdown("**录制开始时自动初始化。** 在任务开始和结束时分别点击对应按钮记录时间戳。可记录多个任务。")
        with gr.Row():
            task_start_btn = gr.Button("🟢 记录任务开始", variant="primary", size="lg")
            task_end_btn = gr.Button("🔴 记录任务结束", variant="stop", size="lg")
        task_status_box = gr.Textbox(
            label="状态",
            interactive=False,
            lines=2,
            value="就绪。开始录制时将创建 CSV 文件。"
        )
        task_summary_box = gr.Textbox(
            label="摘要",
            interactive=False,
            lines=2,
            value=""
        )

    # Batch Analysis Section
    with gr.Accordion("Batch Analysis", open=False):
        run_batch_btn = gr.Button("Batch Analysis (All Raw Folders)")
        batch_summary_box = gr.Textbox(
            label="FPS Summary Result",
            interactive=False,
            lines=10,
            value="Run batch analysis to see results..."
        )

    with gr.Row():
        run_btn7 = gr.Button("Restart ZED")
        run_btn8 = gr.Button("Reset RealSense")
    
    # Shutdown Section - 单独折叠防止误触
    with gr.Accordion("⚠️ 系统关机 (危险操作)", open=False):
        gr.Markdown("**警告：点击关机按钮后，系统将在 60 秒后自动关闭！**")
        shutdown_btn = gr.Button("⚠️ 立即关机", variant="stop")

    # Hidden text box for triggering alerts
    hidden_error_box = gr.Textbox(visible=False, label="Hidden Error")

    # JavaScript to show alert when hidden_error_box changes
    hidden_error_box.change(
        fn=None,
        inputs=[hidden_error_box],
        js="(val) => { if (val && val.length > 0) { alert(val); } }"
    )

    output_box = gr.Textbox(
        label="Terminal Output",
        interactive=False,
        lines=5,
        autoscroll=True,
    )
    output_rs = gr.Textbox(
        label="RealSense Output",
        interactive=False,
        lines=5,
        autoscroll=True,
    )
    output_zed = gr.Textbox(
        label="ZED Output",
        interactive=False,
        lines=5,
        autoscroll=True,
    )
    output_egocap = gr.Textbox(
        label="EgoCap Output",
        interactive=False,
        lines=5,
        autoscroll=True,
    )

    with gr.Accordion("Recorded Data Folders", open=False):
        with gr.Row():
            gr.Markdown(f"### Today's Recordings ({datetime.now().strftime('%Y%m%d')})")
            refresh_btn = gr.Button("Refresh List")
        
        recorded_table = gr.Dataframe(
            headers=["Recording Name", "Modification Time", "Size", "RealSense_Size", "EgoCap_Size"],
            datatype=["str", "str", "str", "str", "str"],
            label="Today's Recording List",
            interactive=False,
            value=get_recorded_folders  # Initial load
        )
    
    # 定义Web Preview更新函数
    def update_rs_frames(device_idx):
        """实时更新RealSense预览帧"""
        if not device_idx: return None, None, "No camera selected"
        # 从"index (SN: xxx)"格式中提取index
        idx = device_idx.split()[0] if ' ' in device_idx else device_idx
        rgb, depth = get_depth_and_rgb(int(idx), "realsense")
        return rgb, depth, f"Streaming from RealSense camera {idx}..."
    
    def update_zed_frames(device_idx):
        """实时更新ZED预览帧"""
        if not device_idx: return None, None, "No camera selected"
        idx = device_idx.split()[0] if ' ' in device_idx else device_idx
        rgb, depth = get_depth_and_rgb(int(idx), "zed")
        return rgb, depth, f"Streaming from ZED camera {idx}..."
    
    def on_start_rs_preview(device_idx):
        if not device_idx: return "No camera selected", gr.Timer(value=0)
        idx = device_idx.split()[0] if ' ' in device_idx else device_idx
        result = start_web_preview(int(idx), "realsense")
        return result, gr.Timer(value=0.067)
    
    def on_stop_rs_preview(device_idx):
        if not device_idx: return "No camera selected", gr.Timer(value=0)
        idx = device_idx.split()[0] if ' ' in device_idx else device_idx
        result = stop_web_preview(int(idx), "realsense")
        return result, gr.Timer(value=0)
    
    def on_start_zed_preview(device_idx):
        if not device_idx: return "No camera selected", gr.Timer(value=0)
        idx = device_idx.split()[0] if ' ' in device_idx else device_idx
        result = start_web_preview(int(idx), "zed")
        return result, gr.Timer(value=0.067)
    
    def on_stop_zed_preview(device_idx):
        if not device_idx: return "No camera selected", gr.Timer(value=0)
        idx = device_idx.split()[0] if ' ' in device_idx else device_idx
        result = stop_web_preview(int(idx), "zed")
        return result, gr.Timer(value=0)
    
    def on_refresh_cameras(cam_type="realsense"):
        try:
            cams = list_cameras(cam_type)
            new_choices = [f"{cam['index']} (SN: {cam['sn']})" for cam in cams]
            new_values = [f"{cam['index']}" for cam in cams]
            if not new_choices:
                if cam_type == "realsense":
                    new_choices = ["0 (No camera detected)"]
                    new_values = ["0"]
                else:
                    new_choices = []
                    new_values = []
                val = new_values[0] if new_values else None
                status = f"No {cam_type} cameras found"
            else:
                val = new_choices[0]  # 显示带SN的选项
                status = f"Found {len(new_choices)} {cam_type} camera(s)"
            
            return gr.Dropdown(choices=new_choices, value=val), status
        except Exception as e:
            return gr.Dropdown(), f"Error refreshing: {e}"
    
    # RealSense Web Preview事件绑定
    rs_web_preview_btn.click(
        fn=on_start_rs_preview,
        inputs=[rs_web_camera_choice],
        outputs=[rs_web_status, rs_timer]
    )
    
    rs_web_stop_btn.click(
        fn=on_stop_rs_preview,
        inputs=[rs_web_camera_choice],
        outputs=[rs_web_status, rs_timer]
    )
    
    rs_refresh_cam_btn.click(
        fn=lambda: on_refresh_cameras("realsense"),
        inputs=[],
        outputs=[rs_web_camera_choice, rs_web_status]
    )
    
    rs_timer.tick(
        fn=update_rs_frames,
        inputs=[rs_web_camera_choice],
        outputs=[rs_web_rgb_output, rs_web_depth_output, rs_web_status]
    )
    
    # ZED Web Preview事件绑定（如果有ZED相机）
    if zed_camera_choices:
        zed_web_preview_btn.click(
            fn=on_start_zed_preview,
            inputs=[zed_web_camera_choice],
            outputs=[zed_web_status, zed_timer]
        )
        
        zed_web_stop_btn.click(
            fn=on_stop_zed_preview,
            inputs=[zed_web_camera_choice],
            outputs=[zed_web_status, zed_timer]
        )
        
        zed_refresh_cam_btn.click(
            fn=lambda: on_refresh_cameras("zed"),
            inputs=[],
            outputs=[zed_web_camera_choice, zed_web_status]
        )
        
        zed_timer.tick(
            fn=update_zed_frames,
            inputs=[zed_web_camera_choice],
            outputs=[zed_web_rgb_output, zed_web_depth_output, zed_web_status]
        )


    js_code = """
    () => {
        console.log("start...");
        document.getElementById('btn1_id').click();
        document.getElementById('btn2_id').click();
        document.getElementById('btn3_id').click();
    }
    """
    
    js_code_rs_ego = """
    () => {
        console.log("Starting RealSense + EgoCap...");
        document.getElementById('btn1_id').click();
        document.getElementById('btn3_id').click();
    }
    """

    run_btn1.click(fn=realsense_pipeline, inputs=[record_name_box], outputs=[output_rs, hidden_error_box])
    run_btn2.click(fn=zed_pipeline, inputs=[record_name_box], outputs=[output_zed, hidden_error_box])
    run_btn3.click(fn=egocap_pipeline, inputs=[record_name_box], outputs=[output_egocap, hidden_error_box])
    run_btn4.click(fn=None, inputs=None, outputs=None, js=js_code)
    run_btn4_5.click(fn=None, inputs=None, outputs=None, js=js_code_rs_ego)
    run_btn5.click(fn=stop_func, inputs=[], outputs=[output_box])
    run_btn6.click(fn=timestamp_analysis_pipe, inputs=[record_name_box, btn6_choice], outputs=[output_box])
    
    # Task timestamp recording button events
    task_start_btn.click(fn=record_task_start, inputs=[], outputs=[task_status_box, task_summary_box])
    task_end_btn.click(fn=record_task_end, inputs=[], outputs=[task_status_box, task_summary_box])
    
    run_batch_btn.click(fn=batch_timestamp_analysis_pipe, inputs=[], outputs=[output_box, batch_summary_box])
    run_btn7.click(fn=restart_zed, inputs=[], outputs=[output_box])
    run_btn8.click(fn=reset_realsense, inputs=[], outputs=[output_box])
    shutdown_btn.click(fn=shutdown_system, inputs=[], outputs=[output_box])
    refresh_btn.click(fn=get_recorded_folders, inputs=[], outputs=[recorded_table])
    
    # Config Generator 事件绑定
    detect_cameras_btn.click(
        fn=update_camera_location_ui,
        inputs=[],
        outputs=camera_location_dropdowns + [config_status]
    )
    
    def generate_config_wrapper_manual(record_name, *locations):
        """手动生成配置的包装函数"""
        cameras = detect_all_cameras()
        camera_locations = {}
        
        for i, cam in enumerate(cameras):
            if i < len(locations) and locations[i]:
                camera_locations[cam['sn']] = locations[i]
        
        return generate_config_json(
            task_no=record_name,
            task_name="recording",
            camera_locations=camera_locations,
            record_name=record_name
        )
    
    generate_config_btn.click(
        fn=generate_config_wrapper_manual,
        inputs=[manual_config_record_name] + camera_location_dropdowns,
        outputs=[config_output]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=9067, share=True)

