import gradio as gr
import subprocess
import os
import signal
import time
import threading
import queue
import sys

# 全局子进程对象
p_zed = None
p_realsense = None
p_egocap = None
p_tsa = None

# Use maxsize to prevent unbounded memory growth, though typically small for this use case.
q_rs = queue.Queue(maxsize=1000)
q_zed = queue.Queue(maxsize=1000)
q_ego = queue.Queue(maxsize=1000)
q_terminal = queue.Queue(maxsize=1000)

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
            yield output_text

            q.task_done()
        except queue.Empty:
            # If the process has ended and the queue is empty, break the loop
            is_process_running = (p_zed is not None and global_var_name == "zed") \
                or (p_realsense is not None and global_var_name == "realsense") \
                or (p_egocap is not None and global_var_name == "egocap") \

            if not is_process_running and q.empty():
                break

            # Continue looping if the process is running or queue might get content soon
            time.sleep(0.1)


# ZED pipeline
def zed_pipeline(record_name):
    if not record_name:
        record_name = str(time.time())
    # Use the unified generator
    yield from pipeline_generator(
        command=f"{camera_env} python zed_record.py --record_name {record_name}",
        global_var_name="zed",
        q=q_zed,
    )


# RealSense pipeline
def realsense_pipeline(record_name):
    if not record_name:
        record_name = str(time.time())

    # Use the unified generator
    yield from pipeline_generator(
        command=f"{camera_env} python rs_record.py --record_name {record_name}",
        global_var_name="realsense",
        q=q_rs
    )


# EgoCap pipeline
def egocap_pipeline(record_name):
    if not record_name:
        record_name = str(time.time())

    # Use the unified generator
    yield from pipeline_generator(
        command=f"cd egocap && ./egocap ../raw/{record_name}/egocap",
        global_var_name="egocap",
        q=q_ego
    )


# RealSense pipeline
def run_timestamp_analysis(command, q):

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
    # Clear the queue before starting a new run
    q = q_terminal
    with q.mutex:
        q.queue.clear()
    print(camera_name)
    if camera_name == 'realsense':
        command = f'{camera_env} python timestamp_analysis.py --rs_data_path raw/{record_name}/realsense'
    if camera_name == 'zed':
        command = f'{camera_env} python timestamp_analysis.py --zed_data_path raw/{record_name}/zed'
    if camera_name == 'all':
        command = f'{camera_env} python timestamp_analysis.py --rs_data_path raw/{record_name}/realsense --zed_data_path raw/{record_name}/zed'
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


def restart_zed():
    passwd = "111222"
    command = 'systemctl restart zed_x_daemon.service'
    res =  os.system('echo %s | sudo -S %s' % (passwd, command))
    if res == 0:
      output_text = "restart successed"
    else:
      output_text = "restart failed"
    return output_text


# Gradio UI
with gr.Blocks(title="Jetson Orin AGX - Camera Control") as demo:
    gr.Markdown("# Jetson Orin AGX - Camera Control")
    gr.Markdown("Control the video recording process for ZED and RealSense cameras.")

    with gr.Row():
        record_name_box = gr.Textbox(
            label="Record Item Name",
            placeholder="Enter data item name"
        )
    
    run_btn1 = gr.Button("Run RealSense Pipeline", elem_id="btn1_id")
    run_btn2 = gr.Button("Run ZED Pipeline", elem_id="btn2_id")
    run_btn3 = gr.Button("Run EgoCap Pipeline", elem_id="btn3_id")
    run_btn4 = gr.Button("Run all")
    run_btn5 = gr.Button("Stop all")
    with gr.Row():
        run_btn6 = gr.Button("timestamp analysis")
        btn6_choice = gr.Radio(
            show_label=False,
            choices=["realsense", "zed", "all"],
            value=["realsense"],

        )
    run_btn7 = gr.Button("restart zed")

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

    js_code = """
    () => {
        console.log("start...");
        document.getElementById('btn1_id').click();
        document.getElementById('btn2_id').click();
        document.getElementById('btn3_id').click();
    }
    """

    run_btn1.click(fn=realsense_pipeline, inputs=[record_name_box], outputs=[output_rs])
    run_btn2.click(fn=zed_pipeline, inputs=[record_name_box], outputs=[output_zed])
    run_btn3.click(fn=egocap_pipeline, inputs=[record_name_box], outputs=[output_egocap])
    run_btn4.click(fn=None, inputs=None, outputs=None, js=js_code)
    run_btn5.click(fn=stop_func, inputs=[], outputs=[output_box])
    run_btn6.click(fn=timestamp_analysis_pipe, inputs=[record_name_box, btn6_choice], outputs=[output_box])
    run_btn7.click(fn=restart_zed, inputs=[], outputs=[output_box])


if __name__ == "__main__":
    demo.launch(share=True)



