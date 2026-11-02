"""Postprocessing module: PostprocessWeb.

Args:
    None.

Returns:
    None.

Raises:
    None.
"""

import os
import traceback

try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    gr = None
    GRADIO_AVAILABLE = False

# ============================================================
# 导入你的处理模块
# ============================================================

from EgocapPreprocessing import EgocapPreprocessor
from TimestampSync import TimestampSyncConfig, TimestampSyncProcessor


# ============================================================
# 目录扫描函数
# ============================================================
def scan_subfolders_core(root_path):
    """scan_subfolders_core.

    Args:
        root_path: Input parameter.

    Returns:
        Any: Result of scan_subfolders_core.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    if not root_path or root_path.strip() == "":
        return [], "⚠️ 请输入根目录路径"

    root_path = root_path.strip()

    if not os.path.exists(root_path):
        return [], f"❌ 目录不存在: {root_path}"

    if not os.path.isdir(root_path):
        return [], f"❌ 不是有效的目录: {root_path}"

    folders = []
    try:
        for item in os.listdir(root_path):
            item_path = os.path.join(root_path, item)
            if os.path.isdir(item_path):
                folders.append(item_path)

        folders = sorted(folders)

        if not folders:
            return [], "⚠️ 该目录下没有找到子文件夹"

        return folders, f"✅ 扫描成功，找到 {len(folders)} 个文件夹"
    except OSError as e:
        return [], f"❌ 扫描出错: {str(e)}"


# ============================================================
# 核心处理函数 (支持批量、分步与【实时反馈】)
# ============================================================
def process_pipeline(folder_paths, run_mode, progress_callback=None):
    """process_pipeline.

    Args:
        folder_paths: Input parameter.
        run_mode: Input parameter.
        progress_callback: Input parameter.

    Returns:
        Any: Function result.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    """
    注意：这里使用了 yield 来实现日志和界面的实时刷新。
    当用户点击停止按钮时，Gradio 会自动抛出异常中断这个生成器。
    """
    if not folder_paths or len(folder_paths) == 0:
        yield "❌ 错误：请至少选择一个文件夹", "等待中...", ""
        return

    logs = []
    total_folders = len(folder_paths)
    success_count = 0

    run_step1 = "步骤 1" in run_mode or "全部执行" in run_mode
    run_step2 = "步骤 2" in run_mode or "全部执行" in run_mode

    logs.append(f"[SYSTEM] 准备处理 {total_folders} 个文件夹...")
    logs.append(f"[SYSTEM] 执行模式: {run_mode}")
    yield "🔄 启动处理中...", "初始化...", "\n".join(logs)

    for i, folder_path in enumerate(folder_paths):
        folder_name = os.path.basename(folder_path)
        if progress_callback is not None:
            progress_callback(
                i / total_folders,
                desc=f"🔄 正在处理 ({i+1}/{total_folders}): {folder_name}",
            )

        logs.append(f"\n{'='*50}")
        logs.append(f"[INFO] 正在处理第 {i+1}/{total_folders} 个任务: {folder_path}")
        # 实时刷新界面
        yield f"▶️ 正在处理 ({i+1}/{total_folders}): {folder_name}", folder_path, "\n".join(
            logs
        )

        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            logs.append(f"❌ 错误：路径不存在或不是文件夹 - {folder_path}")
            yield f"⚠️ 路径错误 ({i+1}/{total_folders})", folder_path, "\n".join(logs)
            continue

        try:
            # ---------------- 步骤 1 ----------------
            if run_step1:
                logs.append("  -> [1/2] 正在执行: EgocapPreprocessor(...).execute()...")
                yield f"▶️ [步骤1] 处理中...", folder_path, "\n".join(logs)  # 刷新日志

                preprocessor = EgocapPreprocessor(folder_path)
                preprocessor.execute()

                logs.append("  -> [1/2] ✅ 步骤 1 完成")
            else:
                logs.append("  -> [1/2] ⏩ 跳过: 步骤 1 (解析数据)")

            # ---------------- 步骤 2 ----------------
            if run_step2:
                logs.append(
                    "  -> [2/2] 正在执行: TimestampSyncProcessor(...).execute()..."
                )
                yield f"▶️ [步骤2] 处理中...", folder_path, "\n".join(logs)  # 刷新日志

                ts_processor = TimestampSyncProcessor(
                    TimestampSyncConfig(folder_path=folder_path)
                )
                ts_processor.execute()

                logs.append("  -> [2/2] ✅ 步骤 2 完成")
            else:
                logs.append("  -> [2/2] ⏩ 跳过: 步骤 2 (同步与切割)")

            logs.append(f"✅ [SUCCESS] {folder_name} 当前勾选步骤处理成功！")
            success_count += 1

        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
            error_msg = f"❌ [ERROR] {folder_name} 处理失败: {str(e)}"
            logs.append(error_msg)
            logs.append(f"[TRACE] {traceback.format_exc()}")
            # 发生错误也实时刷新，并继续下一个
            yield f"❌ 处理失败 ({i+1}/{total_folders})", folder_path, "\n".join(logs)

    if progress_callback is not None:
        progress_callback(1.0, desc="✅ 全部处理完成！")
    logs.append(f"\n{'='*50}")
    logs.append(f"[DONE] 批量处理结束。成功: {success_count}/{total_folders}")

    status_msg = f"✅ 处理结束 ({success_count}/{total_folders} 成功)"
    path_msg = f"共处理了 {total_folders} 个文件夹"

    # 最终输出
    yield status_msg, path_msg, "\n".join(logs)


# ============================================================
# 辅助函数
# ============================================================
def process_pipeline_cli(folder_paths, run_mode):
    """process_pipeline_cli.

    Args:
        folder_paths: Input parameter.
        run_mode: Input parameter.

    Returns:
        Any: Result of process_pipeline_cli.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    previous_line_count = 0
    for status, current, logs in process_pipeline(folder_paths, run_mode):
        print(f"\n[STATUS] {status}")
        print(f"[PATH] {current}")

        lines = logs.splitlines()
        new_lines = lines[previous_line_count:]
        for line in new_lines:
            print(line)
        previous_line_count = len(lines)


def choose_folders_from_terminal(folders):
    """choose_folders_from_terminal.

    Args:
        folders: Input parameter.

    Returns:
        Any: Result of choose_folders_from_terminal.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    print("\n可选文件夹：")
    for idx, folder in enumerate(folders, start=1):
        print(f"  {idx:>3}. {folder}")

    while True:
        selected = input("\n请输入要处理的序号（如 1,2,5 或 all）: ").strip().lower()
        if selected == "all":
            return folders

        raw_parts = [part.strip() for part in selected.split(",") if part.strip()]
        indices = []
        valid = True

        for part in raw_parts:
            if not part.isdigit():
                valid = False
                break
            index = int(part)
            if index < 1 or index > len(folders):
                valid = False
                break
            indices.append(index)

        if valid and indices:
            return [folders[index - 1] for index in sorted(set(indices))]

        print("⚠️ 输入无效，请重试。")


def choose_run_mode_from_terminal():
    """choose_run_mode_from_terminal.

    Args:
        None.

    Returns:
        Any: Result of choose_run_mode_from_terminal.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    mode_map = {
        "0": "✅ 全部执行 (步骤 1 + 2)",
        "1": "1️⃣ 仅执行步骤 1 (解析数据)",
        "2": "2️⃣ 仅执行步骤 2 (同步与切割)",
    }
    print("\n请选择执行模式：")
    print("  0. 全部执行 (步骤 1 + 2)")
    print("  1. 仅执行步骤 1")
    print("  2. 仅执行步骤 2")

    while True:
        choice = input("请输入模式编号 [0/1/2] (默认 0): ").strip() or "0"
        if choice in mode_map:
            return mode_map[choice]
        print("⚠️ 输入无效，请输入 0、1 或 2。")


def run_terminal_interface():
    """run_terminal_interface.

    Args:
        None.

    Returns:
        Any: Result of run_terminal_interface.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    print("\n🖥️ Gradio 不可用或启动失败，已自动切换到终端模式。")
    print("提示：按 Ctrl+C 可随时中断。")

    while True:
        default_root = "/mnt/ssd/ACE-Datatool"
        root_path = input(
            f"\n请输入数据根目录（默认 {default_root}）: "
        ).strip() or default_root

        folders, status = scan_subfolders_core(root_path)
        print(status)
        if not folders:
            retry = input("是否重新输入目录？[Y/n]: ").strip().lower()
            if retry in ("", "y", "yes"):
                continue
            return

        selected_folders = choose_folders_from_terminal(folders)
        run_mode = choose_run_mode_from_terminal()

        print("\n🚀 开始处理...\n")
        process_pipeline_cli(selected_folders, run_mode)

        again = input("\n是否继续处理其他任务？[y/N]: ").strip().lower()
        if again not in ("y", "yes"):
            print("✅ 终端模式处理结束。")
            return


def update_status_from_selection(paths):
    """update_status_from_selection.

    Args:
        paths: Input parameter.

    Returns:
        Any: Result of update_status_from_selection.

    Raises:
        Exception: If an unexpected runtime error occurs.
    """
    if not paths or len(paths) == 0:
        return "等待选择..."
    return f"✅ 已选择 {len(paths)} 个文件夹"


if GRADIO_AVAILABLE:

    def scan_subfolders(root_path):
        """scan_subfolders.

        Args:
            root_path: Input parameter.

        Returns:
            Any: Result of scan_subfolders.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        folders, status = scan_subfolders_core(root_path)
        return gr.update(choices=folders, value=[]), status


    def clear_interface():
        """clear_interface.

        Args:
            None.

        Returns:
            Any: Result of clear_interface.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        return (
            "",
            gr.update(choices=[], value=[]),
            "✅ 全部执行 (步骤 1 + 2)",
            "等待输入...",
            "未选择",
            "",
        )


    def process_pipeline_gradio(folder_paths, run_mode, progress=gr.Progress()):
        """process_pipeline_gradio.

        Args:
            folder_paths: Input parameter.
            run_mode: Input parameter.
            progress: Input parameter.

        Returns:
            Any: Result of process_pipeline_gradio.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        yield from process_pipeline(folder_paths, run_mode, progress_callback=progress)


    def build_gradio_app():
        """build_gradio_app.

        Args:
            None.

        Returns:
            Any: Result of build_gradio_app.

        Raises:
            Exception: If an unexpected runtime error occurs.
        """
        with gr.Blocks(
            title="EgoCap 批量数据处理管道",
            theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        ) as demo:

            gr.Markdown(
                """
            # 🎥 Ace Robotics 批量数据处理管道

            **自动化数据处理工具** | 先输入根目录扫描，再选择要处理的数据文件夹和执行步骤
            """
            )

            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### 📂 第一步：指定根目录")
                    with gr.Row():
                        root_input = gr.Textbox(
                            value="/mnt/ssd/ACE-Datatool",
                            label="数据总目录 (例如: /ACE-Data)",
                            placeholder="在此输入包含多个子文件夹的根目录路径...",
                            lines=1,
                            container=True,
                            scale=4,
                        )
                        scan_btn = gr.Button("🔍 扫描目录", variant="primary", scale=1)

                    gr.Markdown("### 🗂️ 第二步：选择数据文件夹")
                    folder_dropdown = gr.Dropdown(
                        choices=[],
                        multiselect=True,
                        label="选择要处理的文件夹 (支持多选)",
                        info="扫描根目录后，这里将列出所有可用的子文件夹",
                        container=True,
                    )

                    gr.Markdown("### ⚙️ 第三步：选择执行步骤")
                    run_mode = gr.Radio(
                        choices=[
                            "✅ 全部执行 (步骤 1 + 2)",
                            "1️⃣ 仅执行步骤 1 (解析数据)",
                            "2️⃣ 仅执行步骤 2 (同步与切割)",
                        ],
                        value="✅ 全部执行 (步骤 1 + 2)",
                        label="处理模式",
                        info="如果某一步出错，修复后你可以选择只重跑特定步骤",
                        container=True,
                    )

                    path_status = gr.Textbox(
                        label="扫描/选择状态", value="等待输入根目录...", interactive=False
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### 📊 运行状态")
                    status_display = gr.Textbox(
                        label="当前运行状态",
                        value="⏸️ 就绪 - 等待开始",
                        interactive=False,
                        lines=2,
                    )
                    current_path = gr.Textbox(
                        label="处理概览", value="未选择", interactive=False
                    )

            with gr.Row():
                start_btn = gr.Button("🚀 开始处理", variant="primary", size="lg", scale=3)
                stop_btn = gr.Button("🛑 强制停止", variant="stop", size="lg", scale=1)
                clear_btn = gr.Button("🔄 清空重置", variant="secondary", size="lg", scale=1)

            with gr.Accordion("📋 详细处理日志 (实时更新)", open=True):
                log_display = gr.Textbox(
                    label="日志输出",
                    lines=18,
                    max_lines=30,
                    interactive=False,
                    show_copy_button=True,
                )

            scan_btn.click(
                fn=scan_subfolders,
                inputs=root_input,
                outputs=[folder_dropdown, path_status],
            )
            root_input.submit(
                fn=scan_subfolders,
                inputs=root_input,
                outputs=[folder_dropdown, path_status],
            )
            folder_dropdown.change(
                fn=update_status_from_selection,
                inputs=folder_dropdown,
                outputs=path_status,
            )

            process_event = start_btn.click(
                fn=process_pipeline_gradio,
                inputs=[folder_dropdown, run_mode],
                outputs=[status_display, current_path, log_display],
                show_progress="full",
            )

            stop_btn.click(
                fn=lambda: (
                    "🛑 处理已强制终止",
                    "已停止",
                    "⚠️ 用户点击了强制停止按钮，后台任务已取消。",
                ),
                inputs=None,
                outputs=[status_display, current_path, log_display],
                cancels=[process_event],
            )

            clear_btn.click(
                fn=clear_interface,
                inputs=None,
                outputs=[
                    root_input,
                    folder_dropdown,
                    run_mode,
                    status_display,
                    current_path,
                    log_display,
                ],
            )

        return demo

# ============================================================
# 启动应用
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🎥 Ace Robotics 批量数据处理管道 (支持分步执行 & 强制停止)")
    print("=" * 60)

    if GRADIO_AVAILABLE:
        try:
            demo = build_gradio_app()
            demo.launch(
                server_name="0.0.0.0",
                server_port=7864,
                share=False,
                show_error=True,
                quiet=False,
            )
        except (RuntimeError, OSError, ValueError) as e:
            print(f"⚠️ Gradio 启动失败: {e}")
            print("将自动切换到终端模式。")
            run_terminal_interface()
    else:
        print("⚠️ 检测到 Gradio 不可用。")
        run_terminal_interface()
