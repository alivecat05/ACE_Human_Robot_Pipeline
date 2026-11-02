from __future__ import annotations

import atexit
from datetime import datetime
import json
import os
import threading
import time
import warnings

try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:  # 具体异常
    matplotlib = None

import gradio as gr

from utils import (
    CameraConfigService,
    DataService,
    get_color_sync_web_assets,
    KeypointTimestampRecorder,
    RecordingService,
    RuntimeConfig,
    SystemService,
    TaskTimestampRecorder,
    WebPreviewService,
    parse_device_index,
)


class _NoopMatplotlibBackendManager:
    """No-op context manager to avoid runtime matplotlib imports in Gradio."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False


try:
    gr.utils.MatplotlibBackendMananger = _NoopMatplotlibBackendManager

    import gradio.route_utils as route_utils

    if hasattr(route_utils, "utils"):
        route_utils.utils.MatplotlibBackendMananger = _NoopMatplotlibBackendManager
except AttributeError as error:
    warnings.warn(f"Failed to monkeypatch Gradio matplotlib backend: {error}")


class RecordingWebApp:
    """Gradio UI application for multi-camera recording workflows."""

    def __init__(self, server_name="0.0.0.0", port=9066) -> None:
        self.runtime = RuntimeConfig.detect()
        self.task_recorder = TaskTimestampRecorder()
        self.keypoint_recorder = KeypointTimestampRecorder()
        self.config_service = CameraConfigService(self.runtime)
        self.recording_service = RecordingService(
            runtime=self.runtime,
            task_recorder=self.task_recorder,
            keypoint_recorder=self.keypoint_recorder,
            config_service=self.config_service,
        )
        self.server_name = server_name
        self.port = port
        self.data_service = DataService()
        self.preview_service = WebPreviewService()
        self.system_service = SystemService(
            stop_callback=self.recording_service.stop_all,
            preview_cleanup_callback=self.preview_service.startup_cleanup,
        )
        self._last_camera_locations = {}
        self._camera_snapshot_lock = threading.Lock()
        self._cached_configurable_cameras = []
        self.preview_service.startup_cleanup()
        self.camera_choices = self.preview_service.get_available_camera_choices()
        # 记忆上一次手动生成配置时的相机位置选择：{sn: location}
        self._last_camera_locations = {}

        print(
            f"[app_web.py] Available RealSense cameras: {self.camera_choices['rs_choices']}"
        )
        print(
            f"[app_web.py] Available ZED cameras: {self.camera_choices['zed_choices']}"
        )
        atexit.register(self.shutdown)

    @staticmethod
    def _parse_selected_camera(device_idx: str):
        """Parses dropdown selection while filtering placeholder options."""
        if not device_idx:
            return None

        if "No camera detected" in str(device_idx):
            return None

        return parse_device_index(device_idx)

    def update_drives_ui(self):
        """Refreshes external drive dropdown choices."""
        drives = self.data_service.get_external_drives()
        value = drives[0] if drives and drives[0] != "未检测到外部存储" else None
        return gr.update(choices=drives, value=value)

    def refresh_recorded_data(self):
        """Refreshes today's recorded table and management task choices."""
        try:
            self.data_service.rebuild_recording_summaries()
        except Exception:
            pass
        table_data = self.data_service.get_recorded_folders()
        task_names = self.data_service.get_today_recorded_task_names()
        storage_info = self.data_service.get_storage_space_info(".")
        return table_data, gr.update(choices=task_names), storage_info

    def toggle_data_manage_mode(self, manage_mode: bool):
        """Toggles data-management panel visibility and task selector choices."""
        next_mode = not bool(manage_mode)
        task_names = self.data_service.get_today_recorded_task_names() if next_mode else []
        status_text = (
            "已进入管理模式：可勾选任务进行删除，或统计今日视频时长。"
            if next_mode
            else "已退出管理模式。"
        )

        return (
            next_mode,
            gr.update(visible=next_mode),
            gr.update(value="完成管理" if next_mode else "管理数据"),
            gr.update(choices=task_names, value=[]),
            status_text,
        )

    def delete_selected_tasks(self, selected_tasks, confirmed: bool):
        """Deletes selected tasks in today's list after explicit confirmation."""
        selected = [str(item).strip() for item in (selected_tasks or []) if str(item).strip()]
        if not selected:
            return (
                "❌ 请先勾选要删除的任务",
                self.data_service.get_recorded_folders(),
                gr.update(choices=self.data_service.get_today_recorded_task_names(), value=[]),
                False,
                self.data_service.get_storage_space_info("."),
            )

        if not confirmed:
            return (
                "⚠️ 请先勾选“我确认要永久删除选中任务”",
                self.data_service.get_recorded_folders(),
                gr.update(choices=self.data_service.get_today_recorded_task_names(), value=selected),
                confirmed,
                self.data_service.get_storage_space_info("."),
            )

        results = []
        success_count = 0
        for task_name in selected:
            message = self.data_service.delete_recorded_task(task_name, confirmed=True)
            results.append(message)
            if str(message).startswith("✅"):
                success_count += 1

        failed_count = len(selected) - success_count
        summary = f"删除完成：成功 {success_count} 个，失败 {failed_count} 个"
        details = "\n\n".join(results)
        final_message = f"{summary}\n{'=' * 30}\n{details}" if details else summary

        return (
            final_message,
            self.data_service.get_recorded_folders(),
            gr.update(choices=self.data_service.get_today_recorded_task_names(), value=[]),
            False,
            self.data_service.get_storage_space_info("."),
        )

    def package_selected_tasks_for_upload(self, selected_tasks):
        """Packages selected tasks; if none selected, packages all today's tasks."""
        selected = [str(item).strip() for item in (selected_tasks or []) if str(item).strip()]
        message = self.data_service.package_tasks_for_upload(selected)
        return (
            message,
            self.data_service.get_recorded_folders(),
            gr.update(choices=self.data_service.get_today_recorded_task_names(), value=[]),
            self.data_service.get_storage_space_info("."),
        )


    def update_quality_expected_mb(self, svo2_mb, rgb_mb, depth_mb, mjpeg_mb):
        """Updates quality check expected size settings."""
        return self.data_service.set_quality_expected_mb(
            svo2_mb,
            rgb_mb,
            depth_mb,
            mjpeg_mb,
        )

    def _get_configurable_cameras_snapshot(self, refresh: bool = False):
        """Returns cached configurable camera list; refreshes under lock when requested."""
        with self._camera_snapshot_lock:
            if refresh or not self._cached_configurable_cameras:
                self._cached_configurable_cameras = (
                    self.config_service.get_configurable_cameras()
                )
            return list(self._cached_configurable_cameras)

    def update_camera_location_ui(self):
        """Builds Gradio updates for camera location selectors."""
        detected_cameras = self._get_configurable_cameras_snapshot(refresh=True)

        if detected_cameras:
            camera_lines = [
                f"- {cam.get('type', 'unknown')} | SN: {cam.get('sn', '')} | {cam.get('name', '')}"
                for cam in detected_cameras
            ]
            summary = "检测结果：\n" + "\n".join(camera_lines)
            detected_count = len([camera for camera in detected_cameras if not camera.get("preset")])
            preset_count = len([camera for camera in detected_cameras if camera.get("preset")])
            status = f"检测到 {detected_count} 个相机，提供 {preset_count} 个预设相机可选"
        else:
            summary = "检测结果：未检测到可用相机（RealSense / ZED / EgoCap）"
            status = "未检测到相机"

        if not detected_cameras:
            return [gr.update(visible=False)] * 10 + [status, summary]

        updates = []
        for index in range(10):
            if index < len(detected_cameras):
                camera = detected_cameras[index]
                default_choices = self.config_service.LOCATION_CHOICES
                env_choices = [f"env:{choice}" for choice in default_choices]
                all_choices = ["不使用"] + default_choices + env_choices

                selected_value = "不使用"
                if index < len(detected_cameras):
                    sn = str(detected_cameras[index].get("sn", ""))
                    remembered = self._last_camera_locations.get(sn, "")
                    if remembered:
                        selected_value = remembered
                        if remembered not in all_choices:
                            all_choices = all_choices + [remembered]

                updates.append(
                    gr.update(
                        visible=True,
                        label=(
                            f"{camera.get('type', 'unknown')} (SN: {camera.get('sn', '')})"
                            "（选择“不使用”则跳过该相机）"
                        ),
                        choices=all_choices,
                        value=selected_value,
                    )
                )
            else:
                updates.append(gr.update(visible=False))
        updates.append(status)
        updates.append(summary)
        return updates

    def update_rs_frames(self, device_idx: str):
        """Updates RealSense preview frames."""
        parsed_idx = self._parse_selected_camera(device_idx)
        if parsed_idx is None:
            return None, None, "未选择相机"

        rgb, depth = self.preview_service.update_web_preview_dual(
            parsed_idx, "realsense"
        )
        return rgb, depth, f"正在从 RealSense 相机 {parsed_idx} 串流..."

    def update_zed_frames(self, device_idx: str):
        """Updates ZED preview frames."""
        parsed_idx = self._parse_selected_camera(device_idx)
        if parsed_idx is None:
            return None, None, "未选择相机"

        rgb, depth = self.preview_service.update_web_preview_dual(parsed_idx, "zed")
        return rgb, depth, f"正在从 ZED 相机 {parsed_idx} 串流..."

    def on_start_preview(self, device_idx: str, camera_type: str):
        """Starts preview stream and enables timer."""
        parsed_idx = self._parse_selected_camera(device_idx)
        if parsed_idx is None:
            # 兼容“先开程序后插相机”场景：自动刷新并尝试首个可用相机
            try:
                refreshed_choices, _, _ = self.preview_service.refresh_cameras(camera_type)
                valid_choices = [
                    choice
                    for choice in refreshed_choices
                    if "No camera detected" not in str(choice)
                ]
                if valid_choices:
                    parsed_idx = parse_device_index(valid_choices[0])
            except Exception as error:
                return f"未检测到可用相机，自动刷新失败: {error}", gr.Timer(value=0)

            if parsed_idx is None:
                return "未检测到可用相机，请先刷新设备", gr.Timer(value=0)

        result = self.preview_service.start_preview(parsed_idx, camera_type)
        timer_value = 0.067 if "已启动" in str(result) else 0
        return result, gr.Timer(value=timer_value)

    def on_stop_preview(self, device_idx: str, camera_type: str):
        """Stops preview stream and disables timer."""
        parsed_idx = self._parse_selected_camera(device_idx)
        if parsed_idx is None:
            return "未检测到可用相机", gr.Timer(value=0)

        result = self.preview_service.stop_preview(parsed_idx, camera_type)
        return result, gr.Timer(value=0)

    def on_refresh_cameras(self, cam_type: str = "realsense"):
        """Refreshes camera list for selected type."""
        try:
            new_choices, value, status = self.preview_service.refresh_cameras(cam_type)

            if not new_choices:
                placeholder = "0 (No camera detected)"
                return gr.Dropdown(choices=[placeholder], value=placeholder), status

            if len(new_choices) == 1 and "No camera detected" in new_choices[0]:
                return gr.Dropdown(choices=new_choices, value=new_choices[0]), status

            return gr.Dropdown(choices=new_choices, value=value or new_choices[0]), status
        except Exception as error:
            return gr.Dropdown(), f"刷新错误: {error}"

    def shutdown(self) -> None:
        """Cleans up preview and recording resources on process exit."""
        try:
            self.preview_service.startup_cleanup()
        except Exception:
            pass

        try:
            self.recording_service.stop_all()
        except Exception:
            pass

    def _build_camera_locations_from_ui(self, *locations):
        """Builds camera-location mapping from dynamic UI fields."""
        cameras = self._get_configurable_cameras_snapshot(refresh=False)
        if not cameras:
            cameras = self._get_configurable_cameras_snapshot(refresh=True)
        camera_locations = {}
        inherited_count = 0

        for index, camera in enumerate(cameras):
            sn = str(camera.get("sn", ""))
            selected = ""
            if index < len(locations):
                selected = str(locations[index] or "").strip()

            # 显式选择“不使用”时，不继承历史
            if selected == "不使用":
                selected = ""
                if sn in self._last_camera_locations:
                    self._last_camera_locations.pop(sn, None)
                continue

            # 本次未改该相机时，继承上一次选择
            if not selected:
                remembered = str(self._last_camera_locations.get(sn, "") or "").strip()
                if remembered:
                    selected = remembered
                    inherited_count += 1

            if selected:
                camera_locations[sn] = selected

        return camera_locations, inherited_count

    def generate_config_wrapper_manual(
        self, record_name, task_name, use_group_indexed_keys, *locations
    ):
        """Wraps manual config generation from dynamic location fields."""
        if not record_name or not str(record_name).strip():
            return "❌ 请先填写“录制任务名称(必填)”，再生成配置文件。"

        camera_locations, inherited_count = self._build_camera_locations_from_ui(
            *locations
        )

        result = self.config_service.generate_config_json(
            task_no=record_name,
            task_name=task_name,
            camera_locations=camera_locations,
            record_name=record_name,
            use_group_indexed_keys=bool(use_group_indexed_keys),
        )

        # 仅在成功时更新记忆
        if isinstance(result, str) and result.startswith("✅"):
            self._last_camera_locations = dict(camera_locations)
            if inherited_count > 0:
                result += f"\n已继承 {inherited_count} 个未修改相机配置"

        return result

    def start_realsense_recording_wrapper(
        self, record_name, task_name, use_group_indexed_keys, *locations
    ):
        """Starts RealSense recording and auto-generates config from current selection."""
        # 先释放网页预览占用，避免与录制进程争抢设备导致 SDK 崩溃
        self.preview_service.startup_cleanup()
        # 等待 UVC 设备完成释放，避免紧接着启动录制时报 busy
        time.sleep(1.0)
        camera_locations, _ = self._build_camera_locations_from_ui(*locations)
        self._last_camera_locations = dict(camera_locations)
        yield from self.recording_service.realsense_pipeline(
            record_name,
            task_name,
            camera_locations,
            bool(use_group_indexed_keys),
        )

    def start_zed_recording_wrapper(
        self, record_name, task_name, use_group_indexed_keys, *locations
    ):
        """Starts ZED recording and auto-generates config from current selection."""
        self.preview_service.startup_cleanup()
        camera_locations, _ = self._build_camera_locations_from_ui(*locations)
        self._last_camera_locations = dict(camera_locations)
        yield from self.recording_service.zed_pipeline(
            record_name,
            task_name,
            camera_locations,
            bool(use_group_indexed_keys),
        )

    def start_egocap_recording_wrapper(
        self, record_name, task_name, use_group_indexed_keys, *locations
    ):
        """Starts EgoCap recording and auto-generates config from current selection."""
        # self.preview_service.startup_cleanup()
        camera_locations, _ = self._build_camera_locations_from_ui(*locations)
        self._last_camera_locations = dict(camera_locations)
        yield from self.recording_service.egocap_pipeline(
            record_name,
            task_name,
            camera_locations,
            bool(use_group_indexed_keys),
        )

    def save_color_sync_log(self, logs_json: str) -> str:
        """Saves browser color sync logs to JSON using color_generate.py style path."""
        if not logs_json:
            return "颜色同步已停止，但未获取到颜色日志。"

        try:
            color_changes = json.loads(logs_json)
        except json.JSONDecodeError as error:
            return f"颜色同步已停止，但日志解析失败: {error}"

        if not isinstance(color_changes, list):
            return "颜色同步已停止，但日志格式不正确。"

        now = datetime.now()
        ts_folder = f"ts{now.strftime('%m%d')}"
        base_dir = os.path.join(self.runtime.app_dir, ts_folder)
        os.makedirs(base_dir, exist_ok=True)
        filename = os.path.join(
            base_dir,
            f"color_sync_log_{now.strftime('%Y%m%d_%H%M%S')}.json",
        )

        with open(filename, "w", encoding="utf-8") as log_file:
            json.dump(color_changes, log_file, indent=4, ensure_ascii=False)

        return f"颜色同步已停止，日志已保存到: {filename}"

    def build(self) -> gr.Blocks:
        """Builds Gradio UI and event bindings."""
        zed_camera_choices = self.camera_choices["zed_choices"]
        zed_camera_values = self.camera_choices["zed_values"]
        rs_camera_choices = self.camera_choices["rs_choices"]
        rs_camera_values = self.camera_choices["rs_values"]
        color_sync_html, color_sync_start_js, color_sync_stop_js = (
            get_color_sync_web_assets()
        )
        quality_expected_mb = self.data_service.get_quality_expected_mb()
        rs_placeholder = len(rs_camera_choices) == 1 and "No camera detected" in rs_camera_choices[0]
        zed_has_cameras = bool(zed_camera_choices)
        zed_camera_choices_ui = zed_camera_choices or ["0 (No camera detected)"]

        with gr.Blocks(title="Jetson Orin AGX - 多相机录制控制台") as demo:
            gr.Markdown("# Jetson Orin AGX - 多相机录制控制台")
            gr.Markdown("控制 ZED 和 RealSense 相机的视频录制流程。")

            with gr.Accordion("📝 相机配置 (可选)", open=False):
                gr.Markdown(
                    "**开始录制时会自动生成 config.json。** 您也可以在这里手动配置相机佩戴位置。"
                )
                gr.Markdown(
                    "**交互说明：** 先点“检测相机”，再给每个相机选择位置。"
                    "\n- 选择 `head/chest/...` 表示写入 `device.ego`"
                    "\n- 选择 `env:xxx` 表示写入 `device.env`"
                    "\n- 选择 `不使用` 表示该相机本次不写入 config"
                    "\n- 本按钮会直接使用下方“录制任务名称(必填)”作为输出目录，确保 config 与录制数据同目录"
                )

                detect_cameras_btn = gr.Button("🔍 检测相机", size="sm")
                config_status = gr.Textbox(
                    label="已检测到的相机",
                    value="点击 '检测相机' 查看已连接设备",
                    interactive=False,
                    lines=3,
                )
                camera_detect_summary = gr.Textbox(
                    label="检测结果详情",
                    value="点击 '检测相机' 后，这里会显示 RealSense / ZED / EgoCap 详情",
                    interactive=False,
                    lines=6,
                )

                gr.Markdown("### 配置相机佩戴位置 (可选)")
                camera_location_dropdowns = []
                with gr.Column():
                    for index in range(10):
                        dropdown = gr.Dropdown(
                            label=f"Camera {index}",
                            choices=[],
                            visible=False,
                            interactive=True,
                        )
                        camera_location_dropdowns.append(dropdown)

                config_key_style_checkbox = gr.Checkbox(
                    label="键名使用分组索引（head_0 / hand_0）",
                    value=True,
                )

            with gr.Accordion("📱 Web 实时预览", open=False):
                gr.Markdown("## Web端实时画面预览")
                gr.Markdown(
                    "实时查看相机画面 (RGB + Depth)。支持手机/平板/电脑浏览器访问！"
                )

                with gr.Tab("RealSense 预览"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            rs_web_camera_choice = gr.Dropdown(
                                choices=rs_camera_choices,
                                value=(
                                    rs_camera_choices[0]
                                    if rs_placeholder
                                    else (rs_camera_values[0] if rs_camera_values else "0")
                                ),
                                label="选择 RealSense 相机 (带序列号)",
                                interactive=True,
                            )
                        with gr.Column(scale=1):
                            rs_refresh_cam_btn = gr.Button("🔄 刷新设备")

                    with gr.Row():
                        rs_web_preview_btn = gr.Button(
                            "启动 RealSense 预览", variant="primary"
                        )
                        rs_web_stop_btn = gr.Button(
                            "停止 RealSense 预览", variant="secondary"
                        )

                    with gr.Row():
                        rs_web_rgb_output = gr.Image(label="RGB 画面", type="numpy")
                        rs_web_depth_output = gr.Image(label="深度图", type="numpy")

                    rs_web_status = gr.Textbox(
                        label="状态",
                        interactive=False,
                        value="点击 '启动 RealSense 预览' 开始",
                    )

                    rs_timer = gr.Timer(value=0)

                with gr.Tab("ZED 预览"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            zed_web_camera_choice = gr.Dropdown(
                                choices=zed_camera_choices_ui,
                                value=(
                                    zed_camera_values[0]
                                    if zed_has_cameras and zed_camera_values
                                    else zed_camera_choices_ui[0]
                                ),
                                label="选择 ZED 相机 (带序列号)",
                                interactive=True,
                            )
                        with gr.Column(scale=1):
                            zed_refresh_cam_btn = gr.Button("🔄 刷新设备")

                    with gr.Row():
                        zed_web_preview_btn = gr.Button(
                            "启动 ZED 预览", variant="primary"
                        )
                        zed_web_stop_btn = gr.Button(
                            "停止 ZED 预览", variant="secondary"
                        )

                    with gr.Row():
                        zed_web_rgb_output = gr.Image(label="RGB 画面", type="numpy")
                        zed_web_depth_output = gr.Image(label="深度图", type="numpy")

                    zed_web_status = gr.Textbox(
                        label="状态",
                        interactive=False,
                        value=(
                            "点击 '启动 ZED 预览' 开始"
                            if zed_has_cameras
                            else "未检测到 ZED，相机接入后点击刷新设备"
                        ),
                    )

                    zed_timer = gr.Timer(value=0)

            with gr.Row():
                record_name_box = gr.Textbox(
                    label="录制任务名称 (必填)",
                    placeholder="输入任务名称, 例如: 000001 或 test_recording",
                )
                record_task_name_box = gr.Textbox(
                    label="任务类型 (可选)",
                    placeholder="例如: 走路, 跑步, 跳跃等",
                    value="",
                )

            with gr.Accordion("🎬 动态组合录制", open=True):
                gr.Markdown(
                    "按需选择要启动的设备，支持任意组合：RealSense / ZED / EgoCap。"
                )
                recording_combo_select = gr.CheckboxGroup(
                    label="选择要录制的设备",
                    choices=["realsense", "zed", "egocap"],
                    value=["realsense", "zed", "egocap"],
                    info="可多选；点击“启动所选录制”后，会依次启动所选设备。",
                )
                start_selected_btn = gr.Button("启动所选录制", variant="primary")

                # 隐藏触发按钮：用于复用现有单设备启动逻辑
                run_btn1 = gr.Button(
                    "启动 RealSense 录制", elem_id="btn1_id", visible=False
                )
                run_btn2 = gr.Button("启动 ZED 录制", elem_id="btn2_id", visible=False)
                run_btn3 = gr.Button("启动 EgoCap 录制", elem_id="btn3_id", visible=False)
            run_btn5 = gr.Button("停止所有任务")
            with gr.Accordion("🎨 颜色同步", open=False):
                gr.Markdown("启动浏览器内全屏/窗口色块切换，用于多设备时间对齐（支持手机/平板）。")
                with gr.Row():
                    color_width_box = gr.Number(
                        label="宽度", value=1920, precision=0, interactive=True
                    )
                    color_height_box = gr.Number(
                        label="高度", value=1080, precision=0, interactive=True
                    )
                    color_interval_box = gr.Number(
                        label="切换间隔(秒)", value=0.4, precision=2, interactive=True
                    )
                color_windowed_box = gr.Checkbox(
                    label="窗口模式(不全屏)", value=False
                )
                with gr.Row():
                    color_sync_start_btn = gr.Button("启动颜色同步", variant="primary")
                    color_sync_stop_btn = gr.Button(
                        "停止颜色同步",
                        variant="secondary",
                        elem_id="color_sync_stop_btn_id",
                    )
                gr.HTML(value=color_sync_html)


            with gr.Row():
                run_btn6 = gr.Button("运行时间戳分析")
                btn6_choice = gr.Radio(
                    show_label=False,
                    choices=["realsense", "zed", "all"],
                    value="realsense",
                )

            with gr.Accordion("📍 任务时间戳记录", open=False):
                gr.Markdown(
                    "**录制开始时自动初始化。** 在任务开始和结束时分别点击对应按钮记录时间戳。可记录多个任务。"
                )
                with gr.Row():
                    task_start_btn = gr.Button(
                        "🟢 记录任务开始", variant="primary", size="lg"
                    )
                    task_end_btn = gr.Button(
                        "🔴 记录任务结束", variant="stop", size="lg"
                    )
                task_status_box = gr.Textbox(
                    label="状态",
                    interactive=False,
                    lines=2,
                    value="就绪。开始录制时将创建 CSV 文件。",
                )
                task_summary_box = gr.Textbox(
                    label="摘要", interactive=False, lines=2, value=""
                )

            # with gr.Accordion("📍 关键点时间戳记录", open=False):
            #     gr.Markdown(
            #         "**录制开始时自动初始化。** 每次关键动作点击一次“记录关键点”，索引从 0 开始。"
            #     )
            #     with gr.Row():
            #         keypoint_init_btn = gr.Button("重新初始化", variant="secondary")
            #         keypoint_reset_btn = gr.Button("重置计数器", variant="secondary")
            #     keypoint_record_btn = gr.Button(
            #         "🔴 记录关键点", variant="primary", size="lg"
            #     )
            #     keypoint_status_box = gr.Textbox(
            #         label="状态",
            #         interactive=False,
            #         lines=2,
            #         value="就绪。开始录制时将创建 keypoint CSV 文件。",
            #     )
            #     keypoint_summary_box = gr.Textbox(
            #         label="摘要", interactive=False, lines=2, value=""
            #     )

            # with gr.Accordion("批量分析", open=False):
            #     run_batch_btn = gr.Button("批量时间戳分析 (由于数据量大，可能较慢)")
            #     batch_summary_box = gr.Textbox(
            #         label="FPS 统计结果",
            #         interactive=False,
            #         lines=10,
            #         value="运行批量分析以查看结果...",
            #     )
            with gr.Accordion("已录制数据管理", open=False):
                with gr.Row():
                    gr.Markdown(
                        f"### 今日录制任务 ({datetime.now().strftime('%Y%m%d')})"
                    )
                    refresh_btn = gr.Button("刷新列表")
                    manage_btn = gr.Button("管理数据")

                storage_space_box = gr.Textbox(
                    label="当前机子存储空间",
                    interactive=False,
                    lines=4,
                    value=self.data_service.get_storage_space_info("."),
                )

                recorded_table = gr.Dataframe(
                    headers=["任务名称", "修改时间", "总大小", "详情", "质检"],
                    datatype=["str", "str", "str", "str", "str"],
                    label="今日录制列表",
                    interactive=False,
                    value=self.data_service.get_recorded_folders,
                )

                manage_mode_state = gr.State(False)
                with gr.Column(visible=False) as manage_panel:
                    with gr.Accordion("质检阈值设置 (MB / 10分钟)", open=False):
                        with gr.Row():
                            quality_svo2_mb = gr.Number(
                                label="*.svo2",
                                value=quality_expected_mb.get("*.svo2", 200.0),
                                precision=1,
                                interactive=True,
                            )
                            quality_rgb_mb = gr.Number(
                                label="*_rgb.mp4",
                                value=quality_expected_mb.get("*_rgb.mp4", 100.0),
                                precision=1,
                                interactive=True,
                            )
                        with gr.Row():
                            quality_depth_mb = gr.Number(
                                label="*_depth.mp4",
                                value=quality_expected_mb.get("*_depth.mp4", 800.0),
                                precision=1,
                                interactive=True,
                            )
                            quality_mjpeg_mb = gr.Number(
                                label="mjpeg.bin",
                                value=quality_expected_mb.get("mjpeg.bin", 5052.0),
                                precision=1,
                                interactive=True,
                            )

                        quality_apply_btn = gr.Button("应用质检阈值", variant="primary")
                        quality_apply_status = gr.Textbox(
                            label="质检阈值更新状态",
                            interactive=False,
                            lines=2,
                            value="",
                        )

                    manage_task_selector = gr.CheckboxGroup(
                        label="勾选要管理的任务（可多选）",
                        choices=[],
                        value=[],
                    )
                    delete_confirm_checkbox = gr.Checkbox(
                        label="我确认要永久删除选中任务",
                        value=False,
                    )
                    with gr.Row():
                        delete_selected_btn = gr.Button("🗑️ 删除选中任务", variant="stop")
                        package_upload_btn = gr.Button("📦 打包为上传数据格式", variant="primary")
                        manage_duration_btn = gr.Button("📊 统计视频总时长")

                    manage_action_output = gr.Textbox(
                        label="管理结果",
                        interactive=False,
                        lines=8,
                        value="点击“管理数据”后，可勾选任务进行删除或统计。",
                    )

                    duration_output = gr.Textbox(
                        label="视频时长统计结果",
                        interactive=False,
                        lines=15,
                        value="点击 '统计视频总时长' 查看统计信息",
                    )

            with gr.Row():
                run_btn7 = gr.Button("重启 ZED 服务")
                run_btn8 = gr.Button("重置 RealSense")
                run_btn9 = gr.Button("重启应用")

            with gr.Accordion("⚠️ 系统关机 (危险操作)", open=False):
                gr.Markdown("**警告：点击关机按钮后，系统将在 5 秒后自动关闭！**")
                with gr.Row():
                    shutdown_btn = gr.Button("⚠️ 立即关机", variant="stop")
                    reboot_btn = gr.Button("🔄 立即重启", variant="stop")
                    cancel_shutdown_btn = gr.Button("🛑 取消关机/重启", variant="secondary")

            hidden_error_box = gr.Textbox(visible=False, label="隐形错误提示框")
            color_sync_log_box = gr.Textbox(visible=False, label="颜色同步日志")
            hidden_error_box.change(
                fn=None,
                inputs=[hidden_error_box],
                js="(val) => { if (val && val.length > 0) { alert(val); } }",
            )

            output_box = gr.Textbox(
                label="终端输出", interactive=False, lines=5, autoscroll=True
            )
            output_rs = gr.Textbox(
                label="RealSense 输出", interactive=False, lines=5, autoscroll=True
            )
            output_zed = gr.Textbox(
                label="ZED 输出", interactive=False, lines=5, autoscroll=True
            )
            output_egocap = gr.Textbox(
                label="EgoCap 输出", interactive=False, lines=5, autoscroll=True
            )


            with gr.Accordion("💾 数据导出 (移动硬盘)", open=False):
                gr.Markdown(
                    "将录制的数据一键备份到移动硬盘。建议使用 SSD 移动硬盘以获得更快的速度。"
                )

                with gr.Row():
                    drive_refresh_btn = gr.Button("🔄 1. 刷新/检测硬盘")
                    drive_dropdown = gr.Dropdown(
                        label="选择目标硬盘",
                        choices=self.data_service.get_external_drives(),
                        value=None,
                        interactive=True,
                    )

                with gr.Row():
                    data_scope_radio = gr.Radio(
                        label="选择传输内容",
                        choices=["today", "all"],
                        value="today",
                        type="value",
                        info="today: 仅传输今日数据 | all: 传输所有日期数据",
                    )
                    transfer_btn = gr.Button("🚀 2. 开始一键传输", variant="primary")

                transfer_status = gr.Textbox(
                    label="当前状态", value="就绪", interactive=False
                )
                transfer_output = gr.Textbox(
                    label="传输日志", lines=10, max_lines=20, autoscroll=True
                )

            rs_web_preview_btn.click(
                fn=lambda device_idx: self.on_start_preview(device_idx, "realsense"),
                inputs=[rs_web_camera_choice],
                outputs=[rs_web_status, rs_timer],
            )
            rs_web_stop_btn.click(
                fn=lambda device_idx: self.on_stop_preview(device_idx, "realsense"),
                inputs=[rs_web_camera_choice],
                outputs=[rs_web_status, rs_timer],
            )
            rs_refresh_cam_btn.click(
                fn=lambda: self.on_refresh_cameras("realsense"),
                inputs=[],
                outputs=[rs_web_camera_choice, rs_web_status],
            )
            rs_timer.tick(
                fn=self.update_rs_frames,
                inputs=[rs_web_camera_choice],
                outputs=[rs_web_rgb_output, rs_web_depth_output, rs_web_status],
            )

            zed_web_preview_btn.click(
                fn=lambda device_idx: self.on_start_preview(device_idx, "zed"),
                inputs=[zed_web_camera_choice],
                outputs=[zed_web_status, zed_timer],
            )
            zed_web_stop_btn.click(
                fn=lambda device_idx: self.on_stop_preview(device_idx, "zed"),
                inputs=[zed_web_camera_choice],
                outputs=[zed_web_status, zed_timer],
            )
            zed_refresh_cam_btn.click(
                fn=lambda: self.on_refresh_cameras("zed"),
                inputs=[],
                outputs=[zed_web_camera_choice, zed_web_status],
            )
            zed_timer.tick(
                fn=self.update_zed_frames,
                inputs=[zed_web_camera_choice],
                outputs=[zed_web_rgb_output, zed_web_depth_output, zed_web_status],
            )

            js_code_dynamic_combo = """
            (selected) => {
                const selectedList = Array.isArray(selected) ? selected : [];
                if (selectedList.length === 0) {
                    alert('请至少选择一个设备（realsense / zed / egocap）');
                    return;
                }

                const idMap = {
                    realsense: 'btn1_id',
                    zed: 'btn2_id',
                    egocap: 'btn3_id',
                };

                const startOrder = ['realsense', 'zed', 'egocap'];
                let step = 0;
                for (const key of startOrder) {
                    if (!selectedList.includes(key)) continue;
                    const btn = document.getElementById(idMap[key]);
                    if (btn) {
                        const delayMs = step * 500;
                        setTimeout(() => btn.click(), delayMs);
                        step += 1;
                    }
                }
            }
            """

            run_btn1.click(
                fn=self.start_realsense_recording_wrapper,
                inputs=[record_name_box, record_task_name_box, config_key_style_checkbox]
                + camera_location_dropdowns,
                outputs=[output_rs, hidden_error_box],
            )
            run_btn2.click(
                fn=self.start_zed_recording_wrapper,
                inputs=[record_name_box, record_task_name_box, config_key_style_checkbox]
                + camera_location_dropdowns,
                outputs=[output_zed, hidden_error_box],
            )
            run_btn3.click(
                fn=self.start_egocap_recording_wrapper,
                inputs=[record_name_box, record_task_name_box, config_key_style_checkbox]
                + camera_location_dropdowns,
                outputs=[output_egocap, hidden_error_box],
            )
            start_selected_btn.click(
                fn=None,
                inputs=[recording_combo_select],
                outputs=[],
                js=js_code_dynamic_combo,
            )
            run_btn5.click(
                fn=self.recording_service.stop_all, inputs=[], outputs=[output_box]
            )
            run_btn6.click(
                fn=self.recording_service.timestamp_analysis_pipe,
                inputs=[record_name_box, btn6_choice],
                outputs=[output_box],
            )
            color_sync_start_btn.click(
                fn=None,
                inputs=[
                    color_width_box,
                    color_height_box,
                    color_interval_box,
                    color_windowed_box,
                ],
                outputs=[output_box, hidden_error_box, color_sync_log_box],
                js=color_sync_start_js,
            )
            color_sync_stop_btn.click(
                fn=self.save_color_sync_log,
                inputs=[color_sync_log_box],
                outputs=[output_box],
                js=color_sync_stop_js,
            )

            task_start_btn.click(
                fn=self.task_recorder.record_task_start,
                inputs=[],
                outputs=[task_status_box, task_summary_box],
            )
            task_end_btn.click(
                fn=self.task_recorder.record_task_end,
                inputs=[],
                outputs=[task_status_box, task_summary_box],
            )

            # keypoint_init_btn.click(
            #     fn=self.keypoint_recorder.initialize,
            #     inputs=[record_name_box],
            #     outputs=[keypoint_status_box],
            # )
            # keypoint_record_btn.click(
            #     fn=self.keypoint_recorder.record,
            #     inputs=[],
            #     outputs=[keypoint_status_box, keypoint_summary_box],
            # )
            # keypoint_reset_btn.click(
            #     fn=self.keypoint_recorder.reset,
            #     inputs=[],
            #     outputs=[keypoint_status_box, keypoint_summary_box],
            # )

            # run_batch_btn.click(
            #     fn=self.recording_service.batch_timestamp_analysis_pipe,
            #     inputs=[],
            #     outputs=[output_box, batch_summary_box],
            # )
            run_btn7.click(
                fn=self.system_service.restart_zed, inputs=[], outputs=[output_box]
            )
            run_btn8.click(
                fn=self.system_service.reset_realsense, inputs=[], outputs=[output_box]
            )
            run_btn9.click(
                fn=self.system_service.restart_app, inputs=[], outputs=[output_box]
            )
            shutdown_btn.click(
                fn=self.system_service.shutdown_system, inputs=[], outputs=[output_box]
            )
            reboot_btn.click(
                fn=self.system_service.reboot_system, inputs=[], outputs=[output_box]
            )
            cancel_shutdown_btn.click(
                fn=self.system_service.cancel_shutdown, inputs=[], outputs=[output_box]
            )
            refresh_btn.click(
                fn=self.refresh_recorded_data,
                inputs=[],
                outputs=[recorded_table, manage_task_selector, storage_space_box],
            )

            manage_btn.click(
                fn=self.toggle_data_manage_mode,
                inputs=[manage_mode_state],
                outputs=[
                    manage_mode_state,
                    manage_panel,
                    manage_btn,
                    manage_task_selector,
                    manage_action_output,
                ],
            )

            delete_selected_btn.click(
                fn=self.delete_selected_tasks,
                inputs=[manage_task_selector, delete_confirm_checkbox],
                outputs=[
                    manage_action_output,
                    recorded_table,
                    manage_task_selector,
                    delete_confirm_checkbox,
                    storage_space_box,
                ],
            )

            package_upload_btn.click(
                fn=self.package_selected_tasks_for_upload,
                inputs=[manage_task_selector],
                outputs=[
                    manage_action_output,
                    recorded_table,
                    manage_task_selector,
                    storage_space_box,
                ],
            )

            quality_apply_btn.click(
                fn=self.update_quality_expected_mb,
                inputs=[
                    quality_svo2_mb,
                    quality_rgb_mb,
                    quality_depth_mb,
                    quality_mjpeg_mb,
                ],
                outputs=[quality_apply_status],
            )

            manage_duration_btn.click(
                fn=self.data_service.calculate_total_video_duration,
                inputs=[],
                outputs=[duration_output],
            )

            detect_cameras_btn.click(
                fn=self.update_camera_location_ui,
                inputs=[],
                outputs=camera_location_dropdowns + [config_status, camera_detect_summary],
            )
            drive_refresh_btn.click(
                fn=self.update_drives_ui,
                inputs=[],
                outputs=[drive_dropdown],
            )
            transfer_btn.click(
                fn=self.data_service.transfer_data_to_drive,
                inputs=[drive_dropdown, data_scope_radio],
                outputs=[transfer_output, transfer_status],
            )

        return demo

    def launch(
        self,
        share: bool = False,
    ) -> None:
        """Launches Gradio web app."""
        app = self.build()
        app.launch(server_name=self.server_name, server_port=self.port, share=share)


def main() -> None:
    """Program entrypoint."""
    app = RecordingWebApp(server_name="0.0.0.0", port=9068)
    app.launch()


if __name__ == "__main__":
    main()
