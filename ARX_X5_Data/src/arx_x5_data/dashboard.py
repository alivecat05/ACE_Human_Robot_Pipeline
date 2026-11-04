"""Optional local NiceGUI operator panel; acquisition runs in separate workers."""

import time

from fastapi import HTTPException, Request
from starlette.responses import Response

from .capture_totals import CaptureTotals
from .config import Config
from .model import CAMERAS
from .operator_session import OperatorSession
from .preview import Preview
from .runtime import Runtime


def run(config: Config, *, mock: bool, port: int) -> None:
    from nicegui import app, ui

    runtime = Runtime(config, mock=mock)
    totals = CaptureTotals(config.output)
    session = OperatorSession(runtime)
    from .workbench_control import RecordingControl

    control = RecordingControl(runtime, config.output / ".workbench-control.json")

    @app.post("/control/record/{action}")
    async def workbench_record(action: str, request: Request):
        expected = f"127.0.0.1:{port}"
        if (
            request.headers.get("host") != expected
            or request.headers.get("origin", "http://" + expected) != "http://" + expected
        ):
            raise HTTPException(403, "Only local same-origin recording control is allowed")
        raw = await request.body()
        if len(raw) > 8192:
            raise HTTPException(413, "Request too large")
        try:
            import json

            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("Request must be an object")
            return control.execute(action, body)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(409, str(error)) from error

    preview = Preview(runtime.sources)
    app.on_startup(runtime.start)
    app.on_startup(preview.start)
    app.on_shutdown(preview.close)
    app.on_shutdown(runtime.close)

    @app.get("/preview/{name}.jpg")
    def video(name: str, after: int = -1):
        if name not in CAMERAS:
            raise HTTPException(404, "Unknown camera")
        frame = preview.latest(name, fresh=True)
        if frame is None or frame.sequence == after:
            return Response(status_code=204, headers={"Cache-Control": "no-store"})
        return Response(
            frame.jpeg,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Sequence": str(frame.sequence),
                "X-Frame-Age-Ms": str(max(0, (time.monotonic() - frame.captured_at) * 1000)),
            },
        )

    @app.get("/status")
    def recorder_status():
        return {**runtime.status(), "capture_totals": totals.snapshot()}

    @app.get("/preview-stats")
    def preview_stats():
        return {"cameras": preview.stats(), "error": str(preview.error) if preview.error else None}

    ui.add_css(
        """
body {background:#f3f6f3;color:#1d302a;font-family:Inter,'Noto Sans CJK SC',sans-serif}
.q-page {padding:0!important}.nicegui-content {width:100%;max-width:none;margin:auto;padding:12px 16px 158px;gap:12px}
.page-header {display:flex;align-items:center;justify-content:space-between;width:100%;gap:20px}
.brand-title {font-size:23px;font-weight:700;letter-spacing:1px}.subtle {font-size:12px;color:#849487}
.mode-pill {font-size:12px;padding:8px 12px;border:1px solid #dae4da;border-radius:20px;color:#69806b;background:#fff}
.preview-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;width:100%}
.preview-card {width:100%;padding:9px;border:1px solid #e0e7de;border-radius:12px;box-shadow:none;background:#fff}
.camera-title {font-size:13px;font-weight:600;margin-bottom:4px}.camera-image {width:100%;height:auto;aspect-ratio:1.6;max-height:calc(100dvh - 315px);object-fit:contain;background:#17241e;border-radius:7px;display:block}
.camera-fps {font-size:10px;color:#9aaa9b;margin-top:7px}
.camera-feed {position:relative;width:100%}.camera-stale {opacity:.25;filter:grayscale(1)}
.camera-warning {position:absolute;inset:0;display:flex;align-items:center;justify-content:center;padding:14px;text-align:center;color:#fff;background:#16231ed9;font-size:18px;font-weight:650;border-radius:7px}.camera-warning[hidden] {display:none}
.operator-panel {background:#fff;border:1px solid #dfe7db;border-radius:14px;padding:14px 18px;width:100%;gap:7px;box-shadow:none}
.status-row {display:flex;justify-content:space-between;align-items:center;gap:20px;width:100%}.stage-title {font-size:23px;font-weight:650;letter-spacing:.4px}
.task-field {width:350px;max-width:100%;font-size:16px}.action-grid {position:fixed;left:16px;right:16px;bottom:14px;z-index:50;display:grid;grid-template-columns:minmax(0,1fr) 240px;gap:16px;width:auto;height:120px;padding:12px;background:#fff;border:1px solid #dfe7db;border-radius:14px;box-shadow:0 5px 30px #20362918}
.op-button {width:100%;height:94px;min-height:94px;border-radius:12px!important;font-size:27px;font-weight:650;box-shadow:none!important;letter-spacing:.3px}
.op-button .q-icon {font-size:31px;margin-right:9px}.op-button.disabled {opacity:.33!important}
.op-start {background:#176b55!important;color:white!important}.op-save {background:#315d82!important;color:white!important}.op-reset {background:#f1eadb!important;color:#80672e!important}.op-invalid {background:#fae8e0!important;color:#aa5542!important}
.next-step {font-size:13px;line-height:1.6;color:#657e68}.connection-line {font-size:12px;color:#81917b;line-height:1.9}.warning-line {font-size:12px;color:#ad6344;line-height:1.8}
.background-line {font-size:12px;color:#7f917c;line-height:1.8;width:100%;border-top:1px solid #e7ece3;padding-top:7px}.file-line {font-size:11px;color:#9aa795;word-break:break-all}
@media(min-width:1800px){.camera-title{font-size:15px}.camera-image{max-height:calc(100dvh - 300px)}}
@media(max-width:900px){.nicegui-content{padding:12px 10px 150px}.preview-grid{gap:7px}.action-grid{left:10px;right:10px;grid-template-columns:minmax(0,1fr) 180px;gap:10px}.stage-title{font-size:20px}.task-field{width:260px}.op-button{font-size:23px}}
@media(max-width:560px){.preview-card{padding:5px}.camera-title{font-size:10px}.camera-fps{font-size:8px}.operator-panel{padding:12px}.status-row{align-items:flex-start;flex-direction:column;gap:7px}.task-field{width:100%}.action-grid{grid-template-columns:minmax(0,1fr) 110px;gap:8px}.op-button{font-size:18px}.op-button .q-icon{font-size:23px;margin-right:4px}.stage-title{font-size:19px}.camera-image{max-height:none}}

""",
        shared=True,
    )

    @ui.page("/")
    def page():
        ui.colors(primary="#176b55", positive="#176b55", negative="#bd5d45")
        with ui.row().classes("page-header"):
            with ui.column().classes("gap-1"):
                ui.label("ARX · 数据采集").classes("brand-title")
                ui.label("鼠标停在底部主按钮：开始 → 结束保存 → 场景 reset → 下一条").classes(
                    "subtle"
                )
            ui.label("模拟采集" if mock else "本地采集").classes("mode-pill")
        with ui.element("section").classes("preview-grid"):
            for name, label in zip(CAMERAS, ("主视角", "左腕", "右腕"), strict=True):
                with ui.card().classes("preview-card"):
                    ui.label(label).classes("camera-title")
                    ui.html(
                        f'<div class="camera-feed"><img id="preview-{name}" class="camera-image" alt="{label}">'
                        f'<div id="preview-warning-{name}" class="camera-warning">等待实时画面</div></div>'
                        f'<div id="preview-fps-{name}" class="camera-fps">等待画面</div>'
                    ).classes("w-full")
        # Finite binary responses work with both desktop browsers and the
        # embedded browser proxy. A completed decode is the next request's
        # backpressure signal: each view has at most one request in flight.
        ui.add_body_html("""
<script>
(() => {
  for (const name of ['head', 'left_wrist', 'right_wrist']) {
    let stopped = false, sequence = -1, activeUrl = null, shown = 0;
    let intervalStart = performance.now();
    let lastFreshAt = null;
    function showFreshness() {
      const image = document.getElementById('preview-' + name);
      const warning = document.getElementById('preview-warning-' + name);
      const label = document.getElementById('preview-fps-' + name);
      const age = lastFreshAt === null ? null : performance.now() - lastFreshAt;
      const stale = age === null || age > 500;
      if (image) image.classList.toggle('camera-stale', stale);
      if (warning) {
        warning.hidden = !stale;
        warning.textContent = age === null ? '等待实时画面'
          : '画面已冻结 · 请勿据此判断机械臂位置';
      }
      if (label && stale) label.textContent = age === null ? '尚未收到新画面'
        : '已 ' + (age / 1000).toFixed(1) + ' 秒未更新';
    }
    const watchdog = setInterval(showFreshness, 100);
    const controller = new AbortController();
    window.addEventListener('pagehide', () => {
      stopped = true;
      controller.abort();
      clearInterval(watchdog);
      if (activeUrl) URL.revokeObjectURL(activeUrl);
    }, {once: true});
    async function nextFrame() {
      if (stopped) return;
      const started = performance.now();
      const image = document.getElementById('preview-' + name);
      if (!image || document.hidden) {
        setTimeout(nextFrame, 200);
        return;
      }
      try {
        const response = await fetch('/preview/' + name + '.jpg?after=' + sequence, {
          cache: 'no-store', signal: controller.signal,
        });
        if (response.ok && response.status !== 204) {
          const url = URL.createObjectURL(await response.blob());
          const previous = activeUrl;
          activeUrl = url;
          image.src = url;
          try {
            await image.decode();
            sequence = Number(response.headers.get('X-Sequence'));
            lastFreshAt = started - Number(response.headers.get('X-Frame-Age-Ms') || 0);
            showFreshness();
            shown++;
          } finally {
            if (previous) URL.revokeObjectURL(previous);
          }
          const elapsed = performance.now() - intervalStart;
          if (elapsed >= 1000) {
            const label = document.getElementById('preview-fps-' + name);
            if (label) label.textContent = '浏览器显示 '
              + (shown * 1000 / elapsed).toFixed(1) + ' fps';
            shown = 0;
            intervalStart = performance.now();
          }
        }
      } catch (error) {
        if (error.name === 'AbortError') return;
        const label = document.getElementById('preview-fps-' + name);
        if (label) label.textContent = '预览连接重试中';
      }
      if (!stopped) setTimeout(nextFrame, Math.max(8, 1000 / 30 - (performance.now() - started)));
    }
    nextFrame();
  }
})();
</script>
""")
        with ui.card().classes("operator-panel"):
            with ui.row().classes("status-row"):
                status = ui.label().classes("stage-title")
                task = (
                    ui.input(
                        "任务名称", value=session.data["task"], placeholder="例如：把杯子放入托盘"
                    )
                    .props("outlined dense")
                    .classes("task-field")
                )
            duration = (
                ui.label()
                .classes("text-xl font-semibold")
                .style("color:#176b55;font-variant-numeric:tabular-nums")
            )
            ui.tooltip(
                "按已校验、未作废的正式记录帧数 ÷ FPS 累计；不含试采、录制中、等待和复位时间。"
            )
            message = ui.label().classes("next-step")
            with ui.element("section").classes("action-grid"):
                primary_state = {"action": None}
                primary = (
                    ui.button(
                        "开始",
                        icon="play_arrow",
                        on_click=lambda: perform(
                            lambda: session.advance(primary_state["action"], task.value)
                        ),
                    )
                    .props("unelevated no-caps")
                    .classes("op-button op-start")
                )
                cancel = (
                    ui.button(
                        "本条无效", icon="block", on_click=lambda: perform(session.invalidate)
                    )
                    .props("unelevated no-caps")
                    .classes("op-button op-invalid")
                )
            connections = ui.label().classes("connection-line")
            warning = ui.label().classes("warning-line")
            background_status = ui.label().classes("background-line")
            location = ui.label().classes("file-line")

        def perform(action):
            try:
                action()
            except Exception as error:
                ui.notify(str(error), type="warning", timeout=5000)
            refresh()

        def on_key(event):
            if event.action.keydown and not event.action.repeat:
                action = {
                    "r": lambda: session.start(task.value),
                    "s": session.stop,
                    "d": session.invalidate,
                }.get(event.key.name.lower())
                if action:
                    perform(action)

        ui.keyboard(on_key=on_key, ignore=["input", "textarea", "select", "button"])

        def refresh():
            state = session.status()
            saved = totals.snapshot()
            duration.set_text(
                f"有效采集总时长  {saved['formatted']}  ·  {saved['episodes']} 条正式记录"
            )
            mode = state["mode"]
            if mode == "recording":
                title = f"录制中 · {state['frames']} 帧"
                hint = "任务完成后点击同一主按钮结束保存；本条不满意可点击右侧“本条无效”。"
            elif mode == "saving":
                title = "正在写盘"
                hint = "等待本条文件收尾；后台处理不影响下一条采集。"
            elif mode == "fault":
                title = "采集需要处理"
                hint = state["message"] + (
                    "。本条未成功保存，原始文件已保留；点击“恢复采集”后复位场景。"
                    if state.get("recoverable")
                    else "。请检查设备连接或等待文件写盘完成。"
                )
            elif state["needs_reset"]:
                title = (
                    "本条保存失败 · 请复位场景"
                    if state["incomplete"]
                    else "本条已作废 · 请复位场景"
                    if state["invalid"]
                    else "本条已保存 · 请复位场景"
                )
                hint = "用遥操恢复场景，完成后在原位置点击“场景 reset”；再点一次开始下一条。"
            elif mode == "idle":
                title = "准备好，开始下一条"
                hint = "确认场景就绪，点击“开始”进行采集。"
            else:
                title = "等待设备就绪"
                hint = "检查下方连接状态，相机与遥操数据就绪后即可开始。"
            status.set_text(title)
            message.set_text(hint)
            action = session.primary_action(state)
            primary_state["action"] = action
            labels = {
                "start": ("开始", "play_arrow", "op-start"),
                "stop": ("结束保存", "stop", "op-save"),
                "reset": ("场景 reset", "restart_alt", "op-reset"),
                "recover": ("恢复采集", "refresh", "op-reset"),
            }
            text, icon, color = labels.get(
                action,
                (
                    "正在写盘…"
                    if mode == "saving"
                    else "采集故障"
                    if mode == "fault"
                    else "等待设备就绪",
                    "hourglass_top",
                    "op-save",
                ),
            )
            primary.set_text(text)
            primary.props("icon=" + icon)
            primary.classes(remove="op-start op-save op-reset", add=color)
            import time

            primary.set_enabled(
                action is not None
                and time.monotonic() >= session.primary_until
                and (action != "start" or bool((task.value or "").strip()))
            )
            cancel.set_enabled(
                mode == "recording"
                or (
                    mode in ("idle", "warming")
                    and bool(state["last_episode"])
                    and not state["invalid"]
                )
            )
            task.set_enabled(mode in ("idle", "warming") and not state["needs_reset"])
            labels = [
                label + ("：已连接" if state["inputs"]["cameras"][name] else "：未连接")
                for name, label in zip(CAMERAS, ("主视角", "左腕", "右腕"), strict=True)
            ]
            labels.append("遥操数据：" + ("已连接" if state["inputs"]["telemetry"] else "未连接"))
            connections.set_text("  ·  ".join(labels))
            counts = state["background"].get("counts", {})
            pending = counts.get("queued", 0)
            working = counts.get("validating", 0) + counts.get("converting", 0)
            background_status.set_text(
                f"后台处理：等待 {pending} 条 · 处理中 {working} 条 · 完成 {counts.get('done', 0)} 条 · 失败 {counts.get('failed', 0)} 条。文件写盘后可继续采集。"
            )
            errors = state["background"].get("failures", [])
            warning.set_text(
                "后台处理已停止，请联系负责人恢复；原始文件保留。"
                if not state["worker_alive"]
                else f"后台处理失败：{errors[-1]['error']}"
                if errors
                else "本条无效会保留原始文件，并排除后续处理；场景 reset 仅确认人工复位。"
            )
            location.set_text(
                "本条文件：" + state["last_episode"]
                if state["last_episode"]
                else "本地保存：" + str(config.output)
            )

        refresh()
        ui.timer(0.3, refresh)

    try:
        ui.run(
            host="127.0.0.1",
            port=port,
            title="ARX X5 2023 数据采集",
            reload=False,
            show=False,
            timeout_graceful_shutdown=2,
        )
    finally:
        # Long-lived browser streams must not leave acquisition workers or the
        # telemetry socket behind when the HTTP server is stopped.
        preview.close()
        runtime.close()
