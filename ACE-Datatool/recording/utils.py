from __future__ import annotations  # 如果有

import concurrent.futures
import csv
import json
import os
import queue
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

# 条件导入的第三方库
try:
    import pyzed.sl as sl

    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    print("[Warning] ZED SDK not available")
try:
    import quality_check

    QUALITY_CHECK_AVAILABLE = True
    QUALITY_CHECK_ERROR = ""
except Exception as error:  # pylint: disable=broad-except
    QUALITY_CHECK_AVAILABLE = False
    QUALITY_CHECK_ERROR = str(error)

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes

    CRYPTO_AVAILABLE = True
    CRYPTO_ERROR = ""
except Exception as error:  # pylint: disable=broad-except
    CRYPTO_AVAILABLE = False
    CRYPTO_ERROR = str(error)


def _derive_file_key(master_key: bytes, salt: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=b"ace-file-encryption",
    ).derive(master_key)
_preview_instances = {}
_preview_lock = threading.Lock()
_zed_open_lock = threading.Lock()

EGOCAP_SN = os.getenv("EGOCAP_SN", "0955623785")

DEFAULT_ENCRYPTION_KEY = b"my_secret_key_32_bytes_long_abc!"
RECORDING_SUMMARY_NAME = "recording_summary.json"


class EncryptionService:
    """Background encryptor for completed recording sessions."""

    def __init__(
        self,
        key: bytes,
        output_subdir: str = "_encrypted",
        chunk_size: int = 4 * 1024 * 1024,
    ) -> None:
        self._key = key
        self._output_subdir = output_subdir
        self._chunk_size = chunk_size
        self._max_workers = self._infer_workers()
        self._queue: queue.Queue = queue.Queue()
        self._running = threading.Event()
        self._running.set()
        self._inflight: set[str] = set()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._worker, daemon=True)

    @staticmethod
    def _infer_workers() -> int:
        override = os.getenv("ACE_ENCRYPTION_WORKERS", "4").strip()
        if override:
            try:
                value = int(override)
                if value > 0:
                    return value
            except ValueError:
                pass

        cpu_count = os.cpu_count() or 1
        return max(1, min(4, cpu_count))

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._running.clear()
        self._queue.put(None)
        self._thread.join(timeout=timeout)

    def enqueue_session(self, session_path: str, reason: str = "") -> None:
        if not session_path:
            return

        real_path = os.path.abspath(session_path)
        if not os.path.isdir(real_path):
            return

        output_dir = os.path.join(real_path, self._output_subdir)
        done_marker = os.path.join(output_dir, ".encrypt_done")
        if os.path.exists(done_marker):
            return

        with self._lock:
            if real_path in self._inflight:
                return
            self._inflight.add(real_path)

        self._queue.put((real_path, reason))

    def _worker(self) -> None:
        while self._running.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                continue

            session_path, reason = item
            try:
                self._encrypt_session(session_path, reason)
            except Exception as error:  # pylint: disable=broad-except
                print(f"[Encryption] Failed to encrypt {session_path}: {error}")
            finally:
                with self._lock:
                    self._inflight.discard(session_path)
                self._queue.task_done()

    def _encrypt_session(self, session_path: str, reason: str) -> None:
        output_dir = os.path.join(session_path, self._output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        in_progress = os.path.join(output_dir, ".encrypt_in_progress")
        done_marker = os.path.join(output_dir, ".encrypt_done")
        with open(in_progress, "w", encoding="utf-8") as marker:
            marker.write(
                f"started={datetime.now().isoformat()}\nreason={reason}\n"
            )

        targets = self._collect_targets(session_path)
        if targets:
            self._encrypt_files_parallel(output_dir, targets)

        self._cleanup_empty_dirs(session_path)

        with open(done_marker, "w", encoding="utf-8") as marker:
            marker.write(f"done={datetime.now().isoformat()}\n")

        try:
            os.remove(in_progress)
        except OSError:
            pass

        print(f"[Encryption] Completed session encrypt: {session_path}")

    def _collect_targets(self, session_path: str) -> list[tuple[str, str]]:
        targets: list[tuple[str, str]] = []
        for root, dirs, files in os.walk(session_path, topdown=True):
            if self._output_subdir in dirs:
                dirs.remove(self._output_subdir)

            for filename in files:
                source_path = os.path.join(root, filename)
                if not os.path.isfile(source_path):
                    continue

                rel_path = os.path.relpath(source_path, session_path)
                if rel_path.startswith(self._output_subdir + os.sep):
                    continue

                targets.append((source_path, rel_path))
        return targets

    def _encrypt_files_parallel(
        self, output_dir: str, targets: list[tuple[str, str]]
    ) -> None:
        if self._max_workers <= 1:
            for source_path, rel_path in targets:
                self._encrypt_file(source_path, output_dir, rel_path)
            return

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self._max_workers
            ) as executor:
                futures = [
                    executor.submit(
                        _encrypt_file_task,
                        source_path,
                        output_dir,
                        rel_path,
                        self._key,
                        self._chunk_size,
                    )
                    for source_path, rel_path in targets
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        except Exception as error:  # pylint: disable=broad-except
            print(f"[Encryption] Parallel encrypt failed, fallback: {error}")
            for source_path, rel_path in targets:
                self._encrypt_file(source_path, output_dir, rel_path)

    def _encrypt_file(self, source_path: str, output_dir: str, metadata_path: str) -> None:
        metadata = metadata_path.encode("utf-8")
        if len(metadata) > 65535:
            raise ValueError("metadata path too long for encryption header")

        timestamp = int(time.time() * 1000)
        rand_suffix = os.urandom(2).hex()
        secure_filename = f"dat_{timestamp}_{rand_suffix}"
        output_path = os.path.join(output_dir, secure_filename)

        salt = os.urandom(16)
        file_key = _derive_file_key(self._key, salt)
        nonce = os.urandom(12)
        encryptor = Cipher(
            algorithms.AES(file_key), modes.GCM(nonce)
        ).encryptor()
        encrypted_metadata = encryptor.update(metadata)

        temp_path = output_path + ".tmp"
        with open(source_path, "rb") as source_file, open(
            temp_path, "wb", buffering=4 * 1024 * 1024
        ) as encrypted_file:
            encrypted_file.write(salt)
            encrypted_file.write(nonce)
            encrypted_file.write(struct.pack("!H", len(metadata)))
            encrypted_file.write(encrypted_metadata)

            while True:
                chunk = source_file.read(self._chunk_size)
                if not chunk:
                    break
                encrypted_file.write(encryptor.update(chunk))

            encryptor.finalize()
            encrypted_file.write(encryptor.tag)

        os.replace(temp_path, output_path)
        self._delete_plaintext_file(source_path)

    def _delete_plaintext_file(self, source_path: str) -> None:
        try:
            os.remove(source_path)
        except OSError as error:
            print(f"[Encryption] Failed to delete {source_path}: {error}")

    def _cleanup_empty_dirs(self, session_path: str) -> None:
        for root, dirs, _ in os.walk(session_path, topdown=False):
            rel_root = os.path.relpath(root, session_path)
            if rel_root == self._output_subdir:
                continue
            if rel_root.startswith(self._output_subdir + os.sep):
                continue
            try:
                os.rmdir(root)
            except OSError:
                pass


def _encrypt_file_task(
    source_path: str,
    output_dir: str,
    metadata_path: str,
    key: bytes,
    chunk_size: int,
) -> None:
    metadata = metadata_path.encode("utf-8")
    if len(metadata) > 65535:
        raise ValueError("metadata path too long for encryption header")

    timestamp = int(time.time() * 1000)
    rand_suffix = os.urandom(2).hex()
    secure_filename = f"dat_{timestamp}_{rand_suffix}"
    output_path = os.path.join(output_dir, secure_filename)

    salt = os.urandom(16)
    file_key = _derive_file_key(key, salt)
    nonce = os.urandom(12)
    encryptor = Cipher(
        algorithms.AES(file_key), modes.GCM(nonce)
    ).encryptor()
    encrypted_metadata = encryptor.update(metadata)

    temp_path = output_path + ".tmp"
    with open(source_path, "rb") as source_file, open(
        temp_path, "wb", buffering=4 * 1024 * 1024
    ) as encrypted_file:
        encrypted_file.write(salt)
        encrypted_file.write(nonce)
        encrypted_file.write(struct.pack("!H", len(metadata)))
        encrypted_file.write(encrypted_metadata)

        while True:
            chunk = source_file.read(chunk_size)
            if not chunk:
                break
            encrypted_file.write(encryptor.update(chunk))

        encryptor.finalize()
        encrypted_file.write(encryptor.tag)

    os.replace(temp_path, output_path)
    try:
        os.remove(source_path)
    except OSError as error:
        print(f"[Encryption] Failed to delete {source_path}: {error}")
def get_color_sync_web_assets() -> Tuple[str, str, str]:
    """Returns HTML + start/stop JS for browser-based color sync display."""
    html = """
    <style>
      #color-sync-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 220px;
        border: 1px dashed #888;
        border-radius: 10px;
        padding: 12px;
        box-sizing: border-box;
      }
      #color-sync-panel {
        width: min(90vw, 960px);
        height: min(56vw, 540px);
        max-height: 70vh;
        border-radius: 10px;
        background: #111;
                color: #fff;
                position: relative;
        user-select: none;
        -webkit-user-select: none;
        padding: 16px;
                box-sizing: border-box;
            }
            #color-sync-info {
                position: absolute;
                top: 14px;
                left: 14px;
                font-size: 18px;
                text-align: left;
                color: #fff;
                background: rgba(0, 0, 0, 0.35);
                padding: 8px 10px;
                border-radius: 8px;
            }
            #color-sync-back-btn {
                position: absolute;
                left: 14px;
                bottom: 14px;
                border: none;
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 16px;
                cursor: pointer;
                background: rgba(0, 0, 0, 0.55);
                color: #fff;
      }
    </style>
    <div id="color-sync-wrap">
            <div id="color-sync-panel">
                <div id="color-sync-info">点击“启动颜色同步”后，这里会按蓝绿红紫黄循环变色</div>
                <button id="color-sync-back-btn" type="button">回退</button>
            </div>
    </div>
    """

    start_js = """
    (width, height, interval, windowed) => {
        const panel = document.getElementById('color-sync-panel');
        const info = document.getElementById('color-sync-info');
        const backBtn = document.getElementById('color-sync-back-btn');
        const stopBtn = document.getElementById('color_sync_stop_btn_id');
        if (!panel) {
            return ['', '❌ 未找到颜色同步显示面板，请刷新页面后重试。'];
        }
        if (!info || !backBtn) {
            return ['', '❌ 颜色同步面板元素缺失，请刷新页面后重试。'];
        }

        if (!window.__colorSyncState) {
            window.__colorSyncState = {
                timer: null,
                idx: 0,
                colors: ['#0000FF', '#00FF00', '#FF0000', '#FF00FF', '#FFFF00'],
                names: ['Blue', 'Green', 'Red', 'Magenta', 'Yellow'],
                logs: [],
                keyHandler: null,
                backHandler: null
            };
        }

        const state = window.__colorSyncState;
        state.idx = 0;
        state.logs = [];

        if (state.timer) {
            clearInterval(state.timer);
            state.timer = null;
        }
        if (state.keyHandler) {
            document.removeEventListener('keydown', state.keyHandler);
            state.keyHandler = null;
        }
        if (state.backHandler) {
            backBtn.removeEventListener('click', state.backHandler);
            state.backHandler = null;
        }

        const safeInterval = Math.max(0.05, Number(interval) || 0.4);
        const safeWidth = Math.max(1, Math.floor(Number(width) || 1920));
        const safeHeight = Math.max(1, Math.floor(Number(height) || 1080));

        const updatePanelText = (name) => {
            const now = new Date();
            const ms = String(now.getMilliseconds()).padStart(3, '0');
            const ts = now.toLocaleTimeString() + '.' + ms;
            info.textContent = `Time: ${ts} | Color: ${name} | 按 q 或左下角回退退出`;
        };

        const applyColor = () => {
            const color = state.colors[state.idx];
            const name = state.names[state.idx];
            panel.style.background = color;
            updatePanelText(name);
            state.logs.push({
                timestamp: Date.now() / 1000,
                color_index: state.idx,
                color_name: name
            });
            state.idx = (state.idx + 1) % state.colors.length;
        };

        panel.style.width = windowed ? `${safeWidth}px` : '100vw';
        panel.style.height = windowed ? `${safeHeight}px` : '100vh';
        panel.style.maxWidth = windowed ? `${safeWidth}px` : '100vw';
        panel.style.maxHeight = windowed ? `${safeHeight}px` : '100vh';
        panel.style.borderRadius = windowed ? '10px' : '0';

        if (!windowed) {
            const requestFS = panel.requestFullscreen
                || panel.webkitRequestFullscreen
                || panel.mozRequestFullScreen
                || panel.msRequestFullscreen;
            if (requestFS) {
                try {
                    requestFS.call(panel);
                } catch (error) {
                    console.warn('requestFullscreen failed:', error);
                }
            }
        }

        const performStop = (triggerBackend = true) => {
            if (state.timer) {
                clearInterval(state.timer);
                state.timer = null;
            }

            if (state.keyHandler) {
                document.removeEventListener('keydown', state.keyHandler);
                state.keyHandler = null;
            }
            if (state.backHandler) {
                backBtn.removeEventListener('click', state.backHandler);
                state.backHandler = null;
            }

            panel.style.background = '#111';
            panel.style.width = 'min(90vw, 960px)';
            panel.style.height = 'min(56vw, 540px)';
            panel.style.maxWidth = 'min(90vw, 960px)';
            panel.style.maxHeight = '70vh';
            panel.style.borderRadius = '10px';
            info.textContent = '颜色同步已停止';

            const exitFS = document.exitFullscreen
                || document.webkitExitFullscreen
                || document.mozCancelFullScreen
                || document.msExitFullscreen;
            if (exitFS) {
                try {
                    exitFS.call(document);
                } catch (error) {
                    console.warn('exitFullscreen failed:', error);
                }
            }

            if (triggerBackend && stopBtn) {
                stopBtn.click();
            }
        };

        window.__colorSyncStop = performStop;
        state.keyHandler = (event) => {
            if ((event.key || '').toLowerCase() === 'q') {
                performStop();
            }
        };
        state.backHandler = () => {
            performStop();
        };
        document.addEventListener('keydown', state.keyHandler);
        backBtn.addEventListener('click', state.backHandler);

        applyColor();
        state.timer = setInterval(applyColor, safeInterval * 1000);
        return [`颜色同步已启动（浏览器模式，间隔 ${safeInterval.toFixed(2)} 秒）。`, '', ''];
    }
    """

    stop_js = """
    (logsJsonFromInput) => {
        const logs = (
            window.__colorSyncState
            && Array.isArray(window.__colorSyncState.logs)
        ) ? window.__colorSyncState.logs : [];
        const logsJson = JSON.stringify(logs);

        if (typeof window.__colorSyncStop === 'function') {
            window.__colorSyncStop(false);
            return [logsJson];
        }

        const info = document.getElementById('color-sync-info');
        if (info) {
            info.textContent = '颜色同步已停止';
        }

        return [logsJson];
    }
    """

    return html, start_js, stop_js


class WebRealSensePreview:
    """用于 Web 的 RealSense 预览类"""

    def __init__(
        self, device_index: int = 0, show_depth: bool = True, serial_number: str = None
    ):
        """
        初始化 Web 预览

        Args:
            device_index: 相机索引
            show_depth: 是否显示深度图
            serial_number: 相机序列号 (可选，如果提供则优先使用)
        """
        self.device_index = device_index
        self.serial_number = serial_number
        self.show_depth = show_depth
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.colorizer = rs.colorizer()
        self.align = None
        self.is_running = False
        self.frame_queue: Queue = Queue(maxsize=2)
        self.depth_queue: Queue = Queue(maxsize=2)
        self.thread = None

    def configure(self) -> bool:
        """配置相机参数"""
        try:
            ctx = rs.context()
            devices = ctx.query_devices()

            device = None

            # 1. 优先尝试通过序列号匹配设备
            if self.serial_number:
                print(f"Looking for camera with SN: {self.serial_number}")
                for i in range(len(devices)):
                    try:
                        d = devices[i]
                        if d.supports(rs.camera_info.serial_number):
                            if (
                                d.get_info(rs.camera_info.serial_number)
                                == self.serial_number
                            ):
                                device = d
                                print(
                                    f"Found device with SN {self.serial_number} at index {i}"
                                )
                                break
                    except Exception as e:
                        print(f"Error checking device {i}: {e}")
                        continue

            # 2. 如果没找到或没提供SN，回退到使用索引
            if device is None:
                if self.device_index >= len(devices):
                    print(
                        f"Error: Device {self.device_index} not found (Total: {len(devices)})"
                    )
                    return False
                device = devices[self.device_index]

            # 获取设备信息
            try:
                sn = device.get_info(rs.camera_info.serial_number)
                name = device.get_info(rs.camera_info.name)
                print(f"[Camera Setup] SN: {sn}, Model: {name}")

                # 配置流
                self.config.enable_device(sn)
                self.config.enable_stream(
                    rs.stream.depth, -1, 848, 480, rs.format.z16, 15
                )
                self.config.enable_stream(
                    rs.stream.color, -1, 848, 480, rs.format.rgb8, 15
                )

                # 启动管道
                profile = self.pipeline.start(self.config)

                # 对齐
                self.align = rs.align(rs.stream.color)

                return True
            except Exception as e:
                print(f"Error starting pipeline: {e}")
                return False

        except Exception as e:
            print(f"Error configuring camera: {e}")
            return False

    def _capture_loop(self):
        """内部捕获循环"""
        while self.is_running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)

                if self.align:
                    frames = self.align.process(frames)

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                # 获取 RGB 图像
                rgb_image = np.asanyarray(color_frame.get_data())

                # 尝试放入队列（非阻塞，满了就丢弃旧帧）
                try:
                    self.frame_queue.put_nowait(rgb_image)
                except:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(rgb_image)
                    except:
                        pass

                # 处理深度图
                if self.show_depth and depth_frame:
                    depth_colormap = np.asanyarray(
                        self.colorizer.colorize(depth_frame).get_data()
                    )
                    try:
                        self.depth_queue.put_nowait(depth_colormap)
                    except:
                        try:
                            self.depth_queue.get_nowait()
                            self.depth_queue.put_nowait(depth_colormap)
                        except:
                            pass

            except Exception as e:
                if self.is_running:
                    print(f"Error in capture loop: {e}")
                time.sleep(0.01)

    def start(self) -> bool:
        """启动预览"""
        if not self.configure():
            return False

        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def get_frame(self) -> Optional[np.ndarray]:
        """获取最新的 RGB 帧"""
        try:
            return self.frame_queue.get(timeout=0.1)
        except:
            return None

    def get_depth_frame(self) -> Optional[np.ndarray]:
        """获取最新的深度帧"""
        try:
            return self.depth_queue.get(timeout=0.1)
        except:
            return None

    def stop(self):
        """停止预览"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)

        try:
            self.pipeline.stop()
        except Exception:
            pass

        self.align = None
        self.config = None
        self.pipeline = None

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Exception:
                break

        while not self.depth_queue.empty():
            try:
                self.depth_queue.get_nowait()
            except Exception:
                break

        print(f"Camera {self.device_index} stopped")


class WebZEDPreview:
    """用于 Web 的 ZED 预览类"""

    def __init__(
        self, device_index: int = 0, show_depth: bool = True, serial_number: int = None
    ):
        """
        初始化 ZED Web 预览

        Args:
            device_index: 相机索引
            show_depth: 是否显示深度图
            serial_number: 相机序列号 (可选)
        """
        self.device_index = device_index
        self.serial_number = serial_number
        self.show_depth = show_depth
        self.zed = None
        self.is_running = False
        self.frame_queue: Queue = Queue(maxsize=2)
        self.depth_queue: Queue = Queue(maxsize=2)
        self.thread = None
        self.is_mono = False

    @staticmethod
    def _restart_zed_argus_daemons() -> bool:
        """Best-effort recovery for Jetson Argus timeout before retrying ZED open."""
        passwd = os.getenv("ACE_SUDO_PASSWORD", "3456999")
        commands = [
            "systemctl restart zed_x_daemon.service",
            "systemctl restart zed_daemon.service",
            "systemctl restart zedx-daemon.service",
            "systemctl restart nvargus-daemon.service",
        ]

        all_ok = True
        for cmd in commands:
            try:
                full_cmd = f"echo {shlex.quote(passwd)} | sudo -S {cmd}"
                result = subprocess.run(
                    ["bash", "-lc", full_cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if result.returncode == 0:
                    print(f"[ZED] Recovery command ok: {cmd}")
                else:
                    # 某些机器上 service 名称不同，允许单条失败继续尝试其它候选。
                    print(f"[ZED] Recovery command failed: {cmd} | output: {result.stdout.strip()}")
            except Exception as error:
                print(f"[ZED] Recovery command exception ({cmd}): {error}")

        # 等待服务 socket 稳定
        time.sleep(2.0)
        return all_ok

    def configure(self) -> bool:
        """配置ZED相机参数"""
        if not ZED_AVAILABLE:
            print("[ZED] SDK not available")
            return False

        try:
            # 串行化 open，避免多线程/多请求并发触发 Argus socket InvalidState。
            with _zed_open_lock:
                # 获取设备列表
                dev_stereo_list = sl.Camera.get_device_list()
                dev_mono_list = (
                    sl.CameraOne.get_device_list() if hasattr(sl, "CameraOne") else []
                )

                all_devices = []
                for dev in dev_stereo_list:
                    all_devices.append(("stereo", dev))
                for dev in dev_mono_list:
                    all_devices.append(("mono", dev))

                if not all_devices:
                    print("[ZED] No devices found")
                    return False

                # 选择设备
                cam_type, device = None, None
                if self.serial_number:
                    # 通过序列号查找
                    for ct, dev in all_devices:
                        if dev.serial_number == self.serial_number:
                            cam_type, device = ct, dev
                            break
                else:
                    # 通过索引查找
                    if self.device_index < len(all_devices):
                        cam_type, device = all_devices[self.device_index]

                if not device:
                    print(f"[ZED] Device {self.device_index} not found")
                    return False

                self.is_mono = cam_type == "mono"

                # 创建参数
                if self.is_mono:
                    init_params = sl.InitParametersOne()
                else:
                    init_params = sl.InitParameters()

                # 配置参数
                init_params.set_from_serial_number(device.serial_number)

                # 使用 AUTO 让 SDK 自动选择合适的分辨率
                init_params.camera_resolution = sl.RESOLUTION.SVGA
                init_params.camera_fps = 15

                # 只有双目相机支持深度
                if not self.is_mono:
                    init_params.depth_mode = (
                        sl.DEPTH_MODE.PERFORMANCE if self.show_depth else sl.DEPTH_MODE.NONE
                    )

                retry_keywords = [
                    "TIMEOUT",
                    "CAMERA NOT DETECTED",
                    "FAILURE",
                    "INVALIDSTATE",
                    "RECEIVE THREAD IS NOT RUNNING",
                    "ERROR_CODE.FAILURE",
                ]

                # 多次尝试打开，每次失败后恢复服务并指数退避。
                for attempt in range(3):
                    if self.is_mono:
                        self.zed = sl.CameraOne()
                    else:
                        self.zed = sl.Camera()

                    err = self.zed.open(init_params)
                    if err == sl.ERROR_CODE.SUCCESS:
                        print(f"[ZED] Camera {device.serial_number} ({device.camera_model}) opened")
                        return True

                    err_text = str(err)
                    err_upper = err_text.upper()
                    print(f"[ZED] Failed to open camera (attempt {attempt + 1}/3): {err_text}")

                    # 失败后主动关闭，避免复用异常句柄。
                    try:
                        if self.zed and self.zed.is_opened():
                            self.zed.close()
                    except Exception:
                        pass
                    finally:
                        self.zed = None

                    should_recover = any(keyword in err_upper for keyword in retry_keywords)
                    if not should_recover:
                        return False

                    print("[ZED] Detected Argus/timeout-like failure, trying daemon recovery...")
                    self._restart_zed_argus_daemons()
                    time.sleep(1.2 + attempt * 0.8)

                return False

        except Exception as e:
            print(f"[ZED] Error configuring camera: {e}")
            return False

    def _capture_loop(self):
        """内部捕获循环"""
        image = sl.Mat()
        # 只有双目相机才获取深度
        depth_map = sl.Mat() if self.show_depth and not self.is_mono else None

        while self.is_running:
            try:
                # 采集帧
                if self.is_mono:
                    if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
                        time.sleep(0.01)
                        continue
                else:
                    if self.zed.grab(sl.RuntimeParameters()) != sl.ERROR_CODE.SUCCESS:
                        time.sleep(0.01)
                        continue

                # 获取RGB图像
                self.zed.retrieve_image(image, sl.VIEW.LEFT)
                rgb_data = image.get_data()

                if rgb_data is not None:
                    # 转换为RGB格式
                    if rgb_data.shape[2] == 4:  # BGRA
                        rgb_image = cv2.cvtColor(rgb_data, cv2.COLOR_BGRA2RGB)
                    else:  # BGR
                        rgb_image = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)

                    # 调整到合适大小
                    rgb_image = cv2.resize(rgb_image, (848, 480))

                    try:
                        self.frame_queue.put_nowait(rgb_image)
                    except:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(rgb_image)
                        except:
                            pass

                # 获取深度图（仅双目相机）
                if depth_map is not None:
                    self.zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
                    depth_data = depth_map.get_data()

                    if depth_data is not None:
                        # 归一化深度图到可视范围
                        depth_norm = np.nan_to_num(
                            depth_data, nan=0.0, posinf=0.0, neginf=0.0
                        )
                        depth_norm = np.clip(depth_norm, 0, 10000)  # 限制到10米
                        depth_norm = (depth_norm / 10000 * 255).astype(np.uint8)
                        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                        depth_colormap = cv2.resize(depth_colormap, (848, 480))
                        depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)

                        try:
                            self.depth_queue.put_nowait(depth_colormap)
                        except:
                            try:
                                self.depth_queue.get_nowait()
                                self.depth_queue.put_nowait(depth_colormap)
                            except:
                                pass
                else:
                    # 单目相机没有深度，推送一个占位图
                    placeholder = np.zeros((480, 848, 3), dtype=np.uint8)
                    cv2.putText(
                        placeholder,
                        "No Depth (Mono Camera)",
                        (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (128, 128, 128),
                        2,
                    )
                    try:
                        self.depth_queue.put_nowait(placeholder)
                    except:
                        try:
                            self.depth_queue.get_nowait()
                            self.depth_queue.put_nowait(placeholder)
                        except:
                            pass

            except Exception as e:
                if self.is_running:
                    print(f"[ZED] Error in capture loop: {e}")
                time.sleep(0.01)

    def start(self) -> bool:
        """启动预览"""
        if not self.configure():
            return False

        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def get_frame(self) -> Optional[np.ndarray]:
        """获取最新的 RGB 帧"""
        try:
            return self.frame_queue.get(timeout=0.1)
        except:
            return None

    def get_depth_frame(self) -> Optional[np.ndarray]:
        """获取最新的深度帧"""
        try:
            return self.depth_queue.get(timeout=0.1)
        except:
            return None

    def stop(self):
        """停止预览"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        try:
            if self.zed and self.zed.is_opened():
                self.zed.close()
        except Exception:
            pass
        self.zed = None
        time.sleep(0.2)
        print(f"[ZED] Camera {self.device_index} stopped")


def get_preview_instance(
    device_index: int = 0,
    show_depth: bool = True,
    auto_start: bool = False,
    serial_number: str = None,
    camera_type: str = "realsense",
) -> object:
    """获取或创建预览实例

    Args:
        device_index: 相机索引
        show_depth: 是否显示深度
        auto_start: 是否自动启动
        serial_number: 指定序列号
        camera_type: 相机类型 "realsense" 或 "zed"
    """
    global _preview_instances

    # 构建 key
    if serial_number:
        key = f"{camera_type}_sn_{serial_number}"
    else:
        key = f"{camera_type}_{device_index}"

    with _preview_lock:
        if key not in _preview_instances:
            # 根据类型创建实例
            if camera_type == "zed":
                if not ZED_AVAILABLE:
                    print("[get_preview_instance] ZED SDK not available")
                    return None
                preview = WebZEDPreview(
                    device_index=device_index,
                    show_depth=show_depth,
                    serial_number=int(serial_number) if serial_number else None,
                )
            else:
                preview = WebRealSensePreview(
                    device_index=device_index,
                    show_depth=show_depth,
                    serial_number=serial_number,
                )

            if auto_start:
                if preview.start():
                    _preview_instances[key] = preview
                    print(f"[get_preview_instance] Started {key}")
                    return preview
                else:
                    print(f"[get_preview_instance] Failed to start {key}")
                    return None
            else:
                _preview_instances[key] = preview
                print(f"[get_preview_instance] Created {key} (not started)")
                return preview

        return _preview_instances.get(key)


def stop_all_previews():
    """停止所有预览实例"""
    global _preview_instances

    with _preview_lock:
        for key in list(_preview_instances.keys()):
            try:
                _preview_instances[key].stop()
            except Exception as e:
                print(f"Error stopping {key}: {e}")
            del _preview_instances[key]
        _preview_instances.clear()

    # 给驱动层一点时间释放 /dev/video* 句柄，避免后续录制立即启动时报 busy。
    time.sleep(0.5)


def cleanup_device(device_index: int = 0, camera_type: str = "realsense"):
    """清理特定设备"""
    global _preview_instances

    key = f"{camera_type}_{device_index}"

    with _preview_lock:
        if key in _preview_instances:
            try:
                preview = _preview_instances[key]
                preview.is_running = False
                preview.stop()
                print(f"[cleanup_device] Stopped {key}")
            except Exception as e:
                print(f"Error cleaning up {key}: {e}")
            finally:
                del _preview_instances[key]
        print(f"[cleanup_device] Instance count: {len(_preview_instances)}")


def get_rgb_frame(device_index: int = 0, camera_type: str = "realsense") -> np.ndarray:
    """
    获取 RGB 帧用于 Gradio Image 组件
    """
    try:
        key = f"{camera_type}_{device_index}"
        preview = _preview_instances.get(key)

        if preview is None or not preview.is_running:
            return np.zeros((480, 848, 3), dtype=np.uint8)

        frame = preview.get_frame()
        if frame is not None:
            return frame
        else:
            return np.zeros((480, 848, 3), dtype=np.uint8)
    except Exception as e:
        print(f"Error in get_rgb_frame: {e}")
        return np.zeros((480, 848, 3), dtype=np.uint8)


def get_depth_and_rgb(
    device_index: int = 0, camera_type: str = "realsense"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取深度和 RGB 帧
    """
    try:
        key = f"{camera_type}_{device_index}"
        preview = _preview_instances.get(key)

        if preview is None or not preview.is_running:
            black = np.zeros((480, 848, 3), dtype=np.uint8)
            return black, black

        rgb = preview.get_frame()
        depth = preview.get_depth_frame()

        if rgb is None:
            rgb = np.zeros((480, 848, 3), dtype=np.uint8)
        if depth is None:
            depth = np.zeros((480, 848, 3), dtype=np.uint8)

        return rgb, depth
    except Exception as e:
        print(f"Error in get_depth_and_rgb: {e}")
        black = np.zeros((480, 848, 3), dtype=np.uint8)
        return black, black


def list_cameras(camera_type: str = "realsense") -> list:
    """列出所有可用的相机
    
    Args:
        camera_type: "realsense" 或 "zed" 或 "all"
    """
    cameras = []
    
    # 列出 RealSense 相机
    if camera_type in ["realsense", "all"]:
        try:
            import subprocess
            # Count actual Intel (8086) USB devices plugged in to safely limit query
            try:
                num_intel = int(subprocess.check_output("lsusb | grep -i 8086 | wc -l", shell=True))
            except Exception:
                num_intel = 4
                
            ctx = rs.context()
            devices = ctx.query_devices()
            num_devices = len(devices)
            
            seen_sns = set()
            max_check = min(num_devices, num_intel)
            max_check = min(max_check, 4)
            
            for i in range(max_check):
                try:
                    dev = devices[i]
                    if not dev.supports(rs.camera_info.serial_number):
                        continue
                        
                    sn = dev.get_info(rs.camera_info.serial_number)
                    name = dev.get_info(rs.camera_info.name)
                    
                    if sn in seen_sns:
                        continue
                    
                    seen_sns.add(sn)
                    
                    cameras.append({
                        'index': i,
                        'name': f"RealSense {name}",
                        'sn': sn,
                        'type': 'realsense'
                    })
                except RuntimeError as e:
                    print(f"[list_cameras] RuntimeError for RealSense device {i}: {e}")
                    break
                except Exception as e:
                    print(f"[list_cameras] Error getting RealSense info for device {i}: {e}")
                    continue
        except Exception as e:
            print(f"Error listing RealSense cameras: {e}")
    
    # 列出 ZED 相机
    if camera_type in ["zed", "all"] and ZED_AVAILABLE:
        try:
            dev_stereo_list = sl.Camera.get_device_list()
            dev_mono_list = sl.CameraOne.get_device_list() if hasattr(sl, 'CameraOne') else []
            
            idx = 0
            for dev in dev_stereo_list:
                cameras.append({
                    'index': idx,
                    'name': f"ZED {dev.camera_model}",
                    'sn': str(dev.serial_number),
                    'type': 'zed'
                })
                idx += 1
            
            for dev in dev_mono_list:
                cameras.append({
                    'index': idx,
                    'name': f"ZED {dev.camera_model}",
                    'sn': str(dev.serial_number),
                    'type': 'zed'
                })
                idx += 1
                
        except Exception as e:
            print(f"Error listing ZED cameras: {e}")
    
    return cameras



def format_record_name(name: str) -> str:
    """Formats a record name with a six-digit numeric prefix when applicable."""
    if not name:
        return str(time.time())

    parts = name.split("_", 1)
    if parts[0].isdigit():
        prefix = f"{int(parts[0]):06d}"
        if len(parts) > 1:
            return f"{prefix}_{parts[1]}"
        return prefix
    return name


def parse_device_index(device_index: str) -> Optional[int]:
    """Parses the device index from a dropdown string like '0 (SN: xxx)'."""
    if not device_index:
        return None
    index_str = device_index.split()[0] if " " in device_index else device_index
    try:
        return int(index_str)
    except (TypeError, ValueError):
        return None


def ensure_executable(path: str) -> None:
    """Ensures the file at path is executable."""
    if os.path.exists(path) and not os.access(path, os.X_OK):
        try:
            os.chmod(path, 0o755)
            print(f"Made executable: {path}")
        except OSError as error:
            print(f"Failed to make executable {path}: {error}")


@dataclass(frozen=True)
class RuntimeConfig:
    """Holds runtime path configuration."""

    app_dir: str
    python_cmd: str

    @classmethod
    def detect(cls) -> "RuntimeConfig":
        """Detects runtime paths for frozen or development mode."""
        if getattr(sys, "frozen", False):
            exe_dir = os.path.dirname(os.path.abspath(sys.executable))
            print(f"Running in frozen mode. EXE_DIR: {exe_dir}")
            return cls(app_dir=exe_dir, python_cmd=sys.executable)

        app_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Running in dev mode. APP_DIR: {app_dir}")
        return cls(app_dir=app_dir, python_cmd=sys.executable)


class MachineInfo:
    """Machine information helpers."""

    @staticmethod
    def get_machine_sn() -> str:
        """Returns machine serial number (device-tree or hostname)."""
        try:
            serial_path = "/proc/device-tree/serial-number"
            if os.path.exists(serial_path):
                with open(serial_path, "r", encoding="utf-8") as file:
                    return file.read().strip().replace("\x00", "")
        except OSError:
            pass

        try:
            return socket.gethostname()
        except OSError:
            return "unknown_sn"


class TaskTimestampRecorder:
    """Manages task timestamp CSV creation and writes."""

    def __init__(self) -> None:
        self._task_counter = 0
        self._task_csv_path: Optional[str] = None
        self._task_file_writable = False
        self._current_record_name: Optional[str] = None
        self._lock = threading.Lock()

    def initialize(self, record_name: str, auto_create_dir: bool = False) -> str:
        """Initializes task CSV under date/sn/record_name."""
        if not record_name:
            return "错误: 未指定录制名称"

        with self._lock:
            self._current_record_name = format_record_name(record_name)
            date_str = datetime.now().strftime("%Y%m%d")
            save_dir = os.path.join(date_str, self._current_record_name)

            if auto_create_dir:
                os.makedirs(save_dir, exist_ok=True)
            elif not os.path.exists(save_dir):
                return f"错误: 录制目录不存在: {save_dir}"

            target_csv_path = os.path.join(save_dir, "task_timestamps.csv")
            header_line = "task_id,event,timestamp"

            def _needs_header_repair(path: str) -> bool:
                if not os.path.exists(path):
                    return True
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        first_line = file.readline().strip()
                    return first_line != header_line
                except OSError:
                    return True

            if self._task_csv_path == target_csv_path and self._task_file_writable:
                if _needs_header_repair(target_csv_path):
                    self._task_counter = 0
                    with open(target_csv_path, "w", encoding="utf-8") as file:
                        file.write("task_id,event,timestamp\n")
                    self._task_file_writable = True
                    return f"✅ 任务时间戳记录已重新初始化: {target_csv_path}"
                return f"✅ 任务时间戳记录已初始化 (激活中): {target_csv_path}"

            if os.path.exists(target_csv_path):
                if _needs_header_repair(target_csv_path):
                    self._task_csv_path = target_csv_path
                    self._task_counter = 0
                    with open(target_csv_path, "w", encoding="utf-8") as file:
                        file.write("task_id,event,timestamp\n")
                    self._task_file_writable = True
                    return f"✅ 任务时间戳记录已修复并初始化: {target_csv_path}"

                self._task_csv_path = target_csv_path
                self._task_file_writable = False
                return f"❌ 时间戳文件已存在，将不会写入: {target_csv_path}"

            self._task_csv_path = target_csv_path
            self._task_counter = 0
            with open(self._task_csv_path, "w", encoding="utf-8") as file:
                file.write("task_id,event,timestamp\n")

            self._task_file_writable = True
            return f"✅ 任务时间戳记录已初始化: {self._task_csv_path}"

    def record_task_start(self) -> Tuple[str, str]:
        """Records task start timestamp."""
        with self._lock:
            if self._task_csv_path is None or not os.path.exists(
                os.path.dirname(self._task_csv_path)
            ):
                return "错误: 任务时间戳记录未初始化。请先开始录制。", ""

            if not self._task_file_writable:
                return "错误: task_timestamps.csv 已存在。无法写入。", ""

            timestamp_ms = int(time.time() * 1000)
            task_id = self._task_counter + 1

            try:
                with open(self._task_csv_path, "a", encoding="utf-8") as file:
                    file.write(f"{task_id},start,{timestamp_ms}\n")
            except OSError as error:
                return f"写入 CSV 错误: {error}", ""

            status_msg = f"✅ 任务 #{task_id} 开始时间已记录: {timestamp_ms}"
            summary = f"当前任务: #{task_id}\nCSV: {self._task_csv_path}"
            return status_msg, summary

    def record_task_end(self) -> Tuple[str, str]:
        """Records task end timestamp."""
        with self._lock:
            if self._task_csv_path is None or not os.path.exists(
                os.path.dirname(self._task_csv_path)
            ):
                return "错误: 任务时间戳记录未初始化。请先开始录制。", ""

            if not self._task_file_writable:
                return "错误: task_timestamps.csv 已存在。无法写入。", ""

            timestamp_ms = int(time.time() * 1000)
            task_id = self._task_counter + 1

            try:
                with open(self._task_csv_path, "a", encoding="utf-8") as file:
                    file.write(f"{task_id},end,{timestamp_ms}\n")
            except OSError as error:
                return f"写入 CSV 错误: {error}", ""

            self._task_counter += 1
            status_msg = f"✅ 任务 #{task_id} 结束时间已记录: {timestamp_ms}"
            summary = (
                f"已完成 {self._task_counter} 个任务\n"
                f"下一个任务: #{self._task_counter + 1}\n"
                f"CSV: {self._task_csv_path}"
            )
            return status_msg, summary


class KeypointTimestampRecorder:
    """Manages keypoint timestamp CSV creation and writes."""

    def __init__(self) -> None:
        self._keypoint_counter = 0
        self._keypoint_csv_path: Optional[str] = None
        self._current_record_name: Optional[str] = None
        self._lock = threading.Lock()

    def initialize(self, record_name: str, auto_create_dir: bool = False) -> str:
        """Initializes keypoint CSV under date/sn/record_name."""
        if not record_name:
            return "错误: 未指定录制名称"

        with self._lock:
            self._current_record_name = format_record_name(record_name)
            date_str = datetime.now().strftime("%Y%m%d")
            
            save_dir = os.path.join(date_str,self._current_record_name)

            if auto_create_dir:
                os.makedirs(save_dir, exist_ok=True)
            elif not os.path.exists(save_dir):
                return f"错误: 录制目录不存在: {save_dir}"

            self._keypoint_csv_path = os.path.join(save_dir, "keypoint_timestamps.csv")
            self._keypoint_counter = 0

            with open(self._keypoint_csv_path, "w", encoding="utf-8") as file:
                file.write("index,timestamp\n")

            return f"✅ 关键点记录已初始化: {self._keypoint_csv_path}"

    def record(self) -> Tuple[str, str]:
        """Records one keypoint timestamp with zero-based index."""
        with self._lock:
            if self._keypoint_csv_path is None or not os.path.exists(
                os.path.dirname(self._keypoint_csv_path)
            ):
                return "错误: 关键点记录未初始化。请先开始录制。", ""

            timestamp_ms = int(time.time() * 1000)

            try:
                with open(self._keypoint_csv_path, "a", encoding="utf-8") as file:
                    file.write(f"{self._keypoint_counter},{timestamp_ms}\n")
            except OSError as error:
                return f"写入 CSV 错误: {error}", ""

            status_msg = (
                f"✅ 关键点 #{self._keypoint_counter} 记录成功: {timestamp_ms}"
            )
            self._keypoint_counter += 1
            summary = (
                f"下一个关键点序号: {self._keypoint_counter}\n"
                f"CSV: {self._keypoint_csv_path}"
            )
            return status_msg, summary

    def reset(self) -> Tuple[str, str]:
        """Resets in-memory keypoint state."""
        with self._lock:
            self._keypoint_counter = 0
            self._keypoint_csv_path = None
            self._current_record_name = None
            return "关键点记录已重置", ""


class CameraConfigService:
    """Camera detection and config.json generation."""

    LOCATION_CHOICES = [
        "head",
        "head_left",
        "head_right",
        "chest_left",
        "chest_right",
        "hand_palm_left",
        "hand_palm_right",
        "hand_back_left",
        "hand_back_right",
        "waist_left",
        "waist_right",
    ]

    FIXED_EGO_KEY_BY_LOCATION = {
        "head_left": "head_0",
        "head_right": "head_1",
        "hand_palm_left": "hand_0",
        "hand_palm_right": "hand_1",
        "hand_back_left": "hand_2",
        "hand_back_right": "hand_3",
        "chest_left": "chest_0",
        "chest_right": "chest_1",
    }

    PRESET_CAMERAS = [
        {
            "type": "egocap",
            "sn": "preset:egocap",
            "name": "EgoCap (预设)",
            "resolution": [],
            "fps": 15,
            "modal": ["rgb", "imu"],
            "preset": True,
        },
    ]

    def __init__(self, runtime: RuntimeConfig) -> None:
        self._runtime = runtime

    def _find_pub_json_files(self) -> List[str]:
        """Finds pub*.json from cwd and runtime app dir."""
        files: List[str] = []
        candidates = [getattr(self._runtime, "app_dir", None)]
        print(f"Looking for pub*.json in: {candidates}")
        for base in candidates:
            if not base:
                continue
            try:
                for name in os.listdir(base):
                    if name.startswith("pub") and name.endswith(".json"):
                        pub_path = os.path.join(base, name)
                        print(f"Found candidate pub json: {pub_path}")
                        files.append(pub_path)
            except Exception as e:
                print(f"Error occurred while listing files in {base}: {e}")
                continue
        return sorted(set(files))

    @staticmethod
    def _parse_egocap_from_pub(pub_data: Dict) -> Optional[Dict[str, Dict]]:
        """Parses ego structure from pub json data.

        Returns:
            dict like {"head": {...}} or None
        """
        if not isinstance(pub_data, dict):
            return None
        entry = {
            "type": "egocap",
            "location": "head",
            "cam_type": "egocap_x1",
            "sn": EGOCAP_SN,
            "resolution": [1088, 1280],
            "fps": 15,
            "cam_calibration":pub_data,
            #"cam_intrinsics": camera_intrinsics,
            #"cam_extrinsics": camera_extrinsics,
            "modal": ["rgb", "imu"],
        }
        return entry

    def _load_egocap_from_pub(self) -> Tuple[Optional[Dict[str, Dict]], Optional[str]]:
        """Loads first valid egocap config from pub*.json."""
        for pub_path in self._find_pub_json_files():
            try:
                with open(pub_path, "r", encoding="utf-8") as pf:
                    pub_data = json.load(pf)
                ego = self._parse_egocap_from_pub(pub_data)
                # print(ego)
                if ego:
                    return ego, pub_path
            except Exception:
                continue
        return None, None

    @staticmethod
    def _make_unique_key(container: Dict, base_key: str) -> str:
        """Builds a unique key within dict."""
        key = base_key
        idx = 1
        while key in container:
            key = f"{base_key}_{idx}"
            idx += 1
        return key

    @staticmethod
    def _key_prefix_from_location(location: str) -> str:
        """Builds template-style key prefix from location."""
        text = (location or "").strip().lower()
        if text.startswith("head"):
            return "head"
        if text.startswith("chest"):
            return "chest"
        if text.startswith("hand"):
            return "hand"
        if text.startswith("waist"):
            return "waist"
        return "cam"

    @classmethod
    def _next_template_key(
        cls, container: Dict, location: str, counters: Dict[str, int]
    ) -> str:
        """Returns keys like head_0, hand_1, chest_2 ..."""
        prefix = cls._key_prefix_from_location(location)
        idx = counters.get(prefix, 0)
        key = f"{prefix}_{idx}"
        while key in container:
            idx += 1
            key = f"{prefix}_{idx}"
        counters[prefix] = idx + 1
        return key

    @classmethod
    def _normalize_device_block_keys(cls, block: Dict) -> Dict:
        """Normalizes keys to template style like head_0/hand_0 based on location."""
        if not isinstance(block, dict):
            return {}

        normalized: Dict = {}
        counters: Dict[str, int] = {}
        for old_key, value in block.items():
            if isinstance(value, dict):
                location = str(value.get("location", old_key))
            else:
                location = str(old_key)
            new_key = cls._next_template_key(normalized, location, counters)
            normalized[new_key] = value
        return normalized

    @staticmethod
    def _split_indexed_key(key: str) -> Tuple[str, int]:
        """Splits keys like hand_2 into (hand, 2); non-indexed => (key, large_index)."""
        if not isinstance(key, str):
            return str(key), 10**9

        if "_" not in key:
            return key, 10**9

        prefix, suffix = key.rsplit("_", 1)
        if suffix.isdigit():
            return prefix, int(suffix)
        return key, 10**9

    @classmethod
    def _sort_ego_key_for_output(cls, key: str) -> Tuple[int, int, str]:
        """Sort priority for ego block keys.

        Expected order:
        head -> head_0/head_1/... -> hand_0/hand_1/... -> chest_* -> waist_* -> cam_* -> others
        """
        if key == "head":
            return (0, 0, key)

        prefix, index = cls._split_indexed_key(key)
        prefix_order = {
            "head": 1,
            "hand": 2,
            "chest": 3,
            "waist": 4,
            "cam": 5,
        }
        return (prefix_order.get(prefix, 9), index, key)

    @classmethod
    def _normalize_ego_block_fixed_keys(cls, block: Dict) -> Dict:
        """Normalizes ego keys with fixed mapping required by template semantics.

        Rules:
        - egocap -> key `head`, location `head`
        - head_left -> head_0, head_right -> head_1
        - hand/chest follow fixed indices
        - others fallback to generic indexed naming
        """
        if not isinstance(block, dict):
            return {}

        normalized: Dict = {}
        counters: Dict[str, int] = {}

        for old_key, value in block.items():
            entry = value if isinstance(value, dict) else {"location": str(old_key)}
            cam_type = str(entry.get("cam_type", "")).lower()
            location = str(entry.get("location", old_key)).strip()
            location_lower = location.lower()

            if cam_type.startswith("egocap"):
                entry["location"] = "head"
                target_key = "head"
                if target_key in normalized:
                    target_key = cls._make_unique_key(normalized, target_key)
                normalized[target_key] = entry
                continue

            fixed = cls.FIXED_EGO_KEY_BY_LOCATION.get(location_lower)
            if fixed and fixed not in normalized:
                normalized[fixed] = entry
                continue

            # fallback (e.g. waist_xx)
            fallback_key = cls._next_template_key(normalized, location, counters)
            normalized[fallback_key] = entry

        ordered: Dict = {}
        for key in sorted(normalized.keys(), key=cls._sort_ego_key_for_output):
            ordered[key] = normalized[key]
        return ordered

    @classmethod
    def _ego_key_for_selected_camera(
        cls,
        ego_group: Dict,
        location: str,
        cam_type: str,
        counters: Dict[str, int],
    ) -> Tuple[str, str]:
        """Computes ego key and normalized location for selected camera."""
        cam_type_lower = (cam_type or "").lower()
        location = (location or "").strip()
        location_lower = location.lower()

        if cam_type_lower.startswith("egocap"):
            key = "head" if "head" not in ego_group else cls._make_unique_key(ego_group, "head")
            return key, "head"

        fixed = cls.FIXED_EGO_KEY_BY_LOCATION.get(location_lower)
        if fixed and fixed not in ego_group:
            return fixed, location

        return cls._next_template_key(ego_group, location, counters), location

    @staticmethod
    def _group_from_location(location: str) -> Tuple[str, str]:
        """Parses location prefix to decide group.

        env:xxx -> group env, location xxx
        default -> group ego
        """
        if isinstance(location, str) and location.startswith("env:"):
            stripped = location.split(":", 1)[1].strip() or "env_cam"
            return "env", stripped
        return "ego", location

    def _read_realsense_intrinsics(self, selected_sns: set) -> Dict[str, List[List[float]]]:
        """Reads RealSense intrinsics in subprocess to isolate SDK instability."""
        if not selected_sns:
            return {}

        marker = "__RS_INTRINSICS_JSON__="
        script = (
            "import json\n"
            "import dynamic_config as dc\n"
            "out = {}\n"
            "for item in (dc.get_all_realsense_intrinsics() or []):\n"
            "    sn = str(item.get('serial_number'))\n"
            "    k_color = item.get('K_color')\n"
            "    if hasattr(k_color, 'tolist'):\n"
            "        k_color = k_color.tolist()\n"
            "    if isinstance(k_color, list):\n"
            "        out[sn] = k_color\n"
            f"print('{marker}' + json.dumps(out, ensure_ascii=False))\n"
        )

        try:
            process = subprocess.run(
                [sys.executable, "-c", script],
                cwd=self._runtime.app_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
            )
        except Exception as error:
            print(f"Warning: failed to read RealSense intrinsics subprocess: {error}")
            return {}

        if process.returncode != 0:
            err = (process.stderr or "").strip()
            if err:
                print(f"Warning: RealSense intrinsics subprocess failed: {err}")
            return {}

        payload = None
        for line in reversed((process.stdout or "").splitlines()):
            if line.startswith(marker):
                payload = line[len(marker) :]
                break

        if not payload:
            return {}

        try:
            all_intrinsics = json.loads(payload)
        except Exception:
            return {}

        result: Dict[str, List[List[float]]] = {}
        for sn, k_color in (all_intrinsics or {}).items():
            sn_text = str(sn)
            if sn_text not in selected_sns:
                continue
            if isinstance(k_color, list):
                result[sn_text] = k_color
        return result

    @staticmethod
    def _read_zed_intrinsics(selected_sns: set) -> Dict[str, List[List[float]]]:
        """Reads ZED intrinsics by serial number."""
        if not selected_sns or not ZED_AVAILABLE:
            return {}

        intrinsics: Dict[str, List[List[float]]] = {}

        try:
            stereo_list = sl.Camera.get_device_list()
        except Exception:
            stereo_list = []

        try:
            mono_list = sl.CameraOne.get_device_list() if hasattr(sl, "CameraOne") else []
        except Exception:
            mono_list = []

        for dev in stereo_list:
            sn = str(dev.serial_number)
            if sn not in selected_sns:
                continue
            try:
                cam = sl.Camera()
                init = sl.InitParameters()
                init.set_from_serial_number(int(sn))
                init.camera_resolution = sl.RESOLUTION.HD1080
                status = cam.open(init)
                if status == sl.ERROR_CODE.SUCCESS:
                    info = cam.get_camera_information()
                    calib = info.camera_configuration.calibration_parameters
                    intrinsics[sn] = [
                        [calib.fx, 0.0, calib.cx],
                        [0.0, calib.fy, calib.cy],
                        [0.0, 0.0, 1.0],
                    ]
                cam.close()
            except Exception:
                continue

        for dev in mono_list:
            sn = str(dev.serial_number)
            if sn not in selected_sns:
                continue
            try:
                cam = sl.CameraOne()
                init = sl.InitParametersOne()
                init.set_from_serial_number(int(sn))
                init.camera_resolution = sl.RESOLUTION.HD1080
                status = cam.open(init)
                if status == sl.ERROR_CODE.SUCCESS:
                    info = cam.get_camera_information()
                    calib = info.camera_configuration.calibration_parameters
                    intrinsics[sn] = [
                        [calib.fx, 0.0, calib.cx],
                        [0.0, calib.fy, calib.cy],
                        [0.0, 0.0, 1.0],
                    ]
                cam.close()
            except Exception:
                continue

        return intrinsics

    def detect_all_cameras(self) -> List[Dict]:
        """Detects RealSense/ZED cameras and optional EgoCap from pub*.json."""
        all_cameras = list_cameras("all")
        camera_info = []

        for cam in all_cameras:
            cam_type = cam.get("type")
            serial_number = cam.get("sn")
            name = cam.get("name")

            if cam_type == "egocap" or serial_number == "00000001":
                continue

            if cam_type == "realsense":
                camera_info.append(
                    {
                        "type": "realsense_d435i",
                        "sn": serial_number,
                        "name": name,
                        "resolution": [848, 480],
                        "fps": 15,
                        "modal": ["rgb", "depth"],
                    }
                )
            elif cam_type == "zed":
                camera_info.append(
                    {
                        "type": "zed_x_one_gs",
                        "sn": serial_number,
                        "name": name,
                        "resolution": [960, 600],
                        "fps": 15,
                        "modal": ["rgb","imu"],
                    }
                )

        ego_from_pub, pub_path = self._load_egocap_from_pub()
        if isinstance(ego_from_pub, dict) and ego_from_pub:
            camera_info.append(
                        ego_from_pub
                )

        return camera_info

    def get_configurable_cameras(self) -> List[Dict]:
        """Returns cameras user can select: detected + missing presets."""
        detected = self.detect_all_cameras()
        by_sn = {str(cam.get("sn")): cam for cam in detected}

        configurable = list(detected)
        for preset in self.PRESET_CAMERAS:
            preset_sn = str(preset.get("sn"))
            if preset_sn in by_sn:
                continue

            # 如果已经检测到同类型设备，仍然保留预设选择能力（用户明确要求可选择）
            configurable.append(dict(preset))

        return configurable

    def get_camera_location_fields(
        self, max_fields: int = 10
    ) -> Tuple[List[Dict], str]:
        """Returns UI-agnostic camera location field metadata."""
        cameras = self.get_configurable_cameras()
        if not cameras:
            return [], "未检测到相机"

        fields = []
        for camera in cameras[:max_fields]:
            fields.append(
                {
                    "visible": True,
                    "label": (
                        f"{camera['type']} (SN: {camera['sn']})"
                        if not camera.get("preset")
                        else f"{camera['type']} (SN: {camera['sn']}, 预设可选)"
                    ),
                    "choices": self.LOCATION_CHOICES,
                    "value": "",
                }
            )
        detected_count = len([c for c in cameras if not c.get("preset")])
        preset_count = len([c for c in cameras if c.get("preset")])
        return fields, f"检测到 {detected_count} 个相机，提供 {preset_count} 个预设相机可选"

    def generate_config_json(
        self,
        task_no: str,
        task_name: str,
        camera_locations: Dict[str, str],
        record_name: str,
        use_group_indexed_keys: bool = True,
    ) -> str:
        """Generates config.json based on cameras selected in Gradio."""
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            template_candidates = [
                "template_config.json",
                os.path.join(self._runtime.app_dir, "template_config.json"),
            ]

            template_path = None
            for candidate in template_candidates:
                if os.path.exists(candidate):
                    template_path = candidate
                    break

            if not template_path:
                joined = "\n".join(template_candidates)
                return f"❌ 模板文件不存在\n已尝试路径:\n{joined}"

            try:
                with open(template_path, "r", encoding="utf-8") as file:
                    content = file.read()
                lines = content.split("\n")
                clean_lines = []
                for line in lines:
                    if "#" in line:
                        line = line[: line.index("#")]
                    clean_lines.append(line)
                clean_content = "\n".join(clean_lines)
                config = json.loads(clean_content)
            except (OSError, json.JSONDecodeError) as error:
                return f"❌ 无法读取模板文件: {error}"

            config["no"] = task_no
            if task_name:
                config["task"] = [task_name]
            elif "task" not in config:
                config["task"] = ["recording"]
            config["date"] = date_str

            cameras = self.get_configurable_cameras()
            if not cameras:
                return "❌ 未检测到任何相机！"

            # 仅处理用户在 Gradio 中已选择（location 非空）的相机
            selected_locations = {
                str(sn): loc for sn, loc in (camera_locations or {}).items() if str(loc).strip()
            }

            # 无选择时保持兼容：沿用模板并直接输出
            if not selected_locations:
                selected_count = 0
                selected_rs = set()
                selected_zed = set()
                selected_has_egocap = False
            else:
                selected_count = len(selected_locations)
                detected_by_sn = {str(cam.get("sn")): cam for cam in cameras}
                selected_cameras = [
                    detected_by_sn[sn]
                    for sn in selected_locations
                    if sn in detected_by_sn
                ]

                if not selected_cameras:
                    return "❌ 已选择相机未在当前检测结果中找到，请先点击“检测相机”再生成配置。"

                # 按选择重建 device 结构
                config.setdefault("device", {})
                config["device"]["ego"] = {}
                config["device"]["env"] = {}

                selected_rs = {
                    str(cam.get("sn"))
                    for cam in selected_cameras
                    if str(cam.get("type", "")).startswith("realsense")
                }
                selected_zed = {
                    str(cam.get("sn"))
                    for cam in selected_cameras
                    if str(cam.get("type", "")).startswith("zed")
                }
                selected_has_egocap = any(
                    str(cam.get("type", "")).startswith("egocap")
                    for cam in selected_cameras
                )

                rs_intrinsics = self._read_realsense_intrinsics(selected_rs)
                zed_intrinsics = self._read_zed_intrinsics(selected_zed)

                ego_key_counters: Dict[str, int] = {}
                env_key_counters: Dict[str, int] = {}
                if selected_has_egocap:
                    ego_from_pub, _pub_path = self._load_egocap_from_pub()
                    ego_from_pub = {'head': ego_from_pub} if ego_from_pub else None
                    if not ego_from_pub:
                        return (
                            "❌ 已选择 EgoCap，但未找到有效 pub*.json 参数文件。\n"
                            "需要从 pub 中读取包含 cam_0~cam_3 的 cam_intrinsics 和 cam_extrinsics。"
                        )
                    ego_from_pub.pop("type", None)
                    ego_group = config["device"].setdefault("ego", {})
                    for ego_key, ego_value in ego_from_pub.items():
                        location = "head"
                        if use_group_indexed_keys:
                            target_key = "head"
                            if target_key in ego_group:
                                target_key = self._make_unique_key(ego_group, target_key)
                            if isinstance(ego_value, dict):
                                ego_value["location"] = "head"
                        else:
                            target_key = self._make_unique_key(ego_group, location)
                        ego_group[target_key] = ego_value
                    
                
                for cam in selected_cameras:
                    sn = str(cam.get("sn"))
                    cam_type = str(cam.get("type", ""))
                    location_raw = selected_locations.get(sn, "").strip()
                    if not location_raw:
                        continue

                    # Egocap 单独处理：仅当被选择时才合并 pub 文件
                    if cam_type.startswith("egocap"):
                        continue

                    group, location = self._group_from_location(location_raw)
                    group_dict = config["device"].setdefault(group, {})
                    if use_group_indexed_keys:
                        if group == "ego":
                            key, location = self._ego_key_for_selected_camera(
                                group_dict,
                                location,
                                cam_type,
                                ego_key_counters,
                            )
                        else:
                            counters = env_key_counters
                            key = self._next_template_key(group_dict, location, counters)
                    else:
                        key = self._make_unique_key(group_dict, location or f"cam_{sn}")

                    if cam_type.startswith("realsense"):
                        cam_intrinsics = rs_intrinsics.get(sn, [])
                        default_cam_type = "realsense_d435i"
                        default_modal = ["rgb", "depth"]
                        default_resolution = [848, 480]
                    else:
                        cam_intrinsics = zed_intrinsics.get(sn, [])
                        default_cam_type = "zed_x_one_gs"
                        default_modal = ["rgb","imu"]
                        default_resolution = [960, 600]

                    group_dict[key] = {
                        "location": location,
                        "cam_type": cam_type or default_cam_type,
                        "sn": sn,
                        "resolution": cam.get("resolution", default_resolution),
                        "fps": int(cam.get("fps", 15)),
                        "cam_intrinsics": cam_intrinsics,
                        "cam_extrinsics": [],
                        "modal": cam.get("modal", default_modal),
                    }



            record_name_formatted = (
                format_record_name(record_name) if record_name else task_no
            )
            output_dir = os.path.join(date_str, record_name_formatted)
            os.makedirs(output_dir, exist_ok=True)

            # 可选：统一键名风格为 head_0 / hand_0 / chest_0 ...
            config.setdefault("device", {})
            if use_group_indexed_keys:
                config["device"]["ego"] = self._normalize_ego_block_fixed_keys(
                    config["device"].get("ego", {})
                )
                config["device"]["env"] = self._normalize_device_block_keys(
                    config["device"].get("env", {})
                )

            output_path = os.path.join(output_dir, "config.json")
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(config, file, indent=2, ensure_ascii=False)

            status_msg = f"✅ 配置文件已生成: {output_path}\n检测到 {len(cameras)} 个相机"
            if selected_locations:
                status_msg += f"，已按选择写入 {selected_count} 个相机"
                if selected_rs:
                    status_msg += f"（RealSense: {len(selected_rs)}）"
                if selected_zed:
                    status_msg += f"（ZED: {len(selected_zed)}）"
                if selected_has_egocap:
                    status_msg += "（EgoCap: 已从 pub*.json 合并）"
            else:
                status_msg += "，未选择具体相机，已按模板保持默认设备配置"
            return status_msg

        except Exception as error:  # pylint: disable=broad-except
            return f"❌ 生成配置文件失败: {error}"

    def auto_generate_config(
        self,
        record_name: str,
        task_name: str = "",
        camera_locations: Optional[Dict[str, str]] = None,
        use_group_indexed_keys: bool = True,
    ) -> str:
        """Auto-generates config file for recording start."""
        try:
            result = self.generate_config_json(
                task_no=record_name,
                task_name=task_name,
                camera_locations=camera_locations or {},
                record_name=record_name,
                use_group_indexed_keys=bool(use_group_indexed_keys),
            )
            print(f"[Auto Config] {result}")
            
            return result
        except Exception as error:  # pylint: disable=broad-except
            print(f"[Auto Config Error] {error}")
            return f"配置生成失败: {error}"


class RecordingService:
    """Recording pipelines and timestamp analysis."""

    _CURRENT_PYTHON_BIN = os.path.dirname(os.path.realpath(sys.executable))
    _CURRENT_PATH = os.environ.get("PATH", "")
    _CAMERA_PATH = (
        f"{_CURRENT_PYTHON_BIN}:{_CURRENT_PATH}"
        if _CURRENT_PATH
        else _CURRENT_PYTHON_BIN
    )
    CAMERA_ENV = f"PATH={shlex.quote(_CAMERA_PATH)}"

    @staticmethod
    def _infer_conda_env_name() -> Optional[str]:
        """Infers current conda env name from env vars or interpreter path."""
        override_env = os.environ.get("ACE_RECORD_CONDA_ENV")
        if override_env:
            return override_env

        default_env = os.environ.get("CONDA_DEFAULT_ENV")
        if default_env:
            return default_env

        python_executable = os.path.realpath(sys.executable)
        path_parts = python_executable.split(os.sep)
        if "envs" in path_parts:
            envs_index = path_parts.index("envs")
            if envs_index + 1 < len(path_parts):
                return path_parts[envs_index + 1]

        return None

    CONDA_ENV_NAME = _infer_conda_env_name.__func__()

    def __init__(
        self,
        runtime: RuntimeConfig,
        task_recorder: TaskTimestampRecorder,
        keypoint_recorder: Optional[KeypointTimestampRecorder],
        config_service: CameraConfigService,
    ) -> None:
        self._runtime = runtime
        self._task_recorder = task_recorder
        self._keypoint_recorder = keypoint_recorder
        self._config_service = config_service

        self._processes: Dict[str, Optional[subprocess.Popen]] = {
            "zed": None,
            "realsense": None,
            "egocap": None,
            "timestamp_analysis": None,
            "color_sync": None,
        }
        self.exit_codes: Dict[str, int] = {}
        self.q_rs: queue.Queue = queue.Queue(maxsize=1000)
        self.q_zed: queue.Queue = queue.Queue(maxsize=1000)
        self.q_ego: queue.Queue = queue.Queue(maxsize=1000)
        self.q_terminal: queue.Queue = queue.Queue(maxsize=1000)
        self.q_color_sync: queue.Queue = queue.Queue(maxsize=1000)
        self._config_generation_lock = threading.Lock()
        self._config_generated_records: set[str] = set()
        self._active_sessions: Dict[str, str] = {}
        self._encryption_service = self._build_encryption_service()

    @staticmethod
    def _parse_encryption_key(raw_value: Optional[str]) -> Optional[bytes]:
        if not raw_value:
            return DEFAULT_ENCRYPTION_KEY

        key = None
        cleaned = raw_value.strip()
        if len(cleaned) == 64:
            try:
                key = bytes.fromhex(cleaned)
            except ValueError:
                key = None

        if key is None:
            key = cleaned.encode("utf-8")

        if len(key) != 32:
            print(
                "[Encryption] Invalid key length. Expected 32 bytes, "
                f"got {len(key)}."
            )
            return None

        return key

    def _build_encryption_service(self) -> Optional["EncryptionService"]:
        if not CRYPTO_AVAILABLE:
            print(f"[Encryption] Crypto unavailable: {CRYPTO_ERROR}")
            return None

        key = self._parse_encryption_key(os.getenv("ACE_ENCRYPTION_KEY"))
        if not key:
            return None

        service = EncryptionService(key=key, output_subdir="_encrypted")
        service.start()
        return service

    def _schedule_encryption(self, session_path: str, reason: str) -> None:
        if not self._encryption_service:
            return

        if not session_path:
            return

        self._encryption_service.enqueue_session(session_path, reason=reason)

    def _ensure_config_generated(
        self,
        record_name: str,
        task_name: str,
        camera_locations: Optional[Dict[str, str]],
        use_group_indexed_keys: bool,
    ) -> str:
        """Generates config once per record name during one start session."""
        record_key = format_record_name(record_name)
        with self._config_generation_lock:
            if record_key in self._config_generated_records:
                return "✅ 配置文件已存在，本次启动跳过重复生成"

            result = self._config_service.auto_generate_config(
                record_name=record_key,
                task_name=task_name,
                camera_locations=camera_locations or {},
                use_group_indexed_keys=bool(use_group_indexed_keys),
            )

            if isinstance(result, str) and result.startswith("✅"):
                self._config_generated_records.add(record_key)
            return result

    def stop_all(self) -> str:
        """Stops all running recording and analysis processes."""
        self._processes["zed"] = self._terminate_process(self._processes["zed"], "ZED")
        self._processes["realsense"] = self._terminate_process(
            self._processes["realsense"], "RealSense"
        )
        self._processes["egocap"] = self._terminate_process(
            self._processes["egocap"], "EgoCap"
        )
        self._processes["timestamp_analysis"] = self._terminate_process(
            self._processes["timestamp_analysis"], "time analysis"
        )
        self._processes["color_sync"] = self._terminate_process(
            self._processes["color_sync"], "Color Sync"
        )

        self._clear_queue(self.q_rs)
        self._clear_queue(self.q_zed)
        self._clear_queue(self.q_ego)
        self._clear_queue(self.q_terminal)
        self._clear_queue(self.q_color_sync)
        with self._config_generation_lock:
            self._config_generated_records.clear()

        if self._active_sessions:
            for session_path in sorted(set(self._active_sessions.values())):
                self._schedule_encryption(session_path, reason="stop_all")
            self._active_sessions.clear()
        return "所有录制已停止！准备就绪。"

    @staticmethod
    def _terminate_process(
        proc: Optional[subprocess.Popen], name: str
    ) -> Optional[subprocess.Popen]:
        """Gracefully terminates a process group then force kills if needed."""
        if proc is None:
            return None

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=5)
            print(f"{name} pipeline stopped.")
        except ProcessLookupError:
            print(f"{name} process already terminated.")
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            print(f"{name} pipeline killed after timeout.")
        except Exception as error:  # pylint: disable=broad-except
            print(f"Error stopping {name}: {error}")

        return None

    @staticmethod
    def _clear_queue(target_queue: queue.Queue) -> None:
        """Removes all queued items without blocking."""
        while not target_queue.empty():
            try:
                target_queue.get_nowait()
            except queue.Empty:
                break

    def _run_pipeline(
        self, command: str, process_key: str, target_queue: queue.Queue
    ) -> None:
        """Runs a pipeline command and streams output to queue."""
        print(f"Starting command: {command}")

        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        self._processes[process_key] = proc

        for line in proc.stdout:
            target_queue.put(line.strip() + "\n")

        proc.wait()
        self.exit_codes[process_key] = proc.returncode
        self._processes[process_key] = None

        finished_name = {
            "zed": "ZED",
            "realsense": "RealSense",
            "egocap": "EgoCap",
            "color_sync": "Color Sync",
        }.get(process_key, process_key)
        target_queue.put(f"{finished_name} process finished.\n")

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        """Returns True when process exists."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _terminate_stale_script_processes(
        self, script_keyword: str, process_name: str
    ) -> None:
        """Terminates stale script processes left from previous crashes/runs."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", script_keyword],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except Exception:
            return

        if result.returncode != 0:
            return

        tracked_pids = {
            proc.pid for proc in self._processes.values() if proc is not None
        }
        current_pid = os.getpid()
        stale_pids = []

        for line in (result.stdout or "").splitlines():
            try:
                pid = int(line.strip())
            except (TypeError, ValueError):
                continue

            if pid in tracked_pids or pid == current_pid:
                continue

            stale_pids.append(pid)

        if not stale_pids:
            return

        for pid in stale_pids:
            try:
                os.kill(pid, signal.SIGINT)
            except OSError:
                continue

        deadline = time.time() + 1.5
        for pid in stale_pids:
            while self._is_pid_alive(pid) and time.time() < deadline:
                time.sleep(0.05)
            if self._is_pid_alive(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

        print(f"Cleaned stale {process_name} process(es): {stale_pids}")

    def _get_conda_executable(self) -> Optional[str]:
        """Returns conda executable path if available."""
        conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
        if conda_exe and os.path.exists(conda_exe):
            return conda_exe

        python_executable = os.path.realpath(sys.executable)
        path_parts = python_executable.split(os.sep)
        candidate_paths = []

        if "envs" in path_parts:
            envs_index = path_parts.index("envs")
            if envs_index > 0:
                conda_root = os.sep.join(path_parts[:envs_index])
                if conda_root:
                    candidate_paths.append(os.path.join(conda_root, "bin", "conda"))

        candidate_paths.append("/opt/conda/bin/conda")

        for candidate in candidate_paths:
            if os.path.exists(candidate):
                return candidate
        return None

    def _build_python_script_command(
        self, script_path: str, script_args: str = ""
    ) -> str:
        """Builds command to run a Python script, preferring configured conda env."""
        script_q = shlex.quote(script_path)
        args = f" {script_args}" if script_args else ""

        conda_exe = self._get_conda_executable()
        if conda_exe and self.CONDA_ENV_NAME:
            return (
                f"{shlex.quote(conda_exe)} run -n {shlex.quote(self.CONDA_ENV_NAME)} "
                f"python {script_q}{args}"
            )

        python_cmd = self._runtime.python_cmd or "python3"
        return f"{shlex.quote(python_cmd)} {script_q}{args}"

    def pipeline_generator(
        self,
        command: str,
        process_key: str,
        target_queue: queue.Queue,
    ) -> Generator[Tuple[str, str], None, None]:
        """Streams command output for Gradio components."""
        self.exit_codes[process_key] = 0
        with target_queue.mutex:
            target_queue.queue.clear()

        threading.Thread(
            target=self._run_pipeline,
            args=(command, process_key, target_queue),
            daemon=True,
        ).start()

        output_text = f"{command}\n"
        while True:
            try:
                line = target_queue.get(timeout=0.1)
                output_text += line
                yield output_text, ""
                target_queue.task_done()
            except queue.Empty:
                is_running = self._processes[process_key] is not None
                if not is_running and target_queue.empty():
                    code = self.exit_codes.get(process_key, 0)
                    if code and code > 0:
                        yield output_text, (
                            f"❌ {process_key} 录制失败!\n"
                            f"(退出代码: {code})\n请检查日志。"
                        )
                    break
                time.sleep(0.1)

    def _pipeline_with_quality_check(
        self,
        command: str,
        process_key: str,
        target_queue: queue.Queue,
        session_path: str,
    ) -> Generator[Tuple[str, str], None, None]:
        """Runs pipeline and appends quality check summary after completion."""
        last_output = ""
        last_error = ""
        for output, error in self.pipeline_generator(command, process_key, target_queue):
            last_output = output
            last_error = error
            yield output, error

        exit_code = self.exit_codes.get(process_key, 0)
        if not (exit_code and exit_code > 0):
            quality_summary = DataService.get_quality_summary(session_path)
            if quality_summary:
                combined = (last_output.rstrip() + "\n" + quality_summary).strip()
                yield combined, last_error
            DataService.write_recording_summary(session_path)

        self._schedule_encryption(session_path, reason=process_key)
        self._active_sessions.pop(process_key, None)

    def zed_pipeline(
        self,
        record_name="",
        task_name="",
        camera_locations=None,
        use_group_indexed_keys: bool = True,
    ) -> Generator[Tuple[str, str], None, None]:
        """Runs ZED recording pipeline."""
        date_str = datetime.now().strftime("%Y%m%d")
        record_name = format_record_name(record_name)
        record_name = str(record_name).strip().strip("/\\")
        if not record_name:
            yield "", "❌ 启动失败：录制任务名称不能为空"
            return

        config_result = self._ensure_config_generated(
            record_name,
            task_name,
            camera_locations,
            use_group_indexed_keys,
        )
        if not (isinstance(config_result, str) and config_result.startswith("✅")):
            yield "", f"❌ 启动失败：配置生成失败\n{config_result}"
            return

        self._task_recorder.initialize(record_name, auto_create_dir=True)
        if self._keypoint_recorder is not None:
            self._keypoint_recorder.initialize(record_name, auto_create_dir=True)

        exec_path = os.path.join(self._runtime.app_dir, "zed_record.py")
        ensure_executable(exec_path)

        script_command = self._build_python_script_command(
            exec_path,
            f"--record_name {shlex.quote(record_name)} --save_path {shlex.quote(date_str)}",
        )
        command = f"{self.CAMERA_ENV} {script_command}"
        session_path = os.path.abspath(os.path.join(date_str, record_name))
        self._active_sessions["zed"] = session_path
        yield from self._pipeline_with_quality_check(command, "zed", self.q_zed, session_path)

    def realsense_pipeline(
        self,
        record_name="",
        task_name="",
        camera_locations=None,
        use_group_indexed_keys: bool = True,
    ) -> Generator[Tuple[str, str], None, None]:
        """Runs RealSense recording pipeline."""
        date_str = datetime.now().strftime("%Y%m%d")
        record_name = format_record_name(record_name)
        record_name = str(record_name).strip().strip("/\\")
        if not record_name:
            yield "", "❌ 启动失败：录制任务名称不能为空"
            return

        # 清理上次异常退出遗留的录制进程，避免占用摄像头设备。
        self._terminate_stale_script_processes("recording/realsense_record.py", "RealSense")

        config_result = self._ensure_config_generated(
            record_name,
            task_name,
            camera_locations,
            use_group_indexed_keys,
        )
        if not (isinstance(config_result, str) and config_result.startswith("✅")):
            yield "", f"❌ 启动失败：配置生成失败\n{config_result}"
            return

        self._task_recorder.initialize(record_name, auto_create_dir=True)
        if self._keypoint_recorder is not None:
            self._keypoint_recorder.initialize(record_name, auto_create_dir=True)

        exec_path = os.path.join(self._runtime.app_dir, "realsense_record.py")
        ensure_executable(exec_path)

        script_command = self._build_python_script_command(
            exec_path,
            f"--record_name {shlex.quote(record_name)}",
        )
        command = f"{self.CAMERA_ENV} {script_command}"
        session_path = os.path.abspath(os.path.join(date_str, record_name))
        self._active_sessions["realsense"] = session_path
        yield from self._pipeline_with_quality_check(
            command,
            "realsense",
            self.q_rs,
            session_path,
        )

    def egocap_pipeline(
        self,
        record_name="",
        task_name="",
        camera_locations=None,
        use_group_indexed_keys: bool = True,
    ) -> Generator[Tuple[str, str], None, None]:
        """Runs EgoCap recording pipeline."""
        record_name = format_record_name(record_name)
        record_name = str(record_name).strip().strip("/\\")
        if not record_name:
            yield "", "❌ 启动失败：录制任务名称不能为空"
            return

        date_str = datetime.now().strftime("%Y%m%d")

        config_result = self._ensure_config_generated(
            record_name,
            task_name,
            camera_locations,
            use_group_indexed_keys,
        )
        if not (isinstance(config_result, str) and config_result.startswith("✅")):
            yield "", f"❌ 启动失败：配置生成失败\n{config_result}"
            return

        self._task_recorder.initialize(record_name, auto_create_dir=True)
        if self._keypoint_recorder is not None:
            self._keypoint_recorder.initialize(record_name, auto_create_dir=True)

        egocap_dir = os.path.join(self._runtime.app_dir, "egocap")
        egocap_bin = os.path.join(egocap_dir, "egocap")

        if os.path.exists(egocap_bin):
            ensure_executable(egocap_bin)
        else:
            print(f"Error: EgoCap binary not found at {egocap_bin}")

        output_path = os.path.abspath(
            os.path.join(date_str, record_name, "egocap")
        )
        if os.path.exists(output_path):
            try:
                if os.listdir(output_path):
                    yield (
                        "",
                        "❌ EgoCap 录制失败：该任务下已存在录制数据\n"
                        f"{output_path}\n目录非空(包含文件或文件夹)，请更换任务名。",
                    )
                    return
            except OSError:
                pass

        command = f"cd {egocap_dir} && {egocap_bin} {output_path}"
        session_path = os.path.abspath(os.path.join(date_str, record_name))
        self._active_sessions["egocap"] = session_path
        yield from self._pipeline_with_quality_check(
            command,
            "egocap",
            self.q_ego,
            session_path,
        )


    def stop_color_sync(self) -> str:
        """Stops color synchronization process only."""
        if self._processes.get("color_sync") is None:
            return "颜色同步未在运行。"

        self._processes["color_sync"] = self._terminate_process(
            self._processes["color_sync"], "Color Sync"
        )
        self._clear_queue(self.q_color_sync)
        return "颜色同步已停止。"

    def _run_timestamp_analysis(self, command: str, target_queue: queue.Queue) -> None:
        """Runs timestamp analysis command and writes output to queue."""
        with target_queue.mutex:
            target_queue.queue.clear()

        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        self._processes["timestamp_analysis"] = process

        for line in process.stdout:
            target_queue.put(line)

        process.wait()
        self._processes["timestamp_analysis"] = None
        target_queue.put("timestamp analysis finished.\n")

    def timestamp_analysis_pipe(
        self,
        record_name: str,
        camera_name: str,
    ) -> Generator[str, None, None]:
        """Runs single task timestamp analysis."""
        record_name = format_record_name(record_name)
        date_str = datetime.now().strftime("%Y%m%d")

        target_queue = self.q_terminal
        with target_queue.mutex:
            target_queue.queue.clear()

        
        exec_path = os.path.join(self._runtime.app_dir, "timestamp_analysis.py")
        ensure_executable(exec_path)

        if camera_name == "realsense":
            command = (
                f"{self.CAMERA_ENV} {exec_path} "
                f"--rs_data_path {date_str}/{record_name}/realsense"
            )
        elif camera_name == "zed":
            command = (
                f"{self.CAMERA_ENV} {exec_path} "
                f"--zed_data_path {date_str}/{record_name}/zed"
            )
        else:
            command = (
                f"{self.CAMERA_ENV} {exec_path} "
                f"--rs_data_path {date_str}/{record_name}/realsense "
                f"--zed_data_path {date_str}/{record_name}/zed"
            )

        threading.Thread(
            target=self._run_timestamp_analysis,
            args=(command, target_queue),
            daemon=True,
        ).start()

        output_text = f"{command}\n"
        while True:
            try:
                line = target_queue.get(timeout=5)
                output_text += line
                yield output_text
                target_queue.task_done()
            except queue.Empty:
                if (
                    self._processes["timestamp_analysis"] is None
                    and target_queue.empty()
                ):
                    break
                time.sleep(0.1)

    def batch_timestamp_analysis_pipe(self) -> Generator[Tuple[str, str], None, None]:
        """Runs batch timestamp analysis script."""
        script_path = os.path.join(self._runtime.app_dir, "batch_timestamp_analysis.sh")
        command = f"bash {script_path}"

        target_queue = self.q_terminal
        with target_queue.mutex:
            target_queue.queue.clear()

        threading.Thread(
            target=self._run_timestamp_analysis,
            args=(command, target_queue),
            daemon=True,
        ).start()

        output_text = f"{command}\n"
        summary_text = (
            "File, Type, FPS\n--------------------------------------------------\n"
        )

        while True:
            try:
                line = target_queue.get(timeout=5)
                output_text += line

                if "FPS_SUMMARY:" in line:
                    clean_line = line.strip()
                    if clean_line.startswith("FPS_SUMMARY:"):
                        content = clean_line.split("FPS_SUMMARY:", 1)[1].strip()
                        summary_text += content + "\n"

                if "--------------------------------------------------" in line:
                    summary_text += (
                        "--------------------------------------------------\n"
                    )

                yield output_text, summary_text
                target_queue.task_done()
            except queue.Empty:
                if (
                    self._processes["timestamp_analysis"] is None
                    and target_queue.empty()
                ):
                    break
                time.sleep(0.1)

    @staticmethod
    def get_fps_summary_content() -> str:
        """Reads fps_summary.txt content if present."""
        summary_file = "fps_summary.txt"
        if not os.path.exists(summary_file):
            return "未找到摘要文件 (请先运行批量分析)。"

        try:
            with open(summary_file, "r", encoding="utf-8") as file:
                return file.read()
        except OSError as error:
            return f"读取摘要文件错误: {error}"


class SystemService:
    """System-level operations."""

    def __init__(self, stop_callback, preview_cleanup_callback=None) -> None:
        self._stop_callback = stop_callback
        self._preview_cleanup_callback = preview_cleanup_callback
        self._power_action_lock = threading.Lock()
        self._pending_power_action_process: Optional[subprocess.Popen] = None

    def _terminate_pending_power_action_locked(self) -> bool:
        """Terminates pending delayed power action process if still alive."""
        process = self._pending_power_action_process
        if not process or process.poll() is not None:
            self._pending_power_action_process = None
            return False

        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:  # pylint: disable=broad-except
            try:
                process.kill()
                process.wait(timeout=1)
            except Exception:  # pylint: disable=broad-except
                pass

        self._pending_power_action_process = None
        return True

    def _schedule_power_action(self, action_flag: str, action_name: str, regret_time: int = 5) -> str:
        """Schedules delayed shutdown/reboot and keeps a cancellable handle."""
        passwd = os.getenv("ACE_SUDO_PASSWORD", "3456999")
        output_text = f"=== 正在准备{action_name}系统 ===\n"
        output_text += f"⚠️ 系统将在 {regret_time} 秒后{action_name}...\n"
        output_text += "请确保所有数据已保存！\n"

        try:
            output_text += self._stop_camera_users()
            with self._power_action_lock:
                if self._terminate_pending_power_action_locked():
                    output_text += "已取消上一次未执行的电源操作\n"

                command = f"sleep {regret_time}; shutdown {action_flag} now"
                process = subprocess.Popen(  # pylint: disable=consider-using-with
                    ["sudo", "-S", "bash", "-lc", command],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if process.stdin:
                    process.stdin.write((passwd + "\n").encode("utf-8"))
                    process.stdin.flush()
                    process.stdin.close()

                self._pending_power_action_process = process
            output_text += f"✓ {action_name}命令已发送，可在 {regret_time} 秒内点击“取消关机”撤销\n"
        except Exception as error:  # pylint: disable=broad-except
            output_text += f"✗ {action_name}失败: {error}\n"

        return output_text

    @staticmethod
    def _run_sudo_command(command: str, passwd: str, timeout_sec: int = 30) -> Tuple[int, str]:
        """Runs one sudo shell command and returns (returncode, combined_output)."""
        try:
            proc = subprocess.run(
                ["sudo", "-S", "bash", "-lc", command],
                input=(passwd + "\n").encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_sec,
            )
            return proc.returncode, (proc.stdout or b"").decode("utf-8", errors="ignore")
        except Exception as error:  # pylint: disable=broad-except
            return 1, str(error)

    def _restart_service_with_fallback(
        self,
        candidates: List[str],
        passwd: str,
        label: str,
    ) -> Tuple[bool, str, str]:
        """Restarts first available systemd service from candidates with robust sequence."""
        output = ""
        for svc in candidates:
            # 先判断服务是否存在
            code_exist, _ = self._run_sudo_command(
                f"systemctl status {svc} >/dev/null 2>&1", passwd
            )
            if code_exist != 0:
                output += f"- 跳过 {svc}（服务不存在或不可访问）\n"
                continue

            steps = [
                f"systemctl stop {svc}",
                f"systemctl reset-failed {svc}",
                f"systemctl start {svc}",
            ]
            ok = True
            for step in steps:
                code_step, out_step = self._run_sudo_command(step, passwd, timeout_sec=40)
                if code_step != 0:
                    ok = False
                    output += f"✗ {svc}: `{step}` 失败\n{out_step}\n"
                    break

            if ok:
                # 检查 active 状态
                code_active, out_active = self._run_sudo_command(
                    f"systemctl is-active {svc}", passwd
                )
                if code_active == 0 and "active" in out_active:
                    output += f"✓ {label} 服务已恢复: {svc}\n"
                    return True, svc, output

                output += f"✗ {svc} 启动后非 active 状态\n{out_active}\n"

        return False, "", output

    def _stop_camera_users(self) -> str:
        """Stops recording/preview resources before system-level restart/reset."""
        output = ""
        try:
            self._stop_callback()
            output += "✓ 已停止录制进程\n"
        except Exception as error:  # pylint: disable=broad-except
            output += f"⚠ 停止录制进程失败: {error}\n"

        try:
            if callable(self._preview_cleanup_callback):
                self._preview_cleanup_callback()
                output += "✓ 已停止预览占用\n"
        except Exception as error:  # pylint: disable=broad-except
            output += f"⚠ 停止预览占用失败: {error}\n"

        time.sleep(0.6)
        return output



    def restart_zed(self) -> str:
        """Restarts ZED-related services and releases in-process camera handles first."""
        passwd = os.getenv("ACE_SUDO_PASSWORD", "3456999")
        output_text = "=== 重启 ZED 服务 ===\n"

        # 先释放当前应用内占用，避免 daemon 重启后本进程仍持有旧句柄。
        output_text += self._stop_camera_users()

        ok_zed, used_service, log_zed = self._restart_service_with_fallback(
            [
                "zed_x_daemon.service",
                "zed_daemon.service",
                "zedx-daemon.service",
            ],
            passwd,
            "ZED",
        )
        output_text += log_zed

        if not ok_zed:
            output_text += "❌ ZED 服务重启失败，请检查 systemctl 日志。\n"
            return output_text

        # Argus 与 ZED 在 Jetson 上常互相影响，顺带恢复一次可降低后续 open 超时概率。
        code_argus, out_argus = self._run_sudo_command(
            "systemctl restart nvargus-daemon.service", passwd, timeout_sec=40
        )
        if code_argus == 0:
            output_text += "✓ nvargus-daemon 已重启\n"
        else:
            output_text += f"⚠ nvargus-daemon 重启失败（可忽略）\n{out_argus}\n"

        # 等待服务稳定；多数情况下 2~5 秒足够，超时再提示用户重试。
        stable = False
        for _ in range(10):
            code_active, out_active = self._run_sudo_command(
                f"systemctl is-active {used_service}", passwd, timeout_sec=10
            )
            if code_active == 0 and "active" in out_active:
                stable = True
                break
            time.sleep(0.5)

        if stable:
            output_text += "✅ ZED 服务重启成功，请重新点击“启动预览”。"
        else:
            output_text += "⚠ ZED 服务已执行重启，但状态未稳定，请稍后再试或重启应用。"

        return output_text

    def restart_app(self) -> str:
        """Restarts the recording app via systemd or a custom restart command."""
        passwd = os.getenv("ACE_SUDO_PASSWORD", "3456999")
        output_text = "=== 重启应用 ===\n"
        output_text += self._stop_camera_users()

        restart_cmd = os.getenv("ACE_APP_RESTART_CMD", "").strip()
        service_name = os.getenv("ACE_APP_SERVICE", "").strip()

        if restart_cmd:
            code, out = self._run_sudo_command(restart_cmd, passwd, timeout_sec=40)
            if code == 0:
                output_text += "✅ 应用重启命令已执行\n"
            else:
                output_text += f"❌ 应用重启失败\n{out}\n"
            return output_text

        if service_name:
            code, out = self._run_sudo_command(
                f"systemctl restart {service_name}", passwd, timeout_sec=40
            )
            if code == 0:
                output_text += f"✅ 已重启服务: {service_name}\n"
                return output_text
            output_text += f"❌ 重启服务失败: {service_name}\n{out}\n"
            return output_text

        output_text += "❌ 未配置应用自启服务名或重启命令\n"
        output_text += "请设置环境变量 ACE_APP_SERVICE 或 ACE_APP_RESTART_CMD。"
        return output_text
    def reset_realsense(self) -> str:
        """Resets RealSense permissions and stale processes."""
        passwd = os.getenv("ACE_SUDO_PASSWORD", "3456999")
        output_text = "=== 重置 RealSense 系统 ===\n"
        output_text += self._stop_camera_users()

        try:
            check_cmd = "ps aux | grep '[r]s_record' | awk '{print $2}'"
            pids = os.popen(check_cmd).read().strip()
            if pids:
                output_text += f"发现残留进程: {pids}\n"
                kill_cmd = f"kill -9 {pids}"
                result = os.system("echo %s | sudo -S %s" % (passwd, kill_cmd))
                if result == 0:
                    output_text += "✓ 已终止残留进程\n"
                else:
                    output_text += "✗ 无法终止进程\n"
            else:
                output_text += "✓ 未发现残留进程\n"
        except Exception as error:  # pylint: disable=broad-except
            output_text += f"✗ 进程检查失败: {error}\n"

        try:
            chmod_cmd = "chmod 666 /dev/video*"
            result = os.system("echo %s | sudo -S %s" % (passwd, chmod_cmd))
            if result == 0:
                output_text += "✓ USB 权限已更新\n"
            else:
                output_text += "✗ 权限更新失败\n"
        except Exception as error:  # pylint: disable=broad-except
            output_text += f"✗ 权限更新失败: {error}\n"

        try:
            reset_cmd = (
                'bash -c "for dev in /sys/bus/usb/devices/*/idVendor; '
                'do if [ $(cat $dev) == "8086" ]; '
                "then echo $(dirname $dev) | xargs -I{} "
                'sh -c "echo 0 > {}/authorized && echo 1 > {}/authorized"; '
                'fi; done"'
            )
            result = os.system("echo %s | sudo -S %s" % (passwd, reset_cmd))
            if result == 0:
                output_text += "✓ USB 设备已重置\n"
            else:
                output_text += "提示: USB 重置命令已执行 (可能不支持)\n"
        except Exception as error:  # pylint: disable=broad-except
            output_text += f"Note: USB reset skipped: {error}\n"

        # RealSense reset 后补充重启 Argus，减少后续 ZED 访问时的 EOF 概率。
        code_argus, out_argus = self._run_sudo_command("systemctl restart nvargus-daemon.service", passwd)
        if code_argus == 0:
            output_text += "✓ nvargus-daemon 已重启\n"
        else:
            output_text += f"⚠ nvargus-daemon 重启失败\n{out_argus}\n"

        output_text += "\n=== 重置完成 ===\n您现在可以尝试重新录制。"
        return output_text

    def shutdown_system(self) -> str:
        """Schedules system shutdown and stops all recordings first."""
        return self._schedule_power_action(action_flag="-h", action_name="关闭")

    def reboot_system(self) -> str:
        """Schedules system reboot and stops all recordings first."""
        return self._schedule_power_action(action_flag="-r", action_name="重启")

    def cancel_shutdown(self) -> str:
        """Cancels pending delayed shutdown/reboot action."""
        passwd = os.getenv("ACE_SUDO_PASSWORD", "3456999")
        output_text = "=== 取消电源操作 ===\n"

        with self._power_action_lock:
            canceled_local = self._terminate_pending_power_action_locked()

        code_cancel, out_cancel = self._run_sudo_command("shutdown -c", passwd)
        if canceled_local or code_cancel == 0:
            output_text += "✓ 已取消待执行的关机/重启操作\n"
        else:
            output_text += "⚠ 未检测到可取消的关机/重启任务\n"
            if out_cancel:
                output_text += out_cancel

        return output_text


class DataService:
    """Recorded data listing, statistics, and export."""

    @staticmethod
    def _run_quality_check(task_path: str):
        if not QUALITY_CHECK_AVAILABLE:
            return None, f"质量检查不可用: {QUALITY_CHECK_ERROR}"

        try:
            report = quality_check.inspect_session(Path(task_path))
            return report, ""
        except Exception as error:  # pylint: disable=broad-except
            return None, f"质量检查执行失败: {error}"

    @staticmethod
    def _has_task_timestamps(task_path: str) -> bool:
        """Returns True if task_timestamps.csv exists and has any rows."""
        for name in ("task_timestamps.csv", "task_timestamp.csv"):
            csv_path = os.path.join(task_path, name)
            if not os.path.isfile(csv_path):
                continue
            try:
                with open(csv_path, "r", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    for _ in reader:
                        return True
            except OSError:
                return False
        return False

    @staticmethod
    def _summary_path(task_path: str) -> str:
        return os.path.join(task_path, "_encrypted", RECORDING_SUMMARY_NAME)

    @staticmethod
    def _load_recording_summary(task_path: str) -> Optional[Dict[str, object]]:
        summary_path = DataService._summary_path(task_path)
        if not os.path.isfile(summary_path):
            return None

        try:
            with open(summary_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return None

    @staticmethod
    def _summarize_task_timestamps(task_path: str) -> Tuple[int, float, Optional[int], Optional[int]]:
        csv_path = os.path.join(task_path, "task_timestamps.csv")
        if not os.path.exists(csv_path):
            return 0, 0.0, None, None

        task_events: Dict[str, Dict[str, int]] = {}
        try:
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

                    if not task_id or event not in {"start", "end"}:
                        continue

                    event_map = task_events.setdefault(task_id, {})
                    if event == "start":
                        prev_start = event_map.get("start")
                        if prev_start is None or timestamp_ms < prev_start:
                            event_map["start"] = timestamp_ms
                    else:
                        prev_end = event_map.get("end")
                        if prev_end is None or timestamp_ms > prev_end:
                            event_map["end"] = timestamp_ms
        except OSError:
            return 0, 0.0, None, None

        task_count = 0
        first_timestamp_ms: Optional[int] = None
        last_timestamp_ms: Optional[int] = None
        for _, events in sorted(task_events.items()):
            start_ms = events.get("start")
            end_ms = events.get("end")
            if start_ms is None or end_ms is None or end_ms < start_ms:
                continue
            task_count += 1
            if first_timestamp_ms is None or start_ms < first_timestamp_ms:
                first_timestamp_ms = start_ms
            if last_timestamp_ms is None or end_ms > last_timestamp_ms:
                last_timestamp_ms = end_ms

        if task_count == 0 or first_timestamp_ms is None or last_timestamp_ms is None:
            return task_count, 0.0, None, None

        if last_timestamp_ms < first_timestamp_ms:
            return task_count, 0.0, None, None

        duration_seconds = (last_timestamp_ms - first_timestamp_ms) / 1000.0
        return task_count, duration_seconds, first_timestamp_ms, last_timestamp_ms

    @staticmethod
    def _build_size_detail(task_path: str) -> Tuple[int, str, Dict[str, int]]:
        total_size = DataService._get_dir_size(task_path)
        subdir_info = []
        subdir_sizes: Dict[str, int] = {}

        for subdir in ["realsense", "zed", "egocap"]:
            subdir_path = os.path.join(task_path, subdir)
            if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                subdir_size = DataService._get_dir_size(subdir_path)
                subdir_sizes[subdir] = subdir_size
                subdir_str = f"{subdir}: {DataService._format_size(subdir_size)}"

                if subdir == "realsense":
                    try:
                        files = os.listdir(subdir_path)
                        video_files = [
                            name
                            for name in files
                            if name.endswith((".mp4", ".avi", ".mkv"))
                        ]
                        if video_files:
                            details = []
                            for video_file in sorted(video_files):
                                video_path = os.path.join(subdir_path, video_file)
                                video_size = os.path.getsize(video_path)
                                details.append(
                                    f"{video_file}({DataService._format_size(video_size)})"
                                )
                            subdir_str += f" [{', '.join(details)}]"
                    except OSError:
                        pass

                if subdir == "zed":
                    try:
                        files = os.listdir(subdir_path)
                        zed_files = [
                            name
                            for name in files
                            if name.endswith((".svo", ".svo2"))
                        ]
                        if zed_files:
                            details = []
                            for zed_file in sorted(zed_files):
                                zed_path = os.path.join(subdir_path, zed_file)
                                zed_size = os.path.getsize(zed_path)
                                details.append(
                                    f"{zed_file}({DataService._format_size(zed_size)})"
                                )
                            subdir_str += f" [{', '.join(details)}]"
                    except OSError:
                        pass

                subdir_info.append(subdir_str)

        detail_str = " | ".join(subdir_info) if subdir_info else "-"
        return total_size, detail_str, subdir_sizes

    @staticmethod
    def write_recording_summary(task_path: str) -> None:
        summary_path = DataService._summary_path(task_path)
        try:
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        except OSError:
            return

        report, error = DataService._run_quality_check(task_path)
        quality_status = "⚠️ 质检失败"
        quality_summary = f"⚠️ 质检失败: {error}" if error else ""
        duration_minutes = 0.0
        if report is not None and not error:
            duration_minutes = report.duration_minutes
            if report.all_passed:
                quality_status = f"✅ 通过 (时长 {report.duration_minutes:.1f} 分钟)"
                quality_summary = (
                    f"✅ 质检通过：时长 {report.duration_minutes:.1f} 分钟，"
                    "结构/相机/文件大小检查正常"
                )
            else:
                quality_status = DataService.get_quality_status(task_path)
                quality_summary = DataService.get_quality_summary(task_path)

        total_size, detail_str, subdir_sizes = DataService._build_size_detail(task_path)
        task_count, duration_seconds, first_ts, last_ts = DataService._summarize_task_timestamps(
            task_path
        )
        if duration_seconds > 0 and duration_minutes <= 0:
            duration_minutes = duration_seconds / 60.0

        previous_summary = DataService._load_recording_summary(task_path)
        if previous_summary:
            prev_minutes = float(previous_summary.get("duration_minutes") or 0.0)
            prev_seconds = float(previous_summary.get("duration_seconds") or 0.0)
            prev_task_count = int(previous_summary.get("task_count") or 0)
            prev_first_ts = previous_summary.get("first_timestamp_ms")
            prev_last_ts = previous_summary.get("last_timestamp_ms")

            if duration_minutes <= 0 and prev_minutes > 0:
                duration_minutes = prev_minutes
            if duration_seconds <= 0 and prev_seconds > 0:
                duration_seconds = prev_seconds
            if task_count <= 0 and prev_task_count > 0:
                task_count = prev_task_count
            if first_ts is None and prev_first_ts is not None:
                first_ts = prev_first_ts
            if last_ts is None and prev_last_ts is not None:
                last_ts = prev_last_ts

        summary = {
            "version": 1,
            "created_at": datetime.now().isoformat(),
            "record_time": int(os.path.getmtime(task_path)) if os.path.exists(task_path) else None,
            "duration_minutes": duration_minutes,
            "task_count": task_count,
            "duration_seconds": duration_seconds,
            "first_timestamp_ms": first_ts,
            "last_timestamp_ms": last_ts,
            "total_size_bytes": total_size,
            "total_size_str": DataService._format_size(total_size),
            "detail_str": detail_str,
            "subdir_sizes": subdir_sizes,
            "quality_status": quality_status,
            "quality_summary": quality_summary,
        }

        try:
            with open(summary_path, "w", encoding="utf-8") as file:
                json.dump(summary, file, ensure_ascii=False, indent=2)
        except OSError:
            return

    @staticmethod
    def get_quality_status(task_path: str) -> str:
        """Returns short quality status for one task directory."""
        summary = DataService._load_recording_summary(task_path)
        if summary and summary.get("quality_status"):
            return str(summary.get("quality_status"))

        report, error = DataService._run_quality_check(task_path)
        if error:
            return "⚠️ 质检失败"
        if report is None:
            return "⚠️ 质检不可用"
        if report.duration_minutes <= 0:
            return "未结束" if DataService._has_task_timestamps(task_path) else "未打点"
        if report.all_passed:
            return f"✅ 通过 (时长 {report.duration_minutes:.1f} 分钟)"

        issues: List[str] = []
        for chk in (
            report.structure,
            report.egocap_check,
            report.realsense_check,
            report.zed_check,
        ):
            issues.extend(chk.issues)
        for size_result in report.size_results:
            if not size_result.passed:
                issues.append(f"{size_result.file_path}: {size_result.info}")

        if not issues:
            return "❌ 不合格 (未知原因)"

        summary = issues[0]
        if len(issues) > 1:
            summary = f"{summary} 等 {len(issues)} 项"
        return f"❌ 不合格 ({summary})"

    @staticmethod
    def get_quality_summary(task_path: str) -> str:
        """Returns human-readable quality check summary for a task."""
        summary = DataService._load_recording_summary(task_path)
        if summary and summary.get("quality_summary"):
            return str(summary.get("quality_summary"))

        report, error = DataService._run_quality_check(task_path)
        if error:
            return f"⚠️ 质检失败: {error}"
        if report is None:
            return "⚠️ 质检不可用"
        if report.duration_minutes <= 0:
            if DataService._has_task_timestamps(task_path):
                return "未结束：请记录任务结束时间"
            return "未打点：请先记录任务开始/结束时间"
        if report.all_passed:
            return f"✅ 质检通过：时长 {report.duration_minutes:.1f} 分钟，结构/相机/文件大小检查正常"

        issues: List[str] = []
        for chk in (
            report.structure,
            report.egocap_check,
            report.realsense_check,
            report.zed_check,
        ):
            issues.extend(chk.issues)
        for size_result in report.size_results:
            if not size_result.passed:
                issues.append(f"{size_result.file_path}: {size_result.info}")

        if not issues:
            return "❌ 质检未通过: 未知原因"

        max_items = 8
        shown = issues[:max_items]
        issue_lines = "\n".join(f"- {item}" for item in shown)
        if len(issues) > max_items:
            issue_lines += f"\n- ... 共 {len(issues)} 项"

        return f"❌ 质检未通过:\n{issue_lines}"

    @staticmethod
    def get_quality_expected_mb() -> Dict[str, float]:
        """Returns current EXPECTED_MB_PER_10MIN config."""
        default = {
            "*.svo2": 200.0,
            "*_rgb.mp4": 100.0,
            "*_depth.mp4": 800.0,
            "mjpeg.bin": 5052.0,
        }
        if not QUALITY_CHECK_AVAILABLE:
            return dict(default)
        try:
            return dict(quality_check.EXPECTED_MB_PER_10MIN)
        except Exception:  # pylint: disable=broad-except
            return dict(default)

    @staticmethod
    def set_quality_expected_mb(
        svo2_mb: float,
        rgb_mb: float,
        depth_mb: float,
        mjpeg_mb: float,
    ) -> str:
        """Updates EXPECTED_MB_PER_10MIN config used by quality check."""
        values = {
            "*.svo2": float(svo2_mb),
            "*_rgb.mp4": float(rgb_mb),
            "*_depth.mp4": float(depth_mb),
            "mjpeg.bin": float(mjpeg_mb),
        }

        for key, value in values.items():
            if value <= 0:
                return f"❌ 质检阈值需为正数: {key}={value}"

        if not QUALITY_CHECK_AVAILABLE:
            return f"⚠️ 质检模块不可用，未生效: {QUALITY_CHECK_ERROR}"

        try:
            quality_check.EXPECTED_MB_PER_10MIN.clear()
            quality_check.EXPECTED_MB_PER_10MIN.update(values)
        except Exception as error:  # pylint: disable=broad-except
            return f"❌ 更新失败: {error}"

        return "✅ 已更新质检阈值(按 10 分钟统计)"

    @staticmethod
    def _resolve_template_tag_path() -> Optional[str]:
        """Finds template_tag.json from likely runtime locations."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(current_dir, "template_tag.json"),
            os.path.join(os.getcwd(), "template_tag.json"),
            os.path.join(os.getcwd(), "recording", "template_tag.json"),
        ]

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return None
    @staticmethod
    def _get_first_task_timestamp(task_dir: str) -> Optional[int]:
        """Reads the first valid timestamp from task_timestamps.csv (ms preferred)."""
        csv_path = os.path.join(task_dir, "task_timestamps.csv")
        if not os.path.isfile(csv_path):
            return None

        first_timestamp: Optional[int] = None
        try:
            with open(csv_path, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    raw_value = str(row.get("timestamp", "")).strip()
                    try:
                        ts = int(raw_value)
                    except (TypeError, ValueError):
                        continue
                    if first_timestamp is None:
                        first_timestamp = ts
                        break
        except OSError:
            return None

        return first_timestamp
    @staticmethod
    def _get_last_task_timestamp(task_dir: str) -> Optional[int]:
        """Reads the last valid timestamp from task_timestamps.csv (ms preferred)."""
        csv_path = os.path.join(task_dir, "task_timestamps.csv")
        if not os.path.isfile(csv_path):
            return None

        last_timestamp: Optional[int] = None
        try:
            with open(csv_path, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    raw_value = str(row.get("timestamp", "")).strip()
                    try:
                        ts = int(raw_value)
                    except (TypeError, ValueError):
                        continue
                    last_timestamp = ts
        except OSError:
            return None

        return last_timestamp

    @staticmethod
    def _format_autocollect_name_from_timestamp(timestamp: int) -> str:
        """Builds year_month_day_hour_min_second_AutoCollect name from timestamp."""
        # 兼容毫秒/秒时间戳
        ts_seconds = timestamp / 1000.0 if timestamp > 10_000_000_000 else float(timestamp)
        dt = datetime.fromtimestamp(ts_seconds)
        return dt.strftime("%Y_%m_%d_%H_%M_%S_AutoCollect")

    @staticmethod
    def package_tasks_for_upload(selected_tasks: List[str]) -> str:
        """Renames selected today's tasks into AutoCollect format and writes datatag.json."""
        date_str = datetime.now().strftime("%Y%m%d")
        date_root = os.path.realpath(date_str)
        if not os.path.isdir(date_root):
            return f"❌ 今日目录不存在: {date_str}"

        template_path = DataService._resolve_template_tag_path()
        

        if not template_path:
            return "❌ 未找到 template_tag.json，无法打包"
        # 校验模板是合法 JSON
        try:
            with open(template_path, "r", encoding="utf-8") as file:
                template_datatag = json.load(file)
        except Exception as error:  # pylint: disable=broad-except
            return f"❌ template_tag.json 解析失败: {error}"



        task_map = {
            name: path for name, path in DataService._collect_today_task_dirs(date_str)
        }

        selected = [str(item).strip() for item in (selected_tasks or []) if str(item).strip()]
        # 默认行为：未勾选时自动打包今日全部任务
        if not selected:
            selected = sorted(task_map.keys())

        if not selected:
            return "❌ 今日无可打包任务"

        details: List[str] = []
        success_count = 0
        for task_name in selected:
            source_path = task_map.get(task_name)
            if not source_path:
                details.append(f"❌ {task_name}: 未找到任务目录")
                continue

            source_real = os.path.realpath(source_path)
            if not (
                source_real == date_root
                or source_real.startswith(date_root + os.sep)
            ):
                details.append(f"❌ {task_name}: 非法路径，已拒绝 ({source_real})")
                continue
            config_path = os.path.join(source_real, "config.json")
            try:
                with open(config_path, "r", encoding="utf-8") as file:
                    config_data = json.load(file)
            except Exception as error:  # pylint: disable=broad-except
                return f"❌ config.json 解析失败: {error}"
            first_timestamp = DataService._get_first_task_timestamp(source_real)
            last_timestamp = DataService._get_last_task_timestamp(source_real)
            if last_timestamp is None:
                details.append(f"❌ {task_name}: 缺少有效 task_timestamps.csv")
                continue

            target_basename = DataService._format_autocollect_name_from_timestamp(
                last_timestamp
            )
            parent_dir = os.path.dirname(source_real)
            target_path = os.path.join(parent_dir, target_basename)

            suffix = 1
            while os.path.exists(target_path) and os.path.realpath(target_path) != source_real:
                target_path = os.path.join(parent_dir, f"{target_basename}_{suffix}")
                suffix += 1

            try:
                if os.path.realpath(target_path) != source_real:
                    os.rename(source_real, target_path)
                
                template_datatag['labels'] = config_data.get('task', [])
                template_datatag['test_date'] = datetime.now().strftime("%Y_%m_%d")
                template_datatag['test_car_id'] = f"ACE_{EGOCAP_SN}"
                template_datatag['test_time_end'] = last_timestamp
                template_datatag['test_time_start'] = first_timestamp
                template_datatag['test_duration'] = (last_timestamp - first_timestamp)/1000
                



                # 上传目录内目标文件名要求为 datatag.json
                dst_template = os.path.join(target_path, "datatag.json")
                # shutil.copy2(template_path, dst_template)

                with open(dst_template, "w", encoding="utf-8") as file:
                    json.dump(template_datatag, file, ensure_ascii=False, indent=2)



                legacy_template = os.path.join(target_path, "template_tag.json")
                if os.path.isfile(legacy_template):
                    try:
                        os.remove(legacy_template)
                    except OSError:
                        pass
                success_count += 1
                details.append(
                    f"✅ {task_name} -> {os.path.basename(target_path)}，已写入 datatag.json"
                )
            except Exception as error:  # pylint: disable=broad-except
                details.append(f"❌ {task_name}: 打包失败 ({error})")

        failed_count = len(selected) - success_count
        summary = f"打包完成：成功 {success_count} 个，失败 {failed_count} 个"
        if details:
            return f"{summary}\n{'=' * 30}\n" + "\n".join(details)
        return summary

    @staticmethod
    def get_storage_space_info(path: str = ".") -> str:
        """Returns storage usage summary for the partition containing path."""
        try:
            real_path = os.path.realpath(path or ".")
            if not os.path.exists(real_path):
                real_path = os.path.realpath(".")

            total, used, free = shutil.disk_usage(real_path)
            free_ratio = (free / total * 100.0) if total > 0 else 0.0

            return (
                f"磁盘路径: {real_path}\n"
                f"总容量: {DataService._format_size(total)}\n"
                f"已使用: {DataService._format_size(used)}\n"
                f"剩余可用: {DataService._format_size(free)} ({free_ratio:.1f}%)"
            )
        except Exception as error:  # pylint: disable=broad-except
            return f"❌ 获取磁盘空间失败: {error}"

    @staticmethod
    def _is_likely_external_mount(mount_point: str, source: str, fstype: str) -> bool:
        """Heuristic filter for user-usable external storage mount points."""
        if not mount_point or not os.path.isdir(mount_point):
            return False

        mount_point = os.path.realpath(mount_point)
        source = str(source or "")
        fstype = str(fstype or "").lower()

        # 过滤虚拟/系统文件系统
        skip_fs = {
            "proc",
            "sysfs",
            "tmpfs",
            "devtmpfs",
            "devpts",
            "overlay",
            "squashfs",
            "cgroup",
            "cgroup2",
            "pstore",
            "securityfs",
            "debugfs",
            "tracefs",
            "hugetlbfs",
            "mqueue",
            "ramfs",
            "autofs",
            "fusectl",
            "nsfs",
            "rpc_pipefs",
            "binfmt_misc",
            "configfs",
            "efivarfs",
        }
        if fstype in skip_fs:
            return False

        # 过滤系统关键挂载点
        skip_prefixes = (
            "/",
            "/boot",
            "/boot/efi",
            "/proc",
            "/sys",
            "/dev",
            "/run",
            "/var",
            "/usr",
            "/etc",
            "/snap",
            "/tmp",
        )
        if mount_point in skip_prefixes:
            return False
        if any(mount_point.startswith(prefix + "/") for prefix in skip_prefixes):
            return False

        # 常见网络文件系统不作为“本地外接盘”
        network_fs = {"nfs", "nfs4", "cifs", "smb3", "sshfs", "fuse.sshfs"}
        if fstype in network_fs:
            return False

        # Linux 下常见磁盘设备路径（含 loop 会在后续过滤）
        if source.startswith("/dev/"):
            dev_name = os.path.basename(source)
            if dev_name.startswith("loop"):
                return False
            return True

        # macOS 常见设备路径
        if source.startswith("/dev/disk"):
            return True

        return False

    @staticmethod
    def get_external_drives() -> List[str]:
        """Finds potential external drive mount points with cross-machine robustness."""
        candidates = set()

        # 1) 优先从系统挂载表识别（更通用，不依赖固定目录）
        mount_files = ["/proc/self/mounts", "/proc/mounts", "/etc/mtab"]
        for mount_file in mount_files:
            if not os.path.exists(mount_file):
                continue
            try:
                with open(mount_file, "r", encoding="utf-8", errors="ignore") as file:
                    for line in file:
                        parts = line.split()
                        if len(parts) < 3:
                            continue
                        source, mount_point, fstype = parts[0], parts[1], parts[2]

                        # 处理挂载表中的空格转义（\040）
                        mount_point = mount_point.replace("\\040", " ")

                        if DataService._is_likely_external_mount(
                            mount_point, source, fstype
                        ):
                            candidates.add(os.path.realpath(mount_point))
                break
            except OSError as error:
                print(f"Error reading mount table {mount_file}: {error}")

        # 2) 兜底：扫描常见挂载根目录（兼容一些精简系统或自定义挂载脚本）
        fallback_roots = ["/media", "/mnt", "/Volumes", "/run/media"]
        try:
            user = os.environ.get("USER") or os.getlogin()
            if user:
                fallback_roots.extend([
                    f"/media/{user}",
                    f"/run/media/{user}",
                ])
        except OSError:
            pass

        for root in fallback_roots:
            if not os.path.isdir(root):
                continue
            try:
                for item in os.listdir(root):
                    if item.startswith("."):
                        continue
                    full_path = os.path.realpath(os.path.join(root, item))
                    if not os.path.isdir(full_path):
                        continue
                    if full_path in {"/media/cdrom", "/media/cdrom0"}:
                        continue
                    candidates.add(full_path)
            except OSError as error:
                print(f"Error scanning {root}: {error}")

        if not candidates:
            return ["未检测到外部存储"]

        result = sorted(candidates)
        result.sort(
            key=lambda item: (
                not any(token in item.lower() for token in ["passport", "samsung", "disk", "usb"]),
                item,
            )
        )
        return result

    @staticmethod
    def transfer_data_to_drive(
        target_path: str,
        scope: str = "today",
    ) -> Generator[Tuple[str, str], None, None]:
        """Transfers recording folders to external drive via rsync."""
        if not target_path or target_path == "未检测到外部存储":
            yield "❌ 请先选择有效的移动硬盘路径", "未开始"
            return

        source_dirs = []
        if scope == "today":
            date_str = datetime.now().strftime("%Y%m%d")
            if os.path.exists(date_str):
                source_dirs = [date_str]
            else:
                yield f"⚠️ 今日 ({date_str}) 无录制数据", "无数据"
                return
        elif scope == "all":
            try:
                for item in os.listdir("."):
                    is_date_dir = (
                        os.path.isdir(item)
                        and item.isdigit()
                        and len(item) == 8
                        and item.startswith("20")
                    )
                    if is_date_dir:
                        source_dirs.append(item)
            except OSError:
                pass

            if not source_dirs:
                yield "⚠️ 当前目录下未找到类似 YYYYMMDD 的数据文件夹", "无数据"
                return

        dest_dir = os.path.join(target_path, "ACE_Data_Backup")
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except OSError as error:
            yield f"❌ 无法创建目标目录:\n{dest_dir}\n错误: {error}", "错误"
            return

        command = ["rsync", "-av", "--progress"] + source_dirs + [dest_dir]
        yield (
            f"🚀 开始传输...\n源: {source_dirs}\n目标: {dest_dir}\n",
            "正在准备...",
        )

        try:
            process = subprocess.Popen(
                command,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            output_buffer = ""
            last_yield_time = time.time()

            for line in process.stdout:
                output_buffer += line
                if len(output_buffer) > 20000:
                    output_buffer = output_buffer[-20000:]

                current_time = time.time()
                if current_time - last_yield_time > 0.2:
                    status = "正在传输..."
                    if "%" in line:
                        for part in line.split():
                            if "%" in part:
                                status = f"传输中... {part}"
                                break
                    yield output_buffer, status
                    last_yield_time = current_time

            process.wait()
            if process.returncode == 0:
                yield output_buffer + "\n\n✅ 传输成功完成！", "✅ 完成"
            else:
                yield (
                    output_buffer + f"\n\n❌ 传输失败 (退出代码 {process.returncode})",
                    "❌ 失败",
                )

        except Exception as error:  # pylint: disable=broad-except
            yield f"执行错误: {error}", "❌ 错误"

    @staticmethod
    def _is_task_dir(path: str) -> bool:
        """Checks whether a directory looks like one recording task folder."""
        if not os.path.isdir(path):
            return False

        if os.path.exists(os.path.join(path, "task_timestamps.csv")):
            return True

        summary_path = DataService._summary_path(path)
        if os.path.isfile(summary_path):
            return True

        for subdir in ["realsense", "zed", "egocap"]:
            if os.path.isdir(os.path.join(path, subdir)):
                return True
        return False

    @staticmethod
    def _collect_today_task_dirs(date_str: str) -> List[Tuple[str, str]]:
        """Collects today's task directories across all device folders."""
        day_path = date_str
        if not os.path.isdir(day_path):
            return []

        collected: List[Tuple[str, str]] = []
        seen_paths = set()

        for level1_name in os.listdir(day_path):
            level1_path = os.path.join(day_path, level1_name)
            if not os.path.isdir(level1_path):
                continue

            if DataService._is_task_dir(level1_path):
                real_path = os.path.realpath(level1_path)
                if real_path not in seen_paths:
                    collected.append((level1_name, level1_path))
                    seen_paths.add(real_path)
                continue

            try:
                for level2_name in os.listdir(level1_path):
                    level2_path = os.path.join(level1_path, level2_name)
                    if not os.path.isdir(level2_path):
                        continue
                    if not DataService._is_task_dir(level2_path):
                        continue

                    display_name = f"{level1_name}/{level2_name}"
                    real_path = os.path.realpath(level2_path)
                    if real_path in seen_paths:
                        continue
                    collected.append((display_name, level2_path))
                    seen_paths.add(real_path)
            except OSError:
                continue

        return collected

    @staticmethod
    def get_recorded_folders() -> List[List[str]]:
        """Returns today's task folders with size breakdown."""
        date_str = datetime.now().strftime("%Y%m%d")
        base_path = date_str

        if not os.path.exists(base_path):
            return [
                [
                    f"今日未找到录制记录 ({date_str})",
                    "",
                    "",
                    "",
                    "",
                ]
            ]

        folders = []
        try:
            task_dirs = DataService._collect_today_task_dirs(date_str)

            for entry, full_path in task_dirs:
                summary = DataService._load_recording_summary(full_path)
                if summary:
                    record_time = summary.get("record_time")
                    if isinstance(record_time, (int, float)):
                        time_str = time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(record_time)
                        )
                    else:
                        mod_time = os.path.getmtime(full_path)
                        time_str = time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(mod_time)
                        )

                    size_str = str(summary.get("total_size_str") or "-")
                    detail_str = str(summary.get("detail_str") or "-")
                    quality_status = str(summary.get("quality_status") or "-")
                    folders.append([entry, time_str, size_str, detail_str, quality_status])
                    continue

                mod_time = os.path.getmtime(full_path)
                time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mod_time))
                dir_size, detail_str, _ = DataService._build_size_detail(full_path)
                size_str = DataService._format_size(dir_size)
                quality_status = DataService.get_quality_status(full_path)
                folders.append([entry, time_str, size_str, detail_str, quality_status])

            folders.sort(key=lambda item: item[1], reverse=True)
        except Exception as error:  # pylint: disable=broad-except
            return [[f"错误: {error}", "", "", "", ""]]

        if not folders:
            return [[f"在 {date_str}/ 中未找到录制记录", "", "", "", ""]]
        return folders

    @staticmethod
    def get_today_recorded_task_names() -> List[str]:
        """Returns today's task display names for dropdown selections."""
        date_str = datetime.now().strftime("%Y%m%d")
        task_dirs = DataService._collect_today_task_dirs(date_str)
        names = [name for name, _ in task_dirs]
        names.sort()
        return names

    @staticmethod
    def delete_recorded_task(task_name: str, confirmed: bool = False) -> str:
        """Deletes one recorded task folder under today's date directory."""
        if not task_name or not str(task_name).strip():
            return "❌ 请先选择要删除的任务"

        if not confirmed:
            return "⚠️ 请先勾选“我确认要永久删除选中任务”"

        date_str = datetime.now().strftime("%Y%m%d")
        date_root = os.path.realpath(date_str)
        if not os.path.isdir(date_root):
            return f"❌ 今日目录不存在: {date_str}"

        task_map = {
            name: path for name, path in DataService._collect_today_task_dirs(date_str)
        }
        target_path = task_map.get(str(task_name).strip())
        if not target_path:
            return f"❌ 未找到任务: {task_name}"

        target_real = os.path.realpath(target_path)
        if not (
            target_real == date_root
            or target_real.startswith(date_root + os.sep)
        ):
            return f"❌ 非法路径，拒绝删除: {target_real}"

        if target_real == date_root:
            return "❌ 不允许删除当天根目录"

        if not os.path.isdir(target_real):
            return f"❌ 目录不存在: {target_real}"

        try:
            folder_size = DataService._get_dir_size(target_real)
            shutil.rmtree(target_real)
            return (
                f"✅ 已删除任务: {task_name}\n"
                f"路径: {target_real}\n"
                f"释放空间: {DataService._format_size(folder_size)}"
            )
        except Exception as error:  # pylint: disable=broad-except
            return f"❌ 删除失败: {error}"

    @staticmethod
    def _get_dir_size(path: str) -> int:
        """Calculates recursive directory size."""
        total_size = 0
        try:
            for dirpath, _, filenames in os.walk(path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
        except OSError:
            pass
        return total_size

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Formats bytes into human-readable units."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        if size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

    @staticmethod
    def calculate_total_video_duration() -> str:
        """Calculates task count and duration from task_timestamps.csv."""

        date_str = datetime.now().strftime("%Y%m%d")
        base_path = date_str

        if not os.path.exists(base_path):
            return f"❌ 目录不存在: {base_path}"

        total_duration_seconds = 0.0
        total_task_count = 0
        folder_details: List[str] = []

        try:
            task_folders = DataService._collect_today_task_dirs(date_str)

            if not task_folders:
                return "❌ 未找到任何任务文件夹"

            task_folders.sort(key=lambda item: item[0])

            for folder_name, task_path in task_folders:
                used_csv = False
                csv_path = os.path.join(task_path, "task_timestamps.csv")
                if os.path.exists(csv_path):
                    try:
                        (
                            folder_task_count,
                            folder_duration_seconds,
                            _,
                            _,
                        ) = DataService._summarize_task_timestamps(task_path)

                        if folder_duration_seconds > 0:
                            total_task_count += folder_task_count
                            total_duration_seconds += folder_duration_seconds
                            folder_details.append(
                                f"{folder_name}: 任务 {folder_task_count} 条, 时长 {folder_duration_seconds:.2f} 秒"
                            )
                            used_csv = True
                    except Exception as error:  # pylint: disable=broad-except
                        print(f"Error processing task {folder_name}: {error}")

                if used_csv:
                    continue

                summary = DataService._load_recording_summary(task_path)
                if summary:
                    folder_task_count = int(summary.get("task_count") or 0)
                    folder_duration_seconds = float(summary.get("duration_seconds") or 0.0)
                    if folder_duration_seconds <= 0:
                        summary_minutes = float(summary.get("duration_minutes") or 0.0)
                        if summary_minutes > 0:
                            folder_duration_seconds = summary_minutes * 60.0
                    if folder_duration_seconds <= 0:
                        continue

                    total_task_count += folder_task_count
                    total_duration_seconds += folder_duration_seconds
                    folder_details.append(
                        f"{folder_name}: 任务 {folder_task_count} 条, 时长 {folder_duration_seconds:.2f} 秒"
                    )
                    continue

            total_minutes = int(total_duration_seconds // 60)
            total_seconds = int(total_duration_seconds % 60)
            total_hours = total_minutes // 60
            remaining_minutes = total_minutes % 60

            result = f"📊 当日视频统计 ({date_str})\n"
            result += f"{'=' * 50}\n"
            result += f"总任务数: {total_task_count}\n"
            if total_hours > 0:
                result += (
                    f"总时长: {total_hours}小时{remaining_minutes}分{total_seconds}秒\n"
                )
            else:
                result += f"总时长: {total_minutes}分{total_seconds}秒\n"
            result += f"{'=' * 50}\n\n"
            result += "各任务详情:\n"
            result += "\n".join(folder_details) if folder_details else "无有效任务时间戳数据"
            return result
        except Exception as error:  # pylint: disable=broad-except
            return f"❌ 统计失败: {error}"

class WebPreviewService:
    """Web preview business logic without UI framework coupling."""

    @staticmethod
    def startup_cleanup() -> None:
        """Stops any stale preview instances during startup."""
        try:
            stop_all_previews()
            # 再等待一小段时间，确保 UVC 句柄彻底释放。
            time.sleep(0.8)
        except Exception:  # pylint: disable=broad-except
            pass

    @staticmethod
    def get_available_camera_choices() -> Dict[str, List[str]]:
        """Returns initial camera dropdown choices and values."""
        available_cameras = list_cameras("all")
        rs_cameras = [
            cam for cam in available_cameras if cam.get("type") == "realsense"
        ]
        zed_cameras = [cam for cam in available_cameras if cam.get("type") == "zed"]

        rs_camera_choices = [f"{cam['index']} (SN: {cam['sn']})" for cam in rs_cameras]
        rs_camera_values = [f"{cam['index']}" for cam in rs_cameras]
        zed_camera_choices = [
            f"{cam['index']} (SN: {cam['sn']})" for cam in zed_cameras
        ]
        zed_camera_values = [f"{cam['index']}" for cam in zed_cameras]

        if not rs_camera_choices:
            rs_camera_choices = ["0 (No camera detected)"]
            rs_camera_values = ["0"]
        if not zed_camera_choices:
            zed_camera_choices = []
            zed_camera_values = []

        return {
            "rs_choices": rs_camera_choices,
            "rs_values": rs_camera_values,
            "zed_choices": zed_camera_choices,
            "zed_values": zed_camera_values,
        }

    @staticmethod
    def refresh_cameras(
        cam_type: str = "realsense",
    ) -> Tuple[List[str], Optional[str], str]:
        """Refreshes camera dropdown options for one camera type."""
        cameras = list_cameras(cam_type)
        new_choices = [f"{cam['index']} (SN: {cam['sn']})" for cam in cameras]
        new_values = [f"{cam['index']}" for cam in cameras]

        if not new_choices:
            if cam_type == "realsense":
                new_choices = ["0 (No camera detected)"]
                new_values = ["0"]
            else:
                new_choices = []
                new_values = []
            value = new_values[0] if new_values else None
            status = f"未找到 {cam_type} 相机"
        else:
            value = new_choices[0]
            status = f"发现 {len(new_choices)} 个 {cam_type} 相机"

        return new_choices, value, status

    @staticmethod
    def update_web_preview(device_index: int = 0, camera_type: str = "realsense"):
        """Returns latest RGB frame for preview."""
        return get_rgb_frame(device_index, camera_type)

    @staticmethod
    def update_web_preview_dual(device_index: int = 0, camera_type: str = "realsense"):
        """Returns latest RGB and depth frames for preview."""
        return get_depth_and_rgb(device_index, camera_type)

    @staticmethod
    def start_preview(device_index: int = 0, camera_type: str = "realsense") -> str:
        """Starts one preview stream."""
        try:
            target_index = int(device_index)
            for attempt in range(2):
                cleanup_device(target_index, camera_type)
                time.sleep(0.3 + attempt * 0.4)
                preview = get_preview_instance(
                    target_index,
                    show_depth=True,
                    auto_start=True,
                    camera_type=camera_type,
                )
                if preview and preview.is_running:
                    if attempt == 0:
                        return f"相机 {device_index} 的 {camera_type.upper()} 预览已启动!"
                    return (
                        f"相机 {device_index} 的 {camera_type.upper()} 预览已启动!"
                        "（重试后成功）"
                    )

            return f"启动相机 {device_index} 的 {camera_type} 预览失败"
        except Exception as error:  # pylint: disable=broad-except
            return f"启动 {camera_type} 预览错误: {error}"

    @staticmethod
    def stop_preview(device_index: int = 0, camera_type: str = "realsense") -> str:
        """Stops one preview stream."""
        try:
            cleanup_device(int(device_index), camera_type)
            return f"相机 {device_index} 的 {camera_type.upper()} 预览已停止!"
        except Exception as error:  # pylint: disable=broad-except
            return f"停止 {camera_type} 预览错误: {error}"
