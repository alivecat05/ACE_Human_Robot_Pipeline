#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense 和 ZED 相机 Web 实时预览 (用于 Gradio 集成)
支持在 Web UI 中直接显示摄像头画面
"""
import pyrealsense2 as rs
import cv2
import numpy as np
import threading
import time
from queue import Queue
from typing import Tuple, Optional

# 尝试导入 ZED SDK
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    print("[Warning] ZED SDK not available")


class WebRealSensePreview:
    """用于 Web 的 RealSense 预览类"""
    
    def __init__(self, device_index: int = 0, show_depth: bool = True, serial_number: str = None):
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
                            if d.get_info(rs.camera_info.serial_number) == self.serial_number:
                                device = d
                                print(f"Found device with SN {self.serial_number} at index {i}")
                                break
                    except Exception as e:
                        print(f"Error checking device {i}: {e}")
                        continue
            
            # 2. 如果没找到或没提供SN，回退到使用索引
            if device is None:
                if self.device_index >= len(devices):
                    print(f"Error: Device {self.device_index} not found (Total: {len(devices)})")
                    return False
                device = devices[self.device_index]
            
            # 获取设备信息
            try:
                sn = device.get_info(rs.camera_info.serial_number)
                name = device.get_info(rs.camera_info.name)
                print(f"[Camera Setup] SN: {sn}, Model: {name}")
                
                # 配置流
                self.config.enable_device(sn)
                self.config.enable_stream(rs.stream.depth, -1, 848, 480, rs.format.z16, 15)
                self.config.enable_stream(rs.stream.color, -1, 848, 480, rs.format.rgb8, 15)
                
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
                    depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
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
        self.pipeline.stop()
        print(f"Camera {self.device_index} stopped")


class WebZEDPreview:
    """用于 Web 的 ZED 预览类"""
    
    def __init__(self, device_index: int = 0, show_depth: bool = True, serial_number: int = None):
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
        
    def configure(self) -> bool:
        """配置ZED相机参数"""
        if not ZED_AVAILABLE:
            print("[ZED] SDK not available")
            return False
            
        try:
            # 获取设备列表
            dev_stereo_list = sl.Camera.get_device_list()
            dev_mono_list = sl.CameraOne.get_device_list() if hasattr(sl, 'CameraOne') else []
            
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
            
            self.is_mono = (cam_type == "mono")
            
            # 创建相机对象
            if self.is_mono:
                self.zed = sl.CameraOne()
                init_params = sl.InitParametersOne()
            else:
                self.zed = sl.Camera()
                init_params = sl.InitParameters()
            
            # 配置参数
            init_params.set_from_serial_number(device.serial_number)
            
            # 使用 AUTO 让 SDK 自动选择合适的分辨率
            init_params.camera_resolution = sl.RESOLUTION.SVGA
            init_params.camera_fps = 15
            
            # 只有双目相机支持深度
            if not self.is_mono:
                init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE if self.show_depth else sl.DEPTH_MODE.NONE
            
            # 打开相机
            err = self.zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"[ZED] Failed to open camera: {err}")
                return False
            
            print(f"[ZED] Camera {device.serial_number} ({device.camera_model}) opened")
            return True
            
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
                        depth_norm = np.nan_to_num(depth_data, nan=0.0, posinf=0.0, neginf=0.0)
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
                    cv2.putText(placeholder, "No Depth (Mono Camera)", (200, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
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
        if self.zed and self.zed.is_opened():
            self.zed.close()
        print(f"[ZED] Camera {self.device_index} stopped")


# 全局预览实例
_preview_instances = {}
_preview_lock = threading.Lock()


def get_preview_instance(device_index: int = 0, show_depth: bool = True, auto_start: bool = False, 
                        serial_number: str = None, camera_type: str = "realsense") -> object:
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
                preview = WebZEDPreview(device_index=device_index, show_depth=show_depth, 
                                       serial_number=int(serial_number) if serial_number else None)
            else:
                preview = WebRealSensePreview(device_index=device_index, show_depth=show_depth, 
                                             serial_number=serial_number)
            
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


def get_depth_and_rgb(device_index: int = 0, camera_type: str = "realsense") -> Tuple[np.ndarray, np.ndarray]:
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
            ctx = rs.context()
            devices = ctx.query_devices()
            num_devices = len(devices)
            
            seen_sns = set()
            max_check = min(num_devices, 4)
            
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


def list_realsense_cameras() -> list:
    """仅列出 RealSense 相机（保持向后兼容）"""
    return list_cameras(camera_type="realsense")


def list_zed_cameras() -> list:
    """仅列出 ZED 相机"""
    return list_cameras(camera_type="zed")


if __name__ == "__main__":
    # 测试代码
    print("Available cameras:", list_cameras())
    
    # 启动预览
    preview = get_preview_instance(device_index=0)
    if preview:
        try:
            for i in range(30):
                rgb = preview.get_frame()
                if rgb is not None:
                    print(f"Frame {i}: {rgb.shape}")
                    cv2.imshow("RGB", rgb)
                    if cv2.waitKey(67) & 0xFF == ord('q'):
                        break
                time.sleep(0.01)
        finally:
            stop_preview_instance(0)
            cv2.destroyAllWindows()
    else:
        print("Failed to start preview")
