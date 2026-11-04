"""Device-filtered Windows Raw Input for USB keyboard foot pedals.

No keyboard hooks or extra packages are required. Events reach Tk through a
queue; the native message thread never calls GUI or recording methods.
API reference: https://learn.microsoft.com/en-us/windows/win32/inputdev/about-raw-input
"""
import ctypes as ct
from ctypes import wintypes as wt
import queue
import sys
import threading
import time


WM_INPUT = 0x00FF
WM_INPUT_DEVICE_CHANGE = 0x00FE
WM_CLOSE, WM_DESTROY = 0x0010, 0x0002
RID_INPUT, RIDI_DEVICENAME = 0x10000003, 0x20000007
RIDEV_INPUTSINK, RIDEV_DEVNOTIFY = 0x100, 0x2000
UINT_ERROR = 0xFFFFFFFF
MODIFIERS = {0x10, 0x11, 0x12, 0x5B, 0x5C, *range(0xA0, 0xA6)}


class Device(ct.Structure):
    _fields_ = [("handle", wt.HANDLE), ("kind", wt.DWORD)]


class Registration(ct.Structure):
    _fields_ = [("page", wt.USHORT), ("usage", wt.USHORT),
                ("flags", wt.DWORD), ("target", wt.HWND)]


class Header(ct.Structure):
    _fields_ = [("kind", wt.DWORD), ("size", wt.DWORD),
                ("device", wt.HANDLE), ("wparam", wt.WPARAM)]


class Keyboard(ct.Structure):
    _fields_ = [("scan", wt.USHORT), ("flags", wt.USHORT),
                ("reserved", wt.USHORT), ("vkey", wt.USHORT),
                ("message", wt.UINT), ("extra", wt.ULONG)]


def _api():
    if sys.platform != "win32":
        raise OSError("USB 脚踏原始输入目前仅支持 Windows")
    api = ct.WinDLL("user32", use_last_error=True)
    signatures = {
        "GetRawInputDeviceList": ([ct.POINTER(Device), ct.POINTER(wt.UINT), wt.UINT], wt.UINT),
        "GetRawInputDeviceInfoW": ([wt.HANDLE, wt.UINT, wt.LPVOID, ct.POINTER(wt.UINT)], wt.UINT),
        "GetRawInputData": ([wt.HANDLE, wt.UINT, wt.LPVOID, ct.POINTER(wt.UINT), wt.UINT], wt.UINT),
        "RegisterRawInputDevices": ([ct.POINTER(Registration), wt.UINT, wt.UINT], wt.BOOL),
        "DefWindowProcW": ([wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM], ct.c_ssize_t),
        "CreateWindowExW": ([wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD, ct.c_int, ct.c_int,
                             ct.c_int, ct.c_int, wt.HWND, wt.HMENU, wt.HINSTANCE, wt.LPVOID], wt.HWND),
        "DestroyWindow": ([wt.HWND], wt.BOOL),
        "PostMessageW": ([wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM], wt.BOOL),
        "GetMessageW": ([ct.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT], ct.c_int),
        "TranslateMessage": ([ct.POINTER(wt.MSG)], wt.BOOL),
        "DispatchMessageW": ([ct.POINTER(wt.MSG)], ct.c_ssize_t),
        "UnregisterClassW": ([wt.LPCWSTR, wt.HINSTANCE], wt.BOOL),
        "PostQuitMessage": ([ct.c_int], None),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes, function.restype = arguments, result
    return api


def _device_name(api, handle):
    size = wt.UINT()
    if api.GetRawInputDeviceInfoW(handle, RIDI_DEVICENAME, None, ct.byref(size)) == UINT_ERROR:
        return ""
    name = ct.create_unicode_buffer(size.value + 1)
    if api.GetRawInputDeviceInfoW(handle, RIDI_DEVICENAME, name, ct.byref(size)) == UINT_ERROR:
        return ""
    return name.value


def keyboard_devices(api=None):
    api = api or _api()
    # A device can arrive between the size query and enumeration.
    for _ in range(3):
        size = wt.UINT()
        if api.GetRawInputDeviceList(None, ct.byref(size), ct.sizeof(Device)) == UINT_ERROR:
            raise ct.WinError(ct.get_last_error())
        devices = (Device * size.value)()
        count = api.GetRawInputDeviceList(devices, ct.byref(size), ct.sizeof(Device))
        if count != UINT_ERROR:
            return [{"handle": d.handle, "device_id": _device_name(api, d.handle)}
                    for d in devices[:count] if d.kind == 1]
    raise ct.WinError(ct.get_last_error())


class PressGate:
    """One action per physical press, including repeats and multi-key macros."""
    def __init__(self, debounce_ms=300):
        self.down = set()
        self.last_press = float("-inf")
        self.debounce_s = debounce_ms / 1000

    def feed(self, device, key, released, now):
        identity = (device, key)
        if released:
            self.down.discard(identity)
            return False
        if key in MODIFIERS or key == 255:
            return False
        already_down = bool(self.down)
        self.down.add(identity)
        if already_down or now - self.last_press < self.debounce_s:
            return False
        self.last_press = now
        return True


class RawPedal:
    def __init__(self, device_id, debounce_ms=300, vkey=119):
        if not isinstance(device_id, str) or not device_id.strip():
            raise ValueError("请配置 foot_pedal.device_id，避免将普通键盘作为脚踏")
        if not isinstance(debounce_ms, (int, float)) or not 0 <= debounce_ms <= 5000:
            raise ValueError("foot_pedal.debounce_ms 必须在 0 到 5000 之间")
        if not isinstance(vkey, int) or not 1 <= vkey < 255:
            raise ValueError("foot_pedal.vkey 必须是 1 到 254 的 Windows 虚拟键码")
        self.device_id = device_id.upper()
        self.vkey = vkey
        self.gate = PressGate(debounce_ms)
        self.events = queue.SimpleQueue()
        self.ready = threading.Event()
        self.hwnd = None
        self.thread = None
        self.error = None
        self.connected = None
        self.devices = {}

    def start(self):
        self.api = _api()
        self.thread = threading.Thread(target=self._run, name="usb-foot-pedal", daemon=True)
        self.thread.start()
        if not self.ready.wait(5):
            raise RuntimeError("USB 脚踏监听启动超时")
        if self.error:
            raise self.error
        return self

    def poll(self):
        events = []
        while not self.events.empty():
            events.append(self.events.get_nowait())
        return events

    def close(self):
        if self.hwnd:
            self.api.PostMessageW(self.hwnd, WM_CLOSE, 0, 0)
        if self.thread:
            self.thread.join(timeout=2)

    def _refresh_devices(self):
        devices = {d["handle"]: d["device_id"] for d in keyboard_devices(self.api)
                   if self.device_id in d["device_id"].upper()}
        changed = self.connected is None or devices != self.devices
        self.devices = devices
        self.connected = bool(self.devices)
        self.gate.down.intersection_update({identity for identity in self.gate.down
                                            if identity[0] in self.devices})
        if changed:
            self.events.put(("status", "脚踏已连接" if self.connected else "脚踏未连接"))

    def _input(self, handle):
        size = wt.UINT()
        header_size = ct.sizeof(Header)
        if self.api.GetRawInputData(handle, RID_INPUT, None, ct.byref(size), header_size) == UINT_ERROR:
            return
        buffer = ct.create_string_buffer(size.value)
        if self.api.GetRawInputData(handle, RID_INPUT, buffer, ct.byref(size), header_size) == UINT_ERROR:
            return
        if size.value < header_size + ct.sizeof(Keyboard):
            return
        header = Header.from_buffer(buffer)
        if header.kind != 1 or header.device not in self.devices:
            return
        key = Keyboard.from_buffer(buffer, header_size)
        released = bool(key.flags & 1)
        self.events.put(("key", {"device_id": self.devices[header.device],
                                 "vkey": key.vkey, "released": released}))
        if key.vkey == self.vkey and self.gate.feed(header.device, key.vkey, released, time.monotonic()):
            self.events.put(("press", key.vkey))

    def _run(self):
        api = self.api
        callback_type = ct.WINFUNCTYPE(ct.c_ssize_t, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)

        class WindowClass(ct.Structure):
            _fields_ = [("style", wt.UINT), ("proc", callback_type),
                        ("class_extra", ct.c_int), ("window_extra", ct.c_int),
                        ("instance", wt.HINSTANCE), ("icon", wt.HANDLE),
                        ("cursor", wt.HANDLE), ("background", wt.HANDLE),
                        ("menu", wt.LPCWSTR), ("name", wt.LPCWSTR)]

        @callback_type
        def window_proc(hwnd, message, wparam, lparam):
            try:
                if message == WM_INPUT:
                    self._input(lparam)
                elif message == WM_INPUT_DEVICE_CHANGE:
                    self._refresh_devices()
                elif message == WM_CLOSE:
                    api.DestroyWindow(hwnd)
                    return 0
                elif message == WM_DESTROY:
                    api.PostQuitMessage(0)
                    return 0
            except Exception as error:
                self.events.put(("error", str(error)))
            return api.DefWindowProcW(hwnd, message, wparam, lparam)

        kernel = ct.WinDLL("kernel32", use_last_error=True)
        kernel.GetModuleHandleW.argtypes, kernel.GetModuleHandleW.restype = [wt.LPCWSTR], wt.HMODULE
        instance = kernel.GetModuleHandleW(None)
        name = f"PiperCapturePedal_{id(self)}"
        window_class = WindowClass(proc=window_proc, instance=instance, name=name)
        api.RegisterClassW.argtypes, api.RegisterClassW.restype = [ct.POINTER(WindowClass)], wt.ATOM
        registered = input_registered = False
        try:
            if not api.RegisterClassW(ct.byref(window_class)):
                raise ct.WinError(ct.get_last_error())
            registered = True
            self.hwnd = api.CreateWindowExW(0, name, name, 0, 0, 0, 0, 0, wt.HWND(-3), None, instance, None)
            if not self.hwnd:
                raise ct.WinError(ct.get_last_error())
            registration = Registration(1, 6, RIDEV_INPUTSINK | RIDEV_DEVNOTIFY, self.hwnd)
            if not api.RegisterRawInputDevices(ct.byref(registration), 1, ct.sizeof(Registration)):
                raise ct.WinError(ct.get_last_error())
            input_registered = True
            self._refresh_devices()
            self.ready.set()
            message = wt.MSG()
            while True:
                result = api.GetMessageW(ct.byref(message), None, 0, 0)
                if result == -1:
                    raise ct.WinError(ct.get_last_error())
                if result == 0:
                    break
                api.TranslateMessage(ct.byref(message))
                api.DispatchMessageW(ct.byref(message))
        except Exception as error:
            self.error = error
            self.events.put(("error", str(error)))
        finally:
            self.ready.set()
            if input_registered:
                registration = Registration(1, 6, 1, None)  # RIDEV_REMOVE requires a NULL target.
                api.RegisterRawInputDevices(ct.byref(registration), 1, ct.sizeof(Registration))
            if self.hwnd:
                api.DestroyWindow(self.hwnd)
                self.hwnd = None
            if registered:
                api.UnregisterClassW(name, instance)


def diagnose(device_id, seconds=30, vkey=119):
    import json
    print(json.dumps(keyboard_devices(), indent=2, ensure_ascii=False), flush=True)
    pedal = RawPedal(device_id, vkey=vkey).start()
    print(f"监听脚踏 {seconds} 秒，请踩下并松开（不会录制或控制机械臂）。", flush=True)
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            for kind, value in pedal.poll():
                print(json.dumps({"event": kind, "value": value}, ensure_ascii=False), flush=True)
            time.sleep(.01)
    finally:
        pedal.close()
