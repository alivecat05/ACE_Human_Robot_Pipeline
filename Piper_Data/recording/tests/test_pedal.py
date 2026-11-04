import ctypes as ct
import sys

import pytest

from piper_capture import pedal
from piper_capture.pedal import Header, Keyboard, PressGate, RawPedal


def test_long_hold_bounce_and_release_only_trigger_one_action():
    gate = PressGate(300)
    assert gate.feed(1, 119, False, 0)
    for now in (.03, .1, .31, 2):
        assert not gate.feed(1, 119, False, now)
    assert not gate.feed(1, 119, True, 2.1)
    assert gate.feed(1, 119, False, 2.2)
    assert not gate.feed(1, 119, True, 2.21)
    assert not gate.feed(1, 119, False, 2.25)  # Switch bounce.
    assert not gate.feed(1, 119, False, 3)  # Holding a rejected press cannot rearm it.
    assert not gate.feed(1, 119, True, 3.1)
    assert gate.feed(1, 119, False, 3.2)


def test_device_filter_and_f8_filter_parse_real_raw_input_layout():
    monitor = RawPedal("VID_0483&PID_5750", debounce_ms=0)
    monitor.devices = {123: "pedal"}

    def send(device, key, released=False):
        buffer = ct.create_string_buffer(ct.sizeof(Header) + ct.sizeof(Keyboard))
        header = Header.from_buffer(buffer)
        header.kind, header.device = 1, device
        keyboard = Keyboard.from_buffer(buffer, ct.sizeof(Header))
        keyboard.vkey, keyboard.flags = key, int(released)

        class Api:
            def GetRawInputData(self, handle, command, data, size, header_size):
                ct.cast(size, ct.POINTER(ct.c_uint))[0] = ct.sizeof(buffer)
                if data is not None:
                    ct.memmove(data, buffer, ct.sizeof(buffer))
                return ct.sizeof(buffer) if data is not None else 0

        monitor.api = Api()
        monitor._input(1)
        return monitor.poll()

    assert send(456, 119) == []  # F8 on the ordinary keyboard.
    assert not any(kind == "press" for kind, _ in send(123, 82))  # Pedal programmed as R.
    assert ("press", 119) in send(123, 119)
    assert not any(kind == "press" for kind, _ in send(123, 119))
    assert not any(kind == "press" for kind, _ in send(123, 119, released=True))
    assert ("press", 119) in send(123, 119)


def test_unrelated_hotplug_does_not_rearm_held_pedal_and_removal_clears_it(monkeypatch):
    monitor = RawPedal("VID_0483&PID_5750")
    monitor.api = object()
    devices = [{"handle": 1, "device_id": "HID#VID_0483&PID_5750"}]
    monkeypatch.setattr(pedal, "keyboard_devices", lambda api: devices)
    monitor._refresh_devices()
    assert monitor.poll() == [("status", "脚踏已连接")]
    assert monitor.gate.feed(1, 119, False, 0)
    monitor._refresh_devices()
    assert monitor.poll() == []
    assert not monitor.gate.feed(1, 119, False, 1)
    devices.clear()
    monitor._refresh_devices()
    assert monitor.poll() == [("status", "脚踏未连接")]
    assert not monitor.gate.down
    devices.append({"handle": 2, "device_id": "HID#VID_0483&PID_5750"})
    monitor._refresh_devices()
    assert monitor.poll() == [("status", "脚踏已连接")]
    assert monitor.gate.feed(2, 119, False, 2)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Raw Input")
def test_native_listener_starts_and_releases_message_thread():
    monitor = RawPedal("VID_FFFF&PID_FFFF").start()
    try:
        assert monitor.connected is False
        assert monitor.poll() == [("status", "脚踏未连接")]
    finally:
        monitor.close()
    assert not monitor.thread.is_alive()
    assert monitor.hwnd is None
