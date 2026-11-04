"""Receive the existing ARX teleop's one-way Unix datagrams; never open CAN."""

import json
import math
import os
import socket
import stat
import threading
import time
from pathlib import Path

from .model import (
    ARMS,
    DEFAULT_GRIPPER_FEEDBACK_MARGIN_NATIVE,
    History,
    RobotSample,
    gripper_bounds_native,
)


def _vector(arms: dict, command: bool, feedback_margin: float) -> tuple[float, ...]:
    result = []
    low, high = gripper_bounds_native(command=command, feedback_margin=feedback_margin)
    for side in ARMS:
        arm = arms[side]["submitted_command"] if command else arms[side]
        if command and (
            arm.get("joint_accepted") is not True or arm.get("gripper_accepted") is not True
        ):
            raise ValueError(f"{side}: SDK did not accept both targets")
        joints = arm["joint_positions_rad"]
        gripper = float(arm["gripper_native"])
        if len(joints) != 6 or not low <= gripper <= high:
            raise ValueError(
                f"{side}: expected six joints and X5-2023 "
                f"{'target' if command else 'feedback'} gripper in {low}..{high}; "
                f"received joints={len(joints)}, gripper={gripper!r}"
            )
        result.extend(float(x) for x in joints)
        result.append(gripper / 5)
    if not all(math.isfinite(x) for x in result):
        raise ValueError("Non-finite robot vector")
    return tuple(result)


def decode_packet(
    data: bytes,
    received_at: float,
    max_age_s: float,
    *,
    gripper_feedback_margin_native: float = DEFAULT_GRIPPER_FEEDBACK_MARGIN_NATIVE,
) -> RobotSample:
    packet = json.loads(data)
    stamp, wall = float(packet["at_monotonic_s"]), float(packet["at_s"])
    if not math.isfinite(stamp) or not math.isfinite(wall):
        raise ValueError("Invalid telemetry clock")
    if not -0.01 <= received_at - stamp <= max_age_s:
        raise ValueError("Telemetry publication is stale or from a different clock")
    arms = packet["arms"]
    if set(arms) != set(ARMS):
        raise ValueError("Telemetry must contain both arms")
    feedback_times = []
    for side in ARMS:
        feedback_wall = float(arms[side]["feedback_oldest_at_s"])
        age = wall - feedback_wall
        if not math.isfinite(age) or not 0 <= age <= max_age_s:
            raise ValueError(f"{side}: stale or invalid motor feedback")
        feedback_times.append(stamp - age)
    return RobotSample(
        stamp,
        _vector(arms, False, gripper_feedback_margin_native),
        _vector(arms, True, gripper_feedback_margin_native),
        tuple(feedback_times),
    )


class TelemetryReceiver:
    def __init__(
        self,
        path: Path,
        history: History[RobotSample],
        max_age_s: float,
        *,
        gripper_feedback_margin_native: float = DEFAULT_GRIPPER_FEEDBACK_MARGIN_NATIVE,
    ):
        self.path, self.history, self.max_age_s = path, history, max_age_s
        gripper_bounds_native(command=False, feedback_margin=gripper_feedback_margin_native)
        self.gripper_feedback_margin_native = gripper_feedback_margin_native
        self.error: Exception | None = None
        self.rejected_packets = 0
        self.last_rejection = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._identity: tuple[int, int] | None = None

    def start(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            # Existing paths are never unlinked or stolen from another recorder.
            sock.bind(str(self.path))
            info = self.path.stat()
            self._identity = info.st_dev, info.st_ino
            os.chmod(self.path, 0o600)
            sock.settimeout(0.1)
            self._socket = sock
            self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
            self._thread.start()
        except BaseException:
            sock.close()
            self._unlink_owned()
            raise

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    data, _, flags, _ = self._socket.recvmsg(65536)
                except TimeoutError:
                    continue
                try:
                    if flags & socket.MSG_TRUNC:
                        raise ValueError("Truncated telemetry datagram")
                    sample = decode_packet(
                        data,
                        time.monotonic(),
                        self.max_age_s,
                        gripper_feedback_margin_native=self.gripper_feedback_margin_native,
                    )
                except (ValueError, KeyError, TypeError, OverflowError, AttributeError) as error:
                    # A rejected sample must never poison the receiving worker.
                    # Do not clamp feedback or insert invalid data into history.
                    self.rejected_packets += 1
                    self.last_rejection = str(error)
                    continue
                self.history.add(sample)
        except Exception as error:
            self.error = error

    def _unlink_owned(self) -> None:
        try:
            info = self.path.lstat()
            if stat.S_ISSOCK(info.st_mode) and (info.st_dev, info.st_ino) == self._identity:
                self.path.unlink()
        except FileNotFoundError:
            pass

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._socket:
            self._socket.close()
        self._unlink_owned()
