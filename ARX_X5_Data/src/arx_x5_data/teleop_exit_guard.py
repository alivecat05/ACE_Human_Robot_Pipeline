"""Optional guard for the external teleop; stdlib only, never commands hardware.

Copy this file next to test_remote_encoder.py when using the documented adapter.
The recorder does not import or run it.
"""

import math
import signal
from collections import deque


class ExitSignals:
    """Turn ordinary exit signals into requests, including repeated Ctrl+C.

    Install in the main thread and keep installed through controller cleanup.
    Faults/exceptions and physical emergency stops are not intercepted here.
    """

    def __init__(self):
        self._pending = None
        self._previous = {}

    def __enter__(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.signal(signum, self._request)
        return self

    def _request(self, signum, _frame):
        self._pending = signal.Signals(signum).name

    def take_request(self):
        pending, self._pending = self._pending, None
        return pending

    def __exit__(self, *_exc):
        for signum, previous in self._previous.items():
            signal.signal(signum, previous)


class HomeExitGuard:
    """Require both measured arms and submitted targets near SDK's six-joint zero.

    Feedback from all seven motors must be fresh. Gripper opening is not changed
    or used as the parking target. A rejected request is consumed, never queued
    for automatic exit later.
    """

    POSITION_TOL = math.radians(3)
    SPEED_TOL = 0.1  # rad/s, measured SDK velocities
    STABILITY_TOL = math.radians(0.5)
    STABLE_SECONDS = 1.0
    MAX_FEEDBACK_AGE = 0.1
    MAX_SAMPLE_GAP = 0.1

    def __init__(self):
        self._recent = deque(maxlen=256)
        self.ready = False
        self.reason = "尚未确认双臂回正"

    def _reject(self, reason):
        self._recent.clear()
        self.ready = False
        self.reason = reason

    def observe(self, sides, commands, *, now, wall_time):
        """Read existing controllers only; any unreadable feedback denies exit."""
        try:
            if len(sides) != 2 or not all(map(math.isfinite, (now, wall_time))):
                raise ValueError("需要双臂与有效时间")
            positions = []
            for side, (name, leader, follower, _) in zip(("left", "right"), sides, strict=True):
                if not leader.is_online:
                    raise ValueError(f"{name}主臂离线")
                status = follower.get_status()
                if any(
                    status.get(k) for k in ("fault", "offline", "overlimit", "gripper_offline")
                ) or status.get("can", {}).get("link_down"):
                    raise ValueError(f"{name}从臂状态异常")
                motors = [*follower._ctrl.motors, follower._ctrl.gripper]
                if len(motors) != 7:
                    raise ValueError(f"{name}从臂电机反馈不全")
                for motor in motors:
                    feedback = motor.feedback
                    age = wall_time - float(feedback.ts)
                    if (
                        not math.isfinite(age)
                        or not 0 <= age <= self.MAX_FEEDBACK_AGE
                        or motor.rx_count < 1
                        or feedback.error
                    ):
                        raise ValueError(f"{name}从臂反馈过期或异常")
                command = commands[side]
                if not command["joint_accepted"]:
                    raise ValueError(f"{name}从臂未接受当前目标")
                q, v, target = (
                    tuple(map(float, values))
                    for values in (status["q"], status["v"], command["joint_positions_rad"])
                )
                if any(len(values) != 6 for values in (q, v, target)) or not all(
                    math.isfinite(x) for values in (q, v, target) for x in values
                ):
                    raise ValueError(f"{name}从臂关节数据无效")
                joint = max(range(6), key=lambda j: abs(q[j]))
                if abs(q[joint]) > self.POSITION_TOL:
                    raise ValueError(
                        f"{name}从臂 J{joint + 1} 距零位 {math.degrees(abs(q[joint])):.1f}°"
                    )
                if max(map(abs, target)) > self.POSITION_TOL:
                    raise ValueError(f"{name}主臂目标尚未回正")
                if max(map(abs, v)) > self.SPEED_TOL:
                    raise ValueError(f"{name}从臂仍在运动")
                positions.extend(q)
        except Exception as exc:
            self._reject(f"无法确认回正：{exc}")
            return

        if self._recent and not 0 < now - self._recent[-1][0] <= self.MAX_SAMPLE_GAP:
            self._recent.clear()
        self._recent.append((now, positions))
        while len(self._recent) > 1 and now - self._recent[1][0] >= self.STABLE_SECONDS:
            self._recent.popleft()
        if any(
            max(values) - min(values) > self.STABILITY_TOL
            for values in zip(*(q for _, q in self._recent), strict=True)
        ):
            self._recent.clear()
            self._recent.append((now, positions))
        elapsed = now - self._recent[0][0]
        self.ready = elapsed >= self.STABLE_SECONDS
        self.reason = (
            "双臂已回正并稳定，可按 Ctrl+C 退出"
            if self.ready
            else f"双臂回正稳定计时 {elapsed:.1f}/{self.STABLE_SECONDS:.1f} 秒"
        )
