import math
import signal
from types import SimpleNamespace

import pytest

from arx_x5_data.teleop_exit_guard import ExitSignals, HomeExitGuard


@pytest.fixture
def teleop():
    sides, commands = [], {}
    for side, name in (("left", "左"), ("right", "右")):
        status = {
            "q": [0.0] * 6,
            "v": [0.0] * 6,
            "fault": None,
            "offline": [],
            "overlimit": [],
            "gripper_offline": False,
            "can": {"link_down": False},
        }
        motors = [
            SimpleNamespace(rx_count=1, feedback=SimpleNamespace(ts=999.999, error=0))
            for _ in range(7)
        ]
        follower = SimpleNamespace(
            status=status,
            get_status=lambda status=status: status,
            _ctrl=SimpleNamespace(motors=motors[:6], gripper=motors[6]),
        )
        sides.append((name, SimpleNamespace(is_online=True), follower, None))
        commands[side] = {"joint_positions_rad": [0.0] * 6, "joint_accepted": True}
    return sides, commands


def observe(guard, teleop, now):
    guard.observe(*teleop, now=now, wall_time=1000.0)


def park(guard, teleop, start=10.0):
    for step in range(102):
        observe(guard, teleop, start + step * 0.01)
    assert guard.ready


def test_both_arms_need_full_stable_interval(teleop):
    guard = HomeExitGuard()
    for step in range(100):
        observe(guard, teleop, 10.0 + step * 0.01)
        assert not guard.ready
    observe(guard, teleop, 11.0)
    assert guard.ready


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize(
    "change",
    [
        lambda leader, arm, cmd: arm.status["q"].__setitem__(2, math.radians(4)),
        lambda leader, arm, cmd: arm.status["v"].__setitem__(2, 0.2),
        lambda leader, arm, cmd: cmd["joint_positions_rad"].__setitem__(0, 0.2),
        lambda leader, arm, cmd: cmd.update(joint_accepted=False),
        lambda leader, arm, cmd: setattr(leader, "is_online", False),
        lambda leader, arm, cmd: arm.status.update(fault="motor fault"),
        lambda leader, arm, cmd: arm.status.update(offline=[2]),
        lambda leader, arm, cmd: arm.status.update(overlimit=[2]),
        lambda leader, arm, cmd: arm.status.update(gripper_offline=True),
        lambda leader, arm, cmd: arm.status["can"].update(link_down=True),
        lambda leader, arm, cmd: setattr(arm._ctrl.motors[0].feedback, "ts", 999.8),
        lambda leader, arm, cmd: setattr(arm._ctrl.gripper.feedback, "ts", 1000.1),
        lambda leader, arm, cmd: setattr(arm._ctrl.gripper.feedback, "error", 1),
        lambda leader, arm, cmd: setattr(arm._ctrl.motors[0], "rx_count", 0),
        lambda leader, arm, cmd: arm.status["q"].__setitem__(0, float("nan")),
        lambda leader, arm, cmd: arm.status["v"].__setitem__(0, float("inf")),
        lambda leader, arm, cmd: arm.status.update(q=[0.0] * 5),
    ],
)
def test_bad_feedback_or_target_revokes_permission(teleop, side, change):
    guard = HomeExitGuard()
    park(guard, teleop)
    _, leader, arm, _ = teleop[0][side]
    change(leader, arm, teleop[1][("left", "right")[side]])
    observe(guard, teleop, 11.02)
    assert not guard.ready
    assert "无法确认回正" in guard.reason


def test_position_drift_restarts_stability_window(teleop):
    guard = HomeExitGuard()
    for step in range(100):
        teleop[0][0][2].status["q"][0] = math.radians(step * 0.02)
        observe(guard, teleop, 10.0 + step * 0.01)
    assert not guard.ready
    park(guard, teleop, start=11.0)


@pytest.mark.parametrize("now", [10.0, 11.5])
def test_clock_reversal_or_loop_gap_requires_new_stable_interval(teleop, now):
    guard = HomeExitGuard()
    park(guard, teleop)
    observe(guard, teleop, now)
    assert not guard.ready
    park(guard, teleop, start=now + 0.01)


def test_unreadable_feedback_denies_exit_without_ending_teleop(teleop):
    guard = HomeExitGuard()
    park(guard, teleop)

    def broken():
        raise RuntimeError("read failed")

    teleop[0][1][2].get_status = broken
    observe(guard, teleop, 11.02)
    assert not guard.ready
    assert "read failed" in guard.reason


def test_repeated_signals_are_consumed_and_handlers_restored():
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    with ExitSignals() as requests:
        for _ in range(3):
            signal.raise_signal(signal.SIGINT)
        assert requests.take_request() == "SIGINT"
        assert requests.take_request() is None
        signal.raise_signal(signal.SIGTERM)
        assert requests.take_request() == "SIGTERM"
        assert requests.take_request() is None
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


def test_exception_is_not_swallowed_by_signal_context():
    previous = signal.getsignal(signal.SIGINT)
    with pytest.raises(RuntimeError, match="controller fault"), ExitSignals():
        raise RuntimeError("controller fault")
    assert signal.getsignal(signal.SIGINT) == previous
