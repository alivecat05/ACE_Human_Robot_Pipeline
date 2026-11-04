from dataclasses import replace

import pytest

from arx_x5_data.model import CAMERAS, CameraFrame, History, RobotSample
from arx_x5_data.synchronization import Synchronizer


def inputs(config):
    cameras = {name: History(20) for name in CAMERAS}
    robot = History(20)
    state = tuple([0.0] * 6 + [0.5]) * 2
    target = tuple([0.2] * 6 + [0.8]) * 2
    packet = RobotSample(1.0, state, target, (0.99, 0.99))
    robot.add(packet)
    for name in CAMERAS:
        cameras[name].add(CameraFrame(1.0, 0, None, 1.001))
    return cameras, robot, packet, Synchronizer(cameras, robot, config.sync)


def test_samples_preserve_the_genuine_same_cycle_target(config):
    _, _, packet, sync = inputs(config)
    row = sync.match(1.0, 0)
    assert row.robot.action == packet.action
    assert row.robot.action != row.robot.state
    assert row.robot.timestamp == 1.0  # No invented future grid timestamp.
    assert sync.match(1.001, 1) is None


def test_joint_camera_matching_can_use_valid_neighbouring_frames(config):
    cameras = {name: History(20) for name in CAMERAS}
    for name in CAMERAS:
        offset = 0.0165 if name == "head" else -0.0125
        for i in range(4):
            stamp = 1 + i / 30 + offset
            cameras[name].add(CameraFrame(stamp, i, None, stamp))
    robot = History(20)
    vector = tuple([0.0] * 6 + [0.5]) * 2
    robot.add(RobotSample(1 + 1 / 30, vector, vector, (1.03, 1.03)))
    sync = Synchronizer(cameras, robot, config.sync)
    row = sync.match(1 + 1 / 30, 1)
    assert row is not None
    stamps = [frame.timestamp for frame in row.cameras.values()]
    assert max(stamps) - min(stamps) <= 0.025


def test_failed_matching_does_not_consume_sources(config):
    cameras, robot, _, sync = inputs(config)
    assert sync.match(2, 0) is None
    assert not sync.last_sequences["head"] >= 0
    assert sync.match(1, 0) is not None
    assert sync.match(1, 1) is None


def test_stale_feedback_is_checked_again_by_synchronizer(config):
    cameras, _, packet, _ = inputs(config)
    robot = History(8)
    robot.add(replace(packet, feedback_times=(0.5, 0.99)))
    sync = Synchronizer(cameras, robot, config.sync)
    assert sync.match(1, 0) is None
    assert sync.rejection == "telemetry_missing_reused_or_stale"


def test_history_cannot_retimestep_cached_data():
    history = History(8)
    frame = CameraFrame(1.0, 0, None, 1.0)
    history.add(frame)
    with pytest.raises(ValueError, match="increase"):
        history.add(frame)


def test_camera_phased_grid_avoids_repeated_span_rejections(config):
    cameras = {name: History(20) for name in CAMERAS}
    offsets = dict(head=0.0, left_wrist=0.013, right_wrist=0.020333)
    for name in CAMERAS:
        for i in range(3):
            stamp = 1 + i / 30 + offsets[name]
            cameras[name].add(CameraFrame(stamp, i, None, stamp))
    robot = History(40)
    sync = Synchronizer(cameras, robot, config.sync)
    click = 1.1
    start = sync.grid_start(click, 30)
    assert click <= start < click + 1 / 30
    assert sync.last_sequences == dict.fromkeys(CAMERAS, -1)
    vector = tuple([0.0] * 6 + [0.5]) * 2
    for i in range(3, 10):
        for name in CAMERAS:
            stamp = 1 + i / 30 + offsets[name]
            cameras[name].add(CameraFrame(stamp, i, None, stamp))
    for i in range(30):
        stamp = 1.08 + i / 1000
        robot.add(RobotSample(stamp, vector, vector, (stamp, stamp)))
    assert sync.match(click, 0) is None
    assert sync.rejection == "camera_span"
    for i in range(30):
        stamp = start - 0.015 + i / 1000
        if stamp > robot.latest().timestamp:
            robot.add(RobotSample(stamp, vector, vector, (stamp, stamp)))
    assert sync.match(start, 0) is not None


def test_grid_start_without_camera_history_does_not_invent_timestamps(config):
    cameras = {name: History(20) for name in CAMERAS}
    sync = Synchronizer(cameras, History(20), config.sync)
    assert sync.grid_start(1.0, 30) == 1.0
