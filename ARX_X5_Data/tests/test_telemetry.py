import json
import socket
import stat
import time

import pytest

from arx_x5_data.model import History
from arx_x5_data.publisher import TelemetryPublisher
from arx_x5_data.telemetry import TelemetryReceiver, decode_packet

from .conftest import packet


def test_actual_targets_are_not_replaced_by_feedback():
    value = packet(10.0)
    sample = decode_packet(json.dumps(value).encode(), 10.01, 0.1)
    assert len(sample.state) == len(sample.action) == 14
    assert sample.state[0] == 0.1 and sample.action[0] == 0.2
    assert sample.state[6] == 0.5 and sample.action[6] == 1
    assert sample.feedback_times == pytest.approx((9.99, 9.99))


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("gripper_native", -0.10001),
        ("gripper_native", 5.10001),
        ("joint_positions_rad", [0.0] * 5),
        ("joint_positions_rad", [float("nan")] * 6),
    ],
)
def test_invalid_vectors_are_rejected(side, field, value):
    payload = packet(10.0)
    payload["arms"][side][field] = value
    with pytest.raises(ValueError):
        decode_packet(json.dumps(payload).encode(), 10.01, 0.1)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(at_monotonic_s=9.0),
        lambda p: p.update(at_monotonic_s=11.0),
        lambda p: p["arms"]["left"].update(feedback_oldest_at_s=98.0),
        lambda p: p["arms"]["right"]["submitted_command"].update(gripper_accepted=False),
        lambda p: p["arms"]["right"]["submitted_command"].update(joint_accepted=False),
    ],
)
def test_stale_or_unaccepted_data_is_not_training_data(mutate):
    payload = packet(10.0)
    mutate(payload)
    with pytest.raises(ValueError):
        decode_packet(json.dumps(payload).encode(), 10.01, 0.1)


def test_socket_round_trip_and_ownership(config):
    history = History(8)
    receiver = TelemetryReceiver(config.socket, history, 0.1)
    receiver.start()
    second = TelemetryReceiver(config.socket, History(8), 0.1)
    publisher = TelemetryPublisher(config.socket)
    try:
        assert stat.S_IMODE(config.socket.stat().st_mode) == 0o600
        with pytest.raises(OSError):
            second.start()
        second.close()
        assert config.socket.exists()
        arms = packet()["arms"]
        for arm in arms.values():
            arm["feedback_oldest_at_s"] = time.time()
        assert publisher.publish(arms)
        deadline = time.monotonic() + 1
        while history.latest() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert history.latest() is not None
        assert receiver.error is None
    finally:
        publisher.close()
        receiver.close()
    assert not config.socket.exists()


def test_missing_recorder_never_blocks_publisher(config):
    publisher = TelemetryPublisher(config.socket)
    try:
        start = time.monotonic()
        assert not publisher.publish(packet()["arms"])
        assert time.monotonic() - start < 0.1
    finally:
        publisher.close()


def test_invalid_datagrams_are_counted_and_valid_telemetry_recovers(config):
    history = History(8)
    receiver = TelemetryReceiver(config.socket, history, 0.1)
    receiver.start()
    sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        invalid = packet()
        invalid["arms"]["left"]["gripper_native"] = -0.2
        for data in (b"not json", b"[]", json.dumps(invalid).encode()):
            sender.sendto(data, str(config.socket))
        deadline = time.monotonic() + 1
        while receiver.rejected_packets < 3 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert receiver.rejected_packets == 3
        assert "gripper=-0.2" in receiver.last_rejection
        assert history.latest() is None
        assert receiver.error is None
        sender.sendto(json.dumps(packet()).encode(), str(config.socket))
        deadline = time.monotonic() + 1
        while history.latest() is None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert history.latest() is not None
        assert receiver.error is None
    finally:
        sender.close()
        receiver.close()


@pytest.mark.parametrize("side,index", [("left", 6), ("right", 13)])
@pytest.mark.parametrize("value", [-0.1, -0.041771572442206306, 5.052834363317313, 5.1])
def test_measured_boundary_feedback_is_preserved_without_clipping(side, index, value):
    payload = packet(10.0)
    payload["arms"][side]["gripper_native"] = value
    sample = decode_packet(json.dumps(payload).encode(), 10.01, 0.1)
    assert sample.state[index] == value / 5
    assert sample.action[index] == 1.0


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("value", [-0.001, 5.001])
def test_feedback_margin_never_expands_the_command_range(side, value):
    payload = packet(10.0)
    payload["arms"][side]["submitted_command"]["gripper_native"] = value
    with pytest.raises(ValueError, match="target"):
        decode_packet(json.dumps(payload).encode(), 10.01, 0.1)


def test_custom_feedback_margin_is_respected():
    payload = packet(10.0)
    payload["arms"]["left"]["gripper_native"] = 5.15
    data = json.dumps(payload).encode()
    with pytest.raises(ValueError):
        decode_packet(data, 10.01, 0.1)
    assert decode_packet(data, 10.01, 0.1, gripper_feedback_margin_native=0.2).state[6] == 1.03
