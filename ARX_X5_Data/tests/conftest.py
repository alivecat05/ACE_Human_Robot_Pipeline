import json
import time
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from arx_x5_data.config import load_config
from arx_x5_data.model import ARMS, CAMERAS, AlignedSample, CameraFrame, RobotSample
from arx_x5_data.recording import EpisodeWriter


@pytest.fixture
def config(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs/mock.toml")
    cameras = {
        name: replace(camera, width=64, height=48) for name, camera in config.cameras.items()
    }
    return replace(
        config,
        output=tmp_path / "raw",
        socket=Path("/tmp") / f"x5-test-{uuid.uuid4().hex}.sock",
        cameras=cameras,
    )


def packet(stamp=None):
    stamp = time.monotonic() if stamp is None else stamp
    arms = {}
    for side in ARMS:
        arms[side] = {
            "joint_positions_rad": [0.1] * 6,
            "gripper_native": 2.5,
            "feedback_oldest_at_s": 100.0,
            "submitted_command": {
                "joint_positions_rad": [0.2] * 6,
                "gripper_native": 5,
                "joint_accepted": True,
                "gripper_accepted": True,
            },
        }
    return {"at_monotonic_s": stamp, "at_s": 100.01, "arms": arms}


def sample(config, stamp, grid, sequence):
    rgb = np.full((48, 64, 3), (40 + sequence * 10, 100, 160), dtype=np.uint8)
    frames = {
        name: CameraFrame(stamp + 0.001, sequence, rgb.copy(), stamp + 0.002, "mock_monotonic")
        for name in CAMERAS
    }
    robot = RobotSample(
        stamp + 0.002,
        tuple([0.1] * 6 + [0.5]) * 2,
        tuple([0.2] * 6 + [0.7]) * 2,
        (stamp - 0.005, stamp - 0.005),
        "mock",
    )
    return AlignedSample(stamp, grid, frames, robot)


@pytest.fixture
def make_episode(config):
    def make(grids=(0, 1, 2, 3), status="complete", task="杯子放入托盘"):
        start = time.monotonic()
        writer = EpisodeWriter(config, task, start, mock=True)
        for sequence, grid in enumerate(grids):
            writer.submit(sample(config, start + grid / config.fps, grid, sequence))
        writer.finish(status)
        writer.wait(5)
        assert writer.result == status, writer.error
        return writer.path

    return make


def edit_jsonl(path, change):
    rows = [json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines()]
    change(rows)
    (path / "samples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
