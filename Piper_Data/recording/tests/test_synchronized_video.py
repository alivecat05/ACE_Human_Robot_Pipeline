"""Verify that the files saved by the recording pipeline are already aligned."""
import csv
import json
import time

import cv2
import numpy as np

from piper_capture.core import CAMERAS
from piper_capture.recording import Recorder
from piper_capture.synchronization import Synchronizer
from test_capture import add_state, cfg, histories


def test_skewed_cameras_and_a_missing_frame_stay_aligned_on_disk(cfg):
    cameras, states = histories()
    start = time.perf_counter()
    offsets = dict(zip(CAMERAS, (-.008, 0., .008)))
    for grid in range(6):
        stamp = start + grid / cfg["record_hz"]
        add_state(states, stamp, x=.1 + grid * .01)
        if grid == 5:
            continue
        for name in CAMERAS:
            if grid == 2 and name == "right_wrist":
                continue
            image = np.full((48, 64, 3), (20 + 40 * grid, 70, 210), np.uint8)
            cameras[name].add(stamp + offsets[name], {"frame_number": grid, "bgr": image})
    sync = Synchronizer(cameras, states, cfg["sync"], period=1 / cfg["record_hz"])
    recorder = Recorder(cfg["output"], cfg, mock=True)
    recorder.start(start)
    for grid in range(5):
        row = sync.sample(start + grid / cfg["record_hz"])
        if row is None:
            recorder.drop(sync.rejection)
        else:
            recorder.add(row, grid)
    recorder.stop(end_time=start + 5 / cfg["record_hz"])
    assert recorder.wait(5)
    assert recorder.result == "complete", recorder.error
    assert recorder.count == 4
    assert recorder.drop_reasons == {"camera_missing:right_wrist": 1}
    rows = [json.loads(line) for line in (recorder.path / "samples.jsonl").read_text().splitlines()]
    assert [row["grid_index"] for row in rows] == [0, 1, 3, 4]
    with (recorder.path / "action.csv").open() as file:
        actions = list(csv.reader(file))[1:]
    assert len(actions) == len(rows)
    for name in CAMERAS:
        video = cv2.VideoCapture(str(recorder.path / f"{name}.mp4"))
        try:
            assert video.isOpened()
            assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == len(rows)
            for row, action in zip(rows, actions):
                ok, image = video.read()
                assert ok
                assert abs(image[:, :, 0].mean() - (20 + 40 * row["grid_index"])) < 10
                assert [float(value) for value in action[1:]] == row["action"]
                stamps = list(row["source_timestamps_unix_s"]["cameras"].values())
                assert max(stamps) - min(stamps) <= cfg["sync"]["camera_span_ms"] / 1000
            assert not video.read()[0]
        finally:
            video.release()
