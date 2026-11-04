import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import teleop_adapter
from arx_x5_data import workbench_control as module


class FakeRecorder:
    def __init__(self):
        self.mode = "idle"
        self.episode = None
        self.starts = 0
        self.stops = 0

    def status(self):
        return dict(mode=self.mode, mock=False, episode=self.episode, message="test")

    def record(self, task):
        self.starts += 1
        self.mode = "recording"
        self.episode = "/data/task/episode_000001"

    def stop_recording(self, cancel=False):
        self.stops += 1
        self.mode = "saving"
        self.cancel = cancel


class RecordingControlTests(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.runtime = FakeRecorder()
        self.journal = Path(self.tmp.name) / "requests.json"
        self.control = module.RecordingControl(self.runtime, self.journal)

    def test_duplicate_start_and_stop_never_replay_after_restart(self):
        request = {"request_id": "start-123", "task": "task"}
        first = self.control.execute("start", request)
        restarted = module.RecordingControl(self.runtime, self.journal)
        self.assertEqual(first, restarted.execute("start", request))
        self.assertEqual(self.runtime.starts, 1)
        stop = {"request_id": "stop-1234", "episode": first["episode"]}
        restarted.execute("stop", stop)
        restarted.execute("stop", stop)
        self.assertEqual(self.runtime.stops, 1)

    def test_wrong_episode_cannot_stop_current_take(self):
        self.control.execute("start", {"request_id": "start-123", "task": "task"})
        with self.assertRaises(RuntimeError):
            self.control.execute("stop", {"request_id": "stop-1234", "episode": "wrong"})
        self.assertEqual(self.runtime.stops, 0)

    def test_cancel_uses_existing_invalid_recording_path_and_is_idempotent(self):
        started = self.control.execute("start", {"request_id": "start-123", "task": "task"})
        body = {"request_id": "cancel-123", "episode": started["episode"]}
        self.control.execute("cancel", body)
        self.control.execute("cancel", body)
        self.assertTrue(self.runtime.cancel)
        self.assertEqual(self.runtime.stops, 1)

    def test_uncertain_start_never_replays(self):
        self.journal.write_text(json.dumps({"start-123": {"action": "start"}}))
        with self.assertRaisesRegex(RuntimeError, "未重复执行"):
            module.RecordingControl(self.runtime, self.journal).execute(
                "start", {"request_id": "start-123", "task": "task"}
            )
        self.assertEqual(self.runtime.starts, 0)

    def test_locally_saved_pending_data_released_but_incomplete_rejected(self):
        base = Path(self.tmp.name)
        episode = base / "task" / "episode"
        episode.mkdir(parents=True)
        state = {"mode": "idle", "episode": str(episode)}
        for status, expected in [
            ("pending_validation", True),
            ("incomplete", False),
            ("cancelled", False),
            ("complete", True),
        ]:
            (episode / "manifest.json").write_text(
                json.dumps({"status": status, "mock": False, "frames": 30})
            )
            with patch.object(teleop_adapter, "DATA_ROOT", base):
                self.assertEqual(bool(teleop_adapter.receipt(state)), expected)

    def test_still_writing_cannot_release_even_with_manifest(self):
        base = Path(self.tmp.name)
        episode = base / "task" / "episode"
        episode.mkdir(parents=True)
        (episode / "manifest.json").write_text(
            json.dumps({"status": "pending_validation", "mock": False, "frames": 30})
        )
        with patch.object(teleop_adapter, "DATA_ROOT", base):
            self.assertIsNone(teleop_adapter.receipt({"mode": "saving", "episode": str(episode)}))
            self.assertIsNotNone(
                teleop_adapter.receipt({"mode": "warming", "episode": str(episode)})
            )
