"""A pending/failed background job must never block the next take."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import teleop_adapter
from server import DemoAdapter, Workbench


class PendingAdapter(DemoAdapter):
    def call(self, action, operation_id, trial=False):
        if action != "record_stop":
            return super().call(action, operation_id, trial)
        path = self.directory / self.ident
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(
            json.dumps({"status": "pending_validation", "frames": 30, "mock": False})
        )
        (path / "job.json").write_text(json.dumps({"state": "queued", "attempts": 0}))
        self.recording = False
        return {
            "teleop_running": True,
            "recording": False,
            "saved": {"id": self.ident, "path": str(path)},
        }


class BackgroundCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = json.loads(Path("config.example.json").read_text())
        self.config["save_dir"] = str(self.root / "data")
        for key in self.config["commands"]:
            self.config["commands"][key] = ["configured"]
        self.adapter = PendingAdapter(self.config["save_dir"])
        self.w = Workbench(self.config, self.root / "state", self.adapter)
        self.count = 0
        self.act("prepare", switches=True, channels=[True] * 4)
        self.act("status")
        self.act("teleop_start")

    def act(self, action, **payload):
        self.count += 1
        return self.w.act(action, payload, f"test-request-{self.count}")

    def take(self):
        self.act("record_start")
        result = self.act("record_stop")
        self.assertEqual(result["phase"], "reset", result["error"])
        self.assertFalse(result["busy"])
        self.assertFalse(result["recording"])
        return result["saved"][-1]

    def test_three_takes_continue_while_every_job_remains_queued(self):
        first = self.take()
        self.act("verify_trial")
        self.act("reset")
        for _ in range(3):
            item = self.take()
            self.assertEqual(item["processing"], "queued")
            self.act("reset")
        result = self.w.snapshot()
        self.assertEqual(result["processing_counts"], {"queued": 4})
        self.assertEqual(result["saved_count"], 3)
        self.assertEqual(result["validated_count"], 0)
        self.assertEqual(result["phase"], "ready")
        self.assertEqual(first["validation"], "pending")

    def test_old_job_finishes_or_fails_during_next_recording_without_interrupting_it(self):
        first = self.take()
        self.act("verify_trial")
        self.act("reset")
        self.act("record_start")
        path = Path(first["path"])
        (path / "manifest.json").write_text(
            json.dumps({"status": "complete", "mock": False, "frames": 30})
        )
        (path / "job.json").write_text(json.dumps({"state": "converting"}))
        state = self.w.snapshot()
        self.assertEqual(state["saved"][0]["validation"], "passed")
        self.assertEqual(state["saved"][0]["processing"], "converting")
        (path / "job.json").write_text(json.dumps({"state": "failed", "error": "export failed"}))
        state = self.w.snapshot()
        self.assertEqual(state["phase"], "recording")
        self.assertTrue(state["recording"])
        self.assertEqual(state["processing_counts"]["failed"], 1)
        self.assertEqual(state["saved"][0]["processing_error"], "export failed")
        self.assertTrue(path.exists())

    def test_reopened_workbench_reads_durable_job_state(self):
        first = self.take()
        path = Path(first["path"])
        (path / "manifest.json").write_text(
            json.dumps({"status": "complete", "mock": False, "frames": 30})
        )
        (path / "job.json").write_text(json.dumps({"state": "done"}))
        reopened = Workbench(self.config, self.root / "state", self.adapter)
        self.assertEqual(reopened.snapshot()["saved"][0]["processing"], "done")

    def test_stop_adapter_returns_on_write_completion_without_waiting_for_validation(self):
        path = self.root / "data" / "episode"
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(
            json.dumps({"status": "pending_validation", "mock": False, "frames": 30})
        )

        def response(data):
            result = MagicMock()
            result.__enter__.return_value.read.return_value = json.dumps(data).encode()
            return result

        saving = dict(mode="saving", episode=str(path))
        idle = dict(mode="idle", episode=str(path), mock=False, inputs={"telemetry": True})
        with (
            patch.object(teleop_adapter, "DATA_ROOT", path.parent),
            patch.object(teleop_adapter, "processes", return_value=[1]),
            patch.object(
                teleop_adapter,
                "urlopen",
                side_effect=[response(saving), response(saving), response(idle)],
            ),
            patch.dict("os.environ", {"ARX_OPERATION_ID": "test-stop-id"}),
            patch.object(teleop_adapter.time, "sleep") as sleep,
        ):
            result = teleop_adapter.record("stop")
            self.assertEqual(result["saved"]["validation"], "pending")
            self.assertEqual(result["saved"]["path"], str(path))
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
