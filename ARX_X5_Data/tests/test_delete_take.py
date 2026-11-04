import fcntl
import json
import tempfile
import unittest
from pathlib import Path

from server import DemoAdapter, Workbench, WorkflowError
from take_recycle import LOCK, MARKER, recycle, restore


class DeleteTakeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = json.loads(Path("config.example.json").read_text())
        self.config["save_dir"] = str(self.root / "data")
        self.adapter = DemoAdapter(self.config["save_dir"])
        self.w = Workbench(self.config, self.root / "state", self.adapter, True)
        self.seq = 0
        self.act("prepare", switches=True, channels=[True] * 4)
        self.act("status")
        self.act("teleop_start")

    def act(self, action, **payload):
        self.seq += 1
        return self.w.act(action, payload, f"delete-test-{self.seq}")

    def saved_formal(self):
        self.act("record_start")
        self.act("record_stop")
        self.act("verify_trial")
        self.act("reset")
        self.act("record_start")
        return self.act("record_stop")["saved"][-1]

    def test_delete_restores_counts_without_touching_files_or_teleop(self):
        item = self.saved_formal()
        path = Path(item["path"])
        before = path.read_bytes()
        deleted = self.act("take_delete", id=item["id"])
        self.assertEqual(deleted["saved_count"], 0)
        self.assertEqual(deleted["saved"][-1]["processing"], "deleted")
        self.assertTrue(self.adapter.teleop)
        self.assertEqual(path.read_bytes(), before)
        restored = self.act("take_restore", id=item["id"])
        self.assertEqual(restored["saved_count"], 1)
        self.assertFalse(restored["saved"][-1]["deleted"])

    def test_recording_discard_ends_only_current_take_and_excludes_count(self):
        self.act("record_start")
        state = self.act("record_discard")
        self.assertEqual(state["phase"], "reset", state["error"])
        self.assertTrue(self.adapter.teleop)
        self.assertFalse(self.adapter.recording)
        self.assertEqual(state["saved_count"], 0)
        self.assertTrue(state["saved"][-1]["discarded"])
        with self.assertRaises(WorkflowError):
            self.act("take_restore", id=state["saved"][-1]["id"])
        self.act("reset")
        self.assertEqual(self.act("record_start")["phase"], "recording")

    def test_delete_older_record_does_not_stop_current_recording(self):
        item = self.saved_formal()
        self.act("reset")
        self.act("record_start")
        state = self.act("take_delete", id=item["id"])
        self.assertEqual(state["phase"], "recording")
        self.assertTrue(self.adapter.recording)

    def test_unknown_id_never_deletes_arbitrary_path(self):
        with self.assertRaises(WorkflowError):
            self.act("take_delete", id="/tmp/unrelated")

    def test_deleted_trial_cannot_pass_first_take_gate(self):
        self.act("record_start")
        item = self.act("record_stop")["saved"][-1]
        self.act("take_delete", id=item["id"])
        with self.assertRaises(WorkflowError):
            self.act("verify_trial")

    def test_recycle_keeps_bytes_and_restore_waits_for_worker(self):
        episode = self.root / "episode"
        episode.mkdir()
        (episode / "manifest.json").write_text(json.dumps({"status": "complete"}))
        raw = episode / "head.mp4"
        raw.write_bytes(b"original data")
        recycle(episode)
        recycle(episode)
        self.assertTrue((episode / MARKER).exists())
        self.assertEqual(raw.read_bytes(), b"original data")
        with (episode / LOCK).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with self.assertRaisesRegex(ValueError, "后台正在收尾"):
                restore(episode)
        (episode / "job.json").write_text(json.dumps({"state": "cancelled"}))
        restore(episode)
        self.assertFalse((episode / MARKER).exists())
        self.assertEqual(json.loads((episode / "job.json").read_text())["state"], "queued")

    def test_cancelled_take_cannot_be_restored_as_valid(self):
        episode = self.root / "cancelled"
        episode.mkdir()
        (episode / "manifest.json").write_text(json.dumps({"status": "cancelled"}))
        recycle(episode)
        with self.assertRaises(ValueError):
            restore(episode)
        self.assertTrue((episode / MARKER).exists())
