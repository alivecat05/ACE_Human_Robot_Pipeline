import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from server import CommandAdapter, DemoAdapter, Workbench, WorkflowError, can_origin, make_handler


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = json.loads(Path("config.example.json").read_text())
        self.config["save_dir"] = str(self.root / "data")
        self.adapter = DemoAdapter(self.config["save_dir"])
        self.w = Workbench(self.config, self.root / "state", self.adapter, True)
        self.counter = 0

    def act(self, action, **kwargs):
        self.counter += 1
        return self.w.act(action, kwargs, f"request-{self.counter:05}")

    def prepare(self):
        self.act("prepare", switches=True, channels=[True] * 4)
        self.act("status")
        self.act("teleop_start")

    def take(self):
        self.act("record_start")
        result = self.act("record_stop")
        self.assertEqual(result["phase"], "reset", result["error"])
        self.assertFalse(self.adapter.recording)
        self.assertTrue(self.adapter.teleop)
        return result

    def test_trial_and_three_takes_exclude_reset(self):
        self.prepare()
        self.take()
        self.act("verify_trial")
        self.act("reset")
        for _ in range(3):
            self.take()
            self.assertFalse(self.adapter.recording)
            self.act("reset")
            self.assertFalse(self.adapter.recording)
        result = self.w.snapshot()
        self.assertEqual(result["saved_count"], 3)
        self.assertEqual(len(result["saved"]), 4)
        for item in result["saved"]:
            self.assertTrue(Path(item["path"]).is_file())

    def test_repeated_request_and_double_start(self):
        self.prepare()
        first = self.w.act("record_start", {}, "same-start-id")
        second = self.w.act("record_start", {}, "same-start-id")
        self.assertEqual(first["recording_id"], second["recording_id"])
        with self.assertRaises(WorkflowError):
            self.act("record_start")
        self.assertEqual(len(self.w.snapshot()["saved"]), 0)

    def test_refresh_reads_without_side_effect(self):
        self.prepare()
        self.act("record_start")
        ident = self.adapter.ident
        for _ in range(5):
            self.w.snapshot()
            self.act("status")
        self.assertEqual(self.adapter.ident, ident)
        self.assertTrue(self.adapter.recording)

    def test_no_next_take_until_reset(self):
        self.prepare()
        self.take()
        with self.assertRaises(WorkflowError):
            self.act("record_start")

    def test_reset_cannot_run_while_recording(self):
        self.prepare()
        self.act("record_start")
        with self.assertRaises(WorkflowError):
            self.act("reset")
        self.assertTrue(self.adapter.recording)

    def test_bad_save_does_not_count_and_recovers_with_receipt(self):
        self.prepare()
        self.act("record_start")
        original = self.adapter.call
        receipt = {}

        def call(action, *args):
            result = original(action, *args)
            if action == "record_stop":
                receipt.update(result)
                return {**result, "saved": {"id": self.adapter.ident, "path": "/missing/file"}}
            return result

        self.adapter.call = call
        result = self.act("record_stop")
        self.assertEqual(result["phase"], "error")
        self.assertEqual(result["saved_count"], 0)
        self.assertEqual(len(result["saved"]), 0)
        self.adapter.call = lambda action, *args: (
            receipt if action == "status" else original(action, *args)
        )
        result = self.act("status")
        self.assertEqual(result["phase"], "reset", result["error"])
        self.assertEqual(len(result["saved"]), 1)

    def test_interruption_never_counts_as_saved(self):
        self.prepare()
        self.act("record_start")
        self.adapter.recording = False
        result = self.act("status")
        self.assertEqual(result["phase"], "error")
        self.assertEqual(len(result["saved"]), 0)

    def test_stop_does_not_repeat_when_already_stopped(self):
        self.prepare()
        self.act("record_start")
        self.adapter.recording = False
        result = self.act("record_stop")
        self.assertEqual(result["phase"], "error")
        self.assertFalse(Path(self.config["save_dir"]).exists())

    def test_teleop_failure_not_success(self):
        self.act("prepare", switches=True, channels=[True] * 4)
        self.act("status")
        original = self.adapter.call
        self.adapter.call = lambda action, *args: (
            (_ for _ in ()).throw(WorkflowError("启动失败"))
            if action == "teleop_start"
            else original(action, *args)
        )
        result = self.act("teleop_start")
        self.assertEqual(result["phase"], "setup")
        self.assertIsNone(result["teleop_running"])
        self.assertIn("启动失败", result["error"])

    def test_busy_blocks_duplicate_inflight(self):
        self.prepare()
        self.act("record_start")
        original = self.adapter.call
        gate = threading.Event()
        entered = threading.Event()

        def call(action, *args):
            if action == "record_stop":
                entered.set()
                gate.wait(3)
            return original(action, *args)

        self.adapter.call = call
        worker = threading.Thread(target=lambda: self.act("record_stop"))
        worker.start()
        self.assertTrue(entered.wait(2))
        self.assertEqual(self.w.snapshot()["phase"], "saving")
        with self.assertRaises(WorkflowError):
            self.act("record_start")
        gate.set()
        worker.join()

    def test_restart_never_replays_and_clears_can_confirmation(self):
        self.prepare()
        self.act("record_start")
        recovered = Workbench(self.config, self.root / "state", self.adapter, True)
        self.assertEqual(recovered.snapshot()["phase"], "error")
        self.assertFalse(recovered.snapshot()["switches"])
        self.assertTrue(self.adapter.recording)
        recovered.act("status", {}, "recover-123")
        self.assertEqual(recovered.snapshot()["phase"], "recording")

    def test_unconfigured_blocks_hardware(self):
        w = Workbench(
            json.loads(Path("config.example.json").read_text()), self.root / "unconfigured"
        )
        with self.assertRaises(WorkflowError):
            w.act("status", {}, "request-123")
        self.assertFalse(w.snapshot()["configured"])

    def test_invalid_status_blocks_success(self):
        self.adapter.call = lambda *args: {"teleop_running": "yes", "recording": False}
        result = self.act("status")
        self.assertIsNone(result["teleop_running"])
        self.assertIsNotNone(result["error"])

    def test_saved_empty_wrong_id_or_external_file_rejected(self):
        self.prepare()
        self.act("record_start")
        base = Path(self.config["save_dir"])
        base.mkdir()
        empty = base / "empty"
        empty.touch()
        nonempty = base / "good"
        nonempty.write_text("data")
        for ident, path in [
            (self.adapter.ident, empty),
            ("wrong", nonempty),
            (self.adapter.ident, Path(__file__)),
        ]:
            with self.assertRaises(WorkflowError):
                self.w.saved_item({"saved": {"id": ident, "path": str(path.resolve())}})

    def test_command_adapter_executes_fixed_argv_and_passes_context(self):
        import sys

        script = self.root / "adapter.py"
        script.write_text(
            "import os,json; print(json.dumps({'teleop_running':False,'recording':False,"
            "'operation':os.environ['ARX_OPERATION_ID'],'trial':os.environ['ARX_TRIAL'],"
            "'script':os.environ['ARX_TELEOP_SCRIPT'],"
            "'recorder':os.environ['ARX_RECORDER_URL'],'save_dir':os.environ['ARX_SAVE_DIR']}))"
        )
        self.config["commands"]["status"] = [sys.executable, str(script)]
        self.config["teleop_script"] = str(self.root / "external_teleop.py")
        self.config["recorder_url"] = "http://127.0.0.1:18080"
        result = CommandAdapter(self.config).call("status", "test-context-123", True)
        self.assertEqual(result["operation"], "test-context-123")
        self.assertEqual(result["trial"], "1")
        self.assertEqual(result["script"], self.config["teleop_script"])
        self.assertEqual(result["recorder"], self.config["recorder_url"])
        self.assertEqual(result["save_dir"], self.config["save_dir"])

    def test_command_failure_and_timeout_are_not_success(self):
        import sys

        self.config["commands"]["status"] = [
            sys.executable,
            "-c",
            'import sys;sys.stderr.write("adapter failed");sys.exit(2)',
        ]
        with self.assertRaisesRegex(WorkflowError, "adapter failed"):
            CommandAdapter(self.config).call("status", "test-123")
        self.config["commands"]["status"] = [sys.executable, "-c", "import time;time.sleep(2)"]
        self.config["command_timeout_seconds"] = 0.02
        with self.assertRaisesRegex(WorkflowError, "超时"):
            CommandAdapter(self.config).call("status", "test-123")

    def test_teleop_stop_and_restart_preserve_saved_data_and_reset(self):
        self.prepare()
        self.take()
        result = self.act("teleop_stop")
        self.assertEqual(result["phase"], "stopped")
        self.assertFalse(self.adapter.teleop)
        self.assertEqual(len(result["saved"]), 1)
        self.assertEqual(self.act("status")["phase"], "stopped")
        result = self.act("teleop_start")
        self.assertEqual(result["phase"], "reset", result["error"])
        self.assertEqual(len(result["saved"]), 1)

    def test_teleop_stop_blocked_during_recording(self):
        self.prepare()
        self.act("record_start")
        with self.assertRaisesRegex(WorkflowError, "先结束并保存"):
            self.act("teleop_stop")
        self.assertTrue(self.adapter.teleop)
        self.assertTrue(self.adapter.recording)

    def test_rejected_teleop_exit_does_not_report_stopped(self):
        self.prepare()
        original = self.adapter.call
        self.adapter.call = lambda action, *args: (
            {
                "teleop_running": True,
                "recording": False,
                "stop_pending": True,
                "teleop_detail": "请手动回正",
            }
            if action == "teleop_stop"
            else original(action, *args)
        )
        result = self.act("teleop_stop")
        self.assertEqual(result["phase"], "ready")
        self.assertTrue(result["teleop_running"])
        self.assertIn("请手动回正", result["error"])

    def test_saved_take_state_recovers_after_recorder_web_restart(self):
        self.prepare()
        self.take()
        self.w.state.update(phase="error", error="recorder restarting")
        state = self.act("status")
        self.assertEqual(state["phase"], "reset", state["error"])
        self.assertIsNone(state["error"])

    def test_can_url_origin_validation(self):
        self.assertEqual(can_origin("http://127.0.0.1:8000/path"), "http://127.0.0.1:8000")
        for url in (
            "javascript:alert(1)",
            "http://",
            "http://user:pass@localhost",
            "http://localhost;evil",
        ):
            with self.assertRaises(ValueError):
                can_origin(url)

    def test_can_open_reuses_existing_service(self):
        from unittest.mock import MagicMock, patch

        self.config["can_doctor_url"] = "http://127.0.0.1:8000"
        response = MagicMock()
        response.__enter__.return_value.status = 200
        with (
            patch("server.urlopen", return_value=response),
            patch("server.subprocess.Popen") as launch,
        ):
            self.assertEqual(self.w.open_can()["url"], self.config["can_doctor_url"])
            launch.assert_not_called()

    def test_can_open_without_configuration(self):
        with self.assertRaisesRegex(WorkflowError, "尚未配置"):
            self.w.open_can()

    def test_http_blocks_cross_origin_and_serves_ui(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.w))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            with urlopen(url) as response:
                self.assertIn("数采工作台", response.read().decode())
            request = Request(
                url + "/api/action",
                data=b"{}",
                headers={"Origin": "http://evil.invalid", "Content-Type": "application/json"},
            )
            with self.assertRaises(HTTPError) as context:
                urlopen(request)
            self.assertEqual(context.exception.code, 403)
            with urlopen(url + "/api/state") as response:
                self.assertTrue(json.load(response)["demo"])
        finally:
            server.shutdown()
            server.server_close()
            worker.join()


if __name__ == "__main__":
    unittest.main()
