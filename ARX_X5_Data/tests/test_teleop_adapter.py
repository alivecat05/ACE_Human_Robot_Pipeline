import json
from unittest import TestCase
from unittest.mock import MagicMock, patch

import teleop_adapter as adapter


class TeleopAdapterTests(TestCase):
    def test_missing_script_does_not_launch_teleop(self):
        import tempfile
        from pathlib import Path

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(adapter, "RUNTIME", Path(directory)),
            patch.object(adapter, "SCRIPT", None),
            patch.object(
                adapter, "status", return_value={"recording": False, "teleop_running": False}
            ),
            patch.object(adapter.subprocess, "Popen") as launch,
        ):
            self.assertEqual(adapter.processes(), [])
            with self.assertRaisesRegex(RuntimeError, "teleop_script"):
                adapter.start()
            launch.assert_not_called()

    def response(self, telemetry, mode="warming"):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"mock": False, "mode": mode, "inputs": {"telemetry": telemetry}}
        ).encode()
        return response

    def test_running_requires_process_and_real_feedback(self):
        for pids, feedback, expected in [
            ([], True, False),
            ([42], False, False),
            ([42], True, True),
        ]:
            with (
                patch.object(adapter, "processes", return_value=pids),
                patch.object(adapter, "urlopen", return_value=self.response(feedback)),
            ):
                self.assertEqual(adapter.status()["teleop_running"], expected)

    def test_mock_recorder_is_rejected(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"mock":true}'
        with (
            patch.object(adapter, "processes", return_value=[]),
            patch.object(adapter, "urlopen", return_value=response),
        ):
            with self.assertRaisesRegex(RuntimeError, "真实设备"):
                adapter.status()

    def test_ready_process_is_never_launched_again(self):
        import tempfile
        from pathlib import Path

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(adapter, "RUNTIME", Path(directory)),
            patch.object(
                adapter, "status", return_value={"teleop_running": True, "recording": False}
            ),
            patch.object(adapter.subprocess, "Popen") as launch,
        ):
            self.assertTrue(adapter.start()["teleop_running"])
            launch.assert_not_called()

    def test_existing_unready_process_is_not_duplicated(self):
        import tempfile
        from pathlib import Path

        statuses = [
            {"teleop_running": False, "recording": False},
            {"teleop_running": True, "recording": False},
        ]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(adapter, "RUNTIME", Path(directory)),
            patch.object(adapter, "status", side_effect=statuses),
            patch.object(adapter, "processes", return_value=[42]),
            patch.object(adapter.subprocess, "Popen") as launch,
        ):
            self.assertTrue(adapter.start()["teleop_running"])
            launch.assert_not_called()

    def test_stop_sends_only_normal_exit_signal(self):
        import signal

        with (
            patch.object(
                adapter, "status", return_value={"teleop_running": False, "recording": False}
            ),
            patch.object(adapter, "processes", side_effect=[[42], [], []]),
            patch.object(adapter.os, "kill") as kill,
        ):
            result = adapter.stop()
            kill.assert_called_once_with(42, signal.SIGTERM)
            self.assertFalse(result["stop_pending"])
            self.assertFalse(result["teleop_process_running"])

    def test_stop_refuses_to_interrupt_recording(self):
        with (
            patch.object(adapter, "status", return_value={"recording": True}),
            patch.object(adapter.os, "kill") as kill,
        ):
            with self.assertRaisesRegex(RuntimeError, "先结束并保存"):
                adapter.stop()
            kill.assert_not_called()
