import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from episode_browser import EpisodeBrowser
from server import ROOT, Workbench, make_handler


class EpisodeBrowserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "data"
        self.path = self.root / "任务/episode_000001"
        self.path.mkdir(parents=True)
        self.manifest = dict(status="complete", mock=False, frames=3, fps=30, dropped_grid_ticks=1)
        self.write_manifest()
        samples = [
            dict(
                grid_index=g,
                **{"observation.state": [0] * 14, "action": [0] * 14},
                timestamp_monotonic_s=g / 30,
                cameras={
                    c: dict(timestamp_monotonic_s=g / 30)
                    for c in ("head", "left_wrist", "right_wrist")
                },
            )
            for g in [0, 1, 3]
        ]
        (self.path / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in samples))
        (self.path / "head.mp4").write_bytes(b"0123456789")
        self.browser = EpisodeBrowser(self.root, Path(self.tmp.name) / "runtime", "/invalid/python")

    def write_manifest(self):
        (self.path / "manifest.json").write_text(json.dumps(self.manifest))

    def test_real_catalogue_qc_and_mock_exclusion(self):
        self.assertEqual(len(self.browser.list()["items"]), 1)
        qc = self.browser.detail("任务/episode_000001")["qc"]
        self.assertEqual(qc["internal_missing"], 1)
        self.assertEqual(qc["samples"], 3)
        self.assertEqual(qc["invalid_vectors"], 0)
        self.manifest["mock"] = True
        self.write_manifest()
        self.assertEqual(self.browser.list()["items"], [])

    def test_review_history_conflict_and_restart(self):
        body = dict(
            id="任务/episode_000001", status="approved", reviewer="审核员", note="成功", revision=0
        )
        first = self.browser.review(body)
        self.assertEqual(first["revision"], 1)
        with self.assertRaisesRegex(ValueError, "其他窗口"):
            self.browser.review(body)
        body.update(revision=1, status="needs_review", note="遮挡")
        second = self.browser.review(body)
        self.assertEqual(len(second["history"]), 2)
        restarted = EpisodeBrowser(self.root, Path(self.tmp.name) / "runtime", "/invalid/python")
        self.assertEqual(restarted.summary(body["id"])["review"]["status"], "needs_review")
        self.assertFalse((self.path / ".workbench-deleted.json").exists())

    def test_review_validation_and_cancelled_approval(self):
        body = dict(id="任务/episode_000001", status="rejected", reviewer="张", note="")
        with self.assertRaisesRegex(ValueError, "原因"):
            self.browser.review(body)
        self.manifest["status"] = "cancelled"
        self.write_manifest()
        body.update(status="approved")
        with self.assertRaisesRegex(ValueError, "不能标记"):
            self.browser.review(body)
        with self.assertRaisesRegex(ValueError, "转换完成"):
            self.browser.export(body["id"])

    def test_path_and_video_symlink_confinement(self):
        for ident in ("../../etc", "/etc/passwd", "任务/../episode_000001", "lerobot/episode_1"):
            with self.assertRaises(ValueError):
                self.browser.episode(ident)
        with self.assertRaises(ValueError):
            self.browser.video("任务/episode_000001", "../manifest")
        (self.path / "head.mp4").unlink()
        (self.path / "head.mp4").symlink_to("/etc/passwd")
        with self.assertRaises(ValueError):
            self.browser.video("任务/episode_000001", "head")

    def test_http_range_seek_and_same_origin_review(self):
        config = json.loads((ROOT / "config.example.json").read_text())
        config["save_dir"] = str(self.root)
        w = Workbench(config, Path(self.tmp.name) / "http-runtime", demo=True)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(w))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join()))
        url = f"http://127.0.0.1:{server.server_port}"
        from urllib.parse import urlencode

        video = (
            url + "/api/episodes/video?" + urlencode(dict(id="任务/episode_000001", camera="head"))
        )
        for header, expected in [
            ("bytes=2-5", b"2345"),
            ("bytes=-3", b"789"),
            ("bytes=7-", b"789"),
        ]:
            with urlopen(Request(video, headers={"Range": header})) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.read(), expected)
        for header in ("bytes=99-", "bytes=1-0", "bytes=-0", "bytes=a-b"):
            with self.assertRaises(HTTPError) as ctx:
                urlopen(Request(video, headers={"Range": header}))
            self.assertEqual(ctx.exception.code, 416)
        req = Request(
            url + "/api/episodes/review",
            data=b"{}",
            headers={"Content-Type": "application/json", "Origin": "http://evil.test"},
        )
        with self.assertRaises(HTTPError) as ctx:
            urlopen(req)
        self.assertEqual(ctx.exception.code, 403)
        with urlopen(url + "/browse") as response:
            self.assertIn("三路", response.read().decode())

    def test_visualization_cache_requires_complete_matching_source(self):
        ident = "任务/episode_000001"
        converted = self.root / "lerobot" / ident / "meta"
        converted.mkdir(parents=True)
        (converted / "info.json").write_text("{}")
        directory = self.browser.viz_dir(ident)
        directory.mkdir(parents=True)
        (directory / "local_episode_episode_0.rrd").write_bytes(b"partial")
        self.assertEqual(self.browser.visualization(ident)["state"], "idle")
        (directory / "ready.json").write_text(
            json.dumps({"source": self.browser.source_signature(ident)})
        )
        self.assertEqual(self.browser.visualization(ident)["state"], "ready")
        (converted / "info.json").write_text('{"changed":true}')
        self.assertEqual(self.browser.visualization(ident)["state"], "idle")
