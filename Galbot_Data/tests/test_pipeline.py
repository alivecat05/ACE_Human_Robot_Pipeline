import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap.writer import Writer

from galbot_data.cli import discover_recordings, main
from galbot_data.extract import CAMERAS, STATE_TOPIC


def make_schema():
    file = descriptor_pb2.FileDescriptorProto(name="fixture.proto", package="fixture", syntax="proto3")
    joint = file.message_type.add(name="JointSensor")
    joint.field.add(name="name", number=2, type=9, label=3)
    joint.field.add(name="position", number=3, type=1, label=3)
    state = file.message_type.add(name="SingoriXSensor")
    entry = state.nested_type.add(name="JointSensorMapEntry")
    entry.options.map_entry = True
    entry.field.add(name="key", number=1, type=9, label=1)
    entry.field.add(name="value", number=2, type=11, label=1, type_name=".fixture.JointSensor")
    state.field.add(name="joint_sensor_map", number=2, type=11, label=3,
                    type_name=".fixture.SingoriXSensor.JointSensorMapEntry")
    image = file.message_type.add(name="CompressedImage")
    image.field.add(name="format", number=2, type=9, label=1)
    image.field.add(name="data", number=3, type=12, label=1)
    urdf = file.message_type.add(name="StringValue")
    urdf.field.add(name="value", number=1, type=9, label=1)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    classes = {name: message_factory.GetMessageClass(pool.FindMessageTypeByName("fixture." + name))
               for name in ("SingoriXSensor", "CompressedImage", "StringValue")}
    return descriptor_pb2.FileDescriptorSet(file=[file]).SerializeToString(), classes


def write_fixture(path, *, invalid=False, videos=False, camera_names=None):
    schema_data, classes = make_schema()
    start = 1784027865346210801
    with path.open("wb") as file:
        writer = Writer(file)
        writer.start()
        schemas = {name: writer.register_schema("fixture." + name, "protobuf", schema_data) for name in classes}
        state_channel = writer.register_channel(STATE_TOPIC, "protobuf", schemas["SingoriXSensor"])
        urdf_channel = writer.register_channel("robot_description", "protobuf", schemas["StringValue"])
        writer.add_message(urdf_channel, start, classes["StringValue"](value='<robot name="fixture"/>').SerializeToString(), start)
        camera_channels = []
        if videos:
            import cv2
            import numpy as np

            ok, jpeg = cv2.imencode(".jpg", np.full((32, 32, 3), 127, dtype=np.uint8))
            assert ok
            camera_channels = [writer.register_channel(topic, "protobuf", schemas["CompressedImage"])
                               for topic, name in CAMERAS.items()
                               if camera_names is None or name in camera_names]
            image = classes["CompressedImage"](format="jpeg", data=jpeg.tobytes()).SerializeToString()
        for index in range(2):
            state = classes["SingoriXSensor"]()
            group = state.joint_sensor_map["left_arm"]
            names, values = (["j1", "j2"], [1.0, 2.0]) if index == 0 else (["j2", "j1"], [20.0, 10.0])
            group.name.extend(names)
            group.position.extend(values[:1] if invalid and index == 1 else values)
            timestamp = start + index * 40_000_000
            writer.add_message(state_channel, timestamp, state.SerializeToString(), timestamp)
            for channel in camera_channels:
                writer.add_message(channel, timestamp, image, timestamp)
        writer.finish()
    return start


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def invoke(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main([str(arg) for arg in args])

    def test_extract_videos_reordered_joints(self):
        source = self.root / "record.mcap"
        timestamp = write_fixture(source, videos=True)
        output = self.root / "out"
        self.assertEqual(self.invoke("-i", source, "-o", output), 0)
        episode = output / "record"
        with (episode / "action.csv").open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(rows[0]["timestamp"], str(timestamp))
        self.assertEqual(rows[1]["left_arm/j1"], "10.000000")
        self.assertEqual(rows[1]["left_arm/j2"], "20.000000")
        self.assertTrue((episode / "robot_description.urdf").exists())
        manifest = json.loads((episode / "manifest.json").read_text())
        self.assertEqual(manifest["action_rows"], 2)
        import cv2

        for camera in manifest["cameras"].values():
            self.assertEqual(camera["fps"], 25)
            capture = cv2.VideoCapture(str(episode / f"{camera['name']}.mp4"))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            finally:
                capture.release()
            with (episode / f"{camera['name']}_timestamps.csv").open() as file:
                times = list(csv.DictReader(file))
            self.assertEqual(times[0]["timestamp"], str(timestamp))
        self.assertFalse((output / "all_actions.csv").exists())
        before = (episode / "action.csv").read_bytes()
        self.assertEqual(self.invoke("-i", source, "-o", output), 1)
        self.assertEqual(before, (episode / "action.csv").read_bytes())

    def test_bad_joint_row_does_not_publish_partial_episode(self):
        source = self.root / "bad.mcap"
        write_fixture(source, invalid=True)
        output = self.root / "out"
        self.assertEqual(self.invoke("-i", source, "-o", output), 1)
        self.assertEqual(list(output.iterdir()), [])

    def test_train_only_three_videos_and_actions_with_head_choice(self):
        import cv2

        for head in ("head_left", "head_right"):
            with self.subTest(head=head):
                source = self.root / f"{head}.mcap"
                write_fixture(source, videos=True)
                output = self.root / f"out_{head}"
                self.assertEqual(self.invoke("-i", source, "-o", output, "--train",
                                             "--head", head.removeprefix("head_")), 0)
                episode = output / source.stem
                self.assertEqual({path.name for path in episode.iterdir()},
                                 {f"{head}.mp4", "left_arm.mp4", "right_arm.mp4", "action.csv"})
                for name in (head, "left_arm", "right_arm"):
                    capture = cv2.VideoCapture(str(episode / f"{name}.mp4"))
                    try:
                        self.assertTrue(capture.isOpened())
                        self.assertTrue(capture.read()[0])
                        self.assertTrue(capture.read()[0])
                        self.assertFalse(capture.read()[0])
                    finally:
                        capture.release()
                normal = self.root / f"normal_{head}"
                self.assertEqual(self.invoke("-i", source, "-o", normal), 0)
                self.assertEqual((episode / "action.csv").read_bytes(),
                                 (normal / source.stem / "action.csv").read_bytes())

    def test_train_defaults_to_left(self):
        source = self.root / "record.mcap"
        write_fixture(source, videos=True)
        output = self.root / "out"
        self.assertEqual(self.invoke("-i", source, "-o", output, "--train"), 0)
        self.assertEqual({path.name for path in (output / "record").iterdir()},
                         {"head_left.mp4", "left_arm.mp4", "right_arm.mp4", "action.csv"})

    def test_train_missing_selected_head_does_not_switch_sides(self):
        for side, other in (("left", "right"), ("right", "left")):
            with self.subTest(side=side):
                source = self.root / f"{side}.mcap"
                write_fixture(source, videos=True, camera_names={f"head_{other}", "left_arm", "right_arm"})
                output = self.root / f"out_{side}"
                self.assertEqual(self.invoke("-i", source, "-o", output, "--train", "--head", side), 1)
                self.assertEqual(list(output.iterdir()), [])

    def test_head_rejects_invalid_choice(self):
        with self.assertRaises(SystemExit) as error:
            self.invoke("-i", self.root, "-o", self.root / "out", "--head", "both")
        self.assertEqual(error.exception.code, 2)

    def test_train_requires_three_camera_streams(self):
        for missing in ("head", "left_arm", "right_arm"):
            with self.subTest(missing=missing):
                names = {"head_left", "left_arm", "right_arm"}
                names.remove("head_left" if missing == "head" else missing)
                source = self.root / f"missing_{missing}.mcap"
                write_fixture(source, videos=True, camera_names=names)
                output = self.root / f"out_{missing}"
                self.assertEqual(self.invoke("-i", source, "-o", output, "--train"), 1)
                self.assertEqual(list(output.iterdir()), [])

    def test_directory_automatically_includes_nested_recordings(self):
        inputs = self.root / "inputs"
        nested = inputs / "nested" / "deeper"
        nested.mkdir(parents=True)
        write_fixture(inputs / "a.mcap")
        write_fixture(nested / "b.mcap")
        output = self.root / "out"
        self.assertEqual(self.invoke("-i", inputs, "-o", output), 0)
        self.assertEqual({path.name for path in output.iterdir()}, {"a", "b"})
        for name in ("a", "b"):
            with (output / name / "action.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)

    def test_failed_episode_does_not_remove_successful_episode(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        write_fixture(inputs / "bad.mcap", invalid=True)
        write_fixture(inputs / "good.mcap")
        output = self.root / "out"
        self.assertEqual(self.invoke("-i", inputs, "-o", output), 1)
        self.assertEqual([path.name for path in output.iterdir()], ["good"])
        self.assertTrue((output / "good" / "action.csv").is_file())

    def test_discovery_duplicate_stems_and_empty_directory(self):
        with self.assertRaisesRegex(ValueError, "没有 .mcap"):
            discover_recordings(self.root)
        for name in ("a", "b"):
            directory = self.root / name
            directory.mkdir()
            (directory / "record.mcap").touch()
        with self.assertRaisesRegex(ValueError, "名称冲突"):
            discover_recordings(self.root)


if __name__ == "__main__":
    unittest.main()
