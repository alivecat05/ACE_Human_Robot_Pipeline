from pathlib import Path

import pytest

from arx_x5_data.config import load_config


def test_local_output_is_relative_to_config_not_cwd(tmp_path):
    source = Path(__file__).parents[1] / "configs/mock.toml"
    path = tmp_path / "machine.toml"
    path.write_text(source.read_text().replace("../data/mock", "records"))
    assert load_config(path).output == tmp_path / "records"


@pytest.mark.parametrize(
    "before,after",
    [
        ("fps = 30", "fps = 0"),
        ("delay_ms = 100", "delay_ms = 1"),
        ("writer_queue_size = 8", "writer_queue_size = 0"),
        ("width = 320", "width = 321"),
        ("crf = 18", "crf = 99"),
        ("gripper_feedback_margin_native = 0.1", "gripper_feedback_margin_native = -0.1"),
        ("gripper_feedback_margin_native = 0.1", "gripper_feedback_margin_native = 0.51"),
        ("gripper_feedback_margin_native = 0.1", "gripper_feedback_margin_native = nan"),
    ],
)
def test_bad_recording_configuration_is_rejected(tmp_path, before, after):
    source = Path(__file__).parents[1] / "configs/mock.toml"
    path = tmp_path / "bad.toml"
    path.write_text(source.read_text().replace(before, after, 1))
    with pytest.raises(ValueError):
        load_config(path)
