import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def installer(tmp_path, monkeypatch, platform="win32"):
    path = Path(__file__).resolve().parents[2] / "install.py"
    spec = importlib.util.spec_from_file_location("piper_install", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "sys", SimpleNamespace(
        executable="python", version_info=(3, 12), platform=platform, stderr=sys.stderr))
    (tmp_path / "recording").mkdir()
    (tmp_path / "recording/config.example.json").write_text('{"example": true}')
    commands = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, check: commands.append(command))
    return module, commands


def test_default_install_and_optional_drivers_share_environment(tmp_path, monkeypatch):
    module, commands = installer(tmp_path, monkeypatch)
    assert module.main([]) == 0
    assert len(commands) == 1
    assert str(tmp_path / "recording") in commands[0]
    assert str(tmp_path / "postprocessing") + "[export]" in commands[0]
    assert not any("vendor" in part or "pyrealsense2" in part for part in commands[0])
    config = tmp_path / "recording/config.json"
    config.write_text('{"user_config": true}')
    commands.clear()
    assert module.main(["--drive"]) == 0
    assert len(commands) == 2
    assert all(command[:3] == ["python", "-m", "pip"] for command in commands)
    assert "pyrealsense2==2.59.1.11053" in commands[1]
    assert str(tmp_path / "recording/vendor/pyAgxArm-master") in commands[1]
    assert config.read_text() == '{"user_config": true}'


def test_windows_driver_install_rejected_before_any_changes_on_other_platforms(tmp_path, monkeypatch):
    module, commands = installer(tmp_path, monkeypatch, platform="linux")
    with pytest.raises(SystemExit):
        module.main(["--drive"])
    assert not commands
    assert not (tmp_path / "recording/config.json").exists()
