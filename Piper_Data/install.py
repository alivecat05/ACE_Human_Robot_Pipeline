"""Install both tools into the active environment; --drive adds hardware SDKs."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="在当前 Python/Conda 环境安装录制和 LeRobot 转换软件")
    parser.add_argument("--drive", action="store_true", help="额外安装 RealSense、CAN 和机械臂 SDK（现有 Windows AGX 方案）")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 12):
        parser.error("请先创建并激活 Python 3.12 Conda 环境")
    if args.drive and sys.platform != "win32":
        parser.error("--drive 使用现有 Windows agx_cando 驱动；此平台请不加 --drive 安装软件")
    commands = [[sys.executable, "-m", "pip", "install", "-e", str(ROOT / "recording"),
                 "-e", str(ROOT / "postprocessing") + "[export]"]]
    if args.drive:
        commands.append([sys.executable, "-m", "pip", "install", "python-can>=4.5,<5",
                         "pyrealsense2==2.59.1.11053",
                         str(ROOT / "recording/vendor/python-can-agx-cando-master"),
                         str(ROOT / "recording/vendor/pyAgxArm-master")])
    try:
        for command in commands:
            subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        print(f"安装失败，退出码 {error.returncode}；修复后可重新运行。", file=sys.stderr)
        return error.returncode
    config = ROOT / "recording/config.json"
    if not config.exists():
        shutil.copyfile(ROOT / "recording/config.example.json", config)
    print("录制和后处理已安装到同一个环境。" + ("硬件 SDK 已安装。" if args.drive else "未安装硬件 SDK；可后续加 --drive 补装。"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
