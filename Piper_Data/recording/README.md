# recording：录制与同步

负责录制界面、RealSense 相机、AGX CAN 机械臂反馈、时间戳同步和 MP4/CSV 保存。
环境安装见 [上层 README](../README.md)，与后处理共用 `piper-data` 环境。
真实设备使用前，在根目录安装时加 `--drive`，并填写本目录的 `config.json`。

从 `Piper_Data` 根目录运行：

```bash
python recording/record.py
# 没有设备时：
python recording/record.py --mock
```

也可在本目录运行 `python record.py`。Windows 自动安装后可运行本目录 `run.cmd`。
默认配置固定在本目录，配置内相对输出路径以配置文件所在目录为基准。

同步流程：`cameras.py` / `robot.py` 采样 → `synchronization.py` 匹配 →
`capture.py` 提交整组 → `recording.py` 同时写入三路视频和对应数据行。
结束保存后，视频已按帧对齐，后处理无需再同步。

默认输出 `outputs/任务名/episode_序号/`，包含三个 MP4、`action.csv`、
`samples.jsonl`、`manifest.json`。时间戳容差、丢帧规则和 action 定义见上层说明。
