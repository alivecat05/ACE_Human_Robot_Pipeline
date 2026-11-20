# ACE_Human_Robot_Pipeline

人类与机器人数据采集、同步、解析和训练数据转换工具集合。

| 目录 | 功能 | 使用说明 |
| --- | --- | --- |
| `PiperX_WS/` | Piper 双臂示教、VR 遥操、策略执行、人类采集与标注 | [README](PiperX_WS/README.md) |
| `Galbot_Data/` | MCAP 解析，输出视频与 action.csv | [README](Galbot_Data/README.md) |
| `ARX_X5_Data/` | ARX X5 2023 三摄同步录制、后台校验与 LeRobot v3 转换，含可选 UI | [README](ARX_X5_Data/README.md) |

## Galbot

```bash
cd Galbot_Data
python -m pip install -r requirements.txt
python process.py -i /path/to/mcap -o /path/to/output
```

训练模式只保存三路视频和 action.csv，头部相机可选择左右：

```bash
python process.py -i /path/to/mcap -o /path/to/output --train --head left
```

## PiperX 工作空间

`PiperX_WS/` 保留原仓库内部结构，包括 PiperCapture、PiperX_Teleop、PiperX_GO、HumanCapture 和 HumanLabel。
请参阅 [PiperX_WS 使用说明](PiperX_WS/README.md)。

## ARX X5 2023

独立录制进程接收已有遥操的单向遥测，不创建机械臂控制器、不打开 CAN。
核心录制使用 Python 3.12；采集 UI 和 LeRobot 转换按需安装。

```bash
cd ARX_X5_Data
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# 无设备试运行：不打开相机或连接机械臂
arx-x5 record --config configs/mock.toml --mock --task mock_demo --seconds 2
# 可选采集 UI
python -m pip install -e '.[ui]'
arx-x5 ui --config configs/mock.toml --mock
```

真实设备接入、后台转换、数据审核与测试见 [ARX X5 使用说明](ARX_X5_Data/README.md)。

## 本地配置与数据

仓库保存代码、说明、配置示例以及原项目附带的 SDK/驱动文件；
录制数据、生成的数据集、本地环境和设备私有配置不提交。
各模块的依赖和运行平台要求见其 README，第三方 SDK 保留原许可证。
