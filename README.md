# ACE_Human_Robot_Pipeline

人类与机器人数据采集、同步、解析和训练数据转换工具集合。

| 目录 | 功能 | 使用说明 |
| --- | --- | --- |
| `ACE-Datatool/` | 多设备录制、时间同步和后处理 | [README](ACE-Datatool/README.md) |
| `ACE-Datatool_iso/` | 原有相机录制、标定、同步和转换脚本 | [README](ACE-Datatool_iso/README.md) |
| `Galbot_Data/` | MCAP 解析，输出视频与 action.csv | [README](Galbot_Data/README.md) |
| `Piper_Data/` | Piper 录制时同步、保存视频和动作、转换 LeRobot | [README](Piper_Data/README.md) |

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

## Piper

`recording/` 负责录制、同步和保存，`postprocessing/` 负责转换 LeRobot。
两部分使用同一个环境，硬件 Python 驱动/SDK 按需安装。

```bash
cd Piper_Data
conda create -n piper-data -c conda-forge python=3.12 pip numpy scipy tk ffmpeg pytest -y
conda activate piper-data
python install.py
```

需要现有 Windows AGX 硬件驱动/SDK 时使用 `python install.py --drive`。
Windows 也可使用 `setup.cmd` 或 `setup.cmd --drive` 完成配置。

```bash
python recording/record.py
python postprocessing/convert.py -i recording/outputs -o postprocessing/datasets/piper
```

## 本地配置与数据

仓库保存代码、说明、配置示例以及原项目附带的 SDK/驱动文件；
录制数据、生成的数据集、本地环境和设备私有配置不提交。
各模块的依赖和运行平台要求见其 README，第三方 SDK 保留原许可证。

`ACE-Datatool_iso/push_feishu.py` 从 `FEISHU_URL` 环境变量读取 webhook，
使用时在本地配置，不要将真实 webhook 或其他凭据写入代码。
