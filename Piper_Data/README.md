# Piper_Data

录制和后处理分开存放，共用一个 `piper-data` 环境：

```text
Piper_Data/
├── recording/               # 录制界面、设备采集、时间戳同步、视频保存
│   ├── record.py
│   ├── run.cmd
│   ├── config.example.json
│   ├── piper_capture/
│   │   └── synchronization.py
│   ├── vendor/              # 原有机械臂和 AGX CAN SDK
│   └── outputs/             # 录制后生成
├── postprocessing/          # 已同步录制数据 → LeRobot
│   ├── convert.py
│   ├── export.cmd
│   └── piper_lerobot/
├── install.py               # 当前环境安装；可选 --drive
├── setup.cmd                # Windows 自动配置；可选 --drive
├── setup.ps1
└── environment.yml
```

## 环境配置：两种方式任选其一

默认安装录制软件与 LeRobot 转换依赖，**加 `--drive` 才安装硬件 Python 驱动/SDK**：
RealSense、python-can、项目内的 AGX CAN 适配层和 pyAgxArm。
录制硬件沿用现有 Windows `agx_cando` 方案；不增加 Linux SocketCAN。
不装硬件依赖也能转换数据、校验录制文件，以及用 `--mock` 运行模拟录制。

### 方式一：Windows 脚本

在 `Piper_Data` 根目录运行，下面两条按需要选一条：

```powershell
.\setup.cmd
# 需要连接真实相机和机械臂时：
.\setup.cmd --drive
```

脚本安装项目内 Conda，并创建同一个 `.conda/envs/piper-data` 环境。
之前不带参数安装过的，也可以再次运行 `setup.cmd --drive` 补装硬件依赖。
已有 `recording/config.json` 不会被覆盖。

安装后运行：

```powershell
.\recording\run.cmd
.\postprocessing\export.cmd -i recording/outputs -o postprocessing/datasets/piper
```

### 方式二：手动 Conda 命令

先打开可使用 `conda` 的终端，进入 `Piper_Data` 根目录，然后执行：

```bash
conda create -n piper-data -c conda-forge python=3.12 pip numpy scipy tk ffmpeg pytest -y
conda activate piper-data
python install.py
```

如果需要硬件依赖，将最后一条替换为下面这条，也可在安装完成后补执行（Windows）：

```bash
python install.py --drive
```

`install.py` 使用当前激活的 Python，不额外创建环境。它会同时安装
`recording` 和 `postprocessing`，其中 LeRobot 固定为 `0.4.2`。
这些命令安装 Python 驱动/SDK，不替代设备所需的系统 USB 驱动安装。

也可使用同一份环境清单代替 `conda create`：

```bash
conda env create -f environment.yml
conda activate piper-data
python install.py
```

手动配置后使用下面的 Python 命令启动，不依赖 `.cmd`：

```bash
python recording/record.py
python postprocessing/convert.py -i recording/outputs -o postprocessing/datasets/piper
```

## 录制与同步

1. 编辑安装时生成的 `recording/config.json`，填三路相机序列号和左右臂 CAN 编号。
2. 启动录制界面，填写任务名、采集序号，点击“开始录制”。
3. 点击“结束保存”，等待保存完成。保存的视频已经按同一时间网格对齐，无需再做后处理同步。

无设备时测试界面：

```bash
python recording/record.py --mock
```

设备编号查询：

```bash
python recording/record.py discover
python recording/record.py discover-arms
```

输出结构：

```text
recording/outputs/任务名/episode_001/
  head.mp4
  left_wrist.mp4
  right_wrist.mp4
  action.csv
  samples.jsonl
  manifest.json
```

三个视频的第 N 帧、`action.csv` 的第 N 条数据、`samples.jsonl` 的第 N 条记录一一对应。
同步在 `recording/piper_capture/synchronization.py` 中完成，只有匹配成功的一整组数据才写盘；
任何一路缺帧或时间差超限，就整组跳过。结束保存会等待末尾匹配和写盘完成，再校验视频帧数。

这是时间戳软件同步：默认每路相机相对采样网格不超过 20 ms，三路跨度不超过 25 ms，
机械臂状态容差为 40 ms，缓冲延迟为 80 ms；均由 `config.json` 的 `sync` 配置。
它不保证三台相机同时曝光。被丢弃的时间段不插帧，三个 MP4 同步省略；
真实时间和丢帧记录保存在 JSONL / manifest 中。

`action.csv` 保留原代码的定义：下一采样时刻的实测双臂 TCP 位姿和夹爪宽度（16 维），
用于示教目标，并非向机械臂发送的命令。当前状态在 `samples.jsonl` 的 `observation.state` 中。
当前主从共线接法仍不提供自定义回位控制。

## 转换 LeRobot

```bash
python postprocessing/convert.py -i recording/outputs -o postprocessing/datasets/piper
```

输入可以是所有任务的根目录、单个任务目录或单段 episode 目录。
转换端读取已同步文件、校验一致性并转换格式，不重新同步视频；
存在丢帧间隔时拆成连续片段，每段至少两帧。不覆盖已有目标目录，也不自动上传。
`action.csv`、三个视频、`samples.jsonl` 和 `manifest.json` 需要一起保留。

只校验、不生成数据集：

```bash
python postprocessing/convert.py -i recording/outputs --validate-only
```

模拟录制默认不进入训练数据；验证模拟数据时显式加 `--allow-mock`。

## 测试

在 `Piper_Data` 根目录、已配置的环境中运行：

```bash
python -m pytest
```

界面测试需要可用的桌面显示。录制端不依赖 LeRobot；转换端运行时不导入相机、CAN 或同步模块。
