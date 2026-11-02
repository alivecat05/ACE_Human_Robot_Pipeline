# Galbot_Data

解析 `.mcap`，把每段录制的 `action.csv` 和四路独立相机视频放在同一个文件夹。

## 使用

```bash
cd /root/ACE_H2R_DATA/Galbot_Data
python process.py -i /path/to/input -o /path/to/output
```

`-i` 可以是一个 `.mcap` 文件，也可以是目录。输入目录时自动搜索所有子目录，逐个处理。
默认只需要 `-i`、`-o` 两个参数，不需要子命令。

只保留训练用的三路视频和 action.csv 时加 `--train`：

```bash
python process.py -i /path/to/input -o /path/to/output --train --head left
```

每段录制目录只输出以下四个文件：

```text
录制A/
├── head_left.mp4
├── left_arm.mp4
├── right_arm.mp4
└── action.csv
```

`--head left` 输出 `head_left.mp4`，`--head right` 输出 `head_right.mp4`，省略时默认 `left`。
指定侧没有数据时会报错，不会自动切换到另一侧。`--head` 只影响训练模式，普通模式仍输出四路视频。
训练模式不生成逐帧时间戳、URDF、元数据或 manifest 文件；`action.csv` 内容与普通模式一致。
缺少指定的头部相机或任一臂部相机时，该段录制报错并清理临时结果，保证成功输出是三路视频加一个 CSV。

首次使用缺少依赖时安装（Python 3.10+）：

```bash
python -m pip install -r requirements.txt
```

## 输出

```text
output/
├── 录制A/
│   ├── action.csv
│   ├── head_left.mp4
│   ├── head_right.mp4
│   ├── left_arm.mp4
│   ├── right_arm.mp4
│   ├── head_left_timestamps.csv
│   ├── head_right_timestamps.csv
│   ├── left_arm_timestamps.csv
│   ├── right_arm_timestamps.csv
│   ├── robot_description.urdf
│   ├── metadata/
│   └── manifest.json
└── 录制B/
    └── ...
```

每段录制独立保存，不合并 CSV，不拼接双目视频。
相机、URDF 和相机标定元数据仅在源文件中存在时导出；缺少相机会提示。
同名录制或已有输出会报错，避免覆盖。失败的录制会清理临时文件，其他成功结果保留。
代码独立于原 SDK 目录，原始 MCAP 不会修改。

## 数据含义

- `action.csv` 延续原脚本逻辑，保存 `singorix/wbcs/sensor` 的实测关节位置，不是控制目标。列名为 `组名/关节名`，数值保留六位小数，单位和夹爪范围保持原样。
- `timestamp` 为 MCAP `log_time` 的整数纳秒值。
- 视频帧率自动按各路消息的时间跨度计算；普通模式下真实时间记录在各自的 `*_timestamps.csv` 中，训练模式不输出这些文件。
- 关节和视频保留各自采样频率，没有按行对齐；不能直接将 CSV 第 N 行视为视频第 N 帧。
- 普通模式提取四路 RGB 视频，训练模式提取三路，均不导出深度图。

原来的关节转末端位姿函数保留在 `galbot_data/kinematics.py` 中供代码调用，主流程不执行它。
使用该函数需另外安装 `requirements-eef.txt`。

## 验证

```bash
python -m unittest discover -s tests -v
```

提取逻辑已用两份原始录制验证：1,689 行和 3,664 行关节数据分别与旧 CSV 完全一致；
视频样本四路各 412 帧，重新解码及逐帧时间戳数量检查通过。
