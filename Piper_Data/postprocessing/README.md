# postprocessing：转换 LeRobot

只处理录制端已经同步的数据，不连接设备，也不重新匹配相机时间戳。
环境安装见 [上层 README](../README.md)，使用同一个 `piper-data` 环境。
只做后处理时无需 `--drive`。

从 `Piper_Data` 根目录运行：

```bash
python postprocessing/convert.py -i recording/outputs -o postprocessing/datasets/piper
```

也可进入本目录运行 `python convert.py -i ../recording/outputs -o datasets/piper`。
Windows 自动安装后可用 `export.cmd`。

默认输入是旁边 `recording/outputs`，默认输出为本目录 `datasets/pipercapture`，
不依赖启动命令时的工作目录。显式 `-i`、`-o` 路径相对于当前工作目录。

输入文件包括 `head.mp4`、`left_wrist.mp4`、`right_wrist.mp4`、`action.csv`、
`samples.jsonl`、`manifest.json`。先校验帧数、数据行、时间戳、字段和配置一致性，
再导出 LeRobot 0.4.2 的 v3 数据集。旧版 JPEG episode 仍可读取。

- `--validate-only`：只校验输入数据，不导入 LeRobot。
- `--allow-mock`：允许模拟数据，仅用于验证。
- `--repo-id`：数据集标识，默认 `local/pipercapture`，不会上传。
- `--image-mode`：使用嵌入图像而非 MP4 保存 LeRobot 图像数据。

录制中的时间缺口会拆成多个连续片段，不重采样；少于两帧的片段不导出。
`schema.py` 只描述录制格式常量，保持与录制端一致，不包含同步算法。
