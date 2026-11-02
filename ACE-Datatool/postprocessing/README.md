# postprocessing 模块说明

本目录用于录制数据的后处理，核心流程是：

1. 解析并整理 EgoCap 原始数据（含颜色同步准备）。
2. 多设备时间戳对齐与视频导出（RealSense / EgoCap / ZED）。

---

## 0. 网页端使用教程（建议先看）

`postprocessing` 自带 Gradio 网页端，适合批量任务和实时查看日志。

### Step 1：启动网页端

```bash
cd /home/ssd/Merge_latest/postprocessing
/home/daxiao/miniconda3/envs/daxiao2/bin/python PostprocessWeb.py
```

浏览器访问：`http://127.0.0.1:7864`

### Step 2：输入根目录并扫描

在页面“第一步：指定根目录”输入包含多个任务子目录的路径，例如：
```text
ACEDATA
   └─ 20260330
      000001
      000002
   └─ 20260401
      000001
      000002
请输入/ACEDATA
```

点击“🔍 扫描目录”，系统会列出所有子文件夹供多选。

### Step 3：选择要处理的任务文件夹

在“第二步：选择数据文件夹”中勾选一个或多个任务目录（支持批量）。

### Step 4：选择执行模式

在“第三步：选择执行步骤”中选择：

- `✅ 全部执行 (步骤 1 + 2)`
- `1️⃣ 仅执行步骤 1 (解析数据)`
- `2️⃣ 仅执行步骤 2 (同步与切割)`

建议首次跑用“全部执行”，某一步失败后可单独重跑该步骤。

### Step 5：开始处理并看实时日志

点击“🚀 开始处理”后，右侧状态区和下方日志会实时更新。

- “🛑 强制停止”可取消当前批处理。
- “🔄 清空重置”可重置输入和页面状态。

### Step 6：查看输出结果

默认输出在 `processed/` 目录（由 `TimestampSyncConfig.output_dir` 默认值决定），典型结构类似：

- `processed/<date>/<sn>/<seq>/...`

其中会包含同步后的视频、时间戳与配置文件。

---

## 1. 文件作用总览

- `PostprocessWeb.py`
  - 后处理总入口（网页端 + 终端模式双入口）。
  - 负责目录扫描、任务多选、分步执行、实时日志与中断控制。

- `EgocapPreprocessing.py`
  - 第一步处理逻辑（EgoCap 预处理）。
  - 主要包含：
    - 读取并清洗 `imu.csv`
    - 解析 `mjpeg.bin` 并写出 `rgb` 帧
    - 按时间戳对齐 IMU 与图像
    - 调用颜色同步（与 `ColorSync.py` 协作）

- `TimestampSync.py`
  - 第二步处理逻辑（跨设备时间同步 + 导出）。
  - 主要包含：
    - 收集各设备时间戳
    - 计算统一 anchor 时间轴
    - 生成同步索引并调度导出
    - 处理已有输出目录、配置文件与任务时间戳文件

- `VideoSyncer.py`
  - 具体视频/IMU 导出实现（被 `TimestampSync.py` 调用）。
  - 包含 RealSense / EgoCap / ZED 的处理器，负责按同步索引写视频与 CSV。

- `ColorSync.py`
  - 颜色事件分析与相机延迟估计。
  - 生成 `sync_color_results/all_camera_latencies.json` 等中间结果，为时序对齐提供偏移量。

- `utils.py`
  - 通用工具函数。
  - 包含编码器选择、进度条、数据清理（如删除冗余目录、删除 `rgb` 缓存目录）等辅助逻辑。

---

## 2. 处理流程（概念图）

```text
原始录制目录
   └─ (步骤1) EgocapPreprocessor
        ├─ 解包/对齐 EgoCap 数据
        └─ 颜色同步准备
   └─ (步骤2) TimestampSyncProcessor
        ├─ 收集多设备时间戳
        ├─ 计算统一时间轴
        └─ 调用 VideoSyncer 导出同步结果
                ↓
            processed/... 输出
```

---

## 3. 常见使用方式

### 3.1 网页端（推荐）

```bash
cd /home/ssd/Merge_latest/postprocessing
/home/daxiao/miniconda3/envs/daxiao2/bin/python PostprocessWeb.py
```

### 3.2 终端模式（无 Gradio 时自动回退）

如果 Gradio 不可用，`PostprocessWeb.py` 会自动进入交互式终端模式：

- 输入根目录
- 选择任务序号
- 选择步骤模式
- 实时打印处理日志

---

## 4. 说明与建议

- 建议先在网页端跑小批量（1~2 个目录）验证流程，再跑全量。
- 若仅某一步失败，使用“仅步骤1/仅步骤2”重跑可节省时间。
- 处理结束后如磁盘紧张，可检查并清理中间缓存目录。
