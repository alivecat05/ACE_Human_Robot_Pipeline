# recording 模块说明

本目录用于**多相机采集、网页控制、预览与数据管理**，覆盖 RealSense、ZED、EgoCap 三类设备。

## 更新说明
1. 新增文件自动加密功能，在录制完后自动加密。同时附带一个解密程序'recording/decrypt_tool.py'.  'DEFAULT_ENCRYPTION_KEY'是需要用户指定的密钥。



## 0. 快速上手（1 分钟）

适合第一次使用，按下面顺序即可完成一次基本录制。

### Step 1：启动网页控制台

```bash
cd /home/ssd/Merge_latest/recording
/home/daxiao/miniconda3/envs/daxiao2/bin/python app.py
```

浏览器打开：`http://127.0.0.1:9066`

### Step 2：确认设备与预览

1. 在“📱 Web 实时预览”中点击“刷新设备”。
2. 选择 RealSense/ZED 相机。
3. 点击“启动预览”，确认 RGB 和深度图正常。

### Step 3：开始录制

1. 在“录制任务名称”输入任务号（例如 `000001`）。
2. 按需点击：
  - “启动 RealSense 录制”
  - “启动 ZED 录制”
  - “启动 EgoCap 录制”
  - 或直接用“启动所有录制 / 组合启动”。

### Step 4：记录任务时间戳（可选但推荐）

在“📍 任务时间戳记录”里：

- 任务开始时点“🟢 记录任务开始”
- 任务结束时点“🔴 记录任务结束”

### Step 5：停止与检查

1. 点击“停止所有任务”。
2. 在“已录制数据管理”里刷新列表。
3. 可点击“📊 统计视频总时长”做快速验收。

### 常见提示

- 多台 RealSense 时，建议使用 `--max_cameras`（例如 2）提高稳定性。
- 若预览正常但录制失败，优先检查 USB 供电/带宽和是否有其他进程占用相机。

---

## 1. 目录与文件作用

### 入口与核心脚本

- `app.py`
  - Gradio 网页端入口。
  - 负责构建控制台 UI、绑定按钮事件、调用服务层执行录制/分析/导出。

- `utils.py`
  - 业务核心服务层（`app.py` 主要依赖这个文件）。
  - 包含：
    - 录制流程编排（`RecordingService`）
    - 相机配置生成（`CameraConfigService`）
    - 任务/关键点时间戳记录（`TaskTimestampRecorder`、`KeypointTimestampRecorder`）
    - 系统运维动作（`SystemService`）
    - 数据统计与传输（`DataService`）
    - Web 预览服务封装（`WebPreviewService`）

### 设备录制脚本

- `realsense_record.py`
  - RealSense 多机录制脚本。
  - 输出：`*_rgb.mp4`、`*_depth.mp4`、`*_ts.csv`，D435i 还会输出 `*_imu.csv`。
  - 支持 `--max_cameras`、曝光模式等参数。

- `zed_record.py`
  - ZED 多机录制脚本。
  - 输出 `.svo2` 及 IMU 时间戳相关文件（具体取决于机型与配置）。

### 分析与预览

- `timestamp_analysis.py`
  - 时间戳分析脚本。
  - 支持分析 RealSense `*_ts.csv` 和 ZED `.svo2` 的帧率、时序信息。


### 配置与依赖

- `template_config.json`
  - 相机配置模板，自动生成 `config.json` 时参考。

- `egocap/`
  - EgoCap 录制侧的二进制与依赖库目录：
    - `egocap`（可执行程序）
    - `libegocap_dep.so`（动态库依赖）
    - `calibration2-camchain-4cam.yaml`（标定配置）
    - `README.md`（EgoCap 子模块说明）

---

## 2. `app.py` 网页端功能说明

`app.py` 通过 `RecordingWebApp` 构建统一的多相机控制台，默认端口 `9066`。

### 2.1 相机配置（可选）

- 检测当前连接的相机。
- 动态生成相机位置下拉框（最多 10 路）。
- 根据任务名/任务类型生成 `config.json`。

### 2.2 Web 实时预览

- RealSense 预览页：
  - 选择相机（带序列号）
  - 启动/停止预览
  - 实时显示 RGB + Depth
  - 支持刷新设备列表

- ZED 预览页：
  - 选择相机（带序列号）
  - 启动/停止预览
  - 实时显示 RGB + Depth
  - 支持刷新设备列表

### 2.3 录制控制

- 单独启动：
  - 仅 RealSense
  - 仅 ZED
  - 仅 EgoCap

- 组合启动：
  - 全部启动
  - RealSense + EgoCap
  - EgoCap + ZED

- 全部停止：
  - 一键停止所有录制进程

### 2.4 时间戳与分析

- 任务时间戳记录：
  - 记录“任务开始/结束”事件
- 关键点时间戳记录：
  - 初始化、记录关键点、重置计数
- 单次时间戳分析：
  - 可选分析 `realsense` / `zed` / `all`
- 批量分析：
  - 对数据目录执行批量时序统计

### 2.5 运维与数据管理

- 系统控制：
  - 重启 ZED 服务
  - 重置 RealSense
  - 系统关机（危险操作）

- 已录制数据管理：
  - 今日录制任务列表
  - 统计视频总时长

- 数据导出：
  - 检测移动硬盘
  - 选择传输范围（today/all）
  - 一键备份并显示日志

---

## 3. 常用启动方式

在 `daxiao2` 环境中启动网页端：

```bash
cd /home/ssd/Merge_latest/recording
/home/daxiao/miniconda3/envs/daxiao2/bin/python app.py
```

启动后浏览器访问（本机）：

- `http://127.0.0.1:9066`

---

## 4. 运行建议（RealSense）

- 如果机器上连接多台 RealSense，建议在录制脚本里明确限制：
  - `--max_cameras 2`（按实际接入数量设置）
- 这样可减少设备枚举异常条目带来的不稳定问题。

示例：

```bash
cd /home/ssd/Merge_latest/recording
/home/daxiao/miniconda3/envs/daxiao2/bin/python realsense_record.py \
  --record_name 1423225027132/000002 \
  --save_path /home/ssd \
  --max_cameras 2
```
