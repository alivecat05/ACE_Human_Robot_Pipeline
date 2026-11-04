# ARX X5 数采工具

面向 ARX X5 2023 的三相机数采工具：同步记录 RGB、双臂实测状态和遥操目标，
保存原始 Episode，后台校验并转换为 LeRobot v3 数据集。

Linux、Python **3.12**。核心录制只依赖 NumPy、PyAV 和 Pillow；
网页采集和 LeRobot 转换分别按需安装。录制进程不创建机器人控制器、不打开 CAN。

## 安装与无设备试运行

在本仓库的 `ARX_X5_Data/` 目录中执行以下命令：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
arx-x5 record --config configs/mock.toml --mock --task mock_demo --seconds 2
```

如果系统提示缺少 `ensurepip`，先安装发行版的 Python 3.12 venv 组件，再创建虚拟环境。

模拟记录在 `data/mock/`，不会打开相机或连接机械臂。视频收尾写盘后先进入
`pending_validation`，后台校验通过后才是 `complete`。安装转换依赖后后台可继续导出；
仅安装核心依赖时，转换失败会记录在 `job.json`，原始文件保留。

## 真实采集

```bash
cp configs/x5_2023.example.toml config.local.toml
arx-x5 devices
```

填写三台相机的稳定设备路径，核对彩色流格式、尺寸和帧率。
输出路径相对于配置文件目录；真实相机序列号和现场配置不提交 Git。

在已有遥操程序中接入[单向遥测协议](docs/telemetry.md)，由操作员独立启动遥操。
已有兼容入口时，可在其原工作目录使用：

```bash
ARX_RECORD_SOCKET=/tmp/arx-record.sock python3 test_remote_encoder.py
```

外部遥操程序及其 SDK 需要另行提供，其启动、运动和退出行为由原程序决定。
本仓库不包含 CAN Doctor、SDK 二进制或完整遥操程序。
另开终端录制：

```bash
arx-x5 camera-check --config config.local.toml -o /tmp/x5-camera-check
arx-x5 record --config config.local.toml --task 把杯子放入托盘
```

收到 `RECORDING` 后开始示教，在录制终端按 Ctrl+C 结束并保存。
也可用 `--seconds 30` 定长录制。已有 Episode 不覆盖。

## 校验与 LeRobot 导出

```bash
python -m pip install -e '.[export]'
arx-x5 validate data
arx-x5 export -i data -o datasets/x5_session --repo-id local/x5_session
```

LeRobot 固定为 [lerobot-main.lock.json](lerobot-main.lock.json) 中的 commit，
不随 main 自动更新。默认导出为 H.264、CRF20、GOP8、2 个编码线程；
支持显式选择 `--mode reuse_mp4` 或 `--mode reencode_av1`。
转换在临时目录完成校验后原子发布，不覆盖目标，不自动上传数据。

后台任务持久化在每条数据的 `job.json`。默认最多两个并行任务、四个物理 CPU 核预算，
可用 `ARX_EXPORT_WORKERS=1`、`ARX_EXPORT_CPU_BUDGET=2` 降低占用。
校验和转换不阻塞下一条录制，失败保留原始文件。分离环境时设置
`ARX_WORKER_PYTHON`，必要时设置 `ARX_WORKER_PYTHONPATH`。
默认软件编码；可选 `ARX_H264_BACKEND=vaapi` 需要本机 FFmpeg、Intel VAAPI
和 `/dev/dri/renderD128`，硬件编码失败会回退软件路径。

模拟数据校验和导出必须显式加 `--allow-mock`；先等待后台完整校验通过。
`complete` 表示文件完整，任务成功和画面质量仍需人工判断。
数据维度、时间戳、采样缺口及状态定义见[数据格式](docs/data-format.md)。

## 可选 UI

UI 源码随仓库保留，命令行录制不需要安装 UI 依赖。

```bash
python -m pip install -e '.[ui]'
arx-x5 ui --config config.local.toml
# 无设备演示：arx-x5 ui --config configs/mock.toml --mock
```

打开 `http://127.0.0.1:8080`：三路预览、任务名、开始、结束保存、人工场景复位、
本条无效和后台队列状态。场景复位只确认人工操作，不发送机械臂动作。
关闭浏览器不会结束录制；请先保存再停止服务。

另有基于标准库的引导工作台与数据审核页面：

```bash
python3 server.py --demo
```

访问 `http://127.0.0.1:8765`，数据浏览位于 `/browse`。
该演示只生成示例 JSON。真实接入见[工作台说明](docs/workbench.md)和
[现场接入协议](docs/接入说明.md)。8765 工作台不替代 8080 录制服务。
这些工作台脚本和静态文件通过 Git 或源码包提供；wheel 安装仅提供 `arx-x5` 包和 8080 可选 UI。

## 仓库结构

```text
src/arx_x5_data/     采集、同步、校验、后台队列、转换和可选采集页
configs/             脱敏配置示例与模拟配置
tests/               离线回归与可选 LeRobot 集成测试
docs/                数据协议、遥测接入、工作台说明和历史验证记录
integrations/        针对特定外部遥操版本的可选补丁
server.py + web/     8765 引导工作台与审核页
```

`work/`、`outputs/`、`data/`、`datasets/`、`.runtime/`、本机配置、截图及日志
只保留在本地，不上传 GitHub。第三方 LeRobot 通过锁定依赖安装，不复制源码入仓库。

## 开发验证

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -m 'not integration'
node --check web/app.js
node --check web/browse.js
python -m build
# 安装 export extra 后运行真正的 LeRobot 写入、加载、视频解码：
ARX_TEST_LEROBOT=1 pytest -m integration
```

GitHub Actions 使用 Python 3.12，分别执行离线检查和锁定版本的 LeRobot 集成测试。
历史现场验证见[验证记录](docs/verification.md)；历史性能数字不代表所有设备的保证。
软件时间戳匹配不等于硬件同步曝光，长期负载、设备映射和任务质量需现场验收。
