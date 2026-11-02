#  ACE Robotics Data Tools 总览

本项目主要包含两个核心模块：

-  `recording`：负责多设备数据采集与录制控制。
-  `postprocessing`：负责录制后的数据整理、同步与导出。

---

## 1️⃣  recording（采集与录制）

`recording` 模块用于统一控制 RealSense、ZED、EgoCap 的采集流程，核心目标是 🎯 稳定完成从设备检查到数据落盘的全链路录制。

###  功能概览（宏观）

-  多设备统一控制（单设备/组合录制）
-  采集前实时预览（RGB/Depth）
-  任务级时间戳与关键点事件记录
-  录制后数据管理、统计与导出

> 📍 典型入口：`recording/app.py`

---

## 2️⃣  postprocessing（后处理与同步）

`postprocessing` 模块用于将原始录制数据转换为可用的同步结果，重点在 ⏱️ 跨设备时间对齐与 📦 标准化导出。

###  功能概览（宏观）

-  EgoCap 数据解析与预处理
-  多设备时间戳同步（RealSense / EgoCap / ZED）
-  对齐后视频与元数据导出
-  网页端批处理与日志查看

> 📍 典型入口：`postprocessing/PostprocessWeb.py`

---

## 3️⃣ ⚙️ 环境配置（统一）

>  推荐使用 Conda 环境

-  环境名：`daixiao`
-  Python 版本：`3.8`

### 3.1  创建与激活环境

```bash
conda create -n daixiao python=3.8 -y
conda activate daixiao
```

### 3.2  安装依赖

```bash
pip install -r v1_requirements.txt
```

---

## 4️⃣  快速启动

###  启动 recording

```bash
cd /recording
python app.py
```

>  默认访问：`http://127.0.0.1:9066`

###  启动 postprocessing

```bash
cd /postprocessing
python PostprocessWeb.py
```

>  默认访问：`http://127.0.0.1:7864`

---

## 5️⃣  建议使用顺序

1.  在 `recording` 完成数据采集与任务打点。
2.  在 `postprocessing` 完成批量后处理与时序对齐。
3. 在 `processed` 目录验收输出结果并进入下游分析流程。

>  遵循上述流程可确保数据完整性与处理效率。

---
