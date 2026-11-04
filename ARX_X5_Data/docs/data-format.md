# 数据合约 v1

只支持 ARX X5 2023，SDK type=0。左右顺序固定为 left、right。
每臂六个关节加一个夹爪值，共 14 维：

~~~text
left.joint_1_rad ... left.joint_6_rad left.gripper_0_to_1
right.joint_1_rad ... right.joint_6_rad right.gripper_0_to_1
~~~

关节单位为 rad。夹爪原生范围 0=闭合、5=张开，保存时除以 5 归一化，
不代表米制物理开度。不使用 2025 款负数范围，不凭空计算 TCP。

0～5 是名义行程与目标范围。实测反馈可有边界偏差：默认容许 −0.1～5.1，
归一化后的 observation.state 夹爪维度对应 −0.02～1.02，保留原始读数，不裁剪。
action 夹爪维度仍严格为 0～1。
余量由清单 config.gripper_feedback_margin_native 明确记录，验证与导出据此检查。
没有该字段的旧记录按原有严格 0～1 状态范围验证；不重写旧文件或旧清单。
向量名称中的 gripper_0_to_1 表示名义归一化刻度，不保证反馈始终位于该闭区间。

observation.state 是遥测包中的实测关节和夹爪值。
action 是同一遥操周期已经向 SDK 提交、且关节/夹爪均获接受的目标。
这不是未来实测状态，也不证明 SDK 平滑后的逐电机设定值或实际执行结果。

## 匹配与时间轴

内部统一使用同一台 Linux 主机的 CLOCK_MONOTONIC。
相机使用 FFmpeg V4L2 原始 PTS，检查其与主机单调时钟是否一致；
缺失、未知或陈旧的 PTS 被拒绝，不降级为 Python 接收时间。
capture_index 是本次 V4L2 流的数据包序号，不冒充内核 buffer sequence。
短帧或损坏包在解码前丢弃，累计数量保存于每帧 camera 元数据中。

旧遥测格式同时包含 at_monotonic_s、at_s 和各臂 feedback_oldest_at_s。
按同一包中的时钟对应关系转换最旧电机接收时间。它是主机 CAN 接收时间，
不是硬件同步时钟；清单中保存该限制。

采样器延迟一段时间后，在网格点附近联合匹配三路图像与遥测。
默认图像距网格不超过 20 ms、三路跨度不超过 25 ms、
遥测距网格不超过 20 ms、反馈相对遥测不超过 100 ms。
不复用相机帧或遥测包；任一路缺失则整组丢弃并记录原因。
这些容差不能代替设备延迟测量。旧包没有精确动作提交时间，
因此不能声称 action 严格晚于图像曝光。

## 文件

- MP4：每个已保存样本一帧，三路按行对应，固定播放帧率。
- samples.jsonl：原始网格序号、帧序号、状态、目标及全部源时间。
- action.csv：Unix 时间及 14 维目标，与 JSONL 对应。
- manifest.json：版本、语义、状态、配置快照、丢帧统计和错误。

Unix 时间采用 Episode 开始时固定的单调时钟映射，避免中途调时改变采样网格。
跨机器、跨开机比较时不要比较单调时间。
原始 MP4 播放会压缩掉丢帧间隔，真实时间必须读取网格和时间戳。
转换端按缺口拆段，并为每个 LeRobot Episode 建立新的均匀时间轴。

pending_validation 表示视频已关闭并同步落盘、等待后台完整校验；
complete 表示完整校验通过；incomplete 表示失败或中断；
cancelled 表示操作员主动标记无效；recording 表示仍在写入。
只有 complete 记录可转换，任务结果和图像质量需另行审核。

后台 job.json 状态为 queued / validating / converting / done / failed。失败不阻塞新录制；重试按钮也可重启已停止的后台进程。关闭录制服务会暂停后台工作，未完成任务在下次启动恢复。
