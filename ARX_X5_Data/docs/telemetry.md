# 遥操数据接入

单向 AF_UNIX / SOCK_DGRAM，默认 /tmp/arx-record.sock，可在本地配置修改。
原有 test_remote_encoder.py 的 Telemetry.publish 使用兼容格式。
夹爪目标仍需在 0～5；实测反馈默认允许 −0.1～5.1 的边界偏差，由录制配置的
gripper_feedback_margin_native 控制。反馈只除以 5，不裁剪或替换为目标值。
录制器不删除已有 socket；异常退出留下路径时，先确认原进程已退出再处理。

~~~json
{
  "at_monotonic_s": 12345.02,
  "at_s": 1790000000.02,
  "arms": {
    "left": {
      "joint_positions_rad": [0, 0, 0, 0, 0, 0],
      "gripper_native": 2.5,
      "feedback_oldest_at_s": 1790000000.01,
      "submitted_command": {
        "joint_positions_rad": [0, 0, 0, 0, 0, 0],
        "gripper_native": 2.5,
        "joint_accepted": true,
        "gripper_accepted": true
      }
    },
    "right": {
      "joint_positions_rad": [0, 0, 0, 0, 0, 0],
      "gripper_native": 2.5,
      "feedback_oldest_at_s": 1790000000.01,
      "submitted_command": {
        "joint_positions_rad": [0, 0, 0, 0, 0, 0],
        "gripper_native": 2.5,
        "joint_accepted": true,
        "gripper_accepted": true
      }
    }
  }
}
~~~

时间来自同一台主机。feedback_oldest_at_s 是本次状态中七个电机反馈最旧的
接收时间。SDK 拒绝目标、主臂离线、从臂故障时不得把旧命令重发为新样本。
原有发布器已经检查相关状态；新的发布端也需保留该约束。

自己的遥操程序还可使用这个独立传输类：

~~~python
from pathlib import Path
from arx_x5_data.publisher import TelemetryPublisher

publisher = TelemetryPublisher(Path("/tmp/arx-record.sock"), hz=50)
try:
    # 在已有控制循环内，完成 SDK 提交、确认返回值并读取对应反馈后，
    # 用上述格式组装 arms。此类自身不会读取或创建机器人控制器。
    publisher.publish(arms)
finally:
    publisher.close()
~~~

发送限频且非阻塞；录制端不存在或队列满时返回 False，不重试、不更改控制模式。
SDK 快照和目标接受结果必须由控制程序准确提供。
不要在录制端另建 SingleArm 读取状态：现场历史表明构造本身可能启动后台发送，
不能视为无副作用的只读接口。
