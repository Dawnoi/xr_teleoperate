# scripts

这里只保留少量 shell 一键脚本，不放 Python wrapper。脚本只负责选择环境、网络口和常用参数，真实业务逻辑仍在对应 Python 业务域。

Python 实现位置：

- 真机遥操入口：`teleop/real/teleop_hand_and_arm.py`
  - 启动装配：`teleop/real/setup.py`
  - 每帧控制：`teleop/control_flow/`
  - 录制流程：`data_pipeline/recording/teleop_recording_flow.py`
- 仿真遥操：`teleop/sim/`
- 回放/数据工具：`data_pipeline/`
- 诊断：`tests/diagnostics`

当前 shell 分组：

- `start/`：真机一键启动
- `debug/`：DDS 调试启动
- `replay/`：回放快捷启动
- `data/`：数据导出快捷启动

## 真机启动脚本

```bash
bash scripts/start/start_real_robot_wired.sh
bash scripts/start/start_real_robot_wifi.sh
bash scripts/start/start_real_robot_wired_3cams_zmq.sh
bash scripts/start/start_real_robot_vla.sh
```

常用环境变量：

```bash
NETWORK_INTERFACE=enx9c69d3212b05 bash scripts/start/start_real_robot_wired.sh
NETWORK_INTERFACE=wlo1 bash scripts/start/start_real_robot_wifi.sh
```

Dex1 夹爪键盘开合 / `tau_est` 反馈探测：

```bash
bash scripts/debug/dex1_gripper_keyboard_probe.sh
```

进入后用 `a` / 左键闭合，`d` / 右键张开，空格把目标位置设为当前实际位置。

默认读取 `scripts/start/start_real_robot_vla.sh` 里的 `NETWORK_INTERFACE` 默认值，保持和 VLA 推理脚本一致。需要临时覆盖时仍可使用 `NETWORK_INTERFACE=...`。

运行前先停掉真机遥操作或 VLA 进程，避免多个进程同时发布 `rt/dex1/*/cmd`。

三相机 ZMQ 录制：

```bash
SENDER_IP=192.168.123.164 \
TASK_DIR=./utils/data \
TASK_NAME=multi_cam_record \
bash scripts/start/start_real_robot_wired_3cams_zmq.sh
```

VLA / online inference 真机推理：

```bash
# 默认允许真机运动：启动前确认 transform config / 工作空间 / 相机链路都正确。
VLA_BASE_URL=http://127.0.0.1:18027 \
SENDER_IP=192.168.123.164 \
bash scripts/start/start_real_robot_vla.sh

# 临时 dry-run：只发观测、解析动作，不下发机械臂运动。
VLA_ENABLE_MOTION=0 \
VLA_BASE_URL=http://127.0.0.1:18027 \
VLA_PROMPT="your task prompt" \
bash scripts/start/start_real_robot_vla.sh
```

常用 VLA 环境变量：

```bash
NETWORK_INTERFACE=eno1
SENDER_IP=192.168.123.164
VLA_BASE_URL=http://127.0.0.1:18027
VLA_PROTOCOL_PROFILE=pi05_dual_arm_20d
VLA_ARM_SIDE=both                    # both / left / right
VLA_ENABLE_MOTION=1                  # 0=dry-run, 1=real motion
VLA_AUTO_START=0                     # 0=按 r 启动, 1=启动后立即开始
VLA_TRANSFORM_CONFIG=configs/inference/unitree_dual_arm_identity_transform.json
MAX_ARM_JOINT_SPEED=1.0
```

## 数据导出脚本

Raw episode -> LeRobot v2（全量导出，不排除问题 episode）：

```bash
RAW_ROOT=utils/data/multi_cam_record \
OUTPUT_ROOT=utils/data/multi_cam_record_lerobot \
TASK="your task prompt" \
bash scripts/data/export_raw_to_lerobot.sh
```

Raw episode -> LeRobot v2：

```bash
RAW_ROOT=utils/data/multi_cam_record \
PROBLEM_FILE=utils/data/multi_cam_record/problem_episodes.txt \
OUTPUT_ROOT=utils/data/multi_cam_record_lerobot_clean \
TASK="your task prompt" \
bash scripts/data/export_clean_multi_cam_lerobot.sh
```

LeRobot v2 -> UMI/DP：

```bash
LEROBOT_ROOT=utils/data/multi_cam_record_lerobot \
OUTPUT_ROOT=utils/data/multi_cam_record_umi_dp \
bash scripts/data/export_lerobot_to_umi_dp.sh
```

只导出部分 episode：

```bash
EPISODES="0 1 2" bash scripts/data/export_lerobot_to_umi_dp.sh
# 或
EPISODES="0,1,2" bash scripts/data/export_lerobot_to_umi_dp.sh
```

高级调试时可覆盖入口路径，默认仍是唯一真机入口：

```bash
REAL_TELEOP_ENTRY=teleop/real/teleop_hand_and_arm.py bash scripts/start/start_real_robot_wired.sh --help
```
