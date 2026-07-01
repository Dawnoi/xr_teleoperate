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
```

常用环境变量：

```bash
NETWORK_INTERFACE=enx9c69d3212b05 bash scripts/start/start_real_robot_wired.sh
NETWORK_INTERFACE=wlo1 bash scripts/start/start_real_robot_wifi.sh
```

三相机 ZMQ 录制：

```bash
SENDER_IP=192.168.123.164 \
TASK_DIR=./utils/data \
TASK_NAME=multi_cam_record \
bash scripts/start/start_real_robot_wired_3cams_zmq.sh
```

高级调试时可覆盖入口路径，默认仍是唯一真机入口：

```bash
REAL_TELEOP_ENTRY=teleop/real/teleop_hand_and_arm.py bash scripts/start/start_real_robot_wired.sh --help
```
