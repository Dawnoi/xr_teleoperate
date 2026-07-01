# 项目目录大纲

原则：目录按业务域组织，但入口不要叠 wrapper。每条流程只保留一个真实 Python 入口。

```text
teleop/                         实时遥操域
  real/
    teleop_hand_and_arm.py      真机遥操唯一入口；只编排启动、主循环、清理
    setup.py                    真机启动装配：DDS/base/arm/EE/camera/record/latency
    args.py                     真机入口 CLI 参数
  sim/                          仿真遥操入口与仿真适配
  control_flow/                 每帧控制流程
    base_command.py             底盘命令流水线
    arm_command_pipeline.py     机械臂目标、IK、限速、重力补偿
    operator_state.py           home/deadman/takeover 状态机
    end_effector_command.py     夹爪/灵巧手命令映射
  runtime/                      操作员按键和运行时命令处理
  debug/                        遥操调试辅助：trace/timing/UI

core/                           可复用实现逻辑：input/control/camera
inference/                      在线推理/VLA 核心逻辑：transport/protocol/session/pose_transform
data_pipeline/                  数据域：recording/replay/export
  recording/
    teleop_recording_flow.py    遥操作业录制：episode 生命周期、对齐、写入
    alignment.py                state/action/camera 时间戳对齐工具
  replay/                       episode 回放
  export/                       数据格式导出
tests/                          调试、数据检查、探针、诊断脚本
scripts/                        少量 shell 一键脚本，不放 Python wrapper
docs/                           文档
assets/                         模型和静态资源
```

推荐入口：

```bash
python teleop/real/teleop_hand_and_arm.py          # 真机遥操
python teleop/sim/xrobotics_mujoco.py              # MuJoCo 遥操
python data_pipeline/replay/raw_episode_real.py    # raw episode 真机回放
python data_pipeline/replay/raw_episode_mujoco.py  # raw episode MuJoCo 回放
```

保留的 shell 快捷入口：

```bash
bash scripts/start/start_real_robot_wired.sh
bash scripts/start/start_real_robot_wifi.sh
bash scripts/start/start_real_robot_wired_3cams_zmq.sh
bash scripts/debug/debug_dds_wired.sh
bash scripts/debug/debug_dds_wifi.sh
bash scripts/replay/replay_raw_episode_real.sh
bash scripts/data/export_clean_multi_cam_lerobot.sh
```
