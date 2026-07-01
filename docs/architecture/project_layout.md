# 项目目录大纲

原则：目录按业务域组织，但入口不要叠 wrapper。每条流程只保留一个真实 Python 入口。

```text
teleop/          实时遥操域
  real/          真机遥操入口：teleop/real/teleop_hand_and_arm.py
  sim/           仿真遥操入口：teleop/sim/xrobotics_mujoco.py
  runtime/       real/sim 共享状态机预留目录

core/            公共能力：input/control/robot/camera/transforms/diagnostics
data_pipeline/   数据域：recording/replay/export/audit/schemas
inference/       在线推理/VLA：clients/protocols/sessions/transforms
scripts/         少量 shell 一键脚本，不放 Python wrapper
docs/            文档
assets/          模型和静态资源
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
