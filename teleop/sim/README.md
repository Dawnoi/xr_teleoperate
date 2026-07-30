# teleop.sim

仿真遥操域。

主要 Python 入口：

```bash
python teleop/sim/xrobotics_mujoco.py
```

用 VIVE Tracker 做真机前预验证（先启动 `vive_locator`、`scripts/vive_keyboard_enable.py`，并准备 AGX 三点标定文件）：

```bash
python teleop/sim/xrobotics_mujoco.py \
  --input-provider vive \
  --controller-deadman grip \
  --controller-orientation-mode relative \
  --vive-calibration-file ~/.config/xr_teleoperate/vive_calibration.json
```

MuJoCo 只复用 VIVE ROS 输入、标定、键盘使能和腕部 IK，不连接 Unitree DDS；按 `R` 开始同步。

其他仿真相关实现：

```text
teleop/sim/real_robot_shadow_mujoco.py
teleop/sim/g1d_mujoco_builder.py
teleop/sim/sim_state_topic.py
```
