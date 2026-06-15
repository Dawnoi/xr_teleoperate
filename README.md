# xr_teleoperate

## Current Scope

This repository has been cleaned up to keep only the **XR-Robotics input + Unitree downstream control** path.

Removed from the main flow:
- `televuer`
- `teleimager`
- websocket / WebRTC / browser image streaming
- related launch parameters such as `--xr-source`, `--display-mode`, `--img-server-ip`

Current main path:

```text
XR-Robotics SDK
-> XRRoboticsWrapper
-> TeleData
-> Arm IK
-> Arm Controller / Hand Controller
-> MuJoCo or real robot DDS
```

---

## Main Entry Points

### Real robot
- `teleop/teleop_hand_and_arm.py`

### MuJoCo
- `teleop/demo_xrobotics_mujoco.py`

### XR input adapter
- `teleop/utils/xr_robotics_wrapper.py`
- `teleop/utils/xr_input_types.py`

---

## Environment

Recommended environment:

```bash
conda activate tv
unset PYTHONPATH
```

Required dependencies are primarily:
- `pinocchio`
- `casadi`
- `unitree_sdk2py`
- `mujoco` (for MuJoCo demo)
- `xrobotoolkit_sdk`

---

## Real Robot Launch

```bash
cd ~/unitree_ws/src/xr_teleoperate
conda activate tv
unset PYTHONPATH

python teleop/teleop_hand_and_arm.py \
  --input-mode controller \
  --controller-deadman grip \
  --head-reference-mode calibrated \
  --max-arm-joint-speed 1.5 \
  --arm G1_29 \
  --ee dex1 \
  --network-interface eno1
```

### Current behavior

```text
startup -> go_home -> wait for r -> calibrate head reference once -> hold -> grip to move -> c to recenter -> q to go_home and exit
```

### Deadman logic
- left grip: enable left arm / left gripper
- right grip: enable right arm / right gripper
- release grip: hold current pose

### Head reference logic
- default: `--head-reference-mode calibrated`
- headset translation is sampled once at startup / recenter
- normal head motion does not move the arms
- press `c` to recenter if your standing position changes

---

## MuJoCo Launch

```bash
cd ~/unitree_ws/src/xr_teleoperate
conda activate tv

python teleop/demo_xrobotics_mujoco.py \
  --frequency 30 \
  --controller-deadman grip \
  --head-reference-mode calibrated \
  --max-arm-joint-speed 1.5
```

This demo keeps the same deadman and outer-loop speed limiting semantics as the real robot path.

---

## Safety Notes

Current safety mechanisms include:
- grip deadman
- outer-loop arm target speed limiting via `--max-arm-joint-speed`

Recommended next improvements if needed:
- XR data timeout -> freeze
- pose jump detection
- configurable exit action

---

## Important Files

- `teleop/teleop_hand_and_arm.py`
- `teleop/demo_xrobotics_mujoco.py`
- `teleop/utils/xr_robotics_wrapper.py`
- `teleop/utils/xr_input_types.py`
- `teleop/utils/arm_target_safety.py`
- `teleop/robot_control/robot_arm.py`
- `teleop/robot_control/robot_arm_ik.py`
- `teleop/robot_control/robot_hand_unitree.py`

---

## Notes

Historical documents and changelogs may still mention the old `televuer/teleimager` pipeline. The active code path is now **XR-Robotics only**.
