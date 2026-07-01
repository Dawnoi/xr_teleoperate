# xr_teleoperate

## 当前范围

这个仓库目前已经清理为只保留：

- **XR-Robotics 输入**
- **Unitree 下游控制**

已经从主流程移除：
- `televuer`
- `teleimager`
- websocket / WebRTC / 浏览器图像链路
- `--xr-source` / `--display-mode` / `--img-server-ip` 等旧参数

当前主链路：

```text
XR-Robotics SDK
-> XRRoboticsWrapper
-> TeleData
-> Arm IK
-> Arm Controller / Hand Controller
-> MuJoCo 或 真机 DDS
```

---

## 主要入口

### 真机入口
- `teleop/real/teleop_hand_and_arm.py`

### MuJoCo 入口
- `teleop/sim/xrobotics_mujoco.py`

### XR 输入适配
- `teleop/input/xr_robotics_wrapper.py`
- `teleop/input/xr_input_types.py`

---

## 环境

推荐：

```bash
conda activate tv
unset PYTHONPATH
```

主要依赖：
- `pinocchio`
- `casadi`
- `unitree_sdk2py`
- `mujoco`（MuJoCo 演示）
- `xrobotoolkit_sdk`

---

## 真机启动命令

```bash
cd ~/unitree_ws/src/xr_teleoperate
conda activate tv
unset PYTHONPATH

python teleop/real/teleop_hand_and_arm.py \
  --input-mode controller \
  --controller-deadman grip \
  --head-reference-mode calibrated \
  --max-arm-joint-speed 1.5 \
  --arm G1_29 \
  --ee dex1 \
  --network-interface eno1
```

### 当前行为

```text
启动 -> go_home -> 等待 r -> 标定一次头参考 -> 保持 -> grip 才动 -> c 重标定 -> q 时 go_home 并退出
```

### deadman 逻辑
- 左 grip：使能左臂 / 左夹爪
- 右 grip：使能右臂 / 右夹爪
- 松开 grip：保持当前姿态

### 头参考逻辑
- 默认：`--head-reference-mode calibrated`
- 启动/重标定时记录一次头显平移参考
- 正常头动不会驱动机械臂
- 如果站位变化了，可以按 `c` 重标定

---

## MuJoCo 启动命令

```bash
cd ~/unitree_ws/src/xr_teleoperate
conda activate tv

python teleop/sim/xrobotics_mujoco.py \
  --frequency 30 \
  --controller-deadman grip \
  --head-reference-mode calibrated \
  --max-arm-joint-speed 1.5
```

这个仿真入口与真机保持一致的：
- grip deadman 语义
- 外环关节限速语义

---

## 当前安全机制

目前已做：
- grip deadman
- 外环关节目标限速 `--max-arm-joint-speed`

后续建议可继续补：
- XR 数据超时 freeze
- pose jump 检测
- 退出动作可配置

---

## 关键文件

- `teleop/real/teleop_hand_and_arm.py`
- `teleop/sim/xrobotics_mujoco.py`
- `teleop/input/xr_robotics_wrapper.py`
- `teleop/input/xr_input_types.py`
- `teleop/control_utils/arm_target_safety.py`
- `teleop/robot_control/robot_arm.py`
- `teleop/robot_control/robot_arm_ik.py`
- `teleop/robot_control/robot_hand_unitree.py`

---

## 说明

仓库里的历史 README / CHANGELOG 可能还保留旧 `televuer/teleimager` 描述；当前实际代码路径已经是 **XR-Robotics only**。
