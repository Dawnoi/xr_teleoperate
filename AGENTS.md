# AGENTS.md

## 当前项目状态

这个仓库当前已经收敛为 **XR-Robotics 输入 + xr_teleoperate 下游控制** 的路线。

已经移除：
- `televuer`
- `teleimager`
- websocket / WebRTC / 图像链路
- `--xr-source` / `--display-mode` / `--img-server-ip` 相关逻辑

当前只保留：
- XR-Robotics SDK 获取头显/手柄数据
- 本地 `TeleData` 语义适配
- Unitree 机械臂 IK
- Unitree 机械臂控制
- Dex1 / Dex3 / Inspire / Brainco 手部控制
- MuJoCo 验证链路
- 真机控制链路

---

## 当前主链路

```text
XR-Robotics SDK
-> XRRoboticsWrapper
-> TeleData
-> Arm IK
-> Arm Controller / Hand Controller
-> MuJoCo 或 真机 DDS
```

---

## 当前关键文件

### 真机入口
- `teleop/teleop_hand_and_arm.py`

### MuJoCo 入口
- `teleop/demo_xrobotics_mujoco.py`

### XR 输入适配
- `teleop/utils/xr_robotics_wrapper.py`
- `teleop/utils/xr_input_types.py`

### 安全相关
- `teleop/utils/arm_target_safety.py`

### 控制器
- `teleop/robot_control/robot_arm.py`
- `teleop/robot_control/robot_arm_ik.py`
- `teleop/robot_control/robot_hand_unitree.py`

---

## 当前真机控制逻辑

状态机：

```text
启动
-> go_home
-> 等待 r
-> 标定一次头参考（默认 calibrated）
-> 按 r 后 hold 当前姿态
-> 按住 grip 才允许运动
-> 按 c 可重标定头参考
-> 松开 grip 则 hold
-> 按 q 时 go_home 后退出
```

### grip deadman
- 左 grip：左臂/左夹爪使能
- 右 grip：右臂/右夹爪使能
- 两边都松开：不跟随，保持当前姿态

### 关节目标限速
外环限速参数：
- `--max-arm-joint-speed`

默认：
- `1.5 rad/s`

---

## 当前最小真机命令

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

---

## 当前最小 MuJoCo 命令

```bash
cd ~/unitree_ws/src/xr_teleoperate
conda activate tv
python teleop/demo_xrobotics_mujoco.py \
  --frequency 30 \
  --controller-deadman grip \
  --head-reference-mode calibrated \
  --max-arm-joint-speed 1.5
```

---

## 接手时注意

1. 不要再恢复 `televuer/teleimager` 相关依赖。
2. `XRRoboticsWrapper` 当前是 controller 模式优先，hand-tracking 不是主路径。
3. MuJoCo 和真机的 deadman / 限速逻辑要保持一致。
4. 若继续做安全增强，优先补：
   - XR 数据超时 freeze
   - pose jump 检测
   - 退出动作可配置
