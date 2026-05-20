# XR-Robotics → xr_teleoperate 当前集成上下文

> 更新时间：2026-05-19

## 当前结论

本仓库已经完成路线切换：

- 仅保留 **XR-Robotics 输入链路**
- 删除了 `televuer / teleimager` 相关模块与主流程依赖
- 当前 `teleop_hand_and_arm.py` 已经是 **xrobotics-only** 真机入口
- 当前 `demo_xrobotics_mujoco.py` 是 **xrobotics-only** 仿真入口

---

## 当前数据流

```text
XR-Robotics SDK
-> XRRoboticsWrapper
-> TeleData
-> Arm IK
-> Arm Controller / Dex1 Controller
-> MuJoCo 或 真机 DDS
```

---

## 已实现内容

### 1. XR 输入适配
文件：
- `teleop/utils/xr_robotics_wrapper.py`
- `teleop/utils/xr_input_types.py`

功能：
- 读取头显 pose / 左右手柄 pose / trigger / grip / buttons / axis
- 做 OpenXR -> robot basis 变换
- 转成项目本地 `TeleData`
- 处理零四元数和无效 pose

### 2. 真机入口清理
文件：
- `teleop/teleop_hand_and_arm.py`

当前已删除：
- `--xr-source`
- `--display-mode`
- `--img-server-ip`
- 图像获取逻辑
- teleimager / televuer 分支

### 3. MuJoCo 入口
文件：
- `teleop/demo_xrobotics_mujoco.py`

当前支持：
- 双臂 IK
- Dex1 可视化
- grip deadman
- 与真机一致的外环关节限速

### 4. 安全逻辑
- grip deadman
- 外环关节限速 `--max-arm-joint-speed`
- 头参考默认使用 `--head-reference-mode calibrated`
  - 启动时标定一次头显平移参考
  - 后续头动默认不驱动机械臂
  - 可按 `c` 重标定

---

## 当前真机行为

```text
启动 -> go_home -> 等待 r -> 标定一次头参考 -> 按 r 后保持 -> grip 才动 -> c 重标定 -> q 时 go_home 退出
```

---

## 当前推荐命令

### 真机
```bash
python teleop/teleop_hand_and_arm.py \
  --input-mode controller \
  --controller-deadman grip \
  --head-reference-mode calibrated \
  --max-arm-joint-speed 1.5 \
  --arm G1_29 \
  --ee dex1 \
  --network-interface eno1
```

### MuJoCo
```bash
python teleop/demo_xrobotics_mujoco.py \
  --frequency 30 \
  --controller-deadman grip \
  --head-reference-mode calibrated \
  --max-arm-joint-speed 1.5
```

---

## 当前遗留项

虽然主代码已经清理完成，但仓库顶层历史 README / CHANGELOG 里仍可能有旧链路描述；如果需要，可以继续做文档层面的彻底清扫。




  # 1）演示“头动跟着动”

  用：

  --head-reference-mode live

  ## MuJoCo

  cd ~/unitree_ws/src/xr_teleoperate
  conda activate tv
  unset PYTHONPATH

  python teleop/demo_xrobotics_mujoco.py \
    --frequency 30 \
    --controller-deadman grip \
    --head-reference-mode live \
    --controller-orientation-mode relative \
    --max-arm-joint-speed 1.2 \
    --home-return-speed 0.5 \
    --ee dex1

  ### 演示方式

  - 按住 grip
  - 然后动头
  - 你会看到手臂目标会跟着变

  ———

  ## 真机

  cd ~/unitree_ws/src/xr_teleoperate
  conda activate tv
  unset PYTHONPATH

  python teleop/teleop_hand_and_arm.py \
    --input-mode controller \
    --arm G1_29 \
    --ee dex1 \
    --network-interface eno1 \
    --controller-deadman grip \
    --head-reference-mode live \
    --controller-orientation-mode relative \
    --max-arm-joint-speed 1.0 \
    --home-return-speed 0.4

  ———

  # 2）演示“头动不跟着动”

  最干净的是：

  --head-reference-mode calibrated
  --calibration-mode manual

  ## MuJoCo

  cd ~/unitree_ws/src/xr_teleoperate
  conda activate tv
  unset PYTHONPATH

  python teleop/demo_xrobotics_mujoco.py \
    --frequency 30 \
    --controller-deadman grip \
    --head-reference-mode calibrated \
    --controller-orientation-mode relative \
    --calibration-mode manual \
    --max-arm-joint-speed 1.2 \
    --home-return-speed 0.5 \
    --ee dex1

  ### 演示方式

  - 启动后按 C 标定
  - 按住 grip
  - 这时动头
  - 手臂不应该因为头动而动

  ———

  ## 真机

  cd ~/unitree_ws/src/xr_teleoperate
  conda activate tv
  unset PYTHONPATH

  python teleop/teleop_hand_and_arm.py \
    --input-mode controller \
    --arm G1_29 \
    --ee dex1 \
    --network-interface eno1 \
    --controller-deadman grip \
    --head-reference-mode calibrated \
    --controller-orientation-mode relative \
    --calibration-mode manual \
    --max-arm-joint-speed 1.0 \
    --home-return-speed 0.4

  ### 真机操作

  - r 开始
  - c 标定
  - 按住 grip 操作
  - 头动不应带手臂动
