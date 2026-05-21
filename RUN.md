# RUN.md

## 1. 真机：仅手臂 + Dex1，不控底盘

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/teleop_hand_and_arm.py \
  --input-mode controller \
  --arm G1_29 \
  --ee dex1 \
  --network-interface eno1 \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative
```

---

## 2. 真机：手臂 + Dex1 + G1D 底盘

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/teleop_hand_and_arm.py \
  --input-mode controller \
  --arm G1_29 \
  --ee dex1 \
  --network-interface eno1 \
  --base-controller g1d_agv \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --base-max-vx 0.15 \
  --base-max-wz 0.40 \
  --base-max-z 1.0 \
  --base-stick-deadzone 0.12
```

### 当前底盘控制语义

- 左摇杆上下：前进/后退
- 左摇杆左右：原地转
- 右摇杆上下：升降柱

---

## 3. 真机：手臂 + 底盘，转向更快

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/teleop_hand_and_arm.py \
  --input-mode controller \
  --arm G1_29 \
  --ee dex1 \
  --network-interface eno1 \
  --base-controller g1d_agv \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --base-max-vx 0.15 \
  --base-max-wz 0.60 \
  --base-max-z 1.0 \
  --base-stick-deadzone 0.12
```

---

## 4. MuJoCo：头动带手动

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/demo_xrobotics_mujoco.py \
  --frequency 30 \
  --viewer-robot g1_d_mobile \
  --controller-deadman grip \
  --head-reference-mode head_coupled \
  --controller-orientation-mode absolute \
  --base-max-vx 0.15 \
  --base-max-wz 0.40 \
  --base-max-z 1.0 \
  --max-arm-joint-speed 1.2 \
  --home-return-speed 0.5 \
  --ee dex1
```

---

## 5. MuJoCo：头动不直接带手动

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/demo_xrobotics_mujoco.py \
  --frequency 30 \
  --viewer-robot g1_d_mobile \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --base-max-vx 0.15 \
  --base-max-wz 0.40 \
  --base-max-z 1.0 \
  --max-arm-joint-speed 1.2 \
  --home-return-speed 0.5 \
  --ee dex1
```

---

## 6. MuJoCo：不需要一直按 grip，便于调试

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/demo_xrobotics_mujoco.py \
  --frequency 30 \
  --viewer-robot g1_d_mobile \
  --controller-deadman none \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --base-max-vx 0.15 \
  --base-max-wz 0.40 \
  --base-max-z 1.0 \
  --max-arm-joint-speed 1.2 \
  --home-return-speed 0.5 \
  --ee dex1
```

---

## 7. 真机：不控 Dex1，只看双臂

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/teleop_hand_and_arm.py \
  --input-mode controller \
  --arm G1_29 \
  --network-interface eno1 \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative
```

---

## 8. 真机：头动带手动演示模式

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/teleop_hand_and_arm.py \
  --input-mode controller \
  --arm G1_29 \
  --ee dex1 \
  --network-interface eno1 \
  --controller-deadman grip \
  --head-reference-mode head_coupled \
  --controller-orientation-mode absolute
```

---

## 常用操作

- 启动程序
- MuJoCo 中初始停在 G1D ready / home pose
- 按 `r` 进入 tracking
- 按住对应手柄 `grip` 才允许运动
- 松开 `grip` 后保持当前位置
- 按 `q` 退出

MuJoCo G1D 带底盘场景下：

- 左摇杆上下：前进 / 后退
- 左摇杆左右：原地转向
- 右摇杆上下：升降柱

如当前分支保留了附加逻辑：

- 按 `y`：回 ready / home
- 按 `c`：重新参考 / 重标定

---

## 不建议

G1D 当前链路不要使用：

```bash
--motion
```

原因：

- 手臂应走 debug 模式
- 底盘应走 `g1d_agv`

---

## 推荐主命令

### 真机主用

```bash
python teleop/teleop_hand_and_arm.py \
  --input-mode controller \
  --arm G1_29 \
  --ee dex1 \
  --network-interface eno1 \
  --base-controller g1d_agv \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --base-max-vx 0.15 \
  --base-max-wz 0.40 \
  --base-max-z 1.0 \
  --base-stick-deadzone 0.12
```
```
python teleop/teleop_hand_and_arm.py \
  --input-mode controller \
  --arm G1_29 \
  --ee dex1 \
  --network-interface wlo1 \  
  --base-controller g1d_agv \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --base-max-vx 0.15 \
  --base-max-wz 0.40 \
  --base-max-z 1.0 \
  --base-stick-deadzone 0.12
```


### MuJoCo 主用

```bash
python teleop/demo_xrobotics_mujoco.py \
  --frequency 30 \
  --viewer-robot g1_d_mobile \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --base-max-vx 0.15 \
  --base-max-wz 0.40 \
  --base-max-z 1.0 \
  --max-arm-joint-speed 1.2 \
  --home-return-speed 0.5 \
  --ee dex1
```

backup
```
mujoco

python teleop/demo_xrobotics_mujoco.py \
    --frequency 30 \
    --viewer-robot g1_d_mobile \
    --controller-deadman grip \
    --head-reference-mode live_head_reference \
    --controller-mapping-mode anchored_safe \
    --controller-orientation-mode relative \
    --base-max-vx 0.15 \
    --base-max-wz 0.40 \
    --base-max-z 1.0 \
    --max-arm-joint-speed 1.2 \
    --home-return-speed 0.5 \
    --ee dex1
```
```
real robot

python teleop/teleop_hand_and_arm.py \
    --input-mode controller \
    --arm G1_29 \
    --ee dex1 \
    --network-interface eno1 \
    --base-controller g1d_agv \
    --controller-deadman grip \
    --head-reference-mode live_head_reference \
    --controller-mapping-mode anchored_safe \
    --controller-orientation-mode relative \
    --base-max-vx 0.15 \
    --base-max-wz 0.40 \
    --base-max-z 1.0 \
    --base-stick-deadzone 0.12

```
