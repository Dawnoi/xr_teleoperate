# RUN.md

仅保留 **不用按 `C` 做标定** 的启动方式。

当前文档只保留两类头参考模式：

- `fixed_per_grip`：**推荐主用**，每次按下 grip 自动建立本次接管参考
- `live_head_reference`：实时头参考，适合实验 / 对比

不再保留需要 `C` 的模式：

- `calibrated` / `head_coupled`
- `hybrid`

---

## 1. 真机主用：手臂 + Dex1 + G1D 底盘（免 C）

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
  --max-arm-joint-speed 5.0 \
  --base-max-vx 0.20 \
  --base-max-wz 0.60 \
  --base-max-z 1.0 \
  --base-stick-deadzone 0.10
```

Wi‑Fi:

```bash
cd ~/unitree_ws/src/xr_teleoperate

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
  --max-arm-joint-speed 5.0 \
  --base-max-vx 0.20 \
  --base-max-wz 0.60 \
  --base-max-z 1.0 \
  --base-stick-deadzone 0.10
```

底盘语义：

- 左摇杆上下：前进 / 后退
- 左摇杆左右：原地转向
- 右摇杆上下：升降柱

### 1.1 开启时延追踪（收到输入 -> DDS下发 -> 执行响应）

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
  --max-arm-joint-speed 5.0 \
  --home-return-speed 1.0 \
  --latency-trace \
  --latency-trace-path ./utils/data/latency_trace.jsonl \
  --timing-debug \
  --timing-debug-interval 2.0
```

运行时会打印：

```text
[LATENCY] seq=7 收到输入->DDS下发=8.31 ms | DDS下发->执行响应=22.12 ms | 收到输入->执行响应=30.43 ms
[TIMING] loop avg/p95/max=12.0/18.4/24.7 ms, tele avg/p95/max=3.2/5.1/6.0 ms, ik avg/p95/max=4.8/8.9/10.2 ms (53 calls), agv avg/p95/max=0.4/0.7/0.9 ms (53 calls)
[TIMING_DDS] publish_hz=249.8, ctrl_loop avg/p95/max=0.18/0.31/0.52 ms, dds_write avg/p95/max=0.042/0.066/0.091 ms, enqueue->publish avg/p95/max=3.21/6.87/9.15 ms
```

生成图：

```bash
python teleop/utils/plot_latency_trace.py \
  ./utils/data/latency_trace.jsonl \
  --output ./utils/data/latency_trace.png
```

图里会直接展示：

- 收到输入 -> DDS下发
- DDS下发 -> 执行响应
- 收到输入 -> 执行响应

最近若干条样本还会画成时间线，方便直接看“从哪一点到哪一点”。

---

## 2. 真机：仅手臂 + Dex1，不控底盘（免 C）

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
  --controller-orientation-mode relative \
  --max-arm-joint-speed 5.0
```

---

## 3. 真机：不控 Dex1，只看双臂（免 C）

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/teleop_hand_and_arm.py \
  --input-mode controller \
  --arm G1_29 \
  --network-interface eno1 \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --max-arm-joint-speed 5.0
```

---

## 4. 真机备选：live 头参考（免 C）

```bash
cd ~/unitree_ws/src/xr_teleoperate

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
  --max-arm-joint-speed 5.0 \
  --base-max-vx 0.20 \
  --base-max-wz 0.60 \
  --base-max-z 1.0 \
  --base-stick-deadzone 0.10
```

---

## 5. MuJoCo 主用：fixed_per_grip（免 C）

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/demo_xrobotics_mujoco.py \
  --frequency 30 \
  --viewer-robot g1_d_mobile \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --base-max-vx 0.20 \
  --base-max-wz 0.60 \
  --base-max-z 1.0 \
  --max-arm-joint-speed 5.0 \
  --home-return-speed 0.5 \
  --ee dex1
```

---

## 6. MuJoCo：调试版，不需要一直按 grip（免 C）

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/demo_xrobotics_mujoco.py \
  --frequency 30 \
  --viewer-robot g1_d_mobile \
  --controller-deadman none \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --base-max-vx 0.20 \
  --base-max-wz 0.60 \
  --base-max-z 1.0 \
  --max-arm-joint-speed 5.0 \
  --home-return-speed 0.5 \
  --ee dex1
```

---

## 7. MuJoCo 备选：live 头参考（免 C）

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/demo_xrobotics_mujoco.py \
  --frequency 30 \
  --viewer-robot g1_d_mobile \
  --controller-deadman grip \
  --head-reference-mode live_head_reference \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --base-max-vx 0.20 \
  --base-max-wz 0.60 \
  --base-max-z 1.0 \
  --max-arm-joint-speed 5.0 \
  --home-return-speed 0.5 \
  --ee dex1
```

---

## 8. 真机状态 MuJoCo 镜像

用途：

- 真机 teleop 启动后
- MuJoCo 实时显示 **真机双臂 + Dex1 夹爪真实状态**
- 只订阅，不发控制，不影响现有使用

先启动真机，再开：

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/demo_real_robot_shadow_mujoco.py \
  --network-interface eno1 \
  --viewer-robot g1_d_mobile \
  --ee dex1
```

Wi‑Fi:

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/demo_real_robot_shadow_mujoco.py \
  --network-interface wlo1 \
  --viewer-robot g1_d_mobile \
  --ee dex1
```

说明：

- `--viewer-robot g1_d_mobile`
  - 与纯 MuJoCo 主用模式保持一致
  - 使用同一套 G1D mobile 场景
  - shadow 模式下只同步 **真机双臂 + Dex1**，底盘保持 ready pose

- `--viewer-robot g1_d`
  - 固定底座 G1D 视图
  - 如果你只想稳定看上半身镜像，也可以用这个

- `--viewer-robot g1`
  - 原始 G1 视图
  - 只看双臂时也可用

---

## 常用操作

- 按 `r`：进入 tracking
- 按住对应手柄 `grip`：该侧手臂 / 夹爪允许运动
- 松开 `grip`：保持当前位置
- 按 `q`：退出

MuJoCo G1D 场景下：

- 左摇杆上下：前进 / 后退
- 左摇杆左右：原地转向
- 右摇杆上下：升降柱

---

## 常用 `--` 参数说明

### 通用

- `--input-mode controller`
  - 使用 XR 控制器输入
  - 当前真机主链路就用这个

- `--arm G1_29`
  - 选择机械臂型号
  - 当前这里统一用 `G1_29`

- `--ee dex1`
  - 选择末端执行器
  - 这里表示使用 Dex1 夹爪

- `--network-interface eno1`
  - 指定 DDS 网卡
  - 有线常用 `eno1`
  - 无线常用 `wlo1`

- `--controller-deadman grip`
  - 只有按住对应侧 `grip` 才允许运动
  - 推荐真机始终使用

- `--controller-deadman none`
  - 不需要一直按 grip
  - 只建议 MuJoCo 调试时使用

### 头参考 / 映射

- `--head-reference-mode fixed_per_grip`
  - **推荐主用**
  - 不需要按 `C`
  - 每次按下 grip 时自动建立当前接管参考
  - 适合真机稳定遥操

- `--head-reference-mode live_head_reference`
  - 不需要按 `C`
  - 头参考始终实时变化
  - 更适合实验 / 对比，不如 `fixed_per_grip` 稳

- `--controller-mapping-mode anchored_safe`
  - 推荐主用
  - 采用 grip-anchor 的更安全接管逻辑

- `--controller-orientation-mode relative`
  - 推荐主用
  - 手腕姿态按“接管时刻”的相对旋转变化
  - 比 `absolute` 更稳

### 真机底盘

- `--base-controller g1d_agv`
  - 开启 G1D 底盘控制
  - 当前推荐链路

- `--base-max-vx 0.20`
  - 最大前后速度
  - 数值越大，前进后退越快

- `--base-max-wz 0.60`
  - 最大转向角速度
  - 数值越大，原地转向越快

- `--base-max-z 1.0`
  - 升降柱控制幅度

- `--base-stick-deadzone 0.10`
  - 摇杆死区
  - 越大越不容易轻微漂移

- `--max-arm-joint-speed 5.0`
  - 真机推荐值
  - 这是外层关节目标限速，单位 rad/s
  - 如果太小会感觉跟手慢、迟滞明显

### MuJoCo

- `--frequency 30`
  - MuJoCo 主循环刷新频率

- `--viewer-robot g1_d_mobile`
  - 使用带移动底盘的 G1D MuJoCo 视图

- `--max-arm-joint-speed 5.0`
  - MuJoCo 推荐值
  - MuJoCo 中手臂目标关节最大速度
  - 越大动作越快，越小越平滑

- `--home-return-speed 0.5`
  - 回 home / ready pose 时的关节速度限制

### 真机状态镜像

- `--network-interface eno1`
  - 指定订阅真机状态使用的 DDS 网卡

- `--viewer-robot g1_d_mobile`
  - 真机 shadow viewer 推荐值
  - 与纯 MuJoCo 主用场景一致
  - 也可选 `g1_d` / `g1`

- `--ee dex1`
  - 在 MuJoCo 中同时显示 Dex1 夹爪状态

---

## 不建议

G1D 当前链路不要使用：

```bash
--motion
```

原因：

- 手臂应走 debug 模式
- 底盘应走 `g1d_agv`


  ### 1）运行时开启追踪

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
    --latency-trace \
    --latency-trace-path ./utils/data/latency_trace.jsonl \
  --timing-debug \
  --timing-debug-interval 2.0

  ### 2）运行时会打印这种日志

  [LATENCY] seq=7 收到输入->DDS下发=8.31 ms | DDS下发->执行响应=22.12 ms | 收到输入->执行响应=30.43 ms

  ### 3）生成图

  python teleop/utils/plot_latency_trace.py \
    ./utils/data/latency_trace.jsonl \
    --output ./utils/data/latency_trace.png
