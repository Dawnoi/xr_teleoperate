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

python teleop/real/teleop_hand_and_arm.py \
  --input-mode controller \
  --arm G1_29 \
  --ee dex1 \
  --network-interface eno1 \
  --base-controller g1d_agv \
  --base-motion \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --max-arm-joint-speed 5.0 \
  --arm-workspace-mode tapered \
  --arm-workspace-z-min -0.05 \
  --arm-workspace-z-max 0.292 \
  --arm-workspace-x-min 0.10 \
  --arm-workspace-x-max-low 0.38 \
  --arm-workspace-x-max-high 0.52 \
  --arm-workspace-y-max-low 0.24 \
  --arm-workspace-y-max-high 0.38 \
  --base-max-vx 0.20 \
  --base-max-wz 0.60 \
  --base-max-z 1.0 \
  --base-stick-deadzone 0.10
```

Wi‑Fi:

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/real/teleop_hand_and_arm.py \
  --input-mode controller \
  --arm G1_29 \
  --ee dex1 \
  --network-interface wlo1 \
  --base-controller g1d_agv \
  --base-motion \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --max-arm-joint-speed 5.0 \
  --arm-workspace-mode tapered \
  --arm-workspace-z-min -0.05 \
  --arm-workspace-z-max 0.292 \
  --arm-workspace-x-min 0.10 \
  --arm-workspace-x-max-low 0.38 \
  --arm-workspace-x-max-high 0.52 \
  --arm-workspace-y-max-low 0.24 \
  --arm-workspace-y-max-high 0.38 \
  --base-max-vx 0.20 \
  --base-max-wz 0.60 \
  --base-max-z 1.0 \
  --base-stick-deadzone 0.10
```
```
python teleop/real/teleop_hand_and_arm.py \
  --input-mode controller \
  --arm G1_29 \
  --ee dex1 \
  --network-interface wlo1 \
  --base-controller g1d_agv \
  --base-motion \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode absolute \
  --max-arm-joint-speed 5.0 \
  --arm-workspace-mode tapered \
  --arm-workspace-z-min -0.05 \
  --arm-workspace-z-max 0.292 \
  --arm-workspace-x-min 0.10 \
  --arm-workspace-x-max-low 0.38 \
  --arm-workspace-x-max-high 0.52 \
  --arm-workspace-y-max-low 0.24 \
  --arm-workspace-y-max-high 0.38 \
  --base-max-vx 0.20 \
  --base-max-wz 0.60 \
  --base-max-z 1.0 \
  --base-stick-deadzone 0.10
```
### 1.0 数据采集（本地 RGB / 远端 ZMQ Raw）

当前录制链路支持两种相机接入方式：

1. 本地 `cv2.VideoCapture`
2. 远端 `ZMQ raw` 图像流

本地相机参数：

- `--head-camera-id`
- `--left-camera-id`
- `--right-camera-id`
- `--camera-width`
- `--camera-height`
- `--camera-fps`
- `--camera-fourcc`
- `--camera-buffer-size`

远端相机参数：

- `--head-zmq-endpoint`
- `--left-zmq-endpoint`
- `--right-zmq-endpoint`

机械臂录制表示参数：

- `--record-arm-repr qpos`：只录关节角（默认）
- `--record-arm-repr pose`：只录左右腕位姿
- `--record-arm-repr both`：同时录关节角和左右腕位姿

移动操作训练采集必须同时启用以下参数：

- `--record-base`：录制底盘 DDS 状态与最终下发命令。
- `--record-slam-map-pose`：从 ROS2 `/tf` 录制 SLAM map 绝对位姿。
- `--slam-pose-source-frame odom`：明确组合 `slamware_map -> odom -> base_link`，记录真实语义的 `slamware_map -> base_link`；两段 raw TF、header 时间戳和 chain skew 会一并写入。当前移动训练采集应使用此模式。
- `--record-mobile-training-state`：新增 `base_link` 下的左右 EEF pose、Dex1.1 两爪中点 TCP pose、升降柱米制位置和腰部状态/目标；保留已有 `pelvis` pose 与 qpos 字段，不改变旧字段语义。该模式要求 `--arm G1_29 --ee dex1`，否则直接拒绝启动。
- `--base-velocity-frame base_link`：声明 `rt/agv/odom` 的 `vx/vy/wz` 使用机体坐标。若上游发布的是世界系速度，必须显式改为 `world`，不能混用。移动训练采集不允许省略该声明。

`--record-mobile-training-state` 强制要求前两个开关和 `--base-height-topic`，避免生成缺失坐标系或升降柱状态的伪完整样本。

录制停止后，UI 会生成并展示 `validation.json`。只要 episode 含有 `states.base`，就必须满足完整 `mobile_tcp_v1`：每帧 `pose_base_link`、`pose_base_link_tcp`、SLAM map pose、底盘/升降柱/腰部状态与命令，以及顶层 Dex1 TCP metadata 都会校验；旧 base-only episode 会明确报错，不能被标记为移动训练可用。SLAM TF 作为独立 `/tf` 历史缓存；每个相机样本按自己的时间戳匹配最近 TF，而不是复用 odom 回调时的 TF。UI 分别校验相机到 base state、相机到 SLAM TF 的对齐误差，以及 TF 的 header 时间新鲜度。`odom` 模式不依赖 `laser == base_link` 假设。

可选 Web UI 参数：

- `--ui`：启动本机 Web UI 控制面
- `--ui-host`：UI 监听地址，默认 `127.0.0.1`
- `--ui-port`：UI 监听端口，默认 `8085`
- `--ui-preview-fps`：UI 状态推送和预览刷新上限，默认 `5.0`

UI 的边界：

- UI 不新建 ROS 节点，不另开机器人控制链路，也不直接发送关节目标、IK 目标或 DDS 消息。
- XR 真机入口默认继续使用 `TeleopUiServer` 和 `teleop/ui/web` 的既有页面布局、HTTP 路由、SSE 状态流及 `UiCommandBus`。真正执行仍在 `teleop/real/teleop_hand_and_arm.py` 的原控制循环中复用原有控制链路。
- `robot_ui_platform` 与 `XrTeleoperateBackend` 是供其他机器人显式接入的通用库，不会替换 XR 默认页面。它们接收高层意图并转入既有命令总线，不满足状态前提时明确返回 `rejected`，不会静默忽略。
- UI 不启动、不停止相机；相机仍由 `--head-camera-id` / `--head-zmq-endpoint` 等启动参数决定。
- UI 预览只读取当前相机 source 的 latest frame，不参与 episode 写盘，不改变录制对齐。

底盘安全约束：当 STOP 未收到确认时，主循环进入 `base_stop_fault`，只会按控制频率重试 STOP，绝不下发新的非零底盘目标；故障会写入 UI 快照和日志。若 G1D bridge 子进程已经退出，Python 已失去到底盘的通信通道，因此 AGV 控制端必须配置“命令超时自动归零”的 watchdog；缺少该 watchdog 时，软件不能保证进程故障后的停车安全。

可选通用 UI API（当前 XR 默认页面不启用）：

- `GET /api/v1/capabilities`：查询当前后端与可用意图。
- `GET /api/v1/snapshot`、`GET /api/v1/events`：读取轮询快照或 SSE 状态流。
- `POST /api/v1/intents`：提交 `{id, type, payload, expected_snapshot_version}`。
- `GET /api/v1/commands/<id>`：查询命令接受/拒绝结果。
- `GET /api/v1/previews/<stream_id>`：读取 `head`、`left_wrist`、`right_wrist` 的 JPEG 预览。

当前 XR 浏览器页面由 `teleop/ui/web` 提供，保持原有布局。未来其他机器人若显式启动 `UiHost`，其页面会按 capabilities 动态生成操作控件；`accepted_by_control_layer` 表示意图已进入该机器人控制线程的既有命令总线，动作是否完成仍须以 snapshot 为准。

接入另一台机器人时，不复制 XR 控制代码：实现 `UiBackend` 的 `capabilities()`、`consume_intent()`、`snapshot()`、`get_preview()`，再将构造函数注册到 `UiBackendFactory`。若机器人服务分布在多个进程，Backend 应调用其 gateway，而不是把控制逻辑搬进 UI 进程。

示例（假设本地三路相机分别是 `/dev/video0 /dev/video2 /dev/video4`）：

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/real/teleop_hand_and_arm.py \
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
    --arm-workspace-mode tapered \
    --arm-workspace-z-min -0.05 \
    --arm-workspace-z-max 0.292 \
    --arm-workspace-x-min 0.10 \
    --arm-workspace-x-max-low 0.38 \
    --arm-workspace-x-max-high 0.52 \
    --arm-workspace-y-max-low 0.24 \
    --arm-workspace-y-max-high 0.38 \
    --base-max-vx 0.20 \
    --base-max-wz 0.60 \
    --base-max-z 1.0 \
    --base-stick-deadzone 0.10 \
    --record \
    --headless \
    --task-dir ./utils/data \
    --task-name pick_cube \
    --task-goal "pick up cube" \
    --task-desc "xr teleop data collection" \
    --task-steps "reach; grasp; lift; place" \
    --head-camera-id 0 \
    --left-camera-id 2 \
    --right-camera-id 4 \
    --camera-width 640 \
    --camera-height 480 \
    --camera-fps 30 \
    --camera-fourcc MJPG \
    --camera-buffer-size 1
```

带 Web UI 的录制示例：

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/real/teleop_hand_and_arm.py \
    --input-mode controller \
    --arm G1_29 \
    --ee dex1 \
    --network-interface eno1 \
    --base-controller g1d_agv \
    --controller-deadman grip \
    --head-reference-mode fixed_per_grip \
    --controller-mapping-mode anchored_safe \
    --controller-orientation-mode relative \
    --record \
    --headless \
    --task-dir ./utils/data \
    --task-name pick_cube \
    --head-camera-id 0 \
    --left-camera-id 2 \
    --right-camera-id 4 \
    --ui \
    --ui-host 0.0.0.0 \
    --ui-port 8085
```

启动后浏览器打开：

```text
http://<robot-host-ip>:8085
```

#### UI 真机回放底盘

在共享 UI 的 `replay.start` 操作中填写 `dataset_root`、`episode_name` 与 `base_source` 后，`base_source` 可选：

- `none`：只回放双臂和末端执行器，不修改底盘。
- `action`：同时回放 `actions.base` 中的 `vx_cmd`、`vy_cmd`、`wz_cmd`、`z_cmd`。

选择 `action` 的启动前提是主程序已使用 `--base-motion --base-controller g1d_agv` 启动，且 episode 每一帧都有 `actions.base`。UI 回放启动时会临时把底盘命令源切到 raw replay provider；停止或完成回放后，底盘同步停车并恢复正常 XR 控制配置。录制 active/armed 时，UI 会明确拒绝开始真机回放。

UI 动态 HTTP 推理示例：进程仍以 XR 作为常驻输入启动；在共享 UI 的 `inference.start` 操作中填写 prompt 后，主循环先 HOLD，再切换到 HTTP pi0.5 provider。推理停止后保持 HOLD，可通过 `provider.xr` 显式恢复 XR。

```bash
cd ~/unitree_ws/src/xr_teleoperate

export SENDER_IP=192.168.123.164

python teleop/real/teleop_hand_and_arm.py \
    --input-provider xr \
    --input-mode controller \
    --arm G1_29 \
    --ee dex1 \
    --network-interface eno1 \
    --base-controller g1d_agv \
    --controller-deadman grip \
    --head-reference-mode fixed_per_grip \
    --controller-mapping-mode anchored_safe \
    --controller-orientation-mode relative \
    --head-zmq-endpoint "tcp://${SENDER_IP}:5556" \
    --left-zmq-endpoint "tcp://${SENDER_IP}:5557" \
    --right-zmq-endpoint "tcp://${SENDER_IP}:5558" \
    --ui \
    --ui-host 0.0.0.0 \
    --ui-port 8085 \
    --online-inference-base-url http://127.0.0.1:18027 \
    --online-inference-transform-config configs/inference/unitree_dual_arm_identity_transform.json
```

动态推理固定使用 HTTP、`pi05_dual_arm_20d`、双臂和真机运动模式；网页只填写 prompt。启动前必须确认 `18027` 推理服务和 head、left_wrist、right_wrist 三路相机已经可用。录制 active/armed 或 raw 真机回放期间，网页会拒绝启动推理。

### 1.0.1 远端单路 ZED 数采（推荐：取 LEFT 作为 RGB）

当前推荐把远端 ZED 的 raw 数采链路配置为：

- sender 侧输入是 ZED 双目图时，`--zmq-raw` 自动取 **LEFT half**
- host 侧按单路 RGB 录入 `colors/head` / `colors/left_wrist` / `colors/right_wrist`

#### 远端 sender（例如 `192.168.123.164`）

目录：

```bash
cd ~/XRoboToolkit-Ubuntu-Video-Sender-Webcam-dev-test
```

编译：

```bash
make clean
make
```

启动单路 ZED raw 发布（这里把它作为 `head` 相机）：

```bash
./OrinVideoSender \
  --send \
  --zmq-raw tcp://*:5556 \
  --camera /dev/video0 \
  --width 640 \
  --height 480 \
  --fps 30
```

说明：

- `--zmq-raw` 现在发送的是与 `xr_teleoperate` 兼容的 `XRAW` 协议。
- 若输入本身是 ZED 的左右拼接图，sender 会自动裁出 **LEFT** 半边用于数采。
- 若要映射成左腕/右腕，只需在 host 侧改用 `--left-zmq-endpoint` 或 `--right-zmq-endpoint`。

#### host 侧（录制 + 保存）

无 Rerun 实时图像，仅录制：

```bash
cd ~/unitree_ws/src/xr_teleoperate

bash scripts/start/start_real_robot_wired.sh \
  --record \
  --headless \
  --task-dir ./utils/data \
  --task-name single_zed_record \
  --head-zmq-endpoint tcp://192.168.123.164:5556
```

带 Rerun 实时图像显示：

```bash
cd ~/unitree_ws/src/xr_teleoperate

bash scripts/start/start_real_robot_wired.sh \
  --record \
  --task-dir ./utils/data \
  --task-name single_zed_record \
  --head-zmq-endpoint tcp://192.168.123.164:5556
```

说明：

- **想看 Rerun 实时图像时不要加 `--headless`**。
- 当前 Rerun 默认布局：
  - 上半部分：`head` 实时图像（居中大图）
  - 下半部分：曲线区
    - `joint_curves` tab：`left/right_arm`、`left/right_ee`
    - `pose_curves` tab：`left/right_arm` 的 `xyz/rpy`
- 时间轴面板默认展开，可直接拖动回放。

说明：

- `camera-id < 0` 表示禁用该路相机。
- ZMQ endpoint 为空字符串表示禁用该路远端相机。
- 推荐采集时加 `--headless`，避免 Rerun Viewer 依赖影响录制。
- 如果要查看实时采集图像，请去掉 `--headless`。
- 当前写入的是 RGB 图像到 `episode_xxxx/colors/`，key 为：
  - `head`
  - `left_wrist`
  - `right_wrist`
- 当前录制的 `states/actions` 只保留：
  - `left_arm`
  - `right_arm`
  - `left_ee`
  - `right_ee`
- 使用 `--record-mobile-training-state` 时会额外写入：
  - `states.left_arm.pose_base_link`、`states.right_arm.pose_base_link`：当前 EEF 在 `base_link` 下的 pose，字段内标明 `frame_id=base_link`。
  - `actions.left_arm.pose_base_link`、`actions.right_arm.pose_base_link`：最终目标 EEF 在当前 `base_link` 下的 pose，字段内标明 `frame_id=base_link`。
  - `states.left_arm.pose_base_link_tcp`、`states.right_arm.pose_base_link_tcp`：相机对齐后的真实 arm q、真实 waist yaw 和真实 column height 经 `assets/g1_d/g1_d.urdf` 完整 FK 得到的 Dex1.1 TCP。TCP 是两爪接触块中点，严格为 `base_link -> left/right_dex1_tcp`。
  - `actions.left_arm.pose_base_link_tcp`、`actions.right_arm.pose_base_link_tcp`：相机对齐后的 arm target q、waist yaw target 和同一时刻 column height 得到的 TCP，严格为 `base_link -> left/right_dex1_tcp_target`。
  - TCP 使用固定 `wrist_yaw_link -> tcp = xyz[0.1201, 0, 0] m, rpy[0, 0, 0] rad`。G1D FK URDF 根为 `AGV_link`；采集安装标定明确 `base_link == AGV_link`，不施加隐藏外参。每个 episode 的 `info` 会写入两份 URDF 路径、frame、单位和该外参。
  - `states.base.slam_map_pose`：`slamware_map -> base_link` 的绝对 `x/y/z/yaw` 与四元数。`--slam-pose-source-frame odom` 时，`source_chain` 保留 `slamware_map -> odom`、`odom -> base_link` 两段 raw transform、header 时间戳和 `max_header_skew_ms`；`tf_header_stamp_ns`、`tf_lookup_wall_time_ns`、`tf_lookup_monotonic_ns` 和 `tf_age_ms` 用于审计最终 transform 是否陈旧。
  - `states.base.column_height_m`、`states.base.waist_yaw`：升降柱米制位置和实际腰关节角。
  - `actions.base.vx_cmd`、`actions.base.wz_cmd`、`actions.base.z_cmd`、`actions.base.waist_yaw_target`：最终下发的底盘、升降柱和腰部命令。其中 `z_cmd` 的单位明确为 `normalized`，并非 m/s；`waist_yaw_target` 是绝对关节角，并非角速度。
- 其中 `left_arm/right_arm` 默认写 `qpos`；若设置 `--record-arm-repr pose/both`，会额外或仅写：
  - `pose.position`
  - `pose.rpy`
  - `pose.rotation_matrix`
  - `pose.matrix4x4`
- 不再录制 `body` / 全身数据。
- 当前 `depths/` 仍未接深度相机录制链路。

### 1.0.2 对齐机制（可靠版）

当前录制不再简单取“latest frame”，而是按 **host monotonic clock** 做近邻对齐：

- `sample_monotonic_ns`：当前样本锚点时间
- `state.host_monotonic_ns`：离该锚点最近的一份 arm state
- `action.host_monotonic_ns`：离该锚点最近的一份 arm action
- `camera.*.host_recv_monotonic_ns` / `camera.*.host_monotonic_ns`：离该锚点最近的一帧图像

具体行为：

- 相机端维护环形缓冲，按 `get_nearest(sample_t)` 选最近帧
- arm state / action 也维护时间缓冲，按 `sample_t` 选最近项
- 录制开始后第一条样本不会吃录制前旧缓存帧
- 若某路已启用相机在允许时间窗内没有找到匹配帧，该条 sample 会被跳过

因此：

- 图像文件名中的时间戳只是辅助排查
- 真正的对齐依据以 `data.json` 中的 `timestamps` 字段为准

### 1.0.3 录制操作

- `r`：启动 teleop
- `s`：开始录制当前 episode
- 再按一次 `s`：停止并保存当前 episode
- `q`：退出

### 1.0.4 时间戳字段

每条 `data.json` item 新增：

```json
"timestamps": {
  "sample_wall_time_ns": ...,
  "sample_monotonic_ns": ...,
  "teleop_input_perf_counter_ns": ...,
  "state": {
    "host_monotonic_ns": ...,
    "delta_to_sample_ns": ...
  },
  "action": {
    "host_monotonic_ns": ...,
    "delta_to_sample_ns": ...
  },
  "camera": {
    "head": {
      "camera_name": "head",
      "camera_id": 0,
      "frame_seq": 123,
      "host_wall_time_ns": ...,
      "host_monotonic_ns": ...,
      "host_recv_monotonic_ns": ...,
      "align_target_monotonic_ns": ...,
      "delta_to_sample_ns": ...,
      "read_latency_ms": ...,
      "shape": [640, 480, 3]
    }
  }
}
```

含义：

- `sample_wall_time_ns`：该条样本写入前的主机墙钟时间。
- `sample_monotonic_ns`：该条样本的统一对齐锚点时间，适合做时差计算。
- `teleop_input_perf_counter_ns`：本轮 teleop 输入到达主循环的高精度计时点。
- `state/action.*.delta_to_sample_ns`：该状态 / 动作与样本锚点的时间差。
- `camera.*.delta_to_sample_ns`：该图像帧与样本锚点的时间差。
- `camera.*`：对应相机匹配帧的 host 侧接收时间与读取耗时。

### 1.0.5 Rerun 回放

每个 episode 保存完成后，目录下会额外生成：

```text
episode_xxxx/rerun.rrd
```

离线回放：

```bash
rerun ./utils/data/<task_name>/episode_0001/rerun.rrd
```

如果 `rerun` 不在 PATH，可直接用 conda 环境里的可执行文件：

```bash
/home/dx/miniconda3/envs/tv/bin/rerun ./utils/data/<task_name>/episode_0001/rerun.rrd
```

一键脚本：

```bash
cd ~/unitree_ws/src/xr_teleoperate

# 有线
bash scripts/start/start_real_robot_wired.sh

# Wi‑Fi
bash scripts/start/start_real_robot_wifi.sh
```

如果网卡名字不同，可以直接覆盖：

```bash
NETWORK_INTERFACE=enp3s0 bash scripts/start/start_real_robot_wired.sh
NETWORK_INTERFACE=wlan0  bash scripts/start/start_real_robot_wifi.sh
```

如果要临时追加额外参数，也可以直接接在后面：

```bash
bash scripts/start/start_real_robot_wifi.sh --timing-debug --timing-debug-interval 2.0
bash scripts/start/start_real_robot_wired.sh --record --headless --task-dir ./utils/data --task-name test_record --head-camera-id 0 --left-camera-id 2 --right-camera-id 4
```

底盘语义：

- 左摇杆上下：前进 / 后退
- 左摇杆左右：原地转向
- 右摇杆上下：升降柱

### 1.1 开启时延追踪（收到输入 -> DDS下发 -> 执行响应）

```bash
python teleop/real/teleop_hand_and_arm.py \
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
  --arm-workspace-mode tapered \
  --arm-workspace-z-min -0.05 \
  --arm-workspace-z-max 0.292 \
  --arm-workspace-x-min 0.10 \
  --arm-workspace-x-max-low 0.38 \
  --arm-workspace-x-max-high 0.52 \
  --arm-workspace-y-max-low 0.24 \
  --arm-workspace-y-max-high 0.38 \
  --home-return-speed 1.0 \
  --latency-trace \
  --latency-trace-path ./utils/data/latency_trace.jsonl \
  --timing-debug \
  --timing-debug-interval 2.0
```

运行时会打印：

```text
[LATENCY] seq=7 recv->pub=8.31 ms | pub->exec=22.12 ms | recv->exec=30.43 ms | tele=3.20 ik=4.80 queue=3.21 wait=3.17 dds=0.042 ms
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
- XR 取数 -> 执行响应（fetch_to_exec）
- takeover / base_control / IK / safety / gravity / controller wait / DDS write / 未归因时间 的拆分
- 最近若干条样本的“分段时间线”，0 点是 tele_data ready，黑线是 publish 时刻

最近若干条样本还会画成时间线，方便直接看“从哪一点到哪一点”。

---

## 2. 真机：仅手臂 + Dex1，不控底盘（免 C）

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/real/teleop_hand_and_arm.py \
  --input-mode controller \
  --arm G1_29 \
  --ee dex1 \
  --network-interface eno1 \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --arm-workspace-mode tapered \
  --arm-workspace-z-min -0.05 \
  --arm-workspace-z-max 0.292 \
  --arm-workspace-x-min 0.10 \
  --arm-workspace-x-max-low 0.38 \
  --arm-workspace-x-max-high 0.52 \
  --arm-workspace-y-max-low 0.24 \
  --arm-workspace-y-max-high 0.38 \
  --max-arm-joint-speed 5.0
```

---

## 3. 真机：不控 Dex1，只看双臂（免 C）

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/real/teleop_hand_and_arm.py \
  --input-mode controller \
  --arm G1_29 \
  --network-interface eno1 \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --arm-workspace-mode tapered \
  --arm-workspace-z-min -0.05 \
  --arm-workspace-z-max 0.292 \
  --arm-workspace-x-min 0.10 \
  --arm-workspace-x-max-low 0.38 \
  --arm-workspace-x-max-high 0.52 \
  --arm-workspace-y-max-low 0.24 \
  --arm-workspace-y-max-high 0.38 \
  --max-arm-joint-speed 5.0
```

---

## 4. 真机备选：live 头参考（免 C）

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/real/teleop_hand_and_arm.py \
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
  --arm-workspace-mode tapered \
  --arm-workspace-z-min -0.05 \
  --arm-workspace-z-max 0.292 \
  --arm-workspace-x-min 0.10 \
  --arm-workspace-x-max-low 0.38 \
  --arm-workspace-x-max-high 0.52 \
  --arm-workspace-y-max-low 0.24 \
  --arm-workspace-y-max-high 0.38 \
  --base-max-vx 0.20 \
  --base-max-wz 0.60 \
  --base-max-z 1.0 \
  --base-stick-deadzone 0.10
```

---

## 4.1 真机工作空间限制

真机现在已经和 MuJoCo 同步支持两类工作空间：

- `--arm-workspace-mode tapered`：**默认推荐**，下窄上宽，适合桌面 / 操作台
- `--arm-workspace-mode box`：固定矩形范围，仅用于对比 / 简单调试

### 推荐：倒梯形（tapered）工作空间

```bash
--arm-workspace-mode tapered \
--arm-workspace-z-min -0.05 \
--arm-workspace-z-max 0.292 \
--arm-workspace-x-min 0.10 \
--arm-workspace-x-max-low 0.38 \
--arm-workspace-x-max-high 0.52 \
--arm-workspace-y-max-low 0.24 \
--arm-workspace-y-max-high 0.38
```

含义：

- 低位（靠近桌面）更窄：
  - `x: [0.10, 0.38]`
  - `y: [-0.24, 0.24]`
- 高位（抬臂后）更宽：
  - `x: [0.10, 0.52]`
  - `y: [-0.38, 0.38]`
- `z: [-0.05, 0.292]`，上边界等于 G1D 肩关节中心相对原 G1_29 IK 原点的高度
- `+z` 是抬臂方向

### 调参建议

如果桌面附近太窄：

- 增大 `--arm-workspace-x-max-low`
- 增大 `--arm-workspace-y-max-low`

如果抬臂后还不够宽：

- 增大 `--arm-workspace-x-max-high`
- 增大 `--arm-workspace-y-max-high`

默认不允许法兰目标高于肩部。只有明确需要高举动作并完成安全验证时，才提高 `--arm-workspace-z-max`。

如果太容易往下扎：

- 增大 `--arm-workspace-z-min`

### 真机双手独立空间与单手 QP 引导

默认 `--arm-workspace-layout shared` 保持左右腕共用同一工作空间。要允许左手更深入右侧、右手更深入左侧，必须显式切换到 `per_arm` 并完整提供两侧参数；缺任意一侧时启动直接报错。

每侧 tapered 参数的九个数依次为：

```text
z_min z_max x_min x_max_low x_max_high y_min_low y_min_high y_max_low y_max_high
```

下面是左右镜像的当前配置。`x_min=0.154` 对齐 G1D URDF 零位 J6（`wrist_pitch_joint`）中心，工作空间不会延伸到 J6 后方。

```bash
--mobile-manipulation-mode mobile_ik_qp \
--arm-workspace-layout per_arm \
--left-arm-workspace-tapered  -0.055 0.245 0.154 0.38 0.52  0.00 0.00  0.20 0.28 \
--right-arm-workspace-tapered -0.055 0.245 0.154 0.38 0.52  -0.20 -0.28  0.00 0.00
```

左 grip 按住时，QP 只检查左手专属空间并为左手移动底盘；右手不按 grip，即使其目标越界也不会触发底盘。右 grip 同理。当前三相机有线启动脚本将跨身边界直接设为身体中线：在当前 G1 IK 坐标中，左手 `y_min=0`，右手 `y_max=0`。QP 的 `0.04 m` 舒适裕量会使底盘在手距中线约 `0.04 m` 时提前开始引导，法兰目标最终不会跨过中线。工作空间高度固定为 G1D URDF 全零位、升降柱零位、腰部零位时 Dex1 TCP 的 IK-frame 高度 `0.095226 m` 上下各 `0.15 m`，即 `z=[-0.055, 0.245] m`（参数保留至毫米）。该功能仍控制原始 G1_29 腕端/法兰目标，不改变 Dex1 TCP 的采集语义。

当前没有完整的机械臂-躯干-另一机械臂自碰撞模型；不要把 `x_min` 继续减小，也不要将跨身区域用于高速真机运动。

### 固定 box 对比模式

```bash
--arm-workspace-mode box \
--arm-workspace-min 0.10 -0.32 -0.08 \
--arm-workspace-max 0.45 0.32 0.292
```

如果完全关闭工作空间限制：

```bash
--disable-arm-workspace-limit
```

---

## 5. MuJoCo 主用：fixed_per_grip（免 C）

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/sim/xrobotics_mujoco.py \
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
  --arm-workspace-mode tapered \
  --arm-workspace-z-min -0.05 \
  --arm-workspace-z-max 0.292 \
  --arm-workspace-x-min 0.10 \
  --arm-workspace-x-max-low 0.38 \
  --arm-workspace-x-max-high 0.52 \
  --arm-workspace-y-max-low 0.24 \
  --arm-workspace-y-max-high 0.38 \
  --ee dex1
```

---

## 6. MuJoCo：调试版，不需要一直按 grip（免 C）

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/sim/xrobotics_mujoco.py \
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
  --arm-workspace-mode tapered \
  --arm-workspace-z-min -0.05 \
  --arm-workspace-z-max 0.292 \
  --arm-workspace-x-min 0.10 \
  --arm-workspace-x-max-low 0.38 \
  --arm-workspace-x-max-high 0.52 \
  --arm-workspace-y-max-low 0.24 \
  --arm-workspace-y-max-high 0.38 \
  --ee dex1
```

---

## 7. MuJoCo 备选：live 头参考（免 C）

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/sim/xrobotics_mujoco.py \
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
  --arm-workspace-mode tapered \
  --arm-workspace-z-min -0.05 \
  --arm-workspace-z-max 0.292 \
  --arm-workspace-x-min 0.10 \
  --arm-workspace-x-max-low 0.38 \
  --arm-workspace-x-max-high 0.52 \
  --arm-workspace-y-max-low 0.24 \
  --arm-workspace-y-max-high 0.38 \
  --ee dex1
```

---

## 8. MuJoCo 工作空间调试与可视化

当前 MuJoCo 已支持两类工作空间：

- `--arm-workspace-mode tapered`：**默认推荐**，下窄上宽，适合桌面 / 操作台
- `--arm-workspace-mode box`：固定矩形范围，仅用于对比 / 简单调试

### 8.1 推荐：倒梯形（tapered）工作空间 + 可视化点

```bash
cd ~/unitree_ws/src/xr_teleoperate

python teleop/sim/xrobotics_mujoco.py \
  --frequency 30 \
  --viewer-robot g1_d_mobile \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode absolute \
  --base-max-vx 0.15 \
  --base-max-wz 0.45 \
  --base-max-z 1.0 \
  --max-arm-joint-speed 5.0 \
  --home-return-speed 0.5 \
  --arm-workspace-mode tapered \
  --arm-workspace-z-min -0.15 \
  --arm-workspace-z-max 0.292 \
  --arm-workspace-x-min -0.15 \
  --arm-workspace-x-max-low 0.25 \
  --arm-workspace-x-max-high 0.65 \
  --arm-workspace-y-max-low 0.28 \
  --arm-workspace-y-max-high 0.42 \
  --arm-workspace-show-targets \
  --ee dex1
```

含义：

- 低位（靠近桌面）更窄：
  - `x: [0.10, 0.38]`
  - `y: [-0.24, 0.24]`
- 高位（抬臂后）更宽：
  - `x: [0.10, 0.52]`
  - `y: [-0.38, 0.38]`
- `z: [-0.05, 0.45]`
- `+z` 是抬臂方向

### 8.2 可视化含义

- 半透明蓝色体：当前工作空间
- 红 / 蓝实心点：**当前机械手真实 EE 位置**
- 黄 / 青点：raw target（原始目标）
- 粉 / 浅蓝点：clamped target（裁剪后目标）

说明：

- 真实 EE 点不是随便画的点，而是按 MuJoCo 中左右手真实 wrist 位姿再加 IK 同样的末端偏移得到
- 所以可以直接拿来对照机械手实际位置

### 8.3 调参建议

如果桌面附近太窄：

- 增大 `--arm-workspace-x-max-low`
- 增大 `--arm-workspace-y-max-low`

如果抬臂后还不够宽：

- 增大 `--arm-workspace-x-max-high`
- 增大 `--arm-workspace-y-max-high`

如果抬不够高：

- 增大 `--arm-workspace-z-max`

如果太容易往下扎：

- 增大 `--arm-workspace-z-min`

### 8.4 固定 box 对比模式

```bash
python teleop/sim/xrobotics_mujoco.py \
  --frequency 30 \
  --viewer-robot g1_d_mobile \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode  \
  --arm-workspace-mode box \
  --arm-workspace-min 0.10 -0.32 -0.08 \
  --arm-workspace-max 0.45 0.32 0.292 \
  --arm-workspace-show-targets \
  --ee dex1
```

如果只想看工作空间体、不想看目标点：

```bash
--hide-arm-workspace-visualization
```

如果完全关闭工作空间限制：

```bash
--disable-arm-workspace-limit
```

---

## 9. MuJoCo：VIVE Tracker 预验证

先启动 `vive_locator` 和 VIVE 专用踏板 deadman 节点，再启动 MuJoCo。MuJoCo 使用与真机相同的 VIVE provider、AGX 三点标定文件、踏板 deadman 和 PICO 式 `relative` 姿态增量；不会连接 Unitree DDS。踏板识别、权限和 `evtest` 验证见 `docs/runbooks/vive_startup_zh-CN.md`。

```bash
# Terminal 1：复用已有缓存；首次无缓存时自动标定，结束后等待 5 秒启动 locator
cd /data/codeBase/src/unitree_ws/src/xr_teleoperate
bash scripts/start/start_vive_locator.sh

# 基站移动后才需要强制重新标定：
# VIVE_RECALIBRATE=1 bash scripts/start/start_vive_locator.sh

# Terminal 2
cd /data/codeBase/src/unitree_ws/src/xr_teleoperate
source /opt/ros/humble/setup.bash
python scripts/vive_keyboard_enable.py \
  --grab-input-devices

# Terminal 3（首次或基站移动后）
python scripts/vive_axis_calibrator.py \
  --pose-topic /vive_pose_l \
  --output-file ~/.config/xr_teleoperate/vive_calibration.json

# Terminal 4
python teleop/sim/xrobotics_mujoco.py \
  --input-provider vive \
  --controller-deadman grip \
  --controller-orientation-mode relative \
  --vive-calibration-file ~/.config/xr_teleoperate/vive_calibration.json
```

踏板节点会自动识别常见 pedal/foot/key08 设备；如果左右相反，在命令末尾增加 `--swap-sides`。MuJoCo 窗口中按 `R` 开始同步；按住踏板发送的 `Alt+L`/`Alt+R` 才使能对应手臂，松开立即保持。未使能、踏板心跳超时、Tracker 超时或断流的手臂保持当前 MuJoCo 腕位姿。仿真通过后，再运行下一节的真机启动命令。

## 10. 真机状态 MuJoCo 镜像

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

- `--motion` 会让双臂改走 `rt/arm_sdk`。
- 当前本体应让双臂保持 Debug mode 的 `rt/lowcmd`。
- 要启用 G1D 底盘，使用 `--base-motion --base-controller g1d_agv`。


  cd ~/unitree_ws/src/xr_teleoperate

  python teleop/real/teleop_hand_and_arm.py \
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

    python teleop/real/teleop_hand_and_arm.py \
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
    --home-return-speed 1.0 \
    --latency-trace \
    --latency-trace-path ./utils/data/latency_trace.jsonl \
    --timing-debug \
  ———

  ## 图也还能继续生成

  python teleop/utils/plot_latency_trace.py \
    ./utils/data/latency_trace.jsonl \
    --output ./utils/data/latency_trace.png




bash scripts/start/start_real_robot_wired_3cams_zmq.sh     --input-provider online_inference     --online-inference-transport http     --online-inference-base-url http://115.190.134.186:8017     --online-inference-protocol-profile pi05_dual_arm_20d     --online-inference-prompt "pick up the purple octagonal prism with the right hand, hand it over to the left hand, and place it in the bowl."     --online-inference-enable-motion     --online-inference-transform-config configs/inference/unitree_dual_arm_identity_transform.json

## VLA + Dex1 force-hold 额外夹紧

Dex1 force-hold 只处理一种情况：

```text
夹爪已经碰到物体 / 边角
Dex1 触发 force-hold
```

它不处理空抓。模型空抓时不会触发 force-hold，系统继续执行 VLA 推理动作。

额外夹紧只在 VLA 推理入口生效：

```text
--input-provider online_inference
```

手柄遥操作、手部追踪遥操作、离线回放不会使用自适应额外夹紧；这些模式触发 force-hold 时只锁住接触位置。

当前行为：

```text
force-hold rising edge
-> 记录接触位置 contact_q
-> VLA 推理: 初始 hold_q = contact_q - initial_offset
-> VLA 推理: 后续按 tau_est 自适应微调，低力矩时继续向物理闭合端小步闭合
-> 非 VLA: 锁存 hold_q = contact_q
-> 默认 initial_offset = 0.20 rad
-> 不再设置最大额外闭合 offset；闭合停止/回松由 tau_est 阈值决定
```

自适应夹持目标：

```text
tau_est < 2.5  -> 每 20ms 小幅继续闭合
2.5 <= tau_est <= 3.7 -> 保持当前 hold_q
tau_est > 3.7  -> 小幅松开
tau_est > 5.0  -> 危险微松并持续打日志
```

对应环境变量：

```bash
DEX1_FORCE_HOLD_INITIAL_OFFSET=0.20
DEX1_FORCE_HOLD_ADJUST_INTERVAL_SEC=0.02
DEX1_FORCE_HOLD_TAU_LOW=2.50
DEX1_FORCE_HOLD_TAU_HIGH=3.70
DEX1_FORCE_HOLD_TAU_DANGER=5.00
DEX1_FORCE_HOLD_Q_STEP_CLOSE=0.03
DEX1_FORCE_HOLD_Q_STEP_RELEASE=0.02
DEX1_FORCE_HOLD_Q_STEP_DANGER_RELEASE=0.05
DEX1_FORCE_HOLD_DIAG=1
DEX1_FORCE_HOLD_DIAG_INTERVAL_SEC=1.0
```

现在没有“最多再夹多少”的 offset 限制。只要 `tau_est < DEX1_FORCE_HOLD_TAU_LOW`，夹爪就会继续小步闭合，直到 `tau_est` 进入目标区间、超过上限触发回松，或者到达 Dex1 物理闭合位置 `q=0.0`。

force-hold 的力矩相关参数目前只用于“触发判断”，不是进入 force-hold 后的持续力矩闭环：

```bash
DEX1_FORCE_HOLD_TAU_ENGAGE_THRESH=1.50
```

含义：

```text
tau_est 绝对值大于该阈值
+ 夹爪仍在闭合
+ 命令位置和实际位置有明显误差
-> 判定为接触
-> 进入 force-hold
```

如果太容易误触发，可以增大该阈值；如果接触后不容易触发，可以减小该阈值。

当前默认 `1.50` 来自实测：稳定夹持时 `tau_est` 在 2.x 左右，完全闭合 / 硬顶时左夹爪 `tau_est` 约 6.9。因此 `1.50` 用作更敏感的进入 force-hold 接触阈值，后续夹持目标不应接近 6.9。

注意：当前不下发非零 `tau`，Dex1 命令里的 `tau` 保持 0.0。VLA 夹持强度由 `tau_est` 反馈调节位置命令，`DEX1_FORCE_HOLD_TAU_ENGAGE_THRESH` 只决定何时进入 force-hold。

已实测 Dex1 的 `tau` 更像保持力矩，不能可靠产生闭合动作。因此纯力矩闭合模式已移除，不作为 VLA 主流程选项。

日志形态：

```text
[Dex1_1_Gripper_Controller] right force-hold diag: active=False, extra_close_enabled=True, closing_intent=True, contact_detected=False, large_command_error=True, driver_ok=True, tau_abs=1.200/1.500, command_error=0.300/0.200, raw_target_q=..., state_q=..., detect_age=none.

[Dex1_1_Gripper_Controller] right gripper force-hold engaged at
contact_q=3.335, hold_q=3.135, extra_close_enabled=True, initial_offset=0.200, physical_close_q=0.000, tau_low=2.500, tau_high=3.700, tau_danger=5.000, tau_est=...
```

其中 `force-hold diag` 表示还没进入，并会直接打印卡在哪个条件：

```text
closing_intent=False       -> VLA 没有继续闭合夹爪
contact_detected=False     -> tau_est 还没超过进入阈值
large_command_error=False  -> 命令位置和实际位置差距还不够
driver_ok=False            -> Dex1 驱动 lost 标志异常
extra_close_enabled=False  -> 当前不是 VLA 推理入口，不会自适应额外夹紧
```

自适应调节日志：

```text
[Dex1_1_Gripper_Controller] right adaptive force-hold reason=close, tau_est=1.900, hold_q=3.115, contact_q=3.335, physical_close_q=0.000.
```
