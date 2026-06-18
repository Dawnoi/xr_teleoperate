# Online Inference Design

## Summary

本设计把 `pika_ros` dataStreamer 的在线推理语义迁移到本仓库现有 Python teleop 主链路中：

```text
state/camera/gripper snapshots
-> OnlineInferenceSession
-> Pika TCP newline JSON observation/action protocol
-> action chunk
-> pose transform to Unitree IK frame
-> OnlineInferenceInputProvider
-> MotionIntent
-> teleop_hand_and_arm.py existing control loop
```

核心语义保持 Pika 风格：

```text
observation -> inference request -> action chunk -> execute chunk -> post-action delay -> next observation
```

但 provider 只是输入适配层，不直接控制机器人。在线推理输出仍然是 `MotionIntent`，后续继续复用现有 IK、workspace clamp、joint speed limit、gravity tauff、DDS 下发、录制和 Rerun 链路。

本阶段不更新 `RUN.md`，也不创建 `RUN_ONLINE_INFERENCE.md`。

## Why Not Copy The C++ dataStreamer

当前仓库主入口是 `teleop/teleop_hand_and_arm.py`，主链路已经掌握 deadman、hold、go_home、IK、限速、录制时间对齐和相机历史。直接照搬 ROS2/C++ dataStreamer 会引入第二套控制链，使真机安全逻辑分散。

迁移后只沿用 Pika wire protocol 和 action chunk 语义，不迁移 ROS2 节点结构。这样可以让 online inference 与 XR、offline replay 一样成为 `TeleopInputProvider`，同时让主循环保持现有控制边界。

## Modules

## `teleop/utils/inference_protocol.py`

协议层只处理 Pika TCP newline JSON：

- `NewlineJsonCodec` 处理粘包、拆包和逐行 JSON。
- `TcpJsonTransport` 负责非阻塞 socket send/recv。
- `encode_jpeg_base64()` 负责 observation 图像 JPEG base64。
- `parse_action_chunk()` 负责 action schema 和数值校验。

协议层不 import provider，不 import 主循环，不懂 DDS 和 IK。

Action 校验规则：

- 每条 JSON 必须以 `\n` 分隔。
- JSON encode/decode 拒绝 NaN/Infinity。
- `type` 必须是 `action`。
- action step 维度必须是 8。
- 数值必须全部 finite。
- 四元数 norm 不能过小，解析后归一化。
- 空 chunk 拒绝。
- `both` 必须提供 `action_l` 和 `action_r`，拒绝泛化 `action`。
- `left/right` 接受对应 `action_l/action_r` 或单臂兼容 `action`。
- 同一 payload 不能混用 `action` 与 `action_l/action_r`。

## `teleop/utils/pose_transform.py`

坐标变换层负责服务端/Pika frame 与本仓库 Unitree IK/base frame 的双向转换。

真机 motion 要求显式配置：

- active side 必须有 `server_to_unitree`。
- active side 必须有 `tcp_to_wrist` 或 `wrist_to_tcp`。
- `left/right/both` 只要求 active side 配置齐全。
- 缺 transform config 时，`--online-inference-enable-motion` 下启动失败。

dry-run 或未启用 motion 时使用 disabled identity transformer，但 `enabled=False`，不会让 provider 输出可执行 motion。

## `teleop/utils/online_inference.py`

`OnlineInferenceSession` 负责 observation history、host monotonic 对齐、Pika-style observation 构造、非阻塞状态机、action chunk 调度、post-action delay 和 fail-closed status。

当前状态包括：

- `collecting_observation`
- `waiting_action`
- `executing_chunk`
- `post_action_delay`
- `failed`

`tick()` 永远不等待网络或相机帧。它每轮只做短轮询并立即返回：

- 有当前 chunk step：返回该 step。
- 等 server：返回 hold。
- post-action delay 中：返回 hold。
- history 不足：返回 hold。
- timeout/断连/invalid action：进入 failed 并返回 hold。

Observation window 使用 host monotonic 时间：

- state timestamp 来自主循环读取 q/dq 的 host monotonic。
- camera timestamp 优先 `host_recv_monotonic_ns`，否则 `host_monotonic_ns`。
- `camera_freq` 和 `n_obs_steps` 决定历史窗口回溯间隔。
- state/camera 都必须覆盖完整窗口，不能只靠最近样本提前发送。

post-action delay 语义：

- chunk 执行结束后计算 `ready_after_ns`。
- delay 期间不阻塞主循环。
- delay 结束后，还要求 state 和所有 required camera 都有不早于 `ready_after_ns` 的样本。
- 覆盖等待超时后 fail-closed。

## `teleop/utils/teleop_input_provider.py`

`OnlineInferenceInputProvider` 是薄 provider：

- 从主循环接收当前双臂 FK wrist pose、q/dq、gripper width、host monotonic 和 shared camera sources。
- 将非空 camera source 名字登记为 session required cameras。
- 调用 session 并把 step 转成 `TeleopInputSample/MotionIntent`。
- `enabled_arms` 放进 `MotionIntent.metadata`，由主循环按侧启用或 hold。
- dry-run 或未 enable motion 时 `enabled_arms=[]`，不会驱动手臂。
- session failed 时返回 `done=True`，让主循环进入停机路径。

Dex1 gripper v1：

- 服务端 action[7] 表示 gripper width meter。
- active side 映射到 Dex1 trigger `5.0..7.0`。
- inactive side 在主循环里写当前 gripper state 对应的 hold trigger，避免旧目标残留。
- 非 Dex1 且没有映射时启动失败，除非 `--no-gripper`。

## Main Loop Hooks

`teleop/teleop_hand_and_arm.py` 只做小范围 hook：

- 新增 `--input-provider online_inference` 和 online inference CLI 参数。
- 相机初始化使用 `needs_camera = args.record or args.input_provider == "online_inference"`，录制和推理共用同一批 camera objects。
- 每 tick 传入当前 FK wrist pose、state timestamp、arm q/dq、Dex1 gripper width 和 camera sources。
- 使用 provider metadata 的 `enabled_arms` gate active side。
- online session failed sample 会使主循环停机。
- workspace clamp、非 finite IK/control target、joint speed limiter 大幅介入会通过 `report_control_feedback()` 让 session fail-closed。

主循环保留现有核心控制链：

```text
read q/dq
-> FK wrist pose
-> provider.get_sample()
-> deadman / enabled arms
-> gripper routing
-> pose IK or joint target
-> speed limit
-> gravity tauff
-> ctrl_dual_arm()
-> record alignment
```

## Pika Protocol

Observation 沿用 Pika schema，不发本仓库自定义 request：

```json
{
  "type": "observation",
  "arm_l": {
    "images": ["<base64 jpeg>"],
    "poses": [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]],
    "grippers": [0.05],
    "init_pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
    "arm_current_pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
  }
}
```

Action：

```json
{
  "type": "action",
  "action_l": [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.05]]
}
```

四元数顺序固定为 `[qx, qy, qz, qw]`。

## Arm Side Semantics

`left`：

- observation 只发 `arm_l`。
- 只执行左臂/左夹爪 action。
- 右臂和右夹爪保持当前状态。

`right` 同理。

`both`：

- observation 发 `arm_l` 和 `arm_r`。
- action 必须同时有 `action_l` 和 `action_r`。
- 左右臂都可执行。

`MotionIntent(kind="pose")` 仍携带左右 4x4 pose，因为现有数据结构要求双臂 pose；inactive side 使用当前 FK pose，并通过 `enabled_arms` 让主循环 hold 该侧。

## Safety And Failure Handling

以下情况 fail-closed：

- socket 断连或 EOF。
- server response timeout。
- JSON parse error。
- invalid action schema。
- action contains NaN/Inf。
- transform 失败。
- required camera/state/gripper history 不足或覆盖超时。
- action pose 超 workspace 后被 clamp。
- joint speed limiter 大幅介入。
- IK/control target 非 finite。
- both 模式缺任一侧 action。

fail-closed 行为：

- 丢弃当前 chunk。
- 返回 hold sample。
- session status 进入 `failed`。
- provider sample `done=True`。
- 主循环记录 error 并停止控制流程。

## Expert Review Notes

机器人软件专家审阅后，本设计收进了四个修正点：

- `get_sample()/tick()` 不等待网络 recv 或 sleep delay。
- post-action delay 后必须等待新 state/camera 覆盖，而不是单纯定时结束。
- provider 必须保持薄层，协议和 request/session 独立在 utils 模块。
- Dex1 inactive side 要写当前 hold trigger，避免旧 gripper target 残留。

## Non-goals

本阶段不做：

- 复刻旧 C++ ROS2 dataStreamer 节点。
- 新增 websocket/WebRTC/旧 viewer 路线。
- 更新 `RUN.md`。
- 创建 `RUN_ONLINE_INFERENCE.md`。
