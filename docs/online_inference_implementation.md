# Online Inference Implementation

## Scope

本实施文档记录当前 online inference 代码落点、测试顺序和真机前验证路线。

本阶段不写 `RUN_ONLINE_INFERENCE.md`，也不更新 `RUN.md`。运行文档等功能实现和验证稳定后再补。

## Files

## `teleop/utils/inference_protocol.py`

职责：

- TCP newline JSON framing。
- strict JSON encode/decode，拒绝 NaN/Infinity。
- JPEG base64 encoding。
- Pika action chunk parser。
- `left/right/both` schema validation。

测试：

- `tests/test_inference_protocol.py`
- 覆盖 newline JSON、粘包/拆包、reset、invalid action、mixed action keys、both 模式拒绝泛化 `action`。

## `teleop/utils/pose_transform.py`

职责：

- 加载 per-side transform config。
- `action_to_unitree(side, pose7)`。
- `observation_to_server(side, matrix4x4)`。
- 真机 motion 缺 config 或 active side 缺显式 transform 时拒绝启动。

测试：

- `tests/test_pose_transform.py`
- 覆盖 quat xyzw、matrix roundtrip、per-side transform、single-arm config、缺 config enable motion 拒绝。

## `teleop/utils/online_inference.py`

职责：

- 维护 state/camera history。
- 按 `n_obs_steps` 和 `camera_freq` 选择 host monotonic observation window。
- 构造 Pika observation。
- 非阻塞 `tick()` 状态机。
- action chunk cadence 和 interpolation。
- post-action delay + 新 observation 覆盖等待。
- timeout/disconnect/invalid action fail-closed。

关键实现要求：

- `tick()` 不阻塞 socket recv。
- `post_action_delay` 不 sleep。
- history 不足返回 hold。
- session failed 后只返回 hold。
- action payload reset 边界后丢弃旧 chunk。

测试：

- `tests/test_online_inference.py`
- 覆盖 history 对齐、base64 JPEG、chunk cadence、dry-run disabled motion、post-action delay 不阻塞、coverage timeout fail-closed。

## `teleop/utils/teleop_input_provider.py`

职责：

- 新增 `OnlineInferenceInputProvider`。
- 创建 `OnlineInferenceConfig`、`PoseTransformer`、`TcpJsonTransport`、`OnlineInferenceSession`。
- 把主循环传入的 current FK pose、gripper width、camera sources 转成 session samples。
- 将非空 camera source 名字登记成 required cameras。
- 把 session step 转成 `TeleopInputSample/MotionIntent`。
- Dex1 gripper width meter 映射到 trigger `5.0..7.0`。
- session failed 时返回 `done=True`。

测试：

- `tests/test_teleop_input_provider.py`
- 覆盖 left/right/both、inactive pose hold、dry-run 不输出 motion、unsupported gripper 拒绝、failed session done、required camera names。

## `teleop/real/teleop_hand_and_arm.py`

最小 hook：

- CLI 增加：
  - `--input-provider online_inference`
  - `--online-inference-host`
  - `--online-inference-port`
  - `--online-inference-arm-side left|right|both`
  - `--online-inference-n-obs-steps`
  - `--online-inference-camera-freq`
  - `--online-inference-jpeg-quality`
  - `--online-inference-action-step-sec`
  - `--online-inference-interp-sec`
  - `--online-inference-post-action-delay-ms`
  - `--online-inference-response-timeout-sec`
  - `--online-inference-transform-config`
  - `--online-inference-enable-motion`
  - `--online-inference-dry-run`
- `needs_camera = args.record or args.input_provider == "online_inference"`。
- 每 tick 传 provider 所需的 current state/camera/gripper snapshot。
- `enabled_arms` metadata 参与 deadman gate。
- provider failed sample 使主循环停机。
- Dex1 online inactive side 写当前 gripper state 对应的 hold trigger。
- control feedback 上报 session fatal。

## TDD / Verification Order

1. Protocol tests:
   `python -m unittest tests.test_inference_protocol`
2. Pose transform tests:
   `python -m unittest tests.test_pose_transform`
3. Session tests:
   `python -m unittest tests.test_online_inference`
4. Provider tests:
   `python -m unittest tests.test_teleop_input_provider`
5. Combined targeted suite:
   `python -m unittest tests.test_inference_protocol tests.test_pose_transform tests.test_online_inference tests.test_teleop_input_provider`
6. Compile check:
   `python -m py_compile inference/transport.py inference/pose_transform.py inference/online_session.py core/input/teleop_input_provider.py teleop/real/teleop_hand_and_arm.py`

当前开发环境使用：

```bash
/home/luopengcheng/miniconda3/envs/tv/bin/python
```

系统 `python3` 可能缺 `numpy/cv2`。

## Mock Server Verification

真机前建议先做本地 fake Pika server：

1. 接收一条 reset。
2. 接收 observation，检查 `arm_l/arm_r`、`images`、`poses`、`grippers`。
3. 返回固定 action chunk。
4. dry-run 下确认主循环不阻塞，session 状态按 `waiting_action -> executing_chunk -> post_action_delay` 推进。
5. 注入 invalid action、断线、timeout，确认 fail-closed。

## Robot Validation Order

1. mock server + dry-run。
2. MuJoCo 或无 DDS publish。
3. 单臂、无夹爪、单 step、低速。
4. 单臂、多 step。
5. 开启 Dex1 gripper。
6. both 双臂。

真机 motion 必须同时满足：

- `--online-inference-enable-motion`
- transform config 有效
- protocol action 合法
- session 正常
- 主循环安全检查未触发 fatal feedback

## Remaining Runtime Notes

- `TcpJsonTransport.connect()` 当前在 provider 创建阶段执行；`get_sample()/tick()` 的网络 recv/send 是非阻塞的。若未来希望 server 可晚启动，需要把 connect 也改成 lazy/nonblocking session 状态。
- docs 目录在当前工作树可能被本地 exclude 忽略，提交时需要确认是否 force add。
- `RUN_ONLINE_INFERENCE.md` 等 mock server 和真机前验证后再写。
