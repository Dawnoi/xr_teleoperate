# `teleop/real/teleop_hand_and_arm.py` 程序流程梳理

> 文件：`teleop/real/teleop_hand_and_arm.py`  
> 当前规模：约 725 行
> 定位：真机遥操唯一入口。它现在主要负责流程编排；DDS/硬件装配、operator 状态、EE 命令、arm/base command、recording 已按职责下沉到对应模块。

## 1. 总体职责

这个文件承担一条实时控制链：

```text
键盘启停
  -> 初始化 DDS / arm / IK / hand / camera / recorder / input provider
  -> 等待 START
  -> 高频主循环
      读取机器人状态
      读取输入源 sample
      处理 operator/runtime 命令
      deadman / home / takeover 安全逻辑
      夹爪控制
      底盘控制
      手腕目标 -> IK / joint target
      workspace clamp / speed limit / gravity compensation
      下发 arm_ctrl.ctrl_dual_arm
      可选 record 对齐写入
  -> finally 安全回 home、关闭线程和资源
```

它不再承载所有实现细节：`if __name__ == '__main__':` 仍是唯一真机入口，但主要只调各职责模块暴露的接口。

## 2. 关键依赖分层

| 依赖 | 来源 | 职责 |
|---|---|---|
| `setup_real_teleop_components` | `teleop.real.setup` | DDS/base/arm/EE/camera/record/latency 启动装配 |
| `G1_29_ArmController` 等 | `teleop.robot_control.robot_arm` | DDS arm 下发、状态读取、go home |
| `G1_29_ArmIK` 等 | `teleop.robot_control.robot_arm_ik` | wrist pose -> 14 维 arm q / tauff |
| `create_teleop_input_provider` | `core.input.teleop_input_provider` | 创建 XR / replay / online inference 输入源 |
| `OperatorRuntime` | `teleop.runtime.*` | 键盘/手柄快捷键、home/record 命令 |
| `OperatorStateFlow` | `teleop.control_flow.operator_state` | home/deadman/provider enabled/takeover 状态 |
| `apply_end_effector_command` | `teleop.control_flow.end_effector_command` | dex1/dex3/inspire/brainco 命令映射 |
| `apply_base_command` | `teleop.control_flow.base_command` | 底盘命令流水线 |
| `build_arm_command` | `teleop.control_flow.arm_command_pipeline` | workspace clamp / IK / 限速 / 重力补偿 |
| `TeleopRecordingFlow` | `data_pipeline.recording.teleop_recording_flow` | 录制 episode 生命周期、相机对齐、样本写入 |
| `alignment` helpers | `data_pipeline.recording.alignment` | 录制时 state/action/camera 时间戳对齐 |
| `SimpleLatencyTracker`, `TimingDebugger` | `teleop.debug.*` | 运行时调试和 latency trace |

## 3. 全局状态

文件顶部维护几个全局状态，主要被键盘回调和主循环共享：

| 变量 | 含义 |
|---|---|
| `START` | 机器人开始跟随输入源 |
| `STOP` | 请求退出 |
| `READY` | 当前是否允许启动/录制 |
| `RECORD_RUNNING` | 当前是否正在录制 |
| `RECORD_TOGGLE` | 录制开始/保存切换请求 |
| `RECORD_CANCEL` | 取消当前 episode 请求 |
| `RECENTER` | 重新标定 head reference 请求 |
| `operator_runtime` | 运行时命令处理器，全局给 `on_press()` 使用 |

## 4. 顶层函数定义

`teleop_hand_and_arm.py` 只保留入口必须的少量 helper：

| 函数 | 行为 | 主要调用方 |
|---|---|---|
| `on_press(key)` | 键盘命令入口：`r/q/s/v/h/c` | `sshkeyboard` |
| `reset_arm_ik_state(arm_ik, arm_q)` | 重置 IK 初值和 smooth filter | home 后、takeover 边沿 |
| `get_robot_wrist_poses(arm_ik, arm_q)` | 用 Pinocchio FK 从 q 算左右 wrist 4x4 pose | 输入 provider anchor、record pose |
| `compute_arm_gravity_tauff()` | 用 Pinocchio RNEA 计算当前 q 的重力补偿，失败回退 0 | arm command pipeline |
| `start_keyboard_listener()` | 启动键盘监听线程 | 入口启动阶段 |
| `cleanup_real_teleop_resources()` | 退出时统一回 home、停线程、关闭 provider/AGV/camera/recorder | `finally` |

## 5. 已下沉的职责模块

| 模块 | 职责 |
|---|---|
| `teleop/real/setup.py` | 真机启动装配，创建硬件控制器、相机、录制、latency tracker |
| `teleop/control_flow/operator_state.py` | 操作员状态、home、deadman、takeover edge |
| `teleop/control_flow/end_effector_command.py` | 夹爪/灵巧手命令下发 |
| `teleop/control_flow/base_command.py` | 底盘命令 |
| `teleop/control_flow/arm_command_pipeline.py` | 机械臂目标求解、安全限幅、重力补偿 |
| `data_pipeline/recording/teleop_recording_flow.py` | recording 命令、首帧等待、pending sample、对齐写入 |
| `data_pipeline/recording/alignment.py` | 时间戳对齐基础工具 |

## 6. CLI 参数分组

### 6.1 基础控制

- `--frequency`
- `--input-mode hand|controller`
- `--arm G1_29|G1_23|H1_2|H1|H2`
- `--ee dex1|dex3|inspire_ftp|inspire_dfx|brainco`
- `--network-interface`
- `--controller-deadman grip|none`
- `--max-arm-joint-speed`
- `--home-return-speed`

### 6.2 底盘控制

- `--motion`
- `--base-controller none|g1d_agv`
- `--base-max-vx/vy/wz/z`
- `--base-stick-deadzone`

注意：`--motion` 和 `--base-controller g1d_agv` 互斥。

### 6.3 controller/head reference 映射

- `--head-reference-mode`
- `--controller-orientation-mode`
- `--controller-mapping-mode`
- `--calibration-mode manual|auto`

### 6.4 输入 provider

- `--input-provider xr|lerobot_offline|online_inference`
- offline replay：`--offline-replay-*`
- online inference：`--online-inference-*`

### 6.5 安全、debug、latency

- `--disable-arm-workspace-limit`
- `--arm-workspace-*`
- `--timing-debug`
- `--latency-trace`

### 6.6 运行模式和录制

- `--sim`
- `--headless`
- `--affinity`
- `--no-gripper`
- `--record`
- `--record-arm-repr qpos|pose|both`
- `--task-*`
- local camera / ZMQ camera 参数

## 7. 启动阶段流程

```text
parse args
  -> OperatorRuntime(record_enabled=args.record)
  -> normalize head_reference_mode
  -> workspace 参数准备
  -> TimingDebugger / state/action history 初始化
  -> 校验互斥参数和 offline dataset
  -> initialize_dds(domain)
  -> keyboard listener
  -> workspace/timing 日志
  -> setup_real_teleop_components()
      -> motion/debug/base 初始化
      -> arm IK + arm controller 初始化
      -> input provider 初始化
      -> end-effector controller 初始化
      -> affinity 可选设置
      -> sim reset publisher/subscriber 可选初始化
      -> recorder/camera 初始化
      -> recording_flow 初始化
      -> latency tracker 可选初始化
  -> arm go home
  -> READY=True，等待 START 或 auto-start
```

## 8. 输入 provider 语义

主循环只依赖统一接口：

```python
sample = tv_wrapper.get_sample(...)
tele_data = sample.tele_data
motion_intent = sample.motion_intent
```

不同 provider 的区别在 `core/input`：

| provider | 输入来源 | 输出语义 |
|---|---|---|
| `xr` | XR controller / hand / tracker | 实时 pose / hand / button |
| `lerobot_offline` | 离线 episode | 回放 `MotionIntent` |
| `online_inference` | 相机 + robot state -> 策略服务 | 策略 action chunk 转 `MotionIntent` |

主程序不直接关心具体 provider，只处理统一的 `tele_data + motion_intent`。

## 9. 主循环核心流程

### 9.1 record 命令处理

每帧开始先通过 `recording_flow.handle_commands(...)` 处理：

- `RECORD_CANCEL`：取消 episode，清空 pending 队列
- `RECORD_TOGGLE`：
  - 未录制：`recorder.create_episode()`，等待首个 post-start camera frame
  - 已录制：`recorder.save_episode()`，仿真时发布 reset

样本写入通过 `recording_flow.process_frame(...)` 完成，主循环不再直接拼 recording payload。

### 9.2 head reference calibration

如果 `head_reference_mode` 需要标定：

- manual：按 `c` 后调用 `tv_wrapper.calibrate_head_reference(require_live=True)`
- auto：启动后自动等待 live pose
- 未标定前持续 hold 当前 arm q，不进入控制

### 9.3 读取机器人状态

```text
arm_ctrl.get_current_dual_arm_q()
arm_ctrl.get_current_dual_arm_dq()
append state_history
get_robot_wrist_poses()
```

这些状态用于：

- XR grip anchor 当前腕位姿
- online inference observation
- latency trace 执行检测
- recording 对齐

### 9.4 获取输入 sample

传给 provider 的上下文包括：

- 当前左右 wrist pose
- 当前 q/dq
- dex1 gripper 当前值
- camera_sources
- `dt`

如果 provider 返回 `None`：

- replay done：设置 `STOP=True`
- online inference fail-closed：设置 `STOP=True`
- 否则 sleep 后下一帧

### 9.5 runtime/operator 处理

```text
operator_runtime.apply_to_tele_data()
  -> 手柄快捷键映射 record/home
  -> keyboard home request 注入 left Y
```

随后处理：

- home button rising edge
- controller deadman grip
- provider `enabled_arms`
- home 后等待 grip release
- takeover rising edge 零增量 settle frames

### 9.6 夹爪/手控制

按 `ee` 和 `input_mode` 分支：

| 条件 | 动作 |
|---|---|
| dex3/inspire/brainco + hand | 写入左右 hand pose array |
| dex1 + controller | 写入 trigger value |
| dex1 + hand | 写入 pinch value |
| online inference + dex1 | 未使能侧保持当前 gripper 状态 |

### 9.7 底盘控制

两条路径：

| 模式 | 条件 | 行为 |
|---|---|---|
| `loco` | `--motion` + controller | `LocoClientWrapper.Move(vx,vy,wz)` |
| `g1d_agv_async` | `--base-controller g1d_agv` | `G1DAgvBridge.set_target(vx,vy,wz,z)` |

home return 时底盘命令清零。

### 9.8 arm 目标生成

根据 `motion_intent.kind`：

| kind | 处理 |
|---|---|
| `pose` | 左右 wrist pose -> workspace clamp -> `arm_ik.solve_ik()` |
| `joint_position` | 直接使用 `motion_intent.arm_q` |
| `joint_velocity` | `current_q + dq * dt` |

如果左右 arm 未使能，会用 `current_hold_q/current_hold_tauff` 覆盖对应 7 维。

### 9.9 安全限幅和下发

```text
limit_arm_joint_target_velocity()
  -> online inference 可反馈超速 fatal
compute_arm_gravity_tauff(sol_q)
  -> current_hold_q/current_hold_tauff 更新
latency trace 可选 begin_trace
append action_history
arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
```

在线推理场景下，如果 workspace clamp、非 finite、或速度限幅触发，会：

- report feedback 给 provider
- hold 当前姿态
- 禁用左右 arm 本帧运动

### 9.10 recording 对齐写入

录制不是直接“当前帧立即写”，而是按主相机帧时间戳对齐：

```text
primary camera latest frame
  -> enqueue pending_record_samples
  -> 等 state_history/action_history 覆盖该时间戳
  -> strict linear interpolation state/action
  -> 找其它相机 nearest frame
  -> 组 states/actions/timestamps/control_extras
  -> recorder.add_item()
```

关键点：

- 以第一个可用 camera 作为 primary source
- 首帧必须在 `record_start_monotonic_ns` 之后
- state/action 优先严格插值，超时后 fallback 最近邻或丢帧
- 可记录 arm `qpos`、`pose` 或 `both`
- `control_extras` 写入 `arm_tauff`

## 10. 退出流程

`finally` 确保资源释放：

```text
arm_ctrl.ctrl_dual_arm_go_home()
  -> hold home 若干秒
  -> stop keyboard listener
  -> close input provider
  -> close G1D AGV bridge
  -> stop sim subscriber
  -> close cameras
  -> close recorder
```

`exit_go_home` 在 offline replay 结束时会根据 `--offline-replay-end-action` 决定是否回 home。

## 11. 当前实现特点

### 好的部分

- 输入源已经统一成 provider + `MotionIntent`，主循环不用关心 XR/replay/inference 细节。
- arm 控制前有 workspace clamp + joint velocity limit。
- online inference 有 fail-closed 反馈机制。
- recording 做了 state/action/camera 时间戳对齐，不是简单按循环频率硬写。
- `finally` 有较完整资源关闭和回 home。

### 主要问题

- `main` 太大：CLI、初始化、控制循环、recording 对齐、底盘、夹爪都在一个文件里。
- 内部 helper 只服务本文件，但被嵌在 main 中，不利于单测。
- record 对齐逻辑和实时控制逻辑耦合在同一个循环尾部。
- `START/STOP/RECORD_*` 是全局变量，键盘/主循环共享，后续并发调试成本高。
- `arm_ctrl`、`gripper_ctrl`、`recorder`、`camera` 等资源生命周期分散在大 try/finally 中。

## 12. 如果后续继续拆，建议顺序

不要一次拆完，按低风险切：

1. CLI 参数定义已抽到 `teleop/real/args.py`。
2. record 时间戳对齐 helper 已抽到 `data_pipeline/recording/alignment.py`。
3. arm 目标生成已抽成 `teleop/control_flow/arm_command_pipeline.py`。
4. base control 已抽成 `teleop/control_flow/base_command.py`。
5. 下一步可以把资源初始化拆成 `teleop/real/session.py`，或继续拆 recording 写入段。

每一步都保持 `teleop/real/teleop_hand_and_arm.py` 仍是唯一真机入口。
