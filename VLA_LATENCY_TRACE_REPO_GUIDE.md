# VLA 延迟 Trace 分支仓库导览

本文档对应分支：

```text
feature/vla-latency-trace
```

本文档的目标是帮助理解整个仓库的模块边界、真机运行链路、数据采集链路和常用命令。当前分支额外强化了 VLA 延迟 trace，所以文档会把延迟分析链路单独展开。

理解仓库时先抓住这条主线：

```text
输入源
-> 统一 provider
-> 真机主循环
-> 控制流水线
-> 硬件控制器
-> 录制 / trace / 数据审查工具
```

---

## 0. 简洁版

这个仓库可以按五层理解：

```text
输入层
  XR / 离线回放 / online inference
  位置：core/input/、inference/

真机主循环
  每 tick 读取状态、取输入、跑状态机、生成命令、记录数据
  位置：teleop/real/teleop_hand_and_arm.py

控制流水线
  deadman / takeover / 底盘 / 夹爪 / IK / workspace / 关节限速 / 重力补偿
  位置：teleop/control_flow/

硬件控制层
  机械臂 DDS 发布线程、lowstate 订阅线程、Dex1 夹爪控制
  位置：teleop/robot_control/

数据和诊断层
  episode 录制、Rerun、数据审查、latency trace 画图
  位置：data_pipeline/、teleop/debug/、tests/diagnostics/
```

真机 VLA 最短链路：

```text
online inference 输出 action step
      |
      v
主循环拿到 MotionIntent
      |
      v
build_arm_command
      |
      | workspace
      | IK
      | safety
      | gravity
      v
ctrl_dual_arm 写给控制线程
      |
      v
控制线程 DDS Write
      |
      v
lowstate 反馈检测到运动
```

最常用命令：

```bash
# VLA dry-run
VLA_ENABLE_MOTION=0 \
VLA_BASE_URL=http://127.0.0.1:18027 \
SENDER_IP=192.168.123.164 \
bash scripts/start/start_real_robot_vla.sh

# VLA 真机运动 + latency trace
VLA_ENABLE_MOTION=1 \
VLA_BASE_URL=http://127.0.0.1:18027 \
SENDER_IP=192.168.123.164 \
ARM_CONTROL_HZ=250 \
bash scripts/start/start_real_robot_vla.sh \
  --latency-trace \
  --latency-trace-path ./utils/data/latency_trace.jsonl \
  --latency-summary-every 10

# latency trace 画图
python tests/diagnostics/plot_latency_trace.py \
  utils/data/latency_trace.jsonl \
  --output utils/data/latency_trace.png \
  --top-report utils/data/latency_trace_top.md

# 三相机 ZMQ 录制
SENDER_IP=192.168.123.164 \
TASK_DIR=./utils/data \
TASK_NAME=multi_cam_record \
RECORD_ARM_REPR=both \
bash scripts/start/start_real_robot_wired_3cams_zmq.sh

# raw 多相机数据审查
DATASET=utils/data/multi_cam_record/transfer_black \
EPISODE_START=207 \
EPISODE_END=239 \
bash scripts/data/audit_multi_cam_record.sh
```

延迟分析先看这几个字段：

```text
online_obs_send_to_action_recv_ms
  发观测到收到动作块，主要是网络/模型推理。

recv_to_pub_ms
  provider 返回动作点到 DDS 发出，主要是主循环和控制线程等待。

pub_to_exec_thread_ms
  DDS 发出到 lowstate 线程检测运动，主要是底层执行反馈。

recv_to_exec_thread_ms
  provider 返回动作点到 lowstate 线程检测运动，是端到端执行反馈。

unknown_pre_pub
  发布前还没拆明白的耗时。
```

---

## 1. 详细版：仓库模块

```text
xr_teleoperate/
  teleop/real/                  真机入口、参数、硬件装配
  teleop/control_flow/          主循环每 tick 的控制流水线
  teleop/robot_control/         机械臂、IK、夹爪、DDS 控制器
  teleop/debug/                 latency trace、timing、夹爪调试工具
  core/input/                   XR / 离线回放 / 在线推理输入 provider
  core/control/                 workspace、安全限速、底盘 bridge
  core/camera/                  本地相机读取
  inference/                    在线推理协议、HTTP/TCP transport、pose transform
  data_pipeline/recording/      episode 录制、时间对齐、Rerun
  data_pipeline/replay/         raw / LeRobot 回放
  data_pipeline/export/         数据格式导出
  data_pipeline/audit/          数据质量审查
  scripts/start/                真机启动脚本
  scripts/debug/                调试脚本
  scripts/data/                 数据审查/导出脚本
  tests/diagnostics/            trace 画图、DDS 诊断脚本
```

模块边界：

```text
teleop/real/
  真机入口和启动装配。

teleop/control_flow/
  主循环每 tick 的控制逻辑，包含状态机、底盘、夹爪、机械臂命令生成。

teleop/robot_control/
  和机器人硬件/DDS 交互，包含机械臂控制线程和 lowstate 订阅线程。

core/input/
  把 XR、离线回放、online inference 统一成 TeleopInputSample。

inference/
  只负责推理服务通信、协议解析和坐标变换，不直接控制机器人。

data_pipeline/
  录制、回放、导出、数据审查。

tests/diagnostics/
  运行后分析工具，例如 latency trace 画图。
```

---

## 2. 真机主流程

唯一真机 Python 入口：

```text
teleop/real/teleop_hand_and_arm.py
```

启动装配：

```text
teleop/real/setup.py
```

参数定义：

```text
teleop/real/args.py
```

主循环简化图：

```text
读取机器人 lowstate
      |
      v
计算当前左右腕 FK pose
      |
      v
tv_wrapper.get_sample()
      |
      +-- XR 输入
      +-- LeRobot / raw episode 离线回放
      +-- online_inference 在线推理
      |
      v
OperatorStateFlow
      |
      | deadman / grip / home / takeover
      v
夹爪/灵巧手命令
      |
      v
底盘命令
      |
      v
build_arm_command
      |
      | workspace 限制
      | IK
      | 左右臂 enable gating
      | 关节限速
      | 重力补偿
      v
arm_ctrl.ctrl_dual_arm()
      |
      v
机械臂控制线程 DDS Write
      |
      v
lowstate 线程检测实际运动
      |
      v
latency trace 记录完成
```

关键文件：

```text
teleop/control_flow/operator_state.py
teleop/control_flow/end_effector_command.py
teleop/control_flow/base_command.py
teleop/control_flow/arm_command_pipeline.py
teleop/robot_control/robot_arm.py
teleop/debug/latency_trace.py
```

---

## 3. 输入 Provider

统一创建入口：

```text
core/input/teleop_input_provider.py
```

支持三类输入：

```text
--input-provider xr
--input-provider lerobot_offline
--input-provider online_inference
```

### 3.1 XR 输入

对应文件：

```text
core/input/xr_provider.py
core/input/xr_robotics_wrapper.py
```

输出统一结构：

```text
TeleopInputSample
  tele_data
  motion_intent
```

### 3.2 离线回放

对应文件：

```text
core/input/lerobot_offline.py
core/input/raw_offline.py
data_pipeline/replay/
```

常用参数：

```text
--input-provider lerobot_offline
--offline-replay-dataset-root <dataset_root>
--offline-replay-episode-index <episode_id>
--offline-replay-arm-source action | state | fk_cmd_pose
--offline-replay-speed-scale 1.0
--offline-replay-end-action home | hold
```

### 3.3 在线推理

对应文件：

```text
core/input/online_inference_provider.py
inference/online_session.py
inference/transport.py
inference/pi05_protocol.py
inference/pose_transform.py
```

状态机：

```text
collecting_observation
  缓存 state / camera，直到覆盖 n_obs_steps。

waiting_action
  发出 observation 后等待服务端动作块。

executing_chunk
  执行动作块。

post_action_delay
  一个动作块执行完，等待新 state / camera 覆盖后再发下一次 observation。
```

当前分支默认：

```text
--online-inference-chunk-step-mode per_tick
```

含义：

```text
主循环每 tick 消费 1 个 action step。
```

不是等待 `action_step_sec` 到点后才消费下一个 step。

---

## 4. 两个频率

### 4.1 `--frequency`

含义：

```text
主循环频率
```

位置：

```text
teleop/real/args.py
teleop/real/teleop_hand_and_arm.py
```

影响：

```text
主循环多久读取一次输入
per_tick 模式下多久消费一个 action step
关节限速每 tick 允许变化量
```

关节限速公式：

```text
每 tick 最大变化量 = max_arm_joint_speed / frequency
```

例子：

```text
max_arm_joint_speed = 1.0 rad/s
frequency = 30 Hz
每 tick 最大变化量约 0.033 rad
```

### 4.2 `--arm-control-hz`

含义：

```text
机械臂 DDS 发布线程频率
```

位置：

```text
teleop/robot_control/robot_arm.py
```

影响：

```text
主循环 ctrl_dual_arm() 写入目标后，控制线程多久能取到并 DDS Write
latency trace 里的 controller_wait_ms / enqueue_to_publish_ms
```

默认值：

```text
--frequency 30
--arm-control-hz 250
```

---

## 5. Latency Trace

开启参数：

```text
--latency-trace
--latency-trace-path ./utils/data/latency_trace.jsonl
--latency-command-threshold 0.02
--latency-exec-q-threshold 0.01
--latency-exec-dq-threshold 0.05
--latency-timeout 2.0
--latency-summary-every 10
```

核心文件：

```text
teleop/debug/latency_trace.py
teleop/robot_control/robot_arm.py
tests/diagnostics/plot_latency_trace.py
```

Trace 起点：

```text
tele_data_recv_ts_ns
```

含义：

```text
输入 provider 返回当前动作点 / XR sample 的时刻。
```

DDS 发布点：

```text
机械臂控制线程执行 lowcmd_publisher.Write() 完成的时刻。
```

执行检测有两个来源：

```text
lowstate thread
  robot_arm.py 的 _subscribe_motor_state() 收到 lowstate 后检测 q_delta / dq_peak。

main loop
  teleop_hand_and_arm.py 下一轮主循环读取 current_lr_arm_q / dq 后检测。
```

优先看 lowstate thread 字段：

```text
pub_to_exec_thread_ms
recv_to_exec_thread_ms
```

main loop 字段会额外受 `--frequency` 影响：

```text
pub_to_exec_ms
recv_to_exec_ms
```

---

## 6. Trace 字段中文解释

### 6.1 顶层延迟

```text
recv_to_pub_ms
  provider 返回动作点 -> DDS 发布完成

pub_to_exec_thread_ms
  DDS 发布完成 -> lowstate 线程检测到运动

recv_to_exec_thread_ms
  provider 返回动作点 -> lowstate 线程检测到运动

pub_to_exec_ms
  DDS 发布完成 -> 主循环检测到运动

recv_to_exec_ms
  provider 返回动作点 -> 主循环检测到运动

fetch_to_exec_ms
  tele_fetch + recv_to_exec_ms
```

### 6.2 online inference 字段

```text
online_obs_send_to_action_recv_ms
  发观测 -> 收到动作块

online_step_output_to_provider_return_ms
  session 输出 action step -> provider 返回 TeleData

online_provider_return_to_pub_ms
  provider 返回 TeleData -> DDS 发布

online_step_output_to_pub_ms
  action step 输出 -> DDS 发布

online_step_output_to_exec_thread_ms
  action step 输出 -> lowstate 线程检测到运动

online_obs_send_to_exec_thread_ms
  发观测 -> lowstate 线程检测到运动
```

### 6.3 主循环拆分字段

```text
tele_fetch_ms
  tv_wrapper.get_sample() 耗时

takeover_logic_ms
  deadman / takeover / home 状态处理

end_effector_command_ms
  夹爪/灵巧手命令写入

base_control_ms
  底盘命令处理

ik_ms
  IK 求解

safety_ms
  关节限速

gravity_ms
  重力补偿

ctrl_dual_arm_call_ms
  主循环写目标给控制线程

controller_wait_ms
  控制线程等待取到新命令的时间

dds_write_ms
  DDS Write 本身耗时

unknown_pre_pub
  recv_to_pub_ms 中尚未被已知字段解释的剩余耗时
```

### 6.4 `build_arm_command` 内部拆分

```text
arm_cmd_input_ms
  输入数组转换 / 基础准备

arm_cmd_takeover_reset_ms
  takeover 时重置 IK / 参考

arm_cmd_target_extra_ms
  除 IK 以外的目标生成耗时

arm_cmd_feedback_gate_ms
  online inference fatal feedback 后禁用运动

arm_cmd_enable_gating_ms
  左右臂 enable gating / hold

arm_cmd_takeover_settle_ms
  takeover settle 帧计数

arm_cmd_speed_feedback_ms
  online inference 速度限制反馈检查

arm_cmd_hold_update_ms
  current_hold_q / current_hold_tauff 更新
```

### 6.5 按拆分标注的时间轴图

下面这张图按 trace 字段标出一次动作从输入返回到机器人反馈的路径。

```text
tele_fetch 开始
      | 代码位置：主循环调用 tv_wrapper.get_sample() 之前
      | 含义：准备从输入源取这一 tick 的输入
      |
      | tele_fetch_ms
      | 中文：取输入耗时
      | 包含：
      |   - XR: 读取 XR 当前手柄/手部/头显状态
      |   - 离线回放: 取 episode 下一帧
      |   - online_inference: session.tick()，可能是取动作点、等动作块、构造观测
      v
t_recv: provider 返回动作点 / XR sample
      | 代码位置：tv_wrapper.get_sample() 返回之后
      | 含义：主循环已经拿到这一 tick 要处理的目标
      | 注意：对 online_inference 来说，这里通常是“一个 action step 已经输出”
      |
      | takeover_logic_ms
      | 中文：接管/状态机耗时
      | 内容：deadman、grip rising edge、home、recenter、takeover settle
      |
      | end_effector_command_ms
      | 中文：夹爪/灵巧手命令耗时
      | 内容：把 tele_data 里的手部/夹爪目标写到共享变量或控制器
      |
      | base_control_ms
      | 中文：底盘命令耗时
      | 内容：把摇杆/输入转成 G1D AGV 或 loco 命令
      |
      | build_arm_command:
      | 中文：机械臂命令生成总块
      | 输入：motion_intent + 当前机器人关节状态
      | 输出：安全后的 sol_q / sol_tauff
      |
      |   arm_cmd_input_ms
      |   中文：输入数组转换和基础准备
      |
      |   arm_cmd_takeover_reset_ms
      |   中文：接管时重置 IK 初值/头参考
      |
      |   arm_cmd_target_extra_ms
      |   中文：除 IK 以外的目标生成耗时
      |
      |   ik_ms
      |   中文：逆解耗时，把左右腕目标 pose 解成 14 维关节目标
      |
      |   arm_cmd_feedback_gate_ms
      |   中文：online inference 安全反馈触发后的禁用逻辑
      |
      |   arm_cmd_enable_gating_ms
      |   中文：左右臂 enable gating，没使能的手臂保持 current_hold_q
      |
      |   arm_cmd_takeover_settle_ms
      |   中文：接管 settle 帧计数
      |
      |   safety_ms
      |   中文：关节速度限制耗时
      |   公式：每 tick 最大关节变化 = max_arm_joint_speed / frequency
      |
      |   arm_cmd_speed_feedback_ms
      |   中文：online inference 速度限制反馈检查
      |
      |   gravity_ms
      |   中文：重力补偿计算耗时
      |
      |   arm_cmd_hold_update_ms
      |   中文：更新 current_hold_q / current_hold_tauff
      |
      | operator_sync_ms
      | 中文：把 arm_command 的 settle 状态同步回 operator_state_flow
      |
      | provider_feedback_ms
      | 中文：把 workspace / speed limit 等 fatal feedback 回报给 online inference provider
      |
      | latency_trace_prepare_ms
      | 中文：准备 trace 记录字段
      |
      | action_history_append_ms
      | 中文：把本 tick 下发动作写入 action_history，供录制对齐使用
      |
      | ctrl_dual_arm_call_ms
      | 中文：主循环调用 arm_ctrl.ctrl_dual_arm() 的耗时
      | 注意：这里只是写 q_target / tauff_target 给控制线程，不等 DDS 真正发完
      |
      v
t_set: 主循环把 q_target / tauff_target 写给机械臂控制线程
      | 代码位置：arm_ctrl.ctrl_dual_arm()
      | 含义：目标已经进入机械臂控制器对象，等待 DDS 发布线程取走
      |
      | controller_wait_ms
      | 中文：控制线程等待时间
      | 含义：主循环写入目标后，DDS 发布线程还没运行到下一次取目标
      | 主要受 arm-control-hz 和线程调度影响
      |
      | dds_write_ms
      | 中文：DDS Write 本身耗时
      | 含义：lowcmd_publisher.Write() 调用消耗的时间
      |
      v
t_pub: DDS 发布完成
      | 代码位置：机械臂控制线程 lowcmd_publisher.Write() 返回后
      | 含义：命令已经从本进程写到 DDS
      |
      | pub_to_exec_thread_ms
      | 中文：DDS 发出后，到 lowstate 线程检测到运动的时间
      | 含义：更接近机器人底层执行反馈延迟
      |
      v
t_exec_thread: lowstate 线程检测到机器人开始动
      | 代码位置：robot_arm.py 的 _subscribe_motor_state()
      | 判断：q_delta 或 dq_peak 超过阈值
      | 优点：不需要等主循环下一 tick，受 frequency 影响较小
      |
      | 主循环下一次读取 lowstate 的额外等待
      | 中文：lowstate 已经变了，但主循环还没读到
      | 主要受主循环 frequency 影响
      |
      v
t_exec_main: 主循环检测到机器人开始动
      | 代码位置：teleop_hand_and_arm.py 主循环开头 maybe_mark_execute()
      | 判断：q_delta 或 dq_peak 超过阈值
      | 注意：这个时间通常会比 t_exec_thread 晚
```

对应总量关系：

```text
recv_to_pub_ms
  = t_recv -> t_pub

pub_to_exec_thread_ms
  = t_pub -> t_exec_thread

recv_to_exec_thread_ms
  = t_recv -> t_exec_thread

pub_to_exec_ms
  = t_pub -> t_exec_main

recv_to_exec_ms
  = t_recv -> t_exec_main
```

`unknown_pre_pub` 的含义：

```text
unknown_pre_pub
  = recv_to_pub_ms
    - 已统计出来的主循环 / 控制线程发布前字段
```

所以当 `unknown_pre_pub` 偏大时，说明发布前还有未拆出来的主循环耗时或时间戳覆盖范围不够细。

---

## 7. VLA 真机启动命令

脚本：

```text
scripts/start/start_real_robot_vla.sh
```

dry-run：

```bash
VLA_ENABLE_MOTION=0 \
VLA_BASE_URL=http://127.0.0.1:18027 \
SENDER_IP=192.168.123.164 \
bash scripts/start/start_real_robot_vla.sh
```

真机运动：

```bash
VLA_ENABLE_MOTION=1 \
VLA_BASE_URL=http://127.0.0.1:18027 \
SENDER_IP=192.168.123.164 \
bash scripts/start/start_real_robot_vla.sh
```

开启延迟 trace：

```bash
VLA_ENABLE_MOTION=1 \
VLA_BASE_URL=http://127.0.0.1:18027 \
SENDER_IP=192.168.123.164 \
ARM_CONTROL_HZ=250 \
bash scripts/start/start_real_robot_vla.sh \
  --latency-trace \
  --latency-trace-path ./utils/data/latency_trace.jsonl \
  --latency-summary-every 10
```

### 7.1 环境变量

```text
CONDA_ENV
  shell 脚本激活的 conda 环境，默认 tv

NETWORK_INTERFACE
  DDS 网卡，默认 enx9c69d3212b05

SENDER_IP
  三路 ZMQ 相机 sender IP，默认 192.168.123.164

HEAD_ZMQ_PORT / LEFT_WRIST_ZMQ_PORT / RIGHT_WRIST_ZMQ_PORT
  三路相机端口，默认 5556 / 5557 / 5558

VLA_TRANSPORT
  推理传输，默认 http

VLA_BASE_URL
  推理服务地址，默认 http://127.0.0.1:18027

VLA_PROTOCOL_PROFILE
  协议 profile，默认 pi05_dual_arm_20d

VLA_PROMPT
  发送给 VLA 的任务文本

VLA_ARM_SIDE
  both / left / right，默认 both

VLA_ENABLE_MOTION
  0 = dry-run，只推理不动机器人
  1 = 允许真机运动

VLA_AUTO_START
  0 = 启动后按 r
  1 = 自动进入 START

ARM_CONTROL_HZ
  机械臂 DDS 发布频率，默认 250

MAX_ARM_JOINT_SPEED
  主循环外层关节限速，默认 1.0
```

### 7.2 脚本实际拼出的关键参数

```text
--input-provider online_inference
--input-mode controller
--arm G1_29
--ee dex1
--network-interface ${NETWORK_INTERFACE}
--base-controller g1d_agv
--controller-deadman grip
--head-reference-mode fixed_per_grip
--controller-mapping-mode anchored_safe
--controller-orientation-mode relative
--arm-control-hz ${ARM_CONTROL_HZ}
--max-arm-joint-speed ${MAX_ARM_JOINT_SPEED}
--arm-workspace-mode tapered
--head-zmq-endpoint tcp://${SENDER_IP}:${HEAD_ZMQ_PORT}
--left-zmq-endpoint tcp://${SENDER_IP}:${LEFT_WRIST_ZMQ_PORT}
--right-zmq-endpoint tcp://${SENDER_IP}:${RIGHT_WRIST_ZMQ_PORT}
--online-inference-transport ${VLA_TRANSPORT}
--online-inference-base-url ${VLA_BASE_URL}
--online-inference-protocol-profile ${VLA_PROTOCOL_PROFILE}
--online-inference-prompt ${VLA_PROMPT}
--online-inference-arm-side ${VLA_ARM_SIDE}
--online-inference-transform-config ${VLA_TRANSFORM_CONFIG}
```

---

## 8. 画图命令

脚本：

```text
tests/diagnostics/plot_latency_trace.py
```

查看参数：

```bash
python tests/diagnostics/plot_latency_trace.py --help
```

典型命令：

```bash
python tests/diagnostics/plot_latency_trace.py \
  utils/data/latency_trace.jsonl \
  --output utils/data/latency_trace.png \
  --top-report utils/data/latency_trace_top.md
```

图里重点看：

```text
Top-level latency
  看 recv_to_pub / pub_to_exec_thread / recv_to_exec_thread

Recent full breakdown timeline
  看最近样本里哪一段最长

Per-segment statistics
  看各段 avg / p95 / max

Base ctrl focus
  看底盘控制是否拖慢主循环
```

---

## 9. 三相机录制命令

脚本：

```text
scripts/start/start_real_robot_wired_3cams_zmq.sh
```

命令：

```bash
SENDER_IP=192.168.123.164 \
TASK_DIR=./utils/data \
TASK_NAME=multi_cam_record \
RECORD_ARM_REPR=both \
bash scripts/start/start_real_robot_wired_3cams_zmq.sh
```

环境变量：

```text
NETWORK_INTERFACE
  DDS 网卡，默认 eno1

SENDER_IP
  ZMQ raw 相机 sender IP

HEAD_ZMQ_PORT / LEFT_WRIST_ZMQ_PORT / RIGHT_WRIST_ZMQ_PORT
  三路相机端口

TASK_DIR
  数据保存根目录

TASK_NAME
  任务目录名

RECORD_ARM_REPR
  qpos / pose / both
```

该脚本实际会加：

```text
--record
--task-dir ${TASK_DIR}
--task-name ${TASK_NAME}
--record-arm-repr ${RECORD_ARM_REPR}
```

---

## 10. Raw 多相机数据审查

脚本：

```text
scripts/data/audit_multi_cam_record.sh
```

基础命令：

```bash
DATASET=utils/data/multi_cam_record/transfer_black \
EPISODE_START=207 \
EPISODE_END=239 \
bash scripts/data/audit_multi_cam_record.sh
```

增强检查：

```bash
DATASET=utils/data/multi_cam_record/transfer_black \
EPISODE_START=207 \
EPISODE_END=239 \
DECODE_IMAGES=1 \
PROGRESS_INTERVAL=1 \
OUTPUT_PREFIX=audit_0207_0239_v2 \
bash scripts/data/audit_multi_cam_record.sh
```

参数含义：

```text
DATASET
  raw episode 根目录

OUTPUT_DIR
  输出报告目录，空则默认写到 dataset 下

OUTPUT_PREFIX
  输出文件名前缀

EPISODE_START / EPISODE_END
  只审查指定 episode 范围

DECODE_IMAGES
  1 时用 cv2 解码图片，检查图片是否真的可读

STATE_GRIPPER_REVIEW
  1 时把 state 夹爪闭合情况也纳入人工复查

PROGRESS_INTERVAL
  每多少个 episode 打一次进度
```

审查结果分两类：

```text
hard fail
  data.json 缺失/损坏
  qpos 长度错误或非有限值
  必要图片缺失
  时间戳缺失、倒退、大断点
  帧数过短

review
  夹爪 action 闭合不足
  长帧数异常
  相机对齐偏大
  idx 不连续
  可选 EE action 跳变
```

---

## 11. Dex1 夹爪调试

脚本：

```text
scripts/debug/dex1_gripper_keyboard_probe.sh
```

命令：

```bash
bash scripts/debug/dex1_gripper_keyboard_probe.sh
```

用途：

```text
直接键盘控制 Dex1 开合
观察 state / action / tau_est
验证 force-hold 参数是否合理
```

按键：

```text
a / 左键     闭合
d / 右键     张开
空格         目标位置设为当前实际位置
```

运行前必须停掉真机遥操作或 VLA 进程，避免多个进程同时发布：

```text
rt/dex1/*/cmd
```

---

## 12. 最短阅读路径

只看 VLA 延迟链路：

```text
1. VLA_LATENCY_TRACE_REPO_GUIDE.md
2. scripts/start/start_real_robot_vla.sh
3. teleop/real/teleop_hand_and_arm.py
4. teleop/debug/latency_trace.py
5. teleop/control_flow/arm_command_pipeline.py
6. teleop/robot_control/robot_arm.py
7. tests/diagnostics/plot_latency_trace.py
```

只看数据采集：

```text
1. scripts/start/start_real_robot_wired_3cams_zmq.sh
2. data_pipeline/recording/teleop_recording_flow.py
3. data_pipeline/recording/episode_writer.py
4. data_pipeline/audit/multi_cam_record.py
```
