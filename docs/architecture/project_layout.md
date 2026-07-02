# 项目结构说明

原则：目录按业务域组织，但入口不要叠 wrapper。每条流程只保留一个真实 Python 入口。

## 1. 顶层结构

```text
assets/                         机器人、手、场景模型资源
configs/                        仓库级配置，当前主要放 inference transform 配置
core/                           可复用核心能力，不绑定某个入口
data_pipeline/                  数据业务域：recording / replay / export
docs/                           架构、数据集、runbook、重构说明
inference/                      在线推理 / VLA 协议、会话和 pose transform
scripts/                        少量 shell 一键脚本；不放 Python wrapper 和业务实现
teleop/                         实时遥操域：real / sim / control_flow / runtime / debug
tests/                          数据检查、诊断探针和调试脚本
```

不建议再新增 `utils/`、`common/` 这类模糊目录。新增代码先判断属于哪条业务链路：

- 实时控制链路：优先放 `teleop/` 或 `core/control`
- 输入源/输入协议：优先放 `core/input`
- 数据采集/回放/导出：放 `data_pipeline/`
- 在线策略/VLA：放 `inference/`
- 调试/检查/探针：放 `tests/` 或 `teleop/debug`

## 2. 遥操域 `teleop/`

```text
teleop/
  real/
    teleop_hand_and_arm.py      真机遥操唯一入口；只编排启动、主循环、清理
    setup.py                    真机启动装配：DDS/base/arm/EE/camera/record/latency
    args.py                     真机入口 CLI 参数
  sim/                          仿真遥操入口与仿真适配
  control_flow/                 每帧控制流程
    base_command.py             底盘命令流水线
    arm_command_pipeline.py     机械臂目标、IK、限速、重力补偿
    operator_state.py           home/deadman/takeover 状态机
    end_effector_command.py     夹爪/灵巧手命令映射
  runtime/                      操作员按键和运行时命令处理
  debug/                        遥操调试辅助：trace/timing/UI

  robot_control/                机器人硬件适配：arm/IK/hand/dex-retargeting vendor
```

职责边界：

- `teleop/real/teleop_hand_and_arm.py` 是真机唯一 Python 入口，不再拆新的 `app.py` wrapper。
- `teleop/real/setup.py` 只负责启动装配，不放每帧控制逻辑。
- `teleop/control_flow/` 放主循环每帧会调用的控制流程：
  - `base_command.py`：底盘命令
  - `arm_command_pipeline.py`：机械臂目标、workspace、IK、限速、重力补偿
  - `operator_state.py`：home、deadman、takeover 状态
  - `end_effector_command.py`：夹爪/灵巧手命令映射
- `teleop/runtime/` 只放操作员运行时命令处理，例如按键、快捷键状态；不放控制算法。
- `teleop/debug/` 放遥操运行时调试辅助，不放正式控制业务。
- `teleop/robot_control/` 目前保留硬件控制和 IK 实现；后续如果继续收敛，可再评估是否迁到 `core/robot/`。

## 3. 公共核心 `core/`

```text
core/
  input/                        输入 provider contract、XR/离线/online inference 输入适配
  control/                      workspace/speed safety、base bridge、motion switcher、滤波
  camera/                       本地相机采集抽象
```

`core/` 放可复用实现逻辑，要求不依赖某一个具体入口。典型调用关系：

```text
teleop/real/teleop_hand_and_arm.py
  -> core.input.create_teleop_input_provider()
  -> core.control.* safety / base bridge
  -> core.camera.LocalCameraStream
```

## 4. 数据域 `data_pipeline/`

```text
data_pipeline/
  recording/
    teleop_recording_flow.py    遥操作业录制：episode 生命周期、对齐、写入
    alignment.py                state/action/camera 时间戳对齐工具
    episode_writer.py           raw episode writer
    lerobot_v2_writer.py        LeRobot v2 writer
    rerun_visualizer.py         rerun 可视化写入
  replay/                       episode 回放
  export/                       数据格式导出
```

边界说明：

- recording 相关逻辑不再放在 `teleop/` 下。
- 真机遥操只调用 `TeleopRecordingFlow` 暴露接口，不直接拼接 recording payload。
- 数据检查不放在 `data_pipeline/`，放 `tests/data_checks/`。

## 5. 在线推理域 `inference/`

```text
inference/
  transport.py                  推理服务通信传输
  pi05_protocol.py              PI0.5 / policy 请求响应协议
  online_session.py             在线推理 session、动作缓存、限速反馈
  pose_transform.py             pose / frame transform
```

`inference/` 不直接控制机器人。真实执行仍回到 `core/input` provider，再进入 `teleop/real/teleop_hand_and_arm.py` 的安全控制链路。

## 6. 脚本和测试

```text
scripts/
  start/                        真机启动 shell
  debug/                        DDS / 网络调试 shell
  replay/                       回放快捷 shell
  data/                         数据导出快捷 shell

tests/
  data_checks/                  数据质量、对齐、probe
  diagnostics/                  运行时诊断和可视化
```

脚本只做三件事：

1. 激活环境
2. 设置常用环境变量和参数
3. 调用唯一 Python 入口或对应业务域 Python 文件

不在 `scripts/` 里新增复杂 Python 业务实现。

## 7. 推荐入口

```bash
python teleop/real/teleop_hand_and_arm.py          # 真机遥操
python teleop/sim/xrobotics_mujoco.py              # MuJoCo 遥操
python data_pipeline/replay/raw_episode_real.py    # raw episode 真机回放
python data_pipeline/replay/raw_episode_mujoco.py  # raw episode MuJoCo 回放
```

保留的 shell 快捷入口：

```bash
bash scripts/start/start_real_robot_wired.sh
bash scripts/start/start_real_robot_wifi.sh
bash scripts/start/start_real_robot_wired_3cams_zmq.sh
bash scripts/debug/debug_dds_wired.sh
bash scripts/debug/debug_dds_wifi.sh
bash scripts/replay/replay_raw_episode_real.sh
bash scripts/data/export_clean_multi_cam_lerobot.sh
```

## 8. 真机主链路

当前真机遥操主流程：

```text
scripts/start/*.sh
  -> python teleop/real/teleop_hand_and_arm.py
      -> parse_args()
      -> initialize_dds()
      -> setup_real_teleop_components()
          -> base / arm / IK / input provider
          -> end-effector
          -> sim / camera / recorder / latency tracker
      -> while not STOP:
          -> 读取机器人状态
          -> tv_wrapper.get_sample()
          -> OperatorStateFlow.apply()
          -> apply_end_effector_command()
          -> apply_base_command()
          -> build_arm_command()
          -> arm_ctrl.ctrl_dual_arm()
          -> recording_flow.process_frame()
      -> cleanup_real_teleop_resources()
```

这条链路里，安全相关逻辑仍集中在真机主循环调用的控制流中：

- deadman / home / takeover：`teleop/control_flow/operator_state.py`
- workspace / IK / joint speed limit / gravity compensation：`teleop/control_flow/arm_command_pipeline.py`
- recording 对齐写入：`data_pipeline/recording/teleop_recording_flow.py`

## 9. 新功能放置规则

| 新功能类型 | 推荐位置 | 说明 |
|---|---|---|
| 新真机启动参数 | `teleop/real/args.py` | 参数定义集中，避免散落在脚本 |
| 新硬件启动装配 | `teleop/real/setup.py` | 只做构造和初始化 |
| 新每帧控制逻辑 | `teleop/control_flow/` | 按 arm/base/operator/EE 分职责 |
| 新输入源 | `core/input/` | 输出统一 `TeleopInputSample` / `MotionIntent` |
| 新在线策略协议 | `inference/` | 不直接下发机器人 |
| 新录制字段或对齐逻辑 | `data_pipeline/recording/` | 主循环只调接口 |
| 新回放方式 | `data_pipeline/replay/` | 不塞进 teleop 主入口 |
| 新数据导出 | `data_pipeline/export/` | 和 recording/replay 分离 |
| 新诊断探针 | `tests/diagnostics/` 或 `tests/data_checks/` | 不放 core |
| 新 shell 一键命令 | `scripts/start|debug|replay|data/` | shell 只负责启动 |

## 10. 当前重构约束

- 不新增多层 wrapper；入口保持清楚。
- 不把 recording、replay、export 塞回 `teleop/`。
- 不把诊断工具放进 `core/`。
- 不为了解耦而拆过细；优先按真实职责拆。
- 真机安全控制链路必须经过 `teleop/real/teleop_hand_and_arm.py` 的主循环。
