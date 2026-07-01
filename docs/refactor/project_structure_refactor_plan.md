# xr_teleoperate 项目结构梳理与重构建议

> 生成日期：2026-07-01  
> 当前分支：`refactor/project-structure`  
> 当前目标：按真实业务边界收敛目录；已清理过渡 shim 和空目录。

## 1. 当前项目主流程

当前仓库已经不是单纯的 XR 遥操 demo，而是混合了以下几条流程：

```text
A. XR 遥操真机
   XR/PICO/Controller/Tracker
   -> core.input
   -> TeleData
   -> teleop_hand_and_arm.py 主循环
   -> IK / 安全限幅 / 夹爪 / 底盘
   -> DDS 真机下发

B. XR 遥操 MuJoCo
   XR/PICO/Controller/Tracker
   -> core.input
   -> teleop/sim/xrobotics_mujoco.py
   -> IK / 安全限幅
   -> MuJoCo qpos/ctrl

C. 离线回放 / 数据集回放
   LeRobot/raw episode
   -> core.input offline provider
   -> replay scripts
   -> MuJoCo 或真机

D. 在线推理/VLA
   Robot state + cameras
   -> inference
   -> online inference transport/protocol
   -> core.input OnlineInferenceInputProvider
   -> 主控制链路

E. 数据采集/导出/检查
   recording writers / rerun visualizer
   -> data_pipeline 中的数据转换/回放
   -> tests 中的数据检查/诊断脚本
```

所以更适合按 **运行流程和边界职责** 组织，而不是按零散工具文件持续堆叠。

## 2. 当前目录职责盘点

```text
xr_teleoperate/
├── assets/                 # 机器人/手/场景模型资源
├── configs/                # 仓库级配置；当前 configs/inference 存放在线推理 transform 配置
├── data/                   # 数据目录，当前职责不够明确
├── docs/                   # 部分设计文档，但根目录仍有较多运行文档
├── img/                    # README 图片资源
├── scripts/                # demo、启动、回放、诊断脚本混在一起
├── teleop/
│   ├── teleop_hand_and_arm.py      # 真机主入口；当前约 2000 行，职责过重
│   ├── input/              # 输入源抽象：XR、offline、online inference
│   ├── inference/          # 在线推理协议、传输、pose transform、session
│   ├── robot_control/      # 机械臂 IK、DDS 控制、夹爪、dex-retargeting vendor
│   ├── control_utils/      # 安全限幅、滤波、底盘桥、motion switcher、timing
│   ├── recording/          # episode/LeRobot/rerun 写入
│   ├── camera/             # 本地相机采集
│   ├── runtime/           # 键盘/操作员运行时
│   ├── sim/                # MuJoCo/G1D 仿真辅助
│   ├── diagnostics/        # latency trace 等诊断
│   ├── cpp/                # C++ 底盘桥源码
│   └── bin/                # 编译产物或可执行文件
└── data_pipeline/          # 数据采集、回放、导出、检查
```

## 3. 主要结构问题

### 3.1 主入口职责过重

`teleop/real/teleop_hand_and_arm.py` 当前同时处理：

- CLI 参数定义
- DDS 初始化
- XR/input provider 初始化
- arm/hand/base/controller 初始化
- runtime/键盘逻辑
- camera/recording 逻辑
- online inference/VLA runtime 接入
- 主循环状态机
- home/takeover/deadman/safety 控制
- latency trace/debug 输出

这导致任何新功能都容易继续往这个文件塞，长期会越来越难测、难回滚。

### 3.2 `scripts/` 目录职责混杂

当前 `scripts/` 同时放了：

- 一键启动：`start_real_robot_*.sh`
- demo：`demo_xrobotics_mujoco.py`, `demo_real_robot_shadow_mujoco.py`
- 回放：`replay_*.py`
- 诊断：`debug_dds_*.sh`, `debug_dds_topics.py`, `offline_dds_replay_probe.py`
- 数据/报告：`audit_multi_cam_record.py`, `generate_alignment_report.py`, `rebuild_rerun_rrd.py`

脚本数量变多后，用户很难知道“正常跑真机/仿真/回放/诊断”分别应该从哪里进入。

### 3.3 `input/` 已经有 provider 抽象，但边界还可以更干净

优点：

- 已有 `BaseTeleopInputProvider` / `TeleopInputSample` / `MotionIntent`
- XR、LeRobot offline、raw offline、online inference 都走 provider 思路
- 新增 PICO motion_tracker 没有破坏 controller 默认逻辑

问题：

- `teleop_input_provider.py` 既做 factory，又混有握手 payload、offline conversion 等逻辑，容易变成第二个“大杂烩”。
- `base.py` 中除了基类，还放了 action/pose/gripper 转换工具；后续可拆成 `types.py` + `conversions.py`。

### 3.4 `control_utils/` 是真实控制逻辑，不应该叫 utils

`arm_target_safety.py`、`arm_workspace_safety.py`、`motion_switcher.py`、`g1d_agv_bridge.py` 都是运行链路里的关键控制模块，不是普通工具函数。建议重命名为 `teleop/control/` 或拆到更明确的 domain。

### 3.5 `teleop/utils/` 已经基本是历史残留

当前代码没有再 import `teleop.utils`。目录里主要是 `__pycache__`，并且在 git status 中显示为 ignored。后续可以清理，避免和 `control_utils/` 混淆。

### 3.6 assets 资源清理边界

`assets/` 下没有删除整个机器人/手模型目录，因为这些目录由 arm 类型、MuJoCo 场景或 hand retargeting 动态使用。当前只清理了 **未被任何 URDF/XML/YAML 引用的孤立 mesh 文件**，删除清单记录在：

```text
docs/refactor/unused_assets_removed.txt
```

清理后已校验所有 `assets` 内的 URDF/XML/YAML 文件引用都能解析到实际文件。

### 3.7 生成物/缓存/日志混在源码树

发现：

- `teleop/MUJOCO_LOG.TXT` 是 git tracked 文件，不适合长期跟踪。
- `g1_29_model_cache.pkl`、`teleop/g1_29_model_cache.pkl` 是 ignored 缓存，但仍在工作区可见。
- 多个 `__pycache__/` 在工作区存在。

建议把运行产物统一放到：

```text
runs/
cache/
logs/
```

或者至少保证不出现在源码结构梳理视野里。

### 3.8 文档分层不清

根目录存在较多文档：

```text
README*.md
RUN.md
RUN_LEROBOT_V2_3CAM.md
Device*.md
docs/datasets/schemas/NERO_LEROBOT_V2_3CAM_DATASET_SCHEMA.md
docs/refactor/xr_teleoperate_项目流程Review.md
```

其中部分是面向用户的入口文档，部分是数据集/方案设计文档，部分是历史 review。建议归档到 `docs/` 下按主题分层，根目录保留少量入口文档。

## 4. 修正后的结构原则：`teleop` 只表示遥操域

你的判断更合理：最终清爽目录不应该把所有东西都塞在 `teleop/` 下面。

建议把项目拆成几个 **业务域**：

```text
teleop          只负责实时遥操运行；内部再分 real / sim
core/common     跨流程复用的基础逻辑；不绑定遥操、采集、回放中的任何一个入口
data_pipeline   数据采集、记录、回放、导出、数据集检查
inference       在线推理/VLA 协议、ob/action schema、client/session
scripts         只放薄启动脚本；复杂逻辑下沉到对应 Python package
configs         按 real/sim/inference/recording/dataset 分层
```

这样语义更清楚：

- `teleop/real`：真机遥操入口和真机遥操 session。
- `teleop/sim`：MuJoCo/仿真遥操入口和仿真 backend。
- `data_pipeline/recording`：episode/LeRobot/rerun 写入逻辑。
- `data_pipeline/replay`：raw/LeRobot episode 回放逻辑。
- `data_pipeline/export`：格式转换、对齐报告、数据检查。
- `inference/`：在线策略/VLA 的协议和 session，不和 XR 手柄遥操绑死。
- `core/` 或 `common/`：输入抽象、pose/transform、安全限幅、robot adapter 等公共模块。

核心原则调整为：

1. **`teleop` 不再等于整个项目**：`teleop` 只代表实时遥操这条业务线。
2. **real/sim 在 teleop 内部分开**：二者共享控制策略，但输出 backend 不同。
3. **采集和回放独立成域**：采集/回放可以复用 teleop 的输入、控制、robot backend，但不属于 teleop 目录。
4. **公共能力单独沉淀**：provider contract、pose transform、control safety、robot arm/hand adapter 不跟某个入口绑定。
5. **脚本只做启动壳**：一键脚本调用 package 内的 app，不再承载业务逻辑。

## 5. 修正后的目标目录：推荐最终形态

最终建议目标是下面这种结构：

```text
xr_teleoperate/
├── teleop/                         # 只负责实时遥操
│   ├── real/                       # 真机遥操
│   │   ├── app.py                  # 真机 CLI/app 拼装
│   │   ├── session.py              # 真机 teleop session 生命周期
│   │   └── backend.py              # DDS 下发、真机状态读取适配
│   ├── sim/                        # 仿真遥操
│   │   ├── app.py                  # MuJoCo CLI/app 拼装
│   │   ├── session.py              # 仿真 teleop session
│   │   └── mujoco_backend.py       # MuJoCo qpos/ctrl 适配
│   ├── runtime/                    # real/sim 共享遥操状态机
│   │   ├── control_loop.py         # 单帧控制流程
│   │   ├── takeover_state.py       # grip/deadman/takeover/home 状态
│   │   └── operator_runtime.py     # 键盘/手柄快捷键
│   └── README.md                   # 遥操运行说明
│
├── core/                           # 被业务入口实际复用的实现逻辑
│   ├── input/                      # XR/controller/tracker/offline/online provider contract
│   ├── control/                    # 安全限幅、workspace、filter、底盘控制策略
│   └── camera/                     # local/zmq camera 抽象
│
├── inference/                      # 在线推理/VLA，不直接属于 teleop
│   ├── transport.py                # tcp/http transport + action chunk 解析
│   ├── pi05_protocol.py            # PI0.5/pose payload 编解码
│   ├── online_session.py           # observation 组包和 chunk 运行节奏
│   └── pose_transform.py           # pose/quaternion/rot6d/matrix 转换
│
├── data_pipeline/                  # 数据采集、回放、导出
│   ├── recording/                  # episode writer / lerobot writer / rerun writer
│   ├── replay/                     # raw/LeRobot 回放到 sim/real 的逻辑
│   └── export/                     # raw -> LeRobot, LeRobot -> UMI/DP 等转换
│
├── tests/                          # 非生产链路：诊断、探针、数据检查
│   ├── diagnostics/
│   └── data_checks/
│
├── scripts/                        # shell wrapper / 一键启动，不放复杂 Python 逻辑
│   ├── start/
│   ├── demo/
│   ├── replay/
│   ├── debug/
│   └── data/
│
├── configs/
│   ├── real/
│   ├── sim/
│   ├── inference/
│   ├── recording/
│   └── datasets/
│
├── docs/
│   ├── architecture/
│   ├── runbooks/
│   ├── datasets/
│   └── refactor/
│
└── assets/
```

这版结构里，`teleop` 的语义变得非常窄：只表示“在线实时遥操”。数据采集、回放、导出、VLA 协议都在自己的业务域里。

## 6. 低风险过渡目录：先不大面积改 import

当前代码已经大量使用 `teleop.xxx` import。为了不一次性打碎运行链路，建议过渡期先这样做：

```text
xr_teleoperate/
├── teleop/
│   ├── real/                      # 新增：真机遥操薄入口
│   ├── sim/                       # 新增：仿真遥操薄入口
│   ├── runtime/                   # 新增：共享遥操状态机
│   ├── input/                     # 暂时保留；后续迁到 core/input
│   ├── inference/                 # 暂时保留；后续迁到 inference/
│   ├── robot_control/             # 暂时保留；后续迁到 core/robot
│   ├── control_utils/             # 暂时保留；后续迁到 core/control
│   ├── recording/                 # 暂时保留；后续迁到 data_pipeline/recording
│   ├── camera/                    # 暂时保留；后续迁到 core/camera
│   ├── diagnostics/               # 暂时保留；后续迁到 tests/diagnostics
│   └── teleop_hand_and_arm.py     # 兼容旧入口，最终变成 wrapper
├── data_pipeline/                 # 先新增空域或迁入新代码，不强行移动旧代码
├── scripts/                       # 暂时保留旧路径，新增分组 wrapper
└── docs/
```

过渡期规则：

- 新增遥操功能：优先放 `teleop/real`、`teleop/sim`、`teleop/runtime`。
- 新增采集/回放/导出功能：优先放 `data_pipeline/`。
- 新增纯公共逻辑：直接放 `core/input`、`core/control`、`core/camera`，不再新增 `teleop/*` 兼容目录。
- 旧入口和旧脚本保留 wrapper，确保已有命令还能跑。

## 7. 按新原则重新归类现有文件

### 7.1 应归到 `teleop/real`

```text
teleop/real/teleop_hand_and_arm.py
scripts/start_real_robot_wired.sh
scripts/start_real_robot_wifi.sh
scripts/start_real_robot_wired_3cams_zmq.sh
```

处理方式：

- 先新增 `teleop/real/app.py`，从旧 `teleop_hand_and_arm.py` 逐步抽 argparse/init 逻辑。
- 旧 `teleop/real/teleop_hand_and_arm.py` 保留为兼容 wrapper。
- `scripts/start_real_robot_*.sh` 后续移动到 `scripts/start/`，只调用 `python teleop/real/teleop_hand_and_arm.py`。

### 7.2 应归到 `teleop/sim`

```text
teleop/sim/xrobotics_mujoco.py
teleop/sim/real_robot_shadow_mujoco.py
teleop/sim/g1d_mujoco_builder.py
teleop/sim/sim_state_topic.py
```

处理方式：

- 先新增 `teleop/sim/app.py`，把 MuJoCo demo 的 CLI 和 session 入口下沉。
- 旧 `teleop/sim/xrobotics_mujoco.py` 保留为 wrapper。

### 7.3 应归到 `data_pipeline/recording`

```text
data_pipeline/recording/episode_writer.py
data_pipeline/recording/lerobot_v2_writer.py
data_pipeline/recording/rerun_visualizer.py
```

注意：数据采集虽然由真机/仿真遥操触发，但 writer 本身不属于 teleop。teleop 只调用 recording API。

### 7.4 应归到 `data_pipeline/replay`

```text
data_pipeline/replay/raw_episode_real.py
data_pipeline/replay/raw_episode_mujoco.py
data_pipeline/replay/lerobot_real.py
core/input/raw_offline.py
core/input/lerobot_offline.py
```

注意：offline provider 可以拆成两层：

```text
core/input/offline_dataset_provider.py      # 把数据集帧转成 TeleopInputSample
data_pipeline/replay/*.py                  # 回放调度、速度控制、目标 backend 选择
```

### 7.5 应归到 `data_pipeline/export` / `audit`

```text
data_pipeline/export/raw_to_lerobot_v2.py
data_pipeline/export/lerobot_to_umi_dp.py
tests/data_checks/probe_umi_online_inference_dataset.py
tests/data_checks/rollout_umi_online_inference_dataset.py
tests/data_checks/visualize_umi_mujoco_probe.py
tests/data_checks/multi_cam_record.py
tests/data_checks/alignment_report.py
data_pipeline/recording/rebuild_rerun_rrd.py
tests/data_checks/offline_dds_replay_probe.py
```

这些都是数据检查、转换、探针或可视化，不应归在 teleop。

### 7.6 应归到 `core/`

```text
core/input/base.py
core/input/xr_input_types.py
core/input/xr_robotics_wrapper.py
core/input/xr_provider.py
core/control/arm_target_safety.py
core/control/arm_workspace_safety.py
core/control/weighted_moving_filter.py
core/control/g1d_agv_bridge.py
core/control/motion_switcher.py
teleop/robot_control/robot_arm.py
teleop/robot_control/robot_arm_ik.py
teleop/robot_control/robot_hand_unitree.py
teleop/robot_control/robot_hand_inspire.py
teleop/robot_control/robot_hand_brainco.py
core/camera/local_camera.py
tests/diagnostics/*.py
```

这些模块可以被 teleop、replay、recording、inference 多方使用，最终不应该只属于 teleop。

### 7.7 应归到 `inference/`

```text
inference/online_session.py
inference/transport.py
inference/pi05_protocol.py
inference/pose_transform.py
core/input/online_inference_provider.py
configs/inference/*
```

在线推理/VLA 是“输入源/策略源”，不是 XR 遥操本体。

## 8. 更新后的迁移顺序

### Phase 0：文档和目录语义先落地

- [x] 新增本结构梳理文档。
- [x] 明确目标：`teleop` 只管实时遥操，采集/回放/导出独立到 `data_pipeline`。
- [x] 新增空目录或 README：`teleop/real/`、`teleop/sim/`、`teleop/runtime/`、`data_pipeline/`。
- [x] 根目录历史 review 移到 `docs/refactor/` 或 `docs/architecture/`。
- [x] 清理 ignored 的 `__pycache__/`、`.pkl`、运行日志。
- [x] 将 `teleop/MUJOCO_LOG.TXT` 从 git 跟踪中移除并加入 `.gitignore`。

### Phase 1：先把遥操入口按 real/sim 收口

- [x] 新增 `teleop/real/app.py`，承接真机 CLI 和启动拼装。
- [x] 新增 `teleop/sim/app.py`，承接 MuJoCo CLI 和启动拼装。
- [ ] 新增 `teleop/runtime/`，抽出 real/sim 共享的 takeover/deadman/home/control-loop 状态机。
- [x] `teleop/real/teleop_hand_and_arm.py` 保留为旧入口 wrapper。
- [x] `teleop/sim/xrobotics_mujoco.py` 保留为旧入口 wrapper。

### Phase 2：把数据采集和回放迁出 teleop

- [x] 新增 `data_pipeline/recording/`，迁移 episode/LeRobot/rerun writer。
- [x] 新增 `data_pipeline/replay/`，迁移 replay raw/LeRobot 的调度逻辑。
- [x] 新增 `data_pipeline/export/` 和 `tests/data_checks/`，迁移 tools 和数据检查脚本。
- [x] teleop 侧只依赖 `data_pipeline.recording` 的 API，不直接拥有 writer 实现。

### Phase 3：沉淀公共 core

- [x] 新增 `core/input/`，迁移 provider contract、XR wrapper、TeleData 类型。
- [x] 新增 `core/control/`，迁移 safety/workspace/filter/base bridge。
- [ ] 新增 `core/robot/`，迁移 arm/IK/hand adapter。
- [x] 新增 `core/camera/` 和 `tests/diagnostics/`。
- [x] 清理旧 `teleop/input`、`teleop/control_utils` 等兼容 import 目录。

### Phase 4：迁移 inference/VLA 独立域

- [x] 新增顶层 `inference/`。
- [x] 将 online session、transport、protocol、pi05 schema、transform config loader 移入。
- [x] teleop、replay、policy execution 都通过公共 provider/API 使用 inference，不互相硬耦合。

### Phase 5：脚本和文档收口

- [x] `scripts/` 按 start/demo/replay/debug/data 分组。
- [x] 旧脚本路径保留薄 wrapper 一个过渡周期。
- [x] 根目录文档只保留 README/RUN；专题文档移到 docs。
- [ ] 增加最小 smoke tests：provider factory、pose transform、pi05 protocol、operator keybinds、real/sim app `--help`。


### 本轮兼容式重构落地

已经完成一版按业务域归位的兼容式重构：

```text
teleop/real/teleop_hand_and_arm.py          # 真机遥操实现
teleop/sim/xrobotics_mujoco.py              # MuJoCo 遥操实现
data_pipeline/recording/*.py                # recording/writer/rerun 实现
data_pipeline/replay/*.py                   # replay 实现
tests/data_checks/*.py                    # 数据检查/探针实现
core/input/*.py                             # 输入 provider 和 TeleData 类型
core/control/*.py                           # safety/workspace/filter/base bridge
core/camera/*.py                            # camera 抽象
teleop/runtime/{latency_trace,timing_debugger}.py  # 遥操运行时可选观测
tests/diagnostics/*.py                       # debug/plot 诊断脚本
inference/{transport,pi05_protocol,online_session,pose_transform}.py
```

旧路径均保留 wrapper/shim，因此旧命令和旧 import 在过渡期仍可用。

## 9. 近期不建议做的事情

- 不建议一次性大规模 rename 所有目录，会导致运行脚本、用户命令、import 全部变化，风险太高。
- 不建议把 VLA/online inference 直接塞回 XR 主循环，应该继续走 provider/runtime 边界。
- 不建议删除旧脚本入口，至少应保留 wrapper 一个过渡周期。
- 不建议为了“结构好看”引入复杂框架；当前项目核心问题是边界不清和入口过重，不是缺框架。

## 10. 本轮结论

当前项目的核心结构问题不是功能不可用，而是业务域边界不够清楚：

1. `teleop` 现在承载了遥操、采集、回放、推理、工具等多种语义；
2. 真机和仿真遥操应该在 `teleop/real`、`teleop/sim` 下收口；
3. 数据采集、回放、导出应该迁到 `data_pipeline/`；
4. provider、control safety、robot adapter、camera、diagnostics 应该沉淀到公共 `core/`；
5. VLA/online inference 应作为独立 `inference/` 域，被 teleop 或 replay 复用，而不是属于 XR 遥操本体。

建议先按 **teleop real/sim 收口 -> data_pipeline 拆出 -> core 公共化 -> inference 独立化 -> scripts/docs 收口** 的顺序做。这样既符合软件工程边界，也能最大程度保留当前真机、仿真、回放、VLA 的可运行性。
