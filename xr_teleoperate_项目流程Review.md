# xr_teleoperate 项目流程 Review

> 生成时间：2026-05-21  
> 代码基线：`/home/dx/unitree_ws/src/xr_teleoperate`

---

## 1. 这套项目现在到底在做什么

这个仓库当前已经收敛成一条非常明确的链路：

```text
XR-Robotics SDK
-> XRRoboticsWrapper
-> TeleData
-> Arm IK
-> Arm Controller / Hand Controller
-> MuJoCo 或真机 DDS
```

也就是说：
- 上游只保留 **XR-Robotics** 的头显/手柄输入。
- 中间把 XR 数据转成项目自己的 `TeleData`。
- 再用 **双臂 IK** 把目标手腕位姿变成机器人关节角。
- 最后下发给：
  - **真机控制器**（DDS）
  - 或 **MuJoCo 仿真**

当前两个主入口：
- 真机入口：`teleop/teleop_hand_and_arm.py`
- MuJoCo 入口：`teleop/demo_xrobotics_mujoco.py`

---

## 2. 项目目录与职责

### 2.1 主入口
- `teleop/teleop_hand_and_arm.py`  
  真机主流程，负责：初始化 DDS、读取 XR、IK、机械臂控制、手部控制、可选录制、可选底盘。

- `teleop/demo_xrobotics_mujoco.py`  
  MuJoCo 演示入口，复用 XR 输入、IK 和安全逻辑，但把输出改成 MuJoCo `qpos/ctrl`。

### 2.2 XR 输入适配
- `teleop/utils/xr_robotics_wrapper.py`
- `teleop/utils/xr_input_types.py`

### 2.3 机械臂控制
- `teleop/robot_control/robot_arm_ik.py`：逆解
- `teleop/robot_control/robot_arm.py`：DDS 下发与机械臂状态读取

### 2.4 手部控制
- `teleop/robot_control/robot_hand_unitree.py`：Dex1 / Dex3
- `teleop/robot_control/robot_hand_inspire.py`：Inspire DFX / FTP
- `teleop/robot_control/robot_hand_brainco.py`：Brainco
- `teleop/robot_control/hand_retargeting.py`：手骨架到硬件关节的映射
- `teleop/robot_control/dex-retargeting/`：实际重定向算法子模块

### 2.5 安全 / 辅助 / 记录
- `teleop/utils/arm_target_safety.py`：外环限速
- `teleop/utils/motion_switcher.py`：切调试模式/运动模式
- `teleop/utils/g1d_agv_bridge.py`：G1D AGV 底盘桥
- `teleop/utils/ipc.py`：IPC 控制与心跳
- `teleop/utils/episode_writer.py`：录制 episode
- `teleop/utils/sim_state_topic.py`：仿真状态订阅
- `teleop/utils/g1d_mujoco_builder.py`：生成可移动 G1D MuJoCo 场景
- `teleop/utils/weighted_moving_filter.py`：平滑滤波

---

## 3. 真机主流程：从启动到控制的完整调用链

文件：`teleop/teleop_hand_and_arm.py`

## 3.1 初始化阶段

主流程在 `if __name__ == '__main__':` 中完成。

### 第一步：解析参数
调用 `argparse.ArgumentParser()`，关键参数包括：
- `--input-mode`：`hand` / `controller`
- `--arm`：`G1_29` / `G1_23` / `H1_2` / `H1` / `H2`
- `--ee`：`dex1` / `dex3` / `inspire_ftp` / `inspire_dfx` / `brainco`
- `--controller-deadman`
- `--head-reference-mode`
- `--max-arm-joint-speed`
- `--motion`
- `--record`
- `--ipc`

### 第二步：初始化 DDS
调用：
- `ChannelFactoryInitialize(0, networkInterface=...)`：真机
- `ChannelFactoryInitialize(1, ...)`：仿真

### 第三步：初始化输入命令通道
二选一：
- `IPC_Server.start()`：如果 `--ipc`
- `threading.Thread(target=listen_keyboard, ...)`：否则用 sshkeyboard

键盘映射函数：`on_press()`
- `r`：启动跟随
- `c`：重标定头参考
- `q`：退出
- `s`：录制开关

### 第四步：初始化 XR 输入包装器
调用：
- `XRRoboticsWrapper(...)`

它内部又调用：
- `xrt.init()`：启动 `xrobotoolkit_sdk`

### 第五步：初始化底盘/模式
- 若 `--motion`：`LocoClientWrapper()`
- 否则：`MotionSwitcher.Enter_Debug_Mode()`
- 若 `--base-controller g1d_agv`：`G1DAgvBridge(...)`

### 第六步：初始化机械臂 IK 与机械臂控制器
根据 `--arm` 选择：
- `G1_29_ArmIK()` + `G1_29_ArmController()`
- `G1_23_ArmIK()` + `G1_23_ArmController()`
- `H1_2_ArmIK()` + `H1_2_ArmController()`
- `H1_ArmIK()` + `H1_ArmController()`
- `H2_ArmIK()` + `H2_ArmController()`

### 第七步：初始化末端执行器
根据 `--ee` 选择：
- `Dex3_1_Controller(...)`
- `Dex1_1_Gripper_Controller(...)`
- `Inspire_Controller_DFX(...)`
- `Inspire_Controller_FTP(...)`
- `Brainco_Controller(...)`

这些控制器都不是在主线程直接算控制，而是：
- 主线程把 XR 输入写进 `multiprocessing.Array/Value`
- 控制器内部线程/子进程持续读取共享内存并发 DDS 命令

### 第八步：初始化录制与仿真附加模块
- `EpisodeWriter(...)`：录制
- `start_sim_state_subscribe()`：仿真状态
- `ChannelPublisher("rt/reset_pose/cmd", String_)`：仿真 reset

---

## 3.2 进入 teleop 前的准备状态

调用：
- `arm_ctrl.ctrl_dual_arm_go_home()`：双臂回零位
- `READY = True`
- 等待 `START`，也就是等 `r`

之后：
- `arm_ctrl.speed_gradual_max()`：速度逐步拉到最大
- `arm_ctrl.get_current_dual_arm_q()`：读当前双臂关节
- `compute_arm_gravity_tauff()`：通过 `pin.rnea(...)` 计算当前重力补偿
- `arm_ctrl.ctrl_dual_arm(current_hold_q, current_hold_tauff)`：先保持住

---

## 3.3 主控制循环 while not STOP

每一帧控制都在这个循环里完成。

### A. 处理录制状态
调用：
- `recorder.create_episode()`
- `recorder.save_episode()`
- `publish_reset_category(...)`（仿真时）

### B. 处理重标定
当按 `c` 时：
- `tv_wrapper.calibrate_head_reference(require_live=True)`
- 若没拿到 live pose，就继续 hold 当前姿态

### C. 读取当前机器人状态
先读机器人，再读 XR，这一点很关键。

调用：
- `arm_ctrl.get_current_dual_arm_q()`
- `arm_ctrl.get_current_dual_arm_dq()`
- `get_robot_wrist_poses(arm_ik, current_lr_arm_q)`

`get_robot_wrist_poses()` 内部调用：
- `pin.framesForwardKinematics(...)`
- `pin.updateFramePlacements(...)`

作用：拿到 **当前机器人左右手腕 pose**，供 XR takeover 锚定。

### D. 读取 XR 输入并转换为 TeleData
调用：
- `tv_wrapper.get_tele_data(current_left_robot_wrist_pose=..., current_right_robot_wrist_pose=...)`

这是整个系统最核心的输入转换点。

### E. deadman / home-return / takeover 管理
主流程使用 `TeleData` 中的字段：
- `left_ctrl_squeeze`
- `right_ctrl_squeeze`
- `left_ctrl_bButton`
- `left/right_ctrl_thumbstickValue`
- `left/right_ctrl_triggerValue`

功能：
- `grip` 作为 deadman
- 左 Y（历史命名落在 `left_ctrl_bButton`）触发双臂回 ready pose
- grip 抬起后才允许 home-return 后重新 takeover
- grip rising edge 时插入 `zero-delta hold`，防止接管抖动

辅助函数：
- `reset_arm_ik_state(arm_ik, current_lr_arm_q)`
- `tv_wrapper.sync_reference_to_current_live_pose(...)`

### F. 把 XR 的手/扳机输入写给末端控制器
- Dex3 / Inspire / Brainco（手骨架模式）：写 `left_hand_pos_array[:]` / `right_hand_pos_array[:]`
- Dex1（controller 模式）：写 `left_gripper_value.value` / `right_gripper_value.value`
- Dex1（hand 模式）：写 pinch value

### G. 可选底盘控制
#### 运动模式 `--motion`
调用：
- `loco_wrapper.Move(vx, vy, wz)`

#### G1D AGV 底盘桥
调用：
- `agv_bridge.move(vx, vy, wz)`
- `agv_bridge.height_adjust(z)`

### H. 机械臂目标生成
这一步是机械臂主链路。

分支如下：
1. **takeover settle 帧**：保持当前关节，不动
2. **home return 中**：目标直接设为 `home_target_q`
3. **至少一侧 arm enabled**：
   - 调用 `arm_ik.solve_ik(left_wrist_pose, right_wrist_pose, current_q, current_dq)`
4. **否则**：保持 `current_hold_q`

### I. 安全限制
外环限速调用：
- `limit_arm_joint_target_velocity(sol_q, current_lr_arm_q, max_joint_speed, control_frequency)`

随后重新算重力补偿：
- `compute_arm_gravity_tauff()` -> `pin.rnea(...)`

### J. 下发机械臂命令
调用：
- `arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)`

### K. 录制当前状态
如果 `--record`：
- `recorder.is_ready()`
- `arm_ctrl.get_current_motor_q()`（controller+dex1 时）
- `sim_state_subscriber.read_data()`（仿真时）
- `recorder.add_item(...)`

---

## 4. MuJoCo 流程：与真机流程哪里相同，哪里不同

文件：`teleop/demo_xrobotics_mujoco.py`

## 4.1 相同点
复用的核心逻辑：
- `XRRoboticsWrapper`
- `G1_29_ArmIK.solve_ik()`
- `limit_arm_joint_target_velocity()`
- deadman 语义
- home-return 语义
- calibration/head-reference 语义
- takeover settle 逻辑

## 4.2 不同点
真实差别只在“输出端”：
- 真机：`arm_ctrl.ctrl_dual_arm()` -> DDS
- MuJoCo：直接写 `data.qpos[...]` 和 `data.ctrl[...]`

### 关键函数
- `resolve_xml_path()`：决定加载哪份 MJCF/XML
- `prepare_g1d_mobile_scene()`：动态生成带移动底盘的 G1D 场景
- `joint_qpos_indices()`：找关节对应的 qpos index
- `apply_dex1_to_qpos()`：把 trigger 值映射成 Dex1 开合量
- `main()`：主循环

### MuJoCo 主循环里实际做的事
1. 读 XR -> `xr_wrapper.get_tele_data()`
2. 求 IK -> `arm_ik.solve_ik()`
3. 限速 -> `limit_arm_joint_target_velocity()`
4. 写机械臂 qpos
5. 若有 Dex1，调用 `apply_dex1_to_qpos()`
6. 若是 G1D 移动底盘，写左右轮 `data.ctrl[...]`
7. `mj.mj_forward()` / `mj.mj_step()` 推进一步物理

---

## 5. 核心模块逐包说明：调用了哪些函数、做什么

## 5.1 `teleop/utils/xr_input_types.py`

### `TeleData`
职责：统一描述一帧 teleop 输入。

核心字段：
- `head_pose`
- `left_wrist_pose` / `right_wrist_pose`
- `left/right_hand_pos`
- `left/right_ctrl_triggerValue`
- `left/right_ctrl_squeeze`
- `left/right_ctrl_thumbstickValue`
- `left/right_ctrl_aButton / bButton`

它是主流程和各控制模块之间的标准数据结构。

---

## 5.2 `teleop/utils/xr_robotics_wrapper.py`

这是 **XR-Robotics SDK -> TeleData** 的适配层。

### 入口函数/方法
- `_bootstrap_xrobotoolkit()`：补充 `xrobotoolkit_sdk` 的搜索路径
- `XRRoboticsWrapper.__init__()`：初始化 SDK 与模式
- `_pose7_to_matrix()`：7维 pose -> 4x4 齐次矩阵
- `_safe_pose_matrix()`：无效 pose 时回退到上一帧
- `_read_robot_basis_poses()`：
  - 调 `xrt.get_headset_pose()`
  - 调 `xrt.get_left_controller_pose()`
  - 调 `xrt.get_right_controller_pose()`
  - 做 `OpenXR -> Robot` 坐标转换
- `calibrate_head_reference()`：记录头参考平移
- `sync_reference_to_current_live_pose()`：把参考软同步到当前 live pose
- `get_tele_data()`：生成一整帧 `TeleData`
- `close()`：`xrt.close()`

### `get_tele_data()` 里实际做了什么
1. 读取头显/左右手柄 pose
2. 读取 trigger/grip/摇杆/按钮：
   - `xrt.get_left_trigger()`
   - `xrt.get_right_trigger()`
   - `xrt.get_left_grip()`
   - `xrt.get_right_grip()`
   - `xrt.get_left_axis()` / `get_right_axis()`
   - `xrt.get_X_button()` / `get_Y_button()` / `get_A_button()` / `get_B_button()`
3. 根据 `head_reference_mode` 处理参考系：
   - `head_coupled`
   - `live_head_reference`
   - `hybrid`
   - `fixed_per_grip`
4. 根据 `controller_mapping_mode` 处理 takeover 锚点：
   - `legacy_main`
   - `anchored_safe`
5. 根据 `controller_orientation_mode` 决定姿态策略：
   - `absolute`
   - `relative`
   - `neutral`
6. 产出 `TeleData`

### 这个包最重要的意义
它不只是“读手柄”，而是在做：
- 坐标系转换
- 头平移解耦
- grip 接管锚定
- controller->robot wrist pose 映射

---

## 5.3 `teleop/robot_control/robot_arm_ik.py`

这是 **手腕目标 pose -> 双臂关节角** 的逆解模块。

每个机器人型号各有一套类：
- `G1_29_ArmIK`
- `G1_23_ArmIK`
- `H1_2_ArmIK`
- `H1_ArmIK`
- `H2_ArmIK`

它们接口几乎一致，主流程主要调用：
- `solve_ik()`

### 关键方法
- `_resolve_asset_paths()`：找 URDF 与 mesh 目录
- `__init__()`：
  - `pin.RobotWrapper.BuildFromURDF(...)`
  - `buildReducedRobot(...)`
  - `model.addFrame(...)` 给左右手添加 EE frame
  - 建 CasADi `Opti()` 优化器
  - 构造：
    - `translational_error`
    - `rotational_error`
    - `regularization_cost`
    - `smooth_cost`
- `save_cache()` / `load_cache()`：缓存 pinocchio model
- `solve_ik()`：真正每帧调用

### `solve_ik()` 具体流程
1. 把上一帧解 `init_data` 作为 warm start
2. `self.opti.set_value(self.param_tf_l, left_wrist)`
3. `self.opti.set_value(self.param_tf_r, right_wrist)`
4. `self.opti.solve()` 运行 IPOPT
5. 对 `sol_q` 做 `WeightedMovingFilter.add_data()` 平滑
6. 调 `pin.rnea(...)` 算 `sol_tauff`
7. 返回 `(sol_q, sol_tauff)`

### 这个包本质上干的事
- 把 XR 给出的左右手腕目标位姿，转成机器人双臂关节角
- 再顺手输出重力补偿扭矩

---

## 5.4 `teleop/robot_control/robot_arm.py`

这是 **真实机械臂 DDS 控制器**。

对应类：
- `G1_29_ArmController`
- `G1_23_ArmController`
- `H1_2_ArmController`
- `H1_ArmController`
- `H2_ArmController`

### 主流程实际调用的方法
- `ctrl_dual_arm_go_home()`：回 home
- `speed_gradual_max()`：逐步提升内部限速
- `get_current_dual_arm_q()`：读当前双臂关节角
- `get_current_dual_arm_dq()`：读当前双臂速度
- `get_current_motor_q()`：录制时读全身电机状态
- `ctrl_dual_arm(q, tauff)`：写目标

### 控制器内部自己调用的方法
- `_subscribe_motor_state()`：订阅 `rt/lowstate`
- `_ctrl_motor_state()`：250Hz 循环发布 `rt/lowcmd` 或 `rt/arm_sdk`
- `clip_arm_q_target()`：再做一层内环速度裁剪
- `get_mode_machine()`：读当前模式

### 这个包到底做了什么
主线程 **并没有直接发 DDS**，而是：
1. `ctrl_dual_arm()` 只更新共享目标 `q_target/tauff_target`
2. 后台线程 `_ctrl_motor_state()` 周期性取目标
3. 调 `CRC.Crc(self.msg)`
4. `lowcmd_publisher.Write(self.msg)` 真正发给机器人

### 安全上要注意两层限速
- 外层：`limit_arm_joint_target_velocity()`（teleop 主循环）
- 内层：`clip_arm_q_target()`（控制器 250Hz 发布线程）

---

## 5.5 `teleop/robot_control/hand_retargeting.py`

这是所有“灵巧手/手爪”共用的重定向配置层。

### 关键类/方法
- `HandType`：不同手型配置文件枚举
- `HandRetargeting.__init__()`：
  - `RetargetingConfig.set_default_urdf_dir(...)`
  - 读 yaml
  - `RetargetingConfig.from_dict(...)`
  - `build()` 得到 `left_retargeting/right_retargeting`

### 输出给下游的核心内容
- `left_retargeting`
- `right_retargeting`
- `left_indices/right_indices`
- `left/right_dex_retargeting_to_hardware`

也就是：
- 先知道要从手骨架取哪些点
- 再知道 retarget 输出后的关节顺序怎么映射到真实硬件顺序

---

## 5.6 `teleop/robot_control/dex-retargeting/`

这是手部重定向算法子模块。

主项目真正用到的核心调用链：
- `RetargetingConfig.from_dict()`
- `RetargetingConfig.build()`
- `SeqRetargeting.retarget(ref_value)`

### 作用
- 从 YAML 配置构造手模型
- 选择优化器（position/vector/dexpilot）
- 建立手骨架向机器人手关节的优化映射
- 每帧根据手关键点差值输出机器人手关节角

这个子模块是 **手部控制器的算法后端**。

---

## 5.7 `teleop/robot_control/robot_hand_unitree.py`

### A. `Dex3_1_Controller`
用于 Unitree Dex3 灵巧手。

主流程把手骨架写入共享数组后，这个控制器内部：
- `_subscribe_hand_state()`：订阅左右手状态
- `control_process()`：
  1. 从共享数组读取 25x3 手关键点
  2. 根据 `left_indices/right_indices` 取参考向量
  3. `left_retargeting.retarget(...)`
  4. `right_retargeting.retarget(...)`
  5. `ctrl_dual_hand(left_q_target, right_q_target)` 发布 DDS

### B. `Dex1_1_Gripper_Controller`
用于 Dex1 夹爪。

主流程写入的是 trigger/pinch 标量，这里内部：
- `_subscribe_gripper_state()`：读夹爪状态
- `control_thread()`：
  1. 读取左右 trigger/pinch 值
  2. `np.interp(...)` 映射成夹爪目标位置
  3. 仿真外时再做 `np.clip(...)` 限制单步变化
  4. 若启用滤波，用 `WeightedMovingFilter.add_data()`
  5. `ctrl_dual_gripper(...)` 下发 DDS

---

## 5.8 `teleop/robot_control/robot_hand_inspire.py`

### `Inspire_Controller_DFX`
- `_subscribe_hand_state()`：读 Inspire DFX 状态
- `control_process()`：
  - 读取手关键点
  - `retarget(...)`
  - 把弧度值归一化到官方 `[0,1]` 控制范围
  - `ctrl_dual_hand(...)`

### `Inspire_Controller_FTP`
- `_subscribe_hand_state()`：读 FTP 版状态
- `_send_hand_command(...)`：把 `[0,1]` 缩放到 `[0,1000]`
- `control_process()`：
  - retarget
  - normalize
  - 缩放成整型命令
  - `_send_hand_command(...)`

---

## 5.9 `teleop/robot_control/robot_hand_brainco.py`

### `Brainco_Controller`
- `_subscribe_hand_state()`：读左右 Brainco 状态
- `control_process()`：
  - 读手关键点
  - `retarget(...)`
  - 按 Brainco 各关节量程做归一化
  - `ctrl_dual_hand(...)`

---

## 5.10 `teleop/utils/arm_target_safety.py`

### `limit_arm_joint_target_velocity()`
输入：
- `target_q`
- `reference_q`
- `max_joint_speed`
- `control_frequency`

作用：
- 把一帧关节变化量裁到 `max_joint_speed / control_frequency`
- 防止 XR 抖动或 IK 突跳直接打到真机

这是主循环里最明确的一层外环安全保护。

---

## 5.11 `teleop/utils/motion_switcher.py`

### `MotionSwitcher`
- `Enter_Debug_Mode()`：退出 AI/motion 模式，进入 debug 控制
- `Exit_Debug_Mode()`：切回 ai

### `LocoClientWrapper`
- `Move(vx, vy, vyaw)`：发送底盘运动命令
- `Enter_Damp_Mode()`：进入阻尼模式

---

## 5.12 `teleop/utils/g1d_agv_bridge.py`

这个包是一个 **Python -> C++ bridge**。

### 关键函数/方法
- `build_g1d_agv_bridge()`：用 g++ 编译 `teleop/cpp/g1d_agv_bridge.cpp`
- `G1DAgvBridge._start()`：拉起子进程桥
- `move(vx, vy, vyaw)`：发送 MOVE 文本协议
- `height_adjust(vz)`：发送 HEIGHT 文本协议
- `stop()` / `close()`：结束桥进程

真机底盘桥链路是：

```text
teleop_hand_and_arm.py
-> G1DAgvBridge.move/height_adjust
-> 子进程 g1d_agv_bridge
-> Unitree SDK2 C++ AgvClient
```

---

## 5.13 `teleop/utils/ipc.py`

### `IPC_Server`
- `_handle_message()`：把外部 IPC 命令映射成键盘语义
  - `CMD_START -> r`
  - `CMD_STOP -> q`
  - `CMD_RECORD_TOGGLE -> s`
- `_hb_loop()`：发布心跳
- `start()` / `stop()`：启动关闭 IPC 服务

### `IPC_Client`
- `send_data()`：发命令
- `latest_state()`：读心跳最新状态
- `is_online()`：判断在线

作用：
- 允许外部进程远程控制启动/停止/录制

---

## 5.14 `teleop/utils/episode_writer.py`

### `EpisodeWriter`
主流程实际会调用：
- `create_episode()`
- `add_item()`
- `save_episode()`
- `close()`

内部机制：
- `add_item()` 只把数据塞进队列
- `process_queue()` 后台线程持续消费
- `_process_item_data()` 负责：
  - 保存图片/深度/音频
  - 追加 JSON
  - 可选 `RerunLogger.log_item_data(...)`
- `_save_episode()` 负责收尾写 JSON 尾部，并保存 `rerun.rrd`

作用：
- 异步录制，尽量不阻塞 teleop 主循环
- 录制时会按 **host monotonic** 做近邻时间对齐，而不是简单拿 latest frame
- 图像文件名现在会带相机 monotonic 时间戳，便于排查

### 当前录制对齐策略

当前数据采集链路已经改成“可靠版”：

1. 主循环定义一个 `sample_monotonic_ns` 作为样本锚点
2. arm state / action 各自维护环形缓冲
3. 本地/远端相机也维护环形缓冲
4. 录制时分别从三类缓冲中取 **最接近 `sample_monotonic_ns`** 的项
5. 若相机在允许时间窗内没有对齐到帧，则跳过这条 sample

因此 `data.json` 中的：

- `timestamps.state`
- `timestamps.action`
- `timestamps.camera.*`

才是离线对齐的正式依据。

### 当前 Rerun 形态

Rerun 现在分为两种用途：

1. **在线监看**
   - 顶部：`head` 图像
   - 底部：曲线 tab
     - `joint_curves`
     - `pose_curves`
   - 时间轴默认展开

2. **离线回放**
   - 每个 episode 保存后会生成 `rerun.rrd`
   - 可直接用 `rerun episode_xxxx/rerun.rrd` 回放

---

## 5.15 `teleop/utils/sim_state_topic.py`

### `SimStateSubscriber`
- `start_subscribe()`：订阅 `rt/sim_state`
- `_subscribe_sim_state()`：持续读仿真状态
- `read_data()`：主线程读取最近一份共享内存里的仿真状态
- `stop_subscribe()`：关闭

作用：
- 仿真录制时把 sim_state 一起录下来

---

## 5.16 `teleop/utils/g1d_mujoco_builder.py`

### `prepare_g1d_mobile_scene()`
作用：
- 从 TWIST2 的 G1D XML 生成本地可移动 MuJoCo 场景
- 可选把 Dex1 手 graft 到 G1D 手腕上
- 增加轮子 actuator、底盘 sensor

主流程里它只在 MuJoCo 入口被调用一次，但作用很关键：
- 决定 MuJoCo 是否支持移动底盘演示

---

## 6. 按“控制链路”再串一次

## 6.1 真机机械臂链路

```text
XRRoboticsWrapper.get_tele_data()
-> TeleData.left_wrist_pose/right_wrist_pose
-> ArmIK.solve_ik()
-> limit_arm_joint_target_velocity()
-> ArmController.ctrl_dual_arm()
-> ArmController._ctrl_motor_state()
-> ChannelPublisher.Write(rt/lowcmd or rt/arm_sdk)
```

## 6.2 真机 Dex1 链路

```text
TeleData.left/right_ctrl_triggerValue
-> left/right_gripper_value (shared memory)
-> Dex1_1_Gripper_Controller.control_thread()
-> np.interp / np.clip
-> ctrl_dual_gripper()
-> ChannelPublisher.Write(rt/dex1/*/cmd)
```

## 6.3 真机灵巧手链路（Dex3 / Inspire / Brainco）

```text
TeleData.left/right_hand_pos
-> shared Array(25x3)
-> HandRetargeting + dex-retargeting
-> retarget(...)
-> hardware joint reorder / normalize
-> ctrl_dual_hand()
-> DDS command topic
```

## 6.4 MuJoCo 链路

```text
XRRoboticsWrapper.get_tele_data()
-> ArmIK.solve_ik()
-> limit_arm_joint_target_velocity()
-> 写 data.qpos / data.ctrl
-> mj_forward + mj_step
```

---

## 7. 当前状态机与操控语义

### 键盘
- `r`：开始 teleop
- `c`：重标定头参考
- `q`：退出
- `s`：录制 toggle

### 手柄
- `grip`：deadman
- 左 Y：双臂回 ready/home
- 右 A：在部分模式下退出
- 左摇杆：底盘平移/前后
- 右摇杆 X：底盘转向
- 右摇杆 Y：G1D 升降柱
- trigger：Dex1 开合

### 头参考模式
- `head_coupled/calibrated`
- `live_head_reference/live`
- `hybrid`
- `fixed_per_grip`

这是当前行为差异最大的配置之一。

---

## 8. review 结论：当前实现的关键特点

## 8.1 优点
1. **链路很清楚**：XR -> TeleData -> IK -> Controller。
2. **真机和 MuJoCo 逻辑统一度高**：核心安全和接管语义一致。
3. **安全机制分层**：
   - deadman
   - outer-loop 关节限速
   - controller 内部再次限速
4. **takeover 逻辑成熟**：使用当前机器人 wrist pose 做锚点，减少接管跳变。
5. **手部控制结构干净**：主线程只负责写共享输入，具体手控制器在各自线程/进程跑。

## 8.2 我认为最值得注意的问题

### 问题 1：`--input-mode` 默认值与当前实现不一致
- `teleop_hand_and_arm.py` 默认 `--input-mode hand`
- 但 `XRRoboticsWrapper.__init__()` 里 `use_hand_tracking=True` 会直接 `raise NotImplementedError`

结论：**当前默认参数实际上不可用**。  
如果不手动传 `--input-mode controller`，主流程会失败。

### 问题 2：`LocoClientWrapper` 方法名和调用名不一致
- 类里定义的是：`Enter_Damp_Mode()`
- 主流程里调用的是：`loco_wrapper.Damp()`

结论：在 `--motion` 模式下，如果触发“双摇杆按下进入阻尼”分支，这里大概率会报错。

### 问题 3：Dex3 左手重定向索引疑似写错
在 `Dex3_1_Controller.control_process()` 里：
- 左手 `left_retargeting.retarget(...)` 的结果使用了 `right_dex_retargeting_to_hardware`

结论：**左手映射疑似误用了右手索引**，需要核对。

### 问题 4：若干文档/默认值与当前推荐流程不完全一致
例如：
- 文档建议主路径是 controller
- 代码默认却是 hand
- 文档推荐 `calibrated`
- 代码默认 `head_reference_mode` 是 `live_head_reference`

结论：**文档、默认参数、当前主路线还有轻微漂移**。

### 问题 5：退出时没有恢复 motion switcher
`finally` 里 `motion_switcher.Exit_Debug_Mode()` 被注释掉了。

结论：退出后模式恢复依赖人工处理。

---

## 9. 如果你要继续读代码，建议顺序

推荐按下面顺序看：

1. `teleop/teleop_hand_and_arm.py`
2. `teleop/utils/xr_robotics_wrapper.py`
3. `teleop/utils/xr_input_types.py`
4. `teleop/robot_control/robot_arm_ik.py`
5. `teleop/robot_control/robot_arm.py`
6. `teleop/robot_control/robot_hand_unitree.py`
7. `teleop/robot_control/hand_retargeting.py`
8. `teleop/robot_control/dex-retargeting/src/dex_retargeting/*`
9. `teleop/demo_xrobotics_mujoco.py`

---

## 10. 一句话总结

这套 `xr_teleoperate` 当前本质上是一个 **“XR-Robotics 输入驱动的双臂/手部远程操作框架”**：
- `XRRoboticsWrapper` 负责把 XR 数据变成项目自己的控制语义；
- `robot_arm_ik.py` 负责把手腕目标位姿变成关节角；
- `robot_arm.py` / `robot_hand_*` 负责把这些目标持续下发到真机 DDS；
- `demo_xrobotics_mujoco.py` 则把同一套控制逻辑映射到 MuJoCo 验证。
