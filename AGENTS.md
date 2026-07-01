# AGENTS.md

## 当前项目定位

这个仓库当前主线是：

- XR-Robotics 输入
- `xr_teleoperate` 真机 / MuJoCo 控制
- 机器人遥操作数据采集
- 本地相机或远端 `ZMQ raw` 图像录制
- Rerun 实时显示与离线回放

当前已经不再依赖：

- `televuer`
- `teleimager`
- websocket / WebRTC 图像链路
- 旧的 `--xr-source` / `--display-mode` 图像显示路线

---

## 当前环境约定

- 当前项目测试默认使用 conda 环境 `dex`
- 远程主机是 `luopengcheng@yxgn-unitree-001`
- 远程对应仓库路径是 `~/programs/xr_teleoperate`
- 远程环境是 `tv`
- 本地对应的同名仓库在 `/home/luopengcheng/Programs/Unitree`
- 进入那个仓库时，优先遵守它自己的 `AGENTS.md`
- 本仓库所有 `python` / 单测相关命令都应在已激活的 `dex` 环境中执行，不要直接用系统 Python

---

## 当前主链路

```text
XR-Robotics SDK
-> XRRoboticsWrapper
-> TeleData
-> Arm IK / Hand Mapping
-> Robot Arm / Hand Controller
-> 真机 DDS 或 MuJoCo
-> 可选 EpisodeWriter 数据录制
-> 可选 Rerun 实时显示 / 离线回放
```

---

## 当前关键文件

### 真机入口

- `teleop/real/teleop_hand_and_arm.py`

### MuJoCo 入口

- `teleop/demo_xrobotics_mujoco.py`

### XR 输入适配

- `teleop/utils/xr_robotics_wrapper.py`
- `teleop/utils/xr_input_types.py`

### 机械臂 / 手控制

- `teleop/robot_control/robot_arm.py`
- `teleop/robot_control/robot_arm_ik.py`
- `teleop/robot_control/robot_hand_unitree.py`

### 数据录制

- `teleop/utils/episode_writer.py`
- `teleop/utils/local_camera.py`

### Rerun

- `teleop/utils/rerun_visualizer.py`

### 运行文档

- `RUN.md`

---

## 当前真机控制逻辑

状态机大意：

```text
启动
-> go_home
-> 等待接管
-> grip 作为 deadman
-> 按住 grip 才允许运动
-> 松开 grip 则 hold
-> 退出时 go_home
```

当前主用头参考模式是：

- `fixed_per_grip`

不再把需要手动 `C` 标定的模式当作主路线。

---

## 当前数据采集逻辑

### 采集入口

录制逻辑主要在：

- `teleop/real/teleop_hand_and_arm.py`
- `EpisodeWriter`

### 相机来源

当前支持两类：

1. 本地相机
   - `LocalCameraStream`
2. 远端相机
   - `ZMQRawCameraReceiver`

对应参数：

- 本地：
  - `--head-camera-id`
  - `--left-camera-id`
  - `--right-camera-id`
  - `--camera-width`
  - `--camera-height`
  - `--camera-fps`
  - `--camera-fourcc`
  - `--camera-buffer-size`
- 远端：
  - `--head-zmq-endpoint`
  - `--left-zmq-endpoint`
  - `--right-zmq-endpoint`

### 机械臂录制表示

当前支持：

- `--record-arm-repr qpos`
- `--record-arm-repr pose`
- `--record-arm-repr both`

其中 pose 模式记录的是：

- `xyz`
- `rpy`
- `rotation_matrix`
- `matrix4x4`

---

## 当前时间对齐逻辑

这是当前数据采集最重要的维护点之一。

### 现状

录制不是“读到什么立刻写什么”，而是按 host 侧时间做对齐：

- `state_history`
- `action_history`
- `camera history`

都会按 host monotonic 时间戳缓存，再取最接近采样时刻的一条。

### 关键实现位置

- `teleop/real/teleop_hand_and_arm.py`
- `teleop/utils/local_camera.py`
- `teleop/utils/episode_writer.py`

### 目前对齐依据

主要用：

- `sample_monotonic_ns`
- `state.host_monotonic_ns`
- `action.host_monotonic_ns`
- `camera.host_recv_monotonic_ns` 或 `host_monotonic_ns`

不是直接依赖 sender 端自己的 monotonic。

### 当前机制

1. 按下录制后，episode 先进入 armed 状态
2. 等待所有已启用相机收到“录制开始之后的第一帧”
3. 再正式开始写样本

对应日志：

- `waiting for first post-start camera frame...`
- `first post-start camera frame received...`

如果后续维护时又把相机链路改成会断流 / 重启，就会重新触发大量：

- `skip sample: no aligned camera frame found after record start.`

---

## 当前远端 ZMQ 相机对接结论

远端 sender 仓库现在按 `XRAW` 协议发送 raw 图像。

接收端在：

- `teleop/utils/episode_writer.py`
  - `ZMQRawCameraReceiver`

当前已确认的工程约定：

1. sender `--stereo-camera` 给 Pico 可走双目 SBS
2. 数采 raw 路只取 **LEFT**
3. host 侧把它作为单路 RGB 录入
4. host 对齐使用接收时刻 `host_recv_monotonic_ns`

因此：

- 若 sender 端 listener / viewer 接入导致 raw 断流或改 shape
- host 侧就会直接表现为对齐 warning

后续看到 warning 时，优先先查 sender 是否在重启 / 改分辨率，不要先怀疑 `xr_teleoperate` 对齐逻辑本身。

---

## 当前数据落盘格式

当前**不是 LeRobot 原生格式**。

落盘格式是项目自定义的：

```text
task_dir/
  episode_0001/
    colors/
    depths/
    audios/
    data.json
    rerun.rrd
```

`data.json` 中每个 item 主要字段有：

- `idx`
- `colors`
- `depths`
- `states`
- `actions`
- `tactiles`
- `audios`
- `sim_state`
- `timestamps`

图片单独保存在 `colors/` 下，再把相对路径写回 JSON。

所以如果用户说“导出 LeRobot”，应理解为：

- 需要额外写 exporter
- 不是当前原生落盘能力

---

## 当前 Rerun 结论

当前已经支持：

1. 实时显示
2. episode 保存后生成 `rerun.rrd`
3. 离线回放

当前默认布局：

- 上半部分：`head` 图像
- 下半部分：曲线区
  - `joint_curves`
  - `pose_curves`

回放入口：

- 直接打开 `episode_xxxx/rerun.rrd`

如果用户说“Rerun 没有视频”，优先检查：

1. 录制时是否真的写入了 `colors/head`
2. 是否保存了 `rerun.rrd`
3. 离线回放是否打开的是正确 episode

---

## 当前修改时的联动规则

### 改录制字段时

至少同时检查：

- `teleop/real/teleop_hand_and_arm.py`
- `teleop/utils/episode_writer.py`
- `teleop/utils/rerun_visualizer.py`
- `RUN.md`

### 改相机时间对齐时

至少同时检查：

- `teleop/real/teleop_hand_and_arm.py`
- `teleop/utils/local_camera.py`
- `teleop/utils/episode_writer.py`

### 改远端 ZMQ 相机协议时

至少同时检查：

- `teleop/utils/episode_writer.py`
- sender 仓库 `zed_webcam_common.cpp`
- sender 仓库 `agents.md`

### 改数据格式时

至少同时检查：

- `EpisodeWriter`
- `rerun_visualizer.py`
- `RUN.md`

因为当前 JSON 结构、图片路径、Rerun 回放三者是绑定的。

---

## 当前维护建议

1. 不要把 `televuer/teleimager` 路线再加回来。
2. 采集稳定性优先级高于“显示花样”。
3. 看到对齐 warning 时，先检查远端 sender 是否断流 / 改 shape / 重启。
4. 若要进一步提升鲁棒性，优先方向是：
   - sender raw 发布更独立
   - host 侧增加相机 shape 变化检测日志
   - exporter 到 LeRobot，而不是直接改原始录制结构

---

## 当前最短阅读路径

如果新接手的是“真机 + 录制 + 远端相机”问题，建议按顺序读：

1. `RUN.md`
2. `teleop/real/teleop_hand_and_arm.py`
3. `teleop/utils/episode_writer.py`
4. `teleop/utils/local_camera.py`
5. `teleop/utils/rerun_visualizer.py`
6. sender 仓库 `XRoboToolkit-Ubuntu-Video-Sender-Webcam-dev-test/agents.md`

这比先读老的 XR 显示链路更有效。
