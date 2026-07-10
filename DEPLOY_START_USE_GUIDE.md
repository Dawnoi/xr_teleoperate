# xr_teleoperate 部署、启动、使用手册

适用分支：

```text
feature/vla-latency-trace
```

---

## 0. quick start

常用目录：

```text
仓库路径              /home/luopengcheng/Programs/xr_teleoperate
真机入口              teleop/real/teleop_hand_and_arm.py
启动脚本              scripts/start/
遥操作脚本            scripts/start/start_real_robot_wired.sh
数采脚本              scripts/start/start_real_robot_wired_3cams_zmq.sh
VLA 启动              scripts/start/start_real_robot_vla.sh
延迟画图              tests/diagnostics/plot_latency_trace.py
数据审查              scripts/data/audit_multi_cam_record.sh
```

运行环境设置：
```bash
cd /home/luopengcheng/Programs/xr_teleoperate

conda activate tv
```
DDS 网络检查：

```bash
# 1. 先看本机有哪些网卡和网段
ip -br addr

# 2. 扫描候选网段，确认机器人在哪个网段
nmap -sn 192.168.123.0/24

# 有线连接条件下通常为 enx9c69d3212b05

# 3. 用实际连到机器人的网卡跑 DDS debug
# NETWORK_INTERFACE=<实际网卡名> bash scripts/debug/debug_dds_wired.sh

NETWORK_INTERFACE=enx9c69d3212b05 bash scripts/debug/debug_dds_wired.sh

```

普通真机遥操作：

```bash
NETWORK_INTERFACE=enx9c69d3212b05 bash scripts/start/start_real_robot_wired.sh
```

启动真机数采和VLA推理需要先在宇树开发机上设置相机sender：

```bash
ssh unitree@192.168.100.78 # 密码默认为123
cd ~/XRoboToolkit-Ubuntu-Video-Sender-Webcam-dev-test
conda activate tv
bash start_four_cams_new.sh
```

真机数采脚本：

```bash
SENDER_IP=192.168.123.164 \
NETWORK_INTERFACE=enx9c69d3212b05 \
TASK_DIR=./utils/data \
TASK_NAME=multi_cam_record \
RECORD_ARM_REPR=both \
bash scripts/start/start_real_robot_wired_3cams_zmq.sh
```

VLA 真机运动：
当前默认使用的推理后端在a100上，但是a100只开放了固定端口，需要ssh转发到实际运行的推理服务端口：
```bash
ssh -L 18027:127.0.0.1:8027 a100 # a100 是已配置别名；切换主机需要重新配置
```
另外启动一个终端窗口运行：
```bash
VLA_ENABLE_MOTION=1 \
VLA_BASE_URL=http://127.0.0.1:18027 \
SENDER_IP=192.168.123.164 \
NETWORK_INTERFACE=enx9c69d3212b05 \
bash scripts/start/start_real_robot_vla.sh
```

VLA 延迟分析：在真机运动命令后追加 trace 参数：

```bash
VLA_ENABLE_MOTION=1 \
VLA_BASE_URL=http://127.0.0.1:18027 \
SENDER_IP=192.168.123.164 \
NETWORK_INTERFACE=enx9c69d3212b05 \
bash scripts/start/start_real_robot_vla.sh \
  --latency-trace \
  --latency-trace-path ./utils/data/latency_trace.jsonl \
  --latency-summary-every 10
```

trace 参数说明：

```text
--latency-trace
  开启延迟记录。

--latency-trace-path ./utils/data/latency_trace.jsonl
  指定 trace 输出文件。

--latency-summary-every 10
  每完成 10 条 trace 在日志里打印一次统计摘要。
```

延迟画图：

```bash
python tests/diagnostics/plot_latency_trace.py \
  utils/data/latency_trace.jsonl \
  --output utils/data/latency_trace.png \
  --top-report utils/data/latency_trace_top.md
```

数据导出：

```bash
RAW_ROOT=utils/data/multi_cam_record \
OUTPUT_ROOT=utils/data/multi_cam_record_lerobot \
TASK="your task prompt" \
bash scripts/data/export_raw_to_lerobot.sh

LEROBOT_ROOT=utils/data/multi_cam_record_lerobot \
OUTPUT_ROOT=utils/data/multi_cam_record_umi_dp \
bash scripts/data/export_lerobot_to_umi_dp.sh
```

数据回放：

```bash

NETWORK_INTERFACE=enx9c69d3212b05 \
bash scripts/replay/replay_raw_episode_real.sh \
  --dataset-root utils/data/multi_cam_record \
  --episode-index 0 \
  --arm-source action \
  --end-action home \
  --motion
```

---

---

## 1. 部署前提

### 1.1 机器和路径

仓库路径：

```text
cd ~/programs/xr_teleoperate
```
真机运行环境：
```text
conda activate tv
```

### 1.2 代码状态检查

```bash
cd /home/luopengcheng/Programs/xr_teleoperate
git status --short --branch
git branch --show-current
```

确认当前分支：

```text
feature/vla-latency-trace
```

## 2. 网络和 DDS 检查

### 2.1 扫 IP 找实际网卡

先看本机当前网卡和 IP：

```bash
ip -br addr
```

输出形态类似：

```text
lo                 UNKNOWN        127.0.0.1/8 ::1/128
eno1               UP             192.168.123.10/24
wlo1               UP             192.168.1.23/24
enx9c69d3212b05    UP             192.168.123.20/24
```

查找 Unitree 所在的网段，默认网段是：192.168.123.0/24
记录对应的网卡名称，有线连接时默认为 enx9c69d3212b05 


判断方式：

```text
哪个网卡的 IP 和机器人 IP 在同一网段，就用哪个网卡。

```bash
nmap -sn 192.168.123.0/24
```

例子：
  机器人在 192.168.123.x
  本机 eno1 是 192.168.123.10/24
  那 NETWORK_INTERFACE=eno1
```

如果不确定机器人 IP，可以先扫描所有本机非 lo 网卡所在网段：

```bash
ip -o -4 addr show scope global | awk '{print $2, $4}'
```

把输出里的 `/24` 网段逐个用 `nmap -sn` 扫即可。

### 2.2 DDS topic 检查

先检查DDS服务是否正常运行。
用上一步确认出来的实际网卡：

```bash
NETWORK_INTERFACE=<实际网卡名> bash scripts/debug/debug_dds_wired.sh
```

例如：

```bash
NETWORK_INTERFACE=enx9c69d3212b05 bash scripts/debug/debug_dds_wired.sh
```

如果是 Wi-Fi 连机器人，也可以直接指定 Wi-Fi 网卡：

```bash
NETWORK_INTERFACE=wlo1 bash scripts/debug/debug_dds_wifi.sh
```

如果 DDS 不通，先不要启动真机运动。先检查：

```text
扫描到的机器人 IP 是否和 NETWORK_INTERFACE 在同一网段
NETWORK_INTERFACE 是否填成了网卡名，而不是 IP
是否有其它进程占用/干扰 DDS
是否在正确 conda 环境
```

---

## 3. 普通真机遥操作启动

### 3.1 有线启动

```bash
NETWORK_INTERFACE=enx9c69d3212b05 bash scripts/start/start_real_robot_wired.sh
```

脚本实际调用：

```text
python teleop/real/teleop_hand_and_arm.py
```

核心参数：

```text
--input-mode controller
--arm G1_29
--ee dex1
--network-interface ${NETWORK_INTERFACE}
--base-controller g1d_agv
--controller-deadman grip
--head-reference-mode fixed_per_grip
--controller-mapping-mode anchored_safe
--controller-orientation-mode relative
--max-arm-joint-speed 5.0
--arm-workspace-mode tapered
```

### 3.2 Wi-Fi 启动

```bash
NETWORK_INTERFACE=wlo1 bash scripts/start/start_real_robot_wifi.sh
```

### 3.3 运行中按键

```text
r
  开始 START，同步机器人跟随输入。

q
  停止并退出。

h
  双臂回 ready/home。

s
  录制模式下开始/保存 episode。

v
  录制模式下取消当前 episode。

c
  有线脚本里有这个按键，但当前 fixed_per_grip 不走头参考标定，平时不用按。
  只有切换到需要标定的头参考模式时才有用。
```

### 3.4 操作逻辑

当前主用模式：

```text
controller-deadman = grip
head-reference-mode = fixed_per_grip
controller-mapping-mode = anchored_safe
controller-orientation-mode = relative
```

含义：

```text
按住 grip 才允许对应手臂运动
每次按下 grip 自动建立本次接管参考
松开 grip 后保持当前位置
```

---

## 4. 相机和录制

### 4.1 宇树相机推流环境

真机数采前，先在宇树机器人上启动相机推流。这个步骤在 `unitree@192.168.100.75` 上执行，不是在 `xr_teleoperate` 仓库所在的 host 上执行。

当前使用的是 `XRoboToolkit-Ubuntu-Video-Sender-Webcam-dev-test` 仓库里的四相机启动脚本。

登录宇树机器人：

```bash
ssh unitree@192.168.100.75
```

先确认相机设备：

```bash
v4l2-ctl --list-devices
```

当前 `unitree001` 的相机对应关系：

```text
001            left_wrist
JR002          right_wrist
USB2.0 Camera  head
```

进入 sender 仓库并启动推流：

```bash
cd ~/XRoboToolkit-Ubuntu-Video-Sender-Webcam-dev-test
conda activate tv
bash start_four_cams_new.sh
```

启动后，host 侧的 `xr_teleoperate` 会通过 ZMQ endpoint 接收相机 raw 流：

```text
head        tcp://${SENDER_IP}:${HEAD_ZMQ_PORT}
left_wrist  tcp://${SENDER_IP}:${LEFT_WRIST_ZMQ_PORT}
right_wrist tcp://${SENDER_IP}:${RIGHT_WRIST_ZMQ_PORT}
```

注意：

```text
SENDER_IP 要填写宇树相机机器在机器人/host 网络里的 IP。
如果 host 侧录制没有图像，先确认 start_four_cams_new.sh 仍在运行。
```

### 4.2 真机数采脚本：三相机 ZMQ 录制

```bash
SENDER_IP=192.168.123.164 \
NETWORK_INTERFACE=enx9c69d3212b05 \
TASK_DIR=./utils/data \
TASK_NAME=multi_cam_record \
RECORD_ARM_REPR=both \
bash scripts/start/start_real_robot_wired_3cams_zmq.sh
```

这个脚本在 host 侧运行，用于真机遥操作数采，会启动：

```text
XR/controller 输入
G1_29 双臂
Dex1 夹爪
G1D AGV 底盘 bridge
三路 ZMQ raw 相机
raw episode 录制
```

录制数据可后续导出到 LeRobot / UMI-DP。

脚本底层仍然调用唯一真机入口：

```text
teleop/real/teleop_hand_and_arm.py
```

脚本默认会加这些录制参数：

```text
--record
--task-dir ${TASK_DIR}
--task-name ${TASK_NAME}
--record-arm-repr ${RECORD_ARM_REPR}
--head-zmq-endpoint tcp://${SENDER_IP}:${HEAD_ZMQ_PORT}
--left-zmq-endpoint tcp://${SENDER_IP}:${LEFT_WRIST_ZMQ_PORT}
--right-zmq-endpoint tcp://${SENDER_IP}:${RIGHT_WRIST_ZMQ_PORT}
```

环境变量：

```text
NETWORK_INTERFACE
  DDS 网卡，默认 enx9c69d3212b05。

SENDER_IP
  ZMQ raw 相机 sender IP。

HEAD_ZMQ_PORT
LEFT_WRIST_ZMQ_PORT
RIGHT_WRIST_ZMQ_PORT
  默认 5556 / 5557 / 5558。

TASK_DIR
  数据保存根目录。

TASK_NAME
  任务名。

RECORD_ARM_REPR
  qpos / pose / both。
```

### 4.3 录制按键

启动后：

```text
r 进入 START
s 开始录制
s 保存当前 episode
v 取消当前 episode
q 退出
```

### 4.4 数据保存结构

当前 raw episode 不是 LeRobot 原生格式。

典型结构：

```text
TASK_DIR/
  TASK_NAME/
    episode_0000/
      colors/
      data.json
      rerun.rrd
```

`data.json` 中包含：

```text
states
actions
colors
timestamps
```

图片保存在 `colors/` 目录，JSON 中保存相对路径。

### 4.5 导出格式

LeRobot / UMI-DP 等 exporter 格式和使用命令详见第 8 节。

---

## 5. VLA 在线推理启动

### 5.1 服务端地址

VLA 脚本使用：

```text
VLA_BASE_URL=http://127.0.0.1:18027
```

当前默认推理后端在 A100 上，实际服务端口是 `8027`。真机侧脚本仍然访问本机 `127.0.0.1:18027`，所以需要先开 SSH 端口转发，把本机 `18027` 转到 A100 的 `8027`。

终端 1：保持端口转发一直运行。

```bash
ssh -L 18027:127.0.0.1:8027 a100
```

`a100` 是当前已配置的 SSH 别名；如果切换主机，或者 A100 服务端口改了，需要同步修改 SSH 目标或端口。

终端 2：启动 VLA。此时 `VLA_BASE_URL` 仍然写本机地址：

```bash
VLA_BASE_URL=http://127.0.0.1:18027
```

如果本机 `18027` 已被占用，可以把本机端口换成其它端口，同时同步修改 `VLA_BASE_URL`。

VLA 启动脚本默认传入的推理接口格式是：

```text
pi05_dual_arm_20d
```

服务返回给机器人侧的每个动作点是 20 维：

```text
前 10 维   左臂
后 10 维   右臂

每只臂 10 维 = xyz 位置 3 维 + rot6d 姿态 6 维 + gripper 夹爪 1 维
```

### 5.2 VLA dry-run

dry-run 只发观测、解析动作，不下发机械臂运动。

```bash
VLA_ENABLE_MOTION=0 \
VLA_BASE_URL=http://127.0.0.1:18027 \
SENDER_IP=192.168.123.164 \
bash scripts/start/start_real_robot_vla.sh
```

建议第一次连接模型服务时先 dry-run。

### 5.3 VLA 真机运动

```bash
VLA_ENABLE_MOTION=1 \
VLA_BASE_URL=http://127.0.0.1:18027 \
SENDER_IP=192.168.123.164 \
NETWORK_INTERFACE=enx9c69d3212b05 \
bash scripts/start/start_real_robot_vla.sh
```

这条命令只负责真机 VLA 运动，不开启延迟记录。需要延迟分析时看第 6 节。


关键环境变量：

```text
NETWORK_INTERFACE
  DDS 网卡。

SENDER_IP
  三路 ZMQ 相机 sender IP。

VLA_BASE_URL
  VLA 服务地址。

VLA_PROMPT
  任务文本。

VLA_ENABLE_MOTION
  0 = dry-run
  1 = 真机运动

VLA_AUTO_START
  0 = 启动后按 r
  1 = 自动 START

ARM_CONTROL_HZ
  机械臂 DDS 发布线程频率，默认 250。

MAX_ARM_JOINT_SPEED
  主循环外层关节限速，默认 1.0。
```

---

## 6. 延迟分析

### 6.1 真机运动命令

```bash
VLA_ENABLE_MOTION=1 \
VLA_BASE_URL=http://127.0.0.1:18027 \
SENDER_IP=192.168.123.164 \
NETWORK_INTERFACE=enx9c69d3212b05 \
bash scripts/start/start_real_robot_vla.sh
```

上面这条只启动真机 VLA，不写 trace 文件。

### 6.2 真机运动 + 延迟 trace

```bash
VLA_ENABLE_MOTION=1 \
VLA_BASE_URL=http://127.0.0.1:18027 \
SENDER_IP=192.168.123.164 \
NETWORK_INTERFACE=enx9c69d3212b05 \
ARM_CONTROL_HZ=250 \
bash scripts/start/start_real_robot_vla.sh \
  --latency-trace \
  --latency-trace-path ./utils/data/latency_trace.jsonl \
  --latency-summary-every 10
```

trace 参数说明：

```text
--latency-trace
  开启延迟 trace。
  不加这个参数时，不会生成 latency_trace.jsonl。

--latency-trace-path ./utils/data/latency_trace.jsonl
  trace 输出路径。
  文件格式是 jsonl：一行一个 JSON，每行对应一次动作延迟记录。

--latency-summary-every 10
  每完成 10 条 trace，在日志里打印一次统计摘要。
  数值越小日志越频繁；数值越大日志越安静。

--latency-command-threshold 0.02
  命令变化大于该值才开始记录 trace。
  如果机器人在动但 trace 很少，可以适当调小。

--latency-exec-q-threshold 0.01
  q 变化超过该值认为检测到运动。
  这是按关节位置变化判断“机器人已经开始动”。

--latency-exec-dq-threshold 0.05
  dq 超过该值也认为检测到运动。
  这是按关节速度判断“机器人已经开始动”。

--latency-timeout 2.0
  单条 trace 超时时间。
  超时还没检测到运动，会写 timeout 记录。
```

推荐显式写全阈值，方便复现实验：

```bash
VLA_ENABLE_MOTION=1 \
VLA_BASE_URL=http://127.0.0.1:18027 \
SENDER_IP=192.168.123.164 \
NETWORK_INTERFACE=enx9c69d3212b05 \
ARM_CONTROL_HZ=250 \
bash scripts/start/start_real_robot_vla.sh \
  --latency-trace \
  --latency-trace-path ./utils/data/latency_trace.jsonl \
  --latency-command-threshold 0.02 \
  --latency-exec-q-threshold 0.01 \
  --latency-exec-dq-threshold 0.05 \
  --latency-timeout 2.0 \
  --latency-summary-every 10
```

### 6.3 画图

```bash
python tests/diagnostics/plot_latency_trace.py \
  utils/data/latency_trace.jsonl \
  --output utils/data/latency_trace.png \
  --top-report utils/data/latency_trace_top.md
```

输出：

```text
utils/data/latency_trace.png
utils/data/latency_trace_top.md
```

### 6.4 优先看哪些字段

```text
online_obs_send_to_action_recv_ms
  发观测到收到动作块，主要看模型/网络。

recv_to_pub_ms
  provider 返回动作点到 DDS 发布完成，主要看主循环和控制线程等待。

pub_to_exec_thread_ms
  DDS 发布完成到 lowstate 线程检测运动，主要看 DDS/底层/反馈。

recv_to_exec_thread_ms
  provider 返回动作点到 lowstate 线程检测运动，是动作点输出后的端到端执行反馈。

unknown_pre_pub
  发布前还没被拆出来的耗时。
```

---

## 7. Dex1 夹爪调试

运行前先停掉真机遥操作或 VLA 进程。

```bash
bash scripts/debug/dex1_gripper_keyboard_probe.sh
```

按键：

```text
a / 左键
  闭合

d / 右键
  张开

空格
  目标位置设为当前实际位置

q / ESC
  退出
```

用途：

```text
观察 state_q / target_q / tau_est
验证 Dex1 是否能正常开合
检查 force-hold 参数是否合理
```

---

## 8. 数据审查和导出

### 8.1 Raw 多相机数据审查

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

审查分类：

```text
hard fail
  JSON 损坏、qpos 非法、图片缺失、时间戳严重错误、帧数过短。

review
  夹爪闭合不足、相机对齐偏大、idx 不连续、长帧数异常。
```

### 8.2 Raw -> LeRobot v2

```bash
RAW_ROOT=utils/data/multi_cam_record \
OUTPUT_ROOT=utils/data/multi_cam_record_lerobot \
TASK="your task prompt" \
bash scripts/data/export_raw_to_lerobot.sh
```

### 8.3 排除问题 episode 后导出 LeRobot

```bash
RAW_ROOT=utils/data/multi_cam_record \
PROBLEM_FILE=utils/data/multi_cam_record/problem_episodes.txt \
OUTPUT_ROOT=utils/data/multi_cam_record_lerobot_clean \
TASK="your task prompt" \
bash scripts/data/export_clean_multi_cam_lerobot.sh
```

### 8.4 LeRobot -> UMI/DP

```bash
LEROBOT_ROOT=utils/data/multi_cam_record_lerobot \
OUTPUT_ROOT=utils/data/multi_cam_record_umi_dp \
bash scripts/data/export_lerobot_to_umi_dp.sh
```

只导出部分 episode：

```bash
EPISODES="0 1 2" bash scripts/data/export_lerobot_to_umi_dp.sh
```

---

## 9. 离线回放

先 dry-run 检查 episode 能否正常读取：

```bash
bash scripts/replay/replay_raw_episode_real.sh \
  --dataset-root utils/data/multi_cam_record \
  --episode-index 0 \
  --dry-run
```

确认数据没问题后，再真机回放：

```bash
NETWORK_INTERFACE=enx9c69d3212b05 \
bash scripts/replay/replay_raw_episode_real.sh \
  --dataset-root utils/data/multi_cam_record \
  --episode-index 0 \
  --arm-source action \
  --end-action home \
  --motion
```


常用参数：

```text
--dataset-root
  raw episode 所在目录。

--episode-index
  回放第几个 episode。

--dry-run
  只检查和播放流程，不下发真机运动。

--motion
  允许真机运动。

--no-gripper
  禁用夹爪回放。

--arm-source action
  使用录制时的 action 回放。

--end-action home
  回放结束后回 home。
```

---

## 10. 退出流程

正常退出：

```text
q
```

退出时程序会：

```text
停止 START
机械臂回 home
保持一段时间
关闭输入 provider
关闭 AGV bridge
关闭相机
关闭 recorder
```

如果正在录制：

```text
s 保存 episode
v 取消 episode
q 退出程序
```

---

## 11. 常见问题排查

### 11.1 DDS 不通

先跑：

```bash
ip -br addr
nmap -sn 192.168.123.0/24
NETWORK_INTERFACE=<实际网卡名> bash scripts/debug/debug_dds_wired.sh
```

检查：

```text
机器人 IP 是否能扫到
NETWORK_INTERFACE 是否和机器人 IP 在同一网段
NETWORK_INTERFACE 是否填网卡名，不是 IP 地址
是否多个进程同时控制 DDS
是否使用 tv 环境
```

### 11.2 VLA 没动作

检查：

```text
VLA_ENABLE_MOTION 是否为 1
是否按 r 进入 START
是否有 grip / deadman 使能
online inference 是否进入 failed 状态
workspace / joint speed 是否触发 fatal feedback
transform config 是否正确
```

### 11.3 推理请求超时

检查：

```text
VLA_BASE_URL 是否正确
服务端 /handshake /infer 是否可访问
相机帧是否正常
response_timeout_sec 是否过小
服务端协议是否匹配 pi05_dual_arm_20d
```

### 11.4 录制没有图像

检查：

```text
SENDER_IP 和端口是否正确
sender 是否真的在发 ZMQ raw
head/left/right endpoint 是否配置
是否加了 --headless，导致没有 Rerun 窗口但仍可能正常录制
episode/colors 是否有图片文件
```

### 11.5 latency trace 没数据

检查：

```text
是否加了 --latency-trace
命令变化是否超过 --latency-command-threshold
机器人是否真的运动
latency trace path 是否写到预期位置
```

### 11.6 Dex1 probe 没反应

检查：

```text
是否停掉了真机遥操作 / VLA 进程
NETWORK_INTERFACE 是否正确
Dex1 DDS topic 是否有 state
键盘终端是否能接收 curses 输入
```

---

## 12. 推荐使用顺序

第一次部署：

```text
1. git status 确认分支
2. compileall 检查 Python import/语法
3. DDS debug 检查网络
4. Dex1 probe 检查夹爪
5. 普通 wired teleop 检查手臂
6. VLA dry-run 检查服务端和相机
7. VLA_ENABLE_MOTION=1 真机运行
8. 加 --latency-trace 做延迟分析
9. 画图看 recv_to_pub / pub_to_exec_thread / unknown_pre_pub
10. 录制数据后跑 audit
```
