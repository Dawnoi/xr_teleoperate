# VIVE Tracker 遥操启动手册

适用路径：

- `xr_teleoperate`: `/data/codeBase/src/unitree_ws/src/xr_teleoperate`
- `vive_locator`: `/data/codeBase/src/unitree_ws/src/vive_locator`
- workspace: `/data/codeBase/src/unitree_ws`

当前链路：

```text
HTC Tracker 3.0 + SteamVR 定位器 2.0
        ↓ libsurvive
vive_locator
        ↓ /vive_pose_l /vive_pose_r
xr_teleoperate Vive provider
        ↓ TeleopInputSample / MotionIntent
Unitree teleop real / MuJoCo
```

---

## 相关启动文件

| 文件 | 作用 |
|---|---|
| `../vive_locator/launch/vive_dual_locator.launch.py` | 启动双 Tracker 定位，发布 `/vive_pose_l`、`/vive_pose_r` |
| `../vive_locator/launch/vive_single_locator.launch.py` | 启动单 Tracker 定位 |
| `../vive_locator/scripts/calibration_first_time.bash` | 第一次 libsurvive / 基站标定 |
| `../vive_locator/scripts/recalibrate_after_move.bash` | 基站移动后重新标定 |
| `../vive_locator/scripts/refresh_lighthouse_ootx.bash` | 只刷新基站 OOTX 数据 |
| `../vive_locator/scripts/set_tracker_codes.bash` | 固定左右 Tracker code |
| `scripts/vive_axis_calibrator.py` | 按 AGX 三点法标定 `vive_world -> robot base` 的旋转和原点偏移 |
| `scripts/vive_keyboard_enable.py` | VIVE 专用左右踏板 deadman，不影响其他 input provider |
| `configs/vive/99-xr-teleoperate-pedal.rules` | 键盘类踏板的通用 udev 权限模板 |
| `scripts/start/start_vive_locator.sh` | 复用/按需执行 libsurvive 标定，等待后自动启动双 Tracker locator |
| `scripts/start/start_real_robot_vive.sh` | 真机 VIVE 遥操启动脚本 |
| `teleop/sim/xrobotics_mujoco.py` | MuJoCo VIVE 输入测试入口 |

---

## 0. 编译

```bash
cd /data/codeBase/src/unitree_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select vive_locator --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

> `xr_teleoperate` 是 Python 项目，通常不需要 colcon 编译；运行时使用 `tv` conda 环境。

---

## 1. libsurvive / 基站标定

第一次使用：

```bash
cd /data/codeBase/src/unitree_ws/src/vive_locator
bash scripts/calibration_first_time.bash --v 5
```

基站移动后：

```bash
cd /data/codeBase/src/unitree_ws/src/vive_locator
bash scripts/recalibrate_after_move.bash --v 5
```

只刷新 OOTX：

```bash
cd /data/codeBase/src/unitree_ws/src/vive_locator
bash scripts/refresh_lighthouse_ootx.bash --v 5
```

正常日志应能看到类似：

```text
Available Posers
Using 'MPFIT' for poser
Loaded drivers: GlobalSceneSolver, HTCVive
Detected LH gen 2 system
Adding tracked object
Adding lighthouse ch 0/1
Global solve
```

不要把“每次启动都强制重做 libsurvive 标定”作为默认流程：它可能改变 `vive_world` 原点，导致已经保存的 robot-frame 三点标定不再对应。推荐让启动脚本默认复用缓存，只有基站移动或缓存损坏时设置 `VIVE_RECALIBRATE=1`。

如果出现：

```text
Cannot find any valid Poser
Cannot find any valid Disambiguator
```

优先检查是否运行的是当前 workspace 的 `vive_locator`，以及 `SURVIVE_PLUGINS` 是否由脚本/launch 注入。

---

## 2. 启动 VIVE locator

推荐使用一键脚本。它会复用已有的 libsurvive 房间标定缓存；首次没有缓存时自动执行标定，标定结束后等待 5 秒再启动双 Tracker ROS 节点：

```bash
cd /data/codeBase/src/unitree_ws/src/xr_teleoperate
bash scripts/start/start_vive_locator.sh
```

基站移动后需要强制重新标定时：

```bash
VIVE_RECALIBRATE=1 bash scripts/start/start_vive_locator.sh
```

可调等待时间：

```bash
VIVE_LOCATOR_WAIT_SEC=5 bash scripts/start/start_vive_locator.sh
```

脚本只负责 libsurvive 房间标定和 `vive_locator` 启动，不会自动重做 `vive_axis_calibrator.py` 的 robot-frame 三点标定；已有 `~/.config/xr_teleoperate/vive_calibration.json` 会继续复用。

自动分配左右 Tracker：

```bash
cd /data/codeBase/src/unitree_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch vive_locator vive_dual_locator.launch.py
```

固定左右 Tracker code：

```bash
cd /data/codeBase/src/unitree_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
source src/vive_locator/scripts/set_tracker_codes.bash T20 T21
ros2 launch vive_locator vive_dual_locator.launch.py auto_assign_empty_codes:=false
```

输出 topic：

```text
/vive_pose_l
/vive_pose_r
/vive_localization_status_l
/vive_localization_status_r
```

检查：

```bash
ros2 topic echo /vive_pose_l --once
ros2 topic echo /vive_pose_r --once
```

---

## 3. 标定 `vive_world -> robot base`

另开终端：

```bash
cd /data/codeBase/src/unitree_ws/src/xr_teleoperate
source /opt/ros/humble/setup.bash
python scripts/vive_axis_calibrator.py \
  --pose-topic /vive_pose_l \
  --output-file ~/.config/xr_teleoperate/vive_calibration.json
```

键盘操作：

```text
o = 记录 robot 原点 / reference 点
x = Tracker 沿 robot +X 方向移动后记录
y = Tracker 沿 robot +Y 方向移动后记录
p = 打印 rotation 和 offset 参数
s = 保存 ~/.config/xr_teleoperate/vive_calibration.json
q = 退出
```

建议：

- `x/y` 至少移动 15-20 cm。
- `+X` 和 `+Y` 不要采得太接近平行。
- 算法与 AGX 一致：三点求 `rotation_robot_from_vive`，并由 robot origin 求 `offset_xyz`。
- 左右 Tracker 安装姿态仍通过标定文件中的 `left_mount_rotation/right_mount_rotation` 单独调整。

生成文件示例：

```json
{
  "rotation_robot_from_vive": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
  "offset_xyz": [0.0, 0.0, 0.0],
  "left_mount_rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
  "right_mount_rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
  "position_scale": 1.0,
  "output_frame": "base_link"
}
```

---

## 4. 配置并启动 VIVE 专用踏板 deadman

### 4.1 正常启动：自动识别

程序默认扫描 `/dev/input/by-path/*-event-kbd`，用物理 USB 端口区分同型号踏板，再读取 sysfs 设备名称，优先选择名称中包含 `pedal`、`foot` 或常见 `key08` 的设备；若系统只有一个键盘类候选，也会直接使用。正常情况不需要查询或填写 USB 路径：

```bash
cd /data/codeBase/src/unitree_ws/src/xr_teleoperate
source /opt/ros/humble/setup.bash
python scripts/vive_keyboard_enable.py \
  --grab-input-devices
```

如果物理左右相反：

```bash
python scripts/vive_keyboard_enable.py \
  --grab-input-devices \
  --swap-sides
```

启动日志会显示自动选中的设备和左右映射。`--grab-input-devices` 只抓取选中的踏板，阻止踏板按键串入终端或其他窗口。

```text
按住 Alt+L = 左臂使能
松开 Alt+L = 左臂保持
按住 Alt+R = 右臂使能
松开 Alt+R = 右臂保持
```

### 4.2 自动识别失败时

若存在多个名称无法判断的键盘设备，节点会拒绝启动并直接打印候选列表，不会猜测或抓取普通键盘。此时才需要插拔对比或 `evtest`：

```bash
ls -l /dev/input/by-path/*-event-kbd
evtest /dev/input/by-path/<your-pedal>-event-kbd
```

确认踏板按住时产生 key-down、松开时产生 key-up，然后用日志中的稳定路径兜底：

```bash
python scripts/vive_keyboard_enable.py \
  --input-device /dev/input/by-path/<your-pedal>-event-kbd \
  --grab-input-devices
```

两只独立踏板可重复传入 `--input-device`。只在踩下时发送一次完整宏、松开时没有 key-up 的设备不能提供 hold-to-enable deadman。

### 4.3 配置读取权限

若节点报告 `Permission denied`，先检查设备属性和当前权限：

```bash
udevadm info --query=property --name=/dev/input/by-path/<your-pedal>-event-kbd | grep '^ID_INPUT_KEYBOARD=1$'
getfacl /dev/input/by-path/<your-pedal>-event-kbd
test -r /dev/input/by-path/<your-pedal>-event-kbd && echo readable
```

通用规则依赖 `ID_INPUT_KEYBOARD=1`。若第一条命令没有输出，不要扩大成匹配所有 input 设备的规则，应按该踏板的 vendor/product 属性单独写一条窄规则。

若不可读，可安装仓库中的通用规则，然后重新插拔踏板：

```bash
sudo install -m 0644 configs/vive/99-xr-teleoperate-pedal.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=input
```

回滚规则：

```bash
sudo rm /etc/udev/rules.d/99-xr-teleoperate-pedal.rules
sudo udevadm control --reload-rules
```

VIVE provider 初始为左右都禁用；只有 Tracker pose 和对应踏板心跳都未超时，该侧才进入控制。设备断开、input 事件丢失、节点退出或心跳超过 `--vive-enable-timeout-sec`（默认 `0.5 s`）都会清理对应 anchor 并保持当前腕位姿。该节点只发布 `/vive/enable_left` 和 `/vive/enable_right`，XR、回放和推理 provider 不订阅这两个 topic。

---

## 5. MuJoCo 测试

```bash
cd /data/codeBase/src/unitree_ws/src/xr_teleoperate
python teleop/sim/xrobotics_mujoco.py \
  --input-provider vive \
  --vive-calibration-file ~/.config/xr_teleoperate/vive_calibration.json
```

如果只想临时传矩阵，不用文件：

```bash
python teleop/sim/xrobotics_mujoco.py \
  --input-provider vive \
  --vive-rotation-robot-from-vive R00 R01 R02 R10 R11 R12 R20 R21 R22
```

---

MuJoCo 同样要求先运行踏板 deadman 节点。

---

## 6. 真机启动

确认 `~/.config/xr_teleoperate/vive_calibration.json` 存在后：

```bash
cd /data/codeBase/src/unitree_ws/src/xr_teleoperate
VIVE_CALIBRATION_FILE=~/.config/xr_teleoperate/vive_calibration.json \
bash scripts/start/start_real_robot_vive.sh
```

常用环境变量：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `CONDA_ENV` | `tv` | Python/机器人运行环境 |
| `NETWORK_INTERFACE` | `eno1` | Unitree DDS 网卡 |
| `VIVE_CALIBRATION_FILE` | `~/.config/xr_teleoperate/vive_calibration.json` | VIVE 坐标轴标定文件 |
| `MAX_ARM_JOINT_SPEED` | `1.0` | 真机遥操最大关节速度 |
| `REAL_TELEOP_ENTRY` | `teleop/real/teleop_hand_and_arm.py` | 真机入口 |

启动脚本内部等价于：

```bash
python teleop/real/teleop_hand_and_arm.py \
  --input-provider vive \
  --input-mode controller \
  --arm G1_29 \
  --ee dex1 \
  --network-interface "$NETWORK_INTERFACE" \
  --base-controller g1d_agv \
  --controller-deadman grip \
  --head-reference-mode fixed_per_grip \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --vive-calibration-file "$VIVE_CALIBRATION_FILE"
```

---

## 推荐终端布局

```text
Terminal 1: start_vive_locator.sh（自动复用/按需标定并启动 locator）
Terminal 2: vive_axis_calibrator，仅首次/基站变动后需要
Terminal 3: vive_keyboard_enable（踏板 deadman）
Terminal 4: MuJoCo 或 start_real_robot_vive.sh
```

真机建议先跑 MuJoCo，确认：

```text
沿 robot +X 移动 Tracker -> 机器人目标主要沿 X 变
沿 robot +Y 移动 Tracker -> 机器人目标主要沿 Y 变
```

再上真机。

---

## 注意事项

1. 基站移动后，先跑 `recalibrate_after_move.bash`，再根据需要重新跑 `vive_axis_calibrator.py`。
2. VIVE provider 使用与 PICO `anchored_safe + relative` 一致的姿态语义：使能上升沿记录 Tracker 和机器人腕部 anchor，之后叠加位置与姿态增量。
3. 标定先按 AGX 公式把 VIVE pose 映射到 robot frame；增量控制下 offset 会在 anchor 差分中自然抵消，但仍保存在标定文件中以保持完整坐标约定。
4. 踏板松开、踏板心跳或 Tracker 数据超时都会清理对应 anchor；再次使能或恢复定位时从机器人当前腕部重新接管。
5. 踏板使能逻辑只存在于 VIVE provider，不改变其他 input provider 的 deadman 语义。
