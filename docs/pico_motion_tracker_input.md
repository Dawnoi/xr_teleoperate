# PICO Motion Tracker 输入模式

该模式只替换左右腕部的位姿来源：

- 原有 `controller` 模式保持不变：左右手柄位姿直接驱动腕部输入。
- 新增 `motion_tracker` 模式：左右腕部位姿来自 PICO Object/Motion Tracker。
- 手柄按键、trigger、grip、摇杆仍然从左右手柄读取，用于 deadman、夹爪、回 home、录制等原有逻辑。

## 启动示例

推荐用 tracker SN 固定左右映射，避免 SDK 返回顺序变化导致左右交换：

```bash
python teleop/teleop_hand_and_arm.py \
  --input-provider xr \
  --input-mode controller \
  --xr-pose-source motion_tracker \
  --left-motion-tracker-sn PC2310MLL2250062G \
  --right-motion-tracker-sn PC2310MLL2250124G \
  --head-reference-mode hybrid \
  --controller-mapping-mode anchored_safe \
  --controller-orientation-mode relative \
  --controller-grip-threshold 0.5 \
  --motion-tracker-max-step-linear 0.03 \
  --motion-tracker-max-step-angular 0.35
```

如果暂时不知道 SN，可以先按 index 试：

```bash
python teleop/teleop_hand_and_arm.py \
  --input-provider xr \
  --input-mode controller \
  --xr-pose-source motion_tracker \
  --left-motion-tracker-index 0 \
  --right-motion-tracker-index 1
```

## 语义

`motion_tracker` 模式对齐 `nero-dual-arm` 的 PICO reader 逻辑：

1. 读取 `xrt.num_motion_data_available()`。
2. 读取 `xrt.get_motion_tracker_pose()`。
3. 读取 `xrt.get_motion_tracker_serial_numbers()`。
4. 优先按 SN 选择左右 tracker；SN 为空时按 index 选择。
5. tracker 位姿仍走原有 XR 坐标变换、head reference、grip anchor、orientation mode 和 IK 链路。

## 注意

- 保持 `--input-mode controller`，因为 grip/trigger/按键仍来自手柄。
- 建议保留 `--controller-deadman grip`，只有握住对应手柄 grip 时才让对应手臂执行。
- 如果不按 grip 也使能，说明 SDK 的 grip 模拟量在松手时仍有非零残留；测试时加 `--controller-grip-threshold 0.5`。
- 如果 tracker 信号冻结后恢复导致目标突跳，使用 `--motion-tracker-max-step-linear` / `--motion-tracker-max-step-angular` 对 tracker 源位姿做逐帧跳变限幅。
- 建议用 SN 固定左右 tracker；如果左右反了，优先交换 SN 配置。
