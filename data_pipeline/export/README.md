# data_pipeline.export

数据格式导出工具。

当前入口：

```bash
python data_pipeline/export/raw_to_lerobot_v2.py
python data_pipeline/export/lerobot_to_umi_dp.py
```

对应职责：

- `raw_to_lerobot_v2.py`：legacy raw episode -> LeRobot v2
- `lerobot_to_umi_dp.py`：LeRobot v2 -> UMI/DP-friendly frame folder

## Dex1.1 TCP

带有 `info.eef_pose_frame="dex1_tcp"` 的移动操作 raw episode 必须使用 `--export-fk 1 --urdf-path assets/g1_d/g1_d.urdf` 导出。导出器会严格验证每帧四个 `pose_base_link_tcp` record，输出以下 `float64[6]` 列，排列为 `[x_m, y_m, z_m, roll_rad, pitch_rad, yaw_rad]`：

- `observation.fk.fb.left.dex1_tcp`
- `observation.fk.fb.right.dex1_tcp`
- `observation.fk.cmd.left.dex1_tcp`
- `observation.fk.cmd.right.dex1_tcp`

旧 `observation.fk.*.gripper_flange` 保留，不表示 Dex1 TCP。一个导出任务不能混用带 TCP metadata 的新 episode 与旧 episode；必须分开导出。

## 移动操作 pi0.5 向量

移动 episode 使用 `--mobile-eef-base 1`。当前在线采集的 `data.json` 已在每个相机采样时刻完成 base、height、SLAM TF 和 base action 对齐，exporter 直接读取 `states.base.{slam_map_pose,velocity,height}` 与 `actions.base`，并要求四条 `timestamps.{base_state,base_height,slam_tf,base_action}` 对齐记录完整。离线对齐 view 仍受支持：额外传入 `--episode-data-file data.mobile_aligned.json` 时，exporter 改读对应的 `*_interpolated` 字段。

导出的训练向量固定为：

- `observation.state: float64[26]`：左右 Dex1 TCP 各 `[xyz, Rot6D, gripper]`（各 10 维）、`slamware_map` 下底盘 `[x, y, yaw]`、`base_link` 下真实 `[vx, wz]`、升降柱真实高度。
- `action: float64[23]`：左右 Dex1 TCP target 各 `[xyz, Rot6D, gripper]`（各 10 维）、`base_link` 下 `[vx_cmd, wz_cmd]` 和 `z_cmd`。

`Rot6D` 是旋转矩阵的前两列按列拼接：`[R[:, 0], R[:, 1]]`。当前底盘不支持横移，`vy/vy_cmd` 不进入训练向量；腰部状态和腰部命令也不进入训练向量。原始 raw 字段仍完整保留。

每个导出 episode 还会把输入 JSON 按字节原样复制到 `extras/raw_episodes/chunk-000/`，并在 `export_summary.json` 记录其 SHA256。该 sidecar 保留全部原始字段；`observation.state` 和 `action` 仅是当前 OpenPI 训练选用的向量，不是 raw 数据的替代品。
