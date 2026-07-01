# NERO LeRobot v2 三路相机数据集结构规范

本文档描述 NERO 双臂任务的 LeRobot v2 原始导出数据集格式，用于规范后续采集和导出。

本版本是三路相机版本：

- `cam0`: 头部第一人称主视角，对应当前采集中的 `head`
- `cam1`: 左腕相机，对应当前采集中的 `left_wrist` / `wrist_left`
- `cam2`: 右腕相机，对应当前采集中的 `right_wrist` / `wrist_right`

这里描述的是采集后的原始数据结构，不是 pi0.5 训练最终使用的 14 维 EEF 数据集。

## 1. 顶层结构

标准数据集目录应包含：

```text
<dataset_root>/
  data/
    chunk-000/
      episode_000000.parquet
      episode_000001.parquet
      ...
  videos/
    chunk-000/
      observation.images.cam0/
        episode_000000.mp4
        episode_000001.mp4
        ...
      observation.images.cam1/
        episode_000000.mp4
        episode_000001.mp4
        ...
      observation.images.cam2/
        episode_000000.mp4
        episode_000001.mp4
        ...
  meta/
    info.json
    episodes.jsonl
    episodes_stats.jsonl
    tasks.jsonl
```

可选目录：

```text
<dataset_root>/
  npy/
    episodes/
      episode_000000.npz
      ...
```

`npy/` 可作为导出中间产物保留，但不属于训练或转换链路的必要格式契约。

## 2. 文件命名

parquet 数据：

```text
data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet
```

视频数据：

```text
videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4
```

标准视频 key：

- `observation.images.cam0`
- `observation.images.cam1`
- `observation.images.cam2`

标准相机语义：

- `cam0`: 头部第一人称主视角，后续转换名可用 `head_fpv`
- `cam1`: 左腕相机，后续转换名可用 `left_wrist`
- `cam2`: 右腕相机，后续转换名可用 `right_wrist`

不同数据集之间必须保持相机 key 和物理视角的映射一致。

## 3. meta/info.json

`info.json` 是数据集的全局描述文件，必须包含：

```json
{
  "codebase_version": "v2.1",
  "robot_type": "nero_dual_arm",
  "total_episodes": 0,
  "total_frames": 0,
  "total_tasks": 0,
  "total_chunks": 1,
  "total_videos": 3,
  "fps": 30,
  "splits": {"train": "0:0"},
  "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4",
  "depth_camera_ids": [],
  "depth_path": "depth/cam{camera_id}/episode_{episode_index:06d}/frame_{frame_index:06d}.png",
  "features": {},
  "camera_roles": {
    "0": "head_fpv",
    "1": "left_wrist",
    "2": "right_wrist"
  },
  "camera_name_map": {
    "0": "头部第一人称主视角",
    "1": "左腕相机",
    "2": "右腕相机"
  },
  "source_camera_map": {
    "observation.images.cam0": "head",
    "observation.images.cam1": "left_wrist/wrist_left",
    "observation.images.cam2": "right_wrist/wrist_right"
  },
  "notes": ""
}
```

约束：

- `total_episodes` 等于 episode 总数。
- `total_frames` 等于所有 episode 的 frame 数之和。
- `total_videos` 固定为 `3`。
- `fps` 必须和 parquet `timestamp`、三路视频 fps 一致。
- `splits.train` 应覆盖全部训练 episode，通常为 `"0:<total_episodes>"`。
- `features` 必须完整描述 parquet 中的每一列。
- `camera_roles`、`camera_name_map`、`source_camera_map` 必须和实际采集相机一致。

## 4. meta/tasks.jsonl

每行描述一个任务：

```json
{"task_index":0,"task":"<task prompt>"}
```

约束：

- `task_index` 从 0 开始连续。
- parquet 中的 `task_index` 必须能在 `tasks.jsonl` 中找到。
- 同一数据集内应保持 task prompt 稳定，避免同一动作目标使用多个近似文本。

## 5. meta/episodes.jsonl

每行描述一个 episode：

```json
{"episode_index":0,"tasks":["<task prompt>"],"length":0}
```

约束：

- `episode_index` 从 0 开始连续。
- `length` 等于该 episode 的 parquet 行数。
- `length` 等于该 episode 每一路相机视频的 frame 数。
- `tasks` 中的文本应来自 `tasks.jsonl`。

## 6. meta/episodes_stats.jsonl

每行对应一个 episode：

```json
{"episode_index":0,"stats":{}}
```

该文件可保存每个 episode 的统计信息。如果导出链路暂不计算 stats，允许 `stats` 为空对象，但不能把空 stats 当作数据质量验证结果。

## 7. parquet 数据结构

每个 episode 对应一个 parquet 文件。每一行对应一个时间步。

基础列：

- `timestamp`: double，单位秒
- `frame_index`: int64，episode 内 frame 编号
- `episode_index`: int64，episode 编号
- `index`: int64，全数据集全局 frame 编号
- `task_index`: int64，任务编号

机器人状态和动作：

- `observation.state`: list<double>，长度 16
- `action`: list<double>，长度 16

16 维顺序固定为：

```text
[
  left_joint1,
  left_joint2,
  left_joint3,
  left_joint4,
  left_joint5,
  left_joint6,
  left_joint7,
  left_gripper,
  right_joint1,
  right_joint2,
  right_joint3,
  right_joint4,
  right_joint5,
  right_joint6,
  right_joint7,
  right_gripper
]
```

FK 反馈列：

```text
observation.fk.fb.left.joint1
...
observation.fk.fb.left.joint7
observation.fk.fb.left.gripper_flange
observation.fk.fb.right.joint1
...
observation.fk.fb.right.joint7
observation.fk.fb.right.gripper_flange
```

FK 指令列：

```text
observation.fk.cmd.left.joint1
...
observation.fk.cmd.left.joint7
observation.fk.cmd.left.gripper_flange
observation.fk.cmd.right.joint1
...
observation.fk.cmd.right.joint7
observation.fk.cmd.right.gripper_flange
```

每个 FK 列是 list<double>，长度 6，顺序固定为：

```text
[x, y, z, roll, pitch, yaw]
```

图像引用列：

- `observation.images.cam0`: struct，包含 `{path: string, timestamp: double}`
- `observation.images.cam1`: struct，包含 `{path: string, timestamp: double}`
- `observation.images.cam2`: struct，包含 `{path: string, timestamp: double}`

图像列中的 `path` 必须是相对数据集根目录的路径，例如：

```text
videos/chunk-000/observation.images.cam0/episode_000000.mp4
videos/chunk-000/observation.images.cam1/episode_000000.mp4
videos/chunk-000/observation.images.cam2/episode_000000.mp4
```

图像列中的 `timestamp` 使用该帧对应的采样时间，单位秒，必须和 parquet 主 `timestamp` 对齐。

## 7.1 可选 control sidecar

部分回放 / 对齐链路会额外写入每个 episode 的控制 sidecar：

```text
<dataset_root>/
  extras/
    control/
      chunk-000/
        episode_000000.jsonl
```

该文件为 JSONL，每行对应 parquet 的一行样本，只记录真机控制回放需要的扩展字段，不改变主 parquet 的 `action` / `observation.state` / 视频 schema。

单行示例：

```json
{
  "schema_version": 1,
  "episode_index": 0,
  "frame_index": 42,
  "timestamp": 1.4,
  "sample_monotonic_ns": 123456789,
  "arm_tauff": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "source": "teleop_runtime"
}
```

字段约束：

- `frame_index` 必须和 parquet 中同名字段一一对应。
- `timestamp` 单位为秒，和 parquet 主 `timestamp` 一致。
- `sample_monotonic_ns` 使用采集 host monotonic 时间戳。
- `arm_tauff` 长度固定为 14，顺序为 `left_arm7 + right_arm7`。
- `source` 固定为 `teleop_runtime`，表示来自遥操运行时下发的重力前馈。

注意：

- control sidecar 是可选的。
- 没有 sidecar 时，主 parquet 结构不变，`action` 仍然按原 16 维关节空间保存。
- 真机 replay 如果没有该 sidecar，必须回退到运行时按 `action` 现场计算 `tauff`。

## 8. 视频数据结构

每个 episode 在每个相机目录下必须有一个 mp4 文件。

三路视频路径示例：

```text
videos/chunk-000/observation.images.cam0/episode_000000.mp4
videos/chunk-000/observation.images.cam1/episode_000000.mp4
videos/chunk-000/observation.images.cam2/episode_000000.mp4
```

推荐视频规范：

- container: `.mp4`
- codec: 默认 OpenCV `MJPG`，即 MJPEG；可选备用 `mp4v` 或 H.264 / libx264
- fps: 与 `info.json.fps` 一致
- frame 数: 等于该 episode 的 `length`
- 不包含音频流

当前默认导出策略是 OpenCV `VideoWriter` 流式写 `.mp4`，fourcc 默认优先 `MJPG`，失败再尝试 `mp4v` / `avc1`。`MJPG` 文件更大，但每帧独立，更接近 `episode_000004` 的可播放表现。如果显式设置 `LEROBOT_VIDEO_BACKEND=ffmpeg`，备用链路会使用 `libx264 + baseline + level 3.0 + yuv420p + CRF 18`。

导出后必须做完整解码检查，不能只检查文件存在或 ffprobe metadata。

## 9. 当前 xr_teleoperate 字段映射

当前 `xr_teleoperate` 原始 episode 使用 `data.json + colors/` 保存。导出到本 schema 时，推荐按下表映射：

| 当前字段 | 目标字段 |
| --- | --- |
| `idx` | `frame_index` |
| episode 目录序号 | `episode_index` |
| 全数据集累计帧号 | `index` |
| `timestamps.sample_monotonic_ns` | `timestamp` |
| `states.left_arm.qpos + states.left_ee.qpos + states.right_arm.qpos + states.right_ee.qpos` | `observation.state` |
| `actions.left_arm.qpos + actions.left_ee.qpos + actions.right_arm.qpos + actions.right_ee.qpos` | `action` |
| `colors.head` | `observation.images.cam0` |
| `colors.left_wrist` 或 `colors.wrist_left` | `observation.images.cam1` |
| `colors.right_wrist` 或 `colors.wrist_right` | `observation.images.cam2` |
| `states.left_arm.pose` / qpos FK 计算结果 | `observation.fk.fb.left.*` |
| `states.right_arm.pose` / qpos FK 计算结果 | `observation.fk.fb.right.*` |
| `actions.left_arm.pose` / qpos FK 计算结果 | `observation.fk.cmd.left.*` |
| `actions.right_arm.pose` / qpos FK 计算结果 | `observation.fk.cmd.right.*` |

如果当前 `left_ee/right_ee` 是单自由度夹爪，则分别填入 `left_gripper/right_gripper`。如果是多自由度灵巧手，导出 16 维关节空间数据时必须明确选择一个标量 gripper 表示，不能把多维手指状态直接塞进 16 维向量。

## 10. 全局一致性约束

每个有效数据集必须满足：

- episode index 连续：`0..total_episodes-1`
- 每个 episode 同时存在 parquet 和三路相机视频
- `episodes.jsonl` 行数等于 `total_episodes`
- parquet 文件数等于 `total_episodes`
- 每个相机目录的视频文件数等于 `total_episodes`
- `sum(episodes.length) == info.total_frames`
- 每个 parquet 行数等于对应 `episodes.length`
- 每一路视频 frame 数等于对应 `episodes.length`
- `frame_index` 在 episode 内从 0 开始连续
- `index` 在全数据集内从 0 开始连续
- `timestamp` 与 `frame_index / fps` 一致或足够接近
- `observation.state` 和 `action` 维度恒为 16
- `observation.images.cam0/cam1/cam2` 每行都必须存在
- 所有数值字段不包含 NaN 或 Inf
- 三路视频都能被解码器完整解码；当前 `xr_teleoperate` 新入口使用 OpenCV/FFmpeg 后端做基础解码验证，不依赖外部 `ffmpeg` / `ffprobe` 命令。

## 11. pi0.5 训练前转换边界

原始导出格式保留 16 维关节空间数据。pi0.5 训练前需要再转换为 14 维 EEF 数据。

## 12. xr_teleoperate 采集入口约定

本仓库新增了专用的三路 LeRobot v2 录制入口：

- `teleop/teleop_hand_and_arm_lerobot.py`
- `scripts/start_real_robot_wired_3cams_lerobot.sh`

约定如下：

- 旧入口 `teleop/real/teleop_hand_and_arm.py` 保持不变，仍然写旧格式。
- 新入口只写本规范中的 LeRobot v2 目录结构。
- 三路相机映射固定为：
  - `cam0=head`
  - `cam1=left_wrist/wrist_left`
  - `cam2=right_wrist/wrist_right`
- 每个 episode 只有在 parquet、三路视频解码、行数一致性、数值有限性都通过后，才允许更新 `meta/info.json`、`meta/tasks.jsonl`、`meta/episodes.jsonl`、`meta/episodes_stats.jsonl`。
- 如果某条 sample 缺少任一路相机，或左右臂 qpos / ee qpos 不完整，必须 skip 该 sample 并 warning，不得写入半残样本。
