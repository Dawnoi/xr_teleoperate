# LeRobot v2 三路相机录制

如果要录制新的 LeRobot v2 三路相机数据集，请使用独立入口，不影响旧的 `data.json + colors/` 链路。

```bash
cd ~/unitree_ws/src/xr_teleoperate
conda activate tv
unset PYTHONPATH

bash scripts/start_real_robot_wired_3cams_lerobot.sh
```

默认行为：

- 使用 `teleop/teleop_hand_and_arm_lerobot.py`
- 默认 `--record-arm-repr both`
- 三路相机固定映射：
  - `cam0=head`
  - `cam1=left_wrist/wrist_left`
  - `cam2=right_wrist/wrist_right`
- 只写 LeRobot v2 新格式，不再写旧的 `episode_000x/data.json`
- 落盘结构：
  - `data/chunk-000/episode_000000.parquet`
  - `videos/chunk-000/observation.images.cam0/episode_000000.mp4`
  - `videos/chunk-000/observation.images.cam1/episode_000000.mp4`
  - `videos/chunk-000/observation.images.cam2/episode_000000.mp4`
  - `meta/info.json`
  - `meta/tasks.jsonl`
  - `meta/episodes.jsonl`
  - `meta/episodes_stats.jsonl`

新 writer 会先完成 episode 的 parquet / mp4 写入与解码验证，只有验证成功后才更新 meta 文件。

视频默认使用和原先 OpenCV mp4 保存更接近的链路：

- backend: `opencv`
- container: `.mp4`
- fourcc: 默认优先 `MJPG`，失败再尝试 `mp4v` / `avc1`
- 写入方式: worker 内流式写 `.partial.mp4`，episode 验证通过后再替换成最终 mp4
- 说明: `MJPG` 会生成 MJPEG 视频，文件更大，但更接近 `episode_000004`，Ubuntu 默认播放器更容易直接打开

如果后续需要显式回到 ffmpeg/libx264 备用链路，可以临时切换：

```bash
LEROBOT_VIDEO_BACKEND=ffmpeg bash scripts/start_real_robot_wired_3cams_lerobot.sh
```

ffmpeg 备用链路仍支持 `LEROBOT_VIDEO_PROFILE=baseline|main` 和 `LEROBOT_VIDEO_CRF=15` 到 `18`。

## 回放 / replay

回放数据时，优先使用已录入的控制 sidecar：

```bash
python teleop/replay_lerobot_real.py \
  --dataset-root /path/to/dataset_root \
  --episode-index 0 \
  --use-recorded-tauff \
  --dry-run
```

真实回放示例：

```bash
python teleop/replay_lerobot_real.py \
  --dataset-root /path/to/dataset_root \
  --episode-index 0 \
  --network-interface eno1 \
  --use-recorded-tauff \
  --max-arm-joint-speed 0.5
```

旧数据集如果没有 `extras/control/chunk-000/episode_000000.jsonl` sidecar，会自动回退到运行时 live tauff 计算，不影响主 parquet / 视频回放。
