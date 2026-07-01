# tests/data_checks

数据检查、在线推理探针和诊断可视化。这里不是数据生产链路，所以不放在 `data_pipeline/`。

常用入口：

```bash
python tests/data_checks/multi_cam_record.py
python tests/data_checks/alignment_report.py
python tests/data_checks/offline_dds_replay_probe.py
python tests/data_checks/probe_umi_online_inference_dataset.py
python tests/data_checks/rollout_umi_online_inference_dataset.py
python tests/data_checks/visualize_umi_mujoco_probe.py
```

这些脚本只做数据检查、服务探测或可视化，不直接下发机器人控制。
