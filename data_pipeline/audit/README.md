# data_pipeline.audit

数据检查、在线推理探针和诊断可视化。

常用入口：

```bash
python data_pipeline/audit/multi_cam_record.py
python data_pipeline/audit/alignment_report.py
python data_pipeline/audit/offline_dds_replay_probe.py
python data_pipeline/audit/probe_umi_online_inference_dataset.py
python data_pipeline/audit/rollout_umi_online_inference_dataset.py
python data_pipeline/audit/visualize_umi_mujoco_probe.py
```

这些脚本只做数据检查、服务探测或可视化，不直接下发机器人控制。
