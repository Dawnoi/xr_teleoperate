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
