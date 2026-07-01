# data_pipeline

数据业务域：数据采集、回放、导出、检查都放这里，不再归到 `teleop/`。

当前结构：

- `recording/`：episode / LeRobot / rerun 相关工具
- `replay/`：raw / LeRobot episode 回放到 sim 或 real
- `export/`：数据格式转换预留
- `audit/`：数据检查、对齐报告、探针
- `schemas/`：数据格式说明或 schema helper

Python 实现直接放在对应业务域；`scripts/` 只保留少量 shell 一键脚本。
