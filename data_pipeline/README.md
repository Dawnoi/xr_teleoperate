# data_pipeline

数据业务域：只放真正参与数据流的 record / replay / export，不放诊断探针。

当前结构：

- `recording/`：episode / LeRobot / rerun 相关工具
- `replay/`：raw / LeRobot episode 回放到 sim 或 real
- `export/`：数据格式转换

数据检查、对齐报告、服务探针、诊断可视化放 `tests/data_checks/`。

Python 实现直接放在对应业务域；`scripts/` 只保留少量 shell 一键脚本。
