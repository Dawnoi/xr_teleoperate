# scripts

这里只保留少量 shell 一键脚本，不放 Python wrapper。

Python 实现直接位于对应业务域：

- 真机/仿真遥操：`teleop/real`, `teleop/sim`
- 回放/数据工具：`data_pipeline/`
- 诊断：`tests/diagnostics`

当前 shell 分组：

- `start/`：真机一键启动
- `debug/`：DDS 调试启动
- `replay/`：回放快捷启动
- `data/`：数据导出快捷启动
