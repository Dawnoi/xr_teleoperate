# teleop

实时遥操业务域。

最终语义：`teleop/` 只负责在线实时遥操，内部按 backend 分为：

- `real/`：真机遥操
- `sim/`：MuJoCo/仿真遥操
- `runtime/`：real/sim 共享状态机和控制循环
- `control_flow/`：每帧控制流程，包含 arm/base command pipeline
- `debug/`：遥操调试辅助，包含 latency、timing、gripper UI

数据采集、回放、导出不放在这里；它们属于顶层 `data_pipeline/`。

动作输入源在 `core/input/`，通用控制/相机在 `core/control/`、`core/camera/`，诊断和数据检查在 `tests/`。`teleop/` 只保留实时遥操入口、运行时和真机/仿真控制相关代码。
