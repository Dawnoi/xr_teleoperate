# teleop

实时遥操业务域。

最终语义：`teleop/` 只负责在线实时遥操，内部按 backend 分为：

- `real/`：真机遥操
- `sim/`：MuJoCo/仿真遥操
- `runtime/`：real/sim 共享状态机和控制循环

数据采集、回放、导出不放在这里；它们属于顶层 `data_pipeline/`。公共输入、控制、机器人、相机、诊断逻辑后续沉淀到顶层 `core/`。
