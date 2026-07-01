# core

可复用实现逻辑。这里不要放“看起来通用”的调试脚本或占位目录，只放会被业务入口实际复用的代码：

- `input/`：输入 provider contract、XR/controller/tracker 适配
- `control/`：安全限幅、workspace、filter、底盘控制策略
- `camera/`：local/zmq camera 抽象

不放这里的内容：

- 在线推理：放 `inference/`
- 数据 record/replay/export：放 `data_pipeline/`
- 诊断、探针、数据检查：放 `tests/`
- 遥操运行时 trace/timing：放 `teleop/runtime/`
