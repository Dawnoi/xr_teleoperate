# teleop.runtime

real/sim 共享的遥操运行时逻辑：

- takeover / deadman / home-return 状态机
- 单帧控制流程
- 运行控制命令与手柄快捷键
- 遥操运行时 trace / timing 这类可选观测逻辑

这里不直接绑定 DDS 真机或 MuJoCo backend。
