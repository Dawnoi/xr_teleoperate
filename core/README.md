# core

跨业务域公共逻辑。这里的代码不应该绑定某个入口：

- `input/`：输入 provider contract、XR/controller/tracker 适配
- `control/`：安全限幅、workspace、filter、底盘控制策略
- `robot/`：arm/hand/IK 适配
- `camera/`：local/zmq camera 抽象
- `transforms/`：pose/quaternion/rot6d/matrix 转换
- `diagnostics/`：trace/timing/latency 公共诊断

当前阶段先建立目录边界，旧实现仍保留在 `teleop/*`，后续分批迁移并保留兼容 import。
