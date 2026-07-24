# 推理平台 UI API 简表

当前 `teleop/real/teleop_hand_and_arm.py --ui` 提供的推理相关接口总览。
默认服务地址：`http://<robot-host>:8085`。当前所有接口均为 `GET`。

## 1. 宇树 XR Teleoperate

### 1.1 共同状态和控制 API

| API | 用途 |
|---|---|
| `GET /snapshot` | 一次性读取完整运行状态，包含 provider、推理、录制和底盘 STOP fault 状态。 |
| `GET /events` | SSE 实时状态流。 |
| `GET /camera/status` | 推理所需相机的健康、时间戳和可用性。 |
| `GET /camera/frame?camera_id=0` | 返回推理 UI 显示的相机最新 JPEG 图片。 |
| `GET /camera/stream?camera_id=0` | 同 `/camera/frame`，当前只返回单帧 JPEG，不是视频流。 |
| `GET /command/start` | 启动遥操作。 |
| `GET /command/stop` | 停止遥操作。 |
| `GET /command/home` | 双臂回 home。 |
| `GET /command/recenter` | 当前模式允许时重置头部参考。 |

### 1.2 在线推理 API

| API | 用途 |
|---|---|
| `GET /inference/status` | 获取在线推理 provider、状态和错误。 |
| `GET /inference/start?prompt=<prompt>` | 使用 prompt 启动在线推理。 |
| `GET /inference/stop` | 停止在线推理。 |
| `GET /ui/provider/hold` | 切换到 hold 输入状态。 |
| `GET /ui/provider/xr` | 切回 XR 实时输入。 |

### 1.3 推理与数采协同 API

| API | 用途 |
|---|---|
| `GET /recording/status` | 判断是否正在录制或 armed，并读取底盘数采/停车故障状态。 |
| `GET /recording/start` | 推理动作需要记录时开始数采。 |
| `GET /recording/stop` | 停止并保存推理过程数采。 |
| `GET /recording/cancel` | 取消当前推理过程数采。 |

## 2. 松灵 Nero

Nero 的推理、相机和数采协同接口尚未从 `~/Programs/nero-dual-arm` 的实际实现中梳理，不能假设与宇树接口一致。

## 3. 详细文档

详见 [推理平台开发 API](inference_api_detailed.md)。
