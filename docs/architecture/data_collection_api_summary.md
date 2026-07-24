# 数采平台 UI API 简表

当前 `teleop/real/teleop_hand_and_arm.py --ui` 提供的数采相关接口总览。
默认服务地址：`http://<robot-host>:8085`。当前所有接口均为 `GET`。

## 1. 宇树 XR Teleoperate

### 1.1 共同状态和控制 API

| API | 用途 |
|---|---|
| `GET /` | 现有 XR UI 页面。 |
| `GET /index.html` | `/` 的页面别名。 |
| `GET /assets/<asset_path>` | 页面静态资源，例如 `GET /assets/app.js`。 |
| `GET /snapshot` | 一次性读取完整运行状态，包含遥操作、双臂、相机、录制、provider、推理、回放和底盘 STOP fault 状态。 |
| `GET /events` | SSE 实时状态流。 |
| `GET /camera/status` | 相机健康、时间戳和可用性。 |
| `GET /camera/frame?camera_id=0` | 返回相机最新一帧 JPEG 图片，`camera_id` 指定相机。 |
| `GET /camera/stream?camera_id=0` | 同 `/camera/frame`，当前只返回单帧 JPEG，不是视频流。 |
| `GET /camera/realsense_list` | 兼容接口，当前返回空设备列表。 |
| `GET /camera/start` | 兼容接口，当前未实现。 |
| `GET /camera/stop` | 兼容接口，当前未实现。 |
| `GET /command/start` | 启动遥操作。 |
| `GET /command/stop` | 停止遥操作。 |
| `GET /command/home` | 双臂回 home。 |
| `GET /command/recenter` | 当前模式允许时重置头部参考。 |

### 1.2 数采 API

| API | 用途 |
|---|---|
| `GET /recording/status` | 录制、相机对齐、数据校验和底盘数采状态。 |
| `GET /recording/start` | 开始或 armed 数采。 |
| `GET /recording/stop` | 停止并保存数采。 |
| `GET /recording/toggle` | 切换录制。 |
| `GET /recording/cancel` | 取消当前或 armed 的录制。 |
| `GET /recording/set_root_dir?root_dir=<root_dir>` | 切换数采根目录。 |
| `GET /recording/episodes?root_dir=<root_dir>&limit=<limit>` | 列出 episode；两个参数均可省略。 |
| `GET /recording/delete_episodes?root_dir=<root_dir>&episode=<episode_name>` | 删除一个或多个 episode；`episode` 可重复。 |
| `GET /recording/set_fps` | 兼容占位；返回成功但不改变采样频率。 |

### 1.3 网页离线回放 API

网页离线回放只在浏览器读取和显示 episode，不向真机下发动作。

| API | 用途 |
|---|---|
| `GET /playback/load?episode=<episode_name>&root_dir=<root_dir>` | 加载 episode；`root_dir` 可省略。 |
| `GET /playback/start` | 开始网页回放。 |
| `GET /playback/pause` | 暂停网页回放。 |
| `GET /playback/stop` | 停止网页回放并回到第 0 帧。 |
| `GET /playback/seek?frame=<frame_index>` | 跳转回放帧。 |
| `GET /playback/status` | 获取网页回放状态。 |
| `GET /playback/curves?max_points=<max_points>` | 获取双臂关节和夹爪曲线。 |
| `GET /playback/image?camera_id=<camera_id>&frame=<frame_index>` | 获取指定相机、指定帧的录制图像。 |

### 1.4 真机轨迹回放 API

真机轨迹回放会将 episode 重新经过现有控制链路下发到真机。

| API | 用途 |
|---|---|
| `GET /replay/real/status` | 获取真机回放状态。 |
| `GET /replay/real/start?episode=<episode_name>&base_source=<base_source>` | 启动真机回放；`episode` 和 `base_source` 必填，后者取 `none` 或 `action`。 |
| `GET /replay/real/stop` | 停止真机回放。 |

### 1.5 数据导出 API

| API | 用途 |
|---|---|
| `GET /convert/status` | 获取 LeRobot 数据导出任务状态。 |
| `GET /convert/start?output_root=<output_root>&dataset_name=<dataset_name>&default_task=<task>` | 启动 LeRobot v2 数据导出；三个参数必填。 |

## 2. 松灵 Nero

Nero 的数采和回放接口尚未从 `~/Programs/nero-dual-arm` 的实际实现中梳理，不能假设与宇树接口一致。

## 3. 详细文档

详见 [数采平台开发 API](data_collection_api_detailed.md)。
