# 推理平台开发 API

本文描述当前 `teleop/real/teleop_hand_and_arm.py --ui` 的推理 UI 接口。默认服务地址为
`http://<robot-host>:8085`。当前实现全部使用 `GET`；`/api/v1/*` 是未来目标，不可调用。

## 1. 宇树 XR Teleoperate

### 1.1 共同状态和控制 API

#### API 总表

| API                              | 响应类型 | 用途                         |
| -------------------------------- | -------- | ---------------------------- |
| `GET /snapshot`                  | JSON     | 推理 UI 的完整初始状态。     |
| `GET /events`                    | SSE      | 推理/provider 状态实时更新。 |
| `GET /camera/status`             | JSON     | 推理相机可用性和帧龄。       |
| `GET /camera/frame?camera_id=N`  | JPEG     | 最新相机预览。               |
| `GET /camera/stream?camera_id=N` | JPEG     | 单帧别名，不是视频流。       |
| `GET /command/start`             | JSON     | 启动遥操作。                 |
| `GET /command/stop`              | JSON     | 停止遥操作。                 |
| `GET /command/home`              | JSON     | 双臂回 home。                |
| `GET /command/recenter`          | JSON     | 重置头部参考。               |

命令成功响应固定形状：

```json
{"ok":true,"queued":true,"command":"start"}
```

| 响应字段  | 类型/取值              | 含义                                         |
| --------- | ---------------------- | -------------------------------------------- |
| `ok`      | 固定 `true`            | 当前 HTTP 请求已通过校验。                   |
| `queued`  | 固定 `true`            | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `start` | 已入队的命令名称。                           |

`queued=true` 只说明已入主控制循环队列，不能说明推理或 provider 已完成切换。拒绝通常是
HTTP `400` 响应：

```json
{"ok":false,"error":"具体原因"}
```

前端必须显示 `error`。

#### 1.1.1 `GET /snapshot`

功能：读取一次完整运行状态，用于页面初始加载或 SSE 重连。

响应示例：

```json
{
  "schema": "data_collector/v1",
  "left": {"q_fb": null, "gripper_q_fb": null, "gripper_q_cmd": null, "stamp_fb_ns": null, "fk_flange_pose": null},
  "right": {"q_fb": null, "gripper_q_fb": null, "gripper_q_cmd": null, "stamp_fb_ns": null, "fk_flange_pose": null},
  "recording_active": false,
  "recording": {
    "active": false,
    "enabled": true,
    "phase": "idle",
    "session_dir": "",
    "root_dir": "/data/task",
    "active_root_dir": "/data/task",
    "fps": 30.0,
    "frame_index": 0,
    "error": "",
    "last_alignment": {},
    "last_alert": {},
    "alert_seq": 0,
    "last_validation": {},
    "base": {"enabled": false, "receiver_alive": false, "odom_topic": "", "height_topic": "", "state_max_age_ms": 0.0, "action_max_age_ms": 0.0, "stop_confirmed": true, "control_fault": false, "fault_reason": ""}
  },
  "playback": {"state": "disabled", "error": "playback is not implemented in xr_teleoperate UI"},
  "convert": {"ok": true, "state": "idle", "phase": "idle", "running": false, "message": "ready: LeRobot v2 raw exporter is available"},
  "provider": {
    "active_provider": "hold",
    "input_provider": "hold",
    "last_error": "",
    "last_reason": "ui_hold",
    "online_inference": {
      "state": "idle",
      "prompt": "",
      "error": "",
      "reason": "",
      "debug": {},
      "runtime_debug": {}
    },
    "real_replay": {"state":"idle","dataset_root":"","episode_name":"","episode_index":-1,"arm_source":"action","base_source":"none","speed_scale":1.0,"frame_index":-1,"error":"","reason":"","runtime_debug":{}}
  },
  "updated_mono": 123.45,
  "teleop": {"started": true, "ready": true, "stopping": false}
}
```

推理相关字段：

| 字段                           | 类型/取值                                     | 含义                                                                                                                                           |
| ------------------------------ | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema`                       | 固定 `data_collector/v1`                      | 页面 payload 标识。                                                                                                                            |
| `left` / `right`               | object                                        | 每侧含 `q_fb`、`gripper_q_fb`、`gripper_q_cmd`、`stamp_fb_ns`、`fk_flange_pose`。`q_fb` 可为 7 个机械臂关节、8 个含夹爪关节的数组，或 `null`。 |
| `recording_active`             | boolean                                       | `recording.active` 的快捷字段。                                                                                                                |
| `teleop.started`               | boolean                                       | 真机遥操作是否已启动。                                                                                                                         |
| `teleop.ready`                 | boolean                                       | 是否已具备接管条件。                                                                                                                           |
| `teleop.stopping`              | boolean                                       | 是否正在停止。                                                                                                                                 |
| `recording.active`             | boolean                                       | 是否在写入或 armed；为 true 时禁止切 provider。                                                                                                |
| `recording.phase`              | `idle` / `armed` / `recording` / `validating` | `armed` 同样禁止切 provider。                                                                                                                  |
| `recording.base.control_fault` | boolean                                       | 底盘控制/停车是否明确故障。                                                                                                                    |
| `recording.base.fault_reason`  | string                                        | 故障原因，非空不可忽略。                                                                                                                       |
| `playback`                     | object                                        | 固定网页回放占位；实际网页回放状态只能从 `/playback/status` 读取。                                                                             |
| `convert`                      | object                                        | 固定 idle 导出占位；实际导出状态只能从 `/convert/status` 读取。                                                                                |
| `provider`                     | object                                        | 当前输入 provider，见下表。                                                                                                                    |
| `updated_mono`                 | number                                        | 本机单调秒数。                                                                                                                                 |

`provider`：

| 字段                                                                                                            | 类型/取值                                                     | 含义                                                             |
| --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------- |
| `active_provider`                                                                                               | `xr_live` / `hold` / `raw_replay` / `online_inference`        | 真正为控制链路提供意图的输入源。                                 |
| `input_provider`                                                                                                | string                                                        | XR 输入实现名、`hold`、`lerobot_offline` 或 `online_inference`。 |
| `last_error` / `last_reason`                                                                                    | string                                                        | 最近错误和切换原因。                                             |
| `online_inference.state` / `prompt` / `error` / `reason`                                                        | string                                                        | 在线推理生命周期、任务文本、错误和状态原因。                     |
| `online_inference.debug` / `runtime_debug`                                                                      | object                                                        | provider 调试快照和主控制循环诊断。                              |
| `real_replay.state` / `dataset_root` / `episode_name` / `episode_index`                                         | string / string / string / integer                            | 真机回放生命周期、数据根、episode 和数字索引。                   |
| `real_replay.arm_source` / `base_source` / `speed_scale` / `frame_index` / `error` / `reason` / `runtime_debug` | string / string / number / integer / string / string / object | 真机回放来源、进度、错误、原因和诊断。                           |

#### 1.1.2 `GET /events`

功能：建立 SSE 连接，持续接收运行状态变化。

| 项目           | 值                                  |
| -------------- | ----------------------------------- |
| 成功状态码     | `200`                               |
| `Content-Type` | `text/event-stream; charset=utf-8`  |
| 消息格式       | `data:<snapshot JSON>\n\n`          |
| 更新频率       | 仅状态变更时发送，默认最多约 5 Hz。 |

`data` 与 1.1.1 的 `/snapshot` 同构。每次新建 SSE 连接都会先发送当前缓存状态；该流不推送相机图像，
没有事件 ID 或历史补偿。客户端可额外读取 `/snapshot`，但重连不依赖这一步。

#### 1.1.3 `GET /camera/status`

功能：读取当前相机缓存、帧龄和采集元数据。

响应示例：

```json
{
  "url": "/camera/frame?camera_id=0",
  "managed_process_alive": true,
  "managed_pid": null,
  "camera_id": 0,
  "camera_name": "head",
  "camera_mode": "rgb",
  "shared_seq": 42,
  "shared_timestamp_ns": 123456789,
  "shared_age_ms": 15,
  "requested_capture": null,
  "actual_capture": {"frame_width": 640, "frame_height": 480, "channels": 3, "transport": "zmq_raw"},
  "active_camera_ids": [0, 1, 2],
  "streams": [{
    "camera_id": 0,
    "camera_name": "head",
    "camera_role": "head",
    "camera_mode": "rgb",
    "url": "/camera/frame?camera_id=0",
    "shared_seq": 42,
    "shared_timestamp_ns": 123456789,
    "shared_age_ms": 15,
    "requested_capture": null,
    "actual_capture": {"frame_width": 640, "frame_height": 480, "channels": 3, "transport": "zmq_raw"}
  }],
  "camera_name_map": {"0": "head"},
  "name_presets": [{"name": "head", "role": "head"}],
  "last_error": ""
}
```

| 字段                                    | 类型/取值       | 含义                                                     |
| --------------------------------------- | --------------- | -------------------------------------------------------- |
| `active_camera_ids`                     | integer[]       | 当前有元数据的相机 ID。                                  |
| `streams`                               | object[]        | 每路相机状态。                                           |
| `streams[].camera_id`                   | `0` / `1` / `2` | 分别是 `head`、`left_wrist`、`right_wrist`。             |
| `streams[].camera_name` / `camera_role` | string          | 相机语义名。                                             |
| `streams[].camera_mode`                 | 当前固定 `rgb`  | 当前图像模式。                                           |
| `streams[].url`                         | string          | 此相机最新单帧 JPEG 地址。                               |
| `streams[].shared_seq`                  | integer         | 最新帧序号；无帧可能为 `-1`。                            |
| `streams[].shared_timestamp_ns`         | integer         | host 侧单调收帧时间。                                    |
| `streams[].shared_age_ms`               | integer         | 帧龄；无时间戳为 `-1`。                                  |
| `streams[].requested_capture`           | 当前 `null`     | 兼容字段。                                               |
| `streams[].actual_capture`              | object          | `frame_width`、`frame_height`、`channels`、`transport`。 |
| `url`                                   | string          | 主相机预览地址。                                         |
| `last_error`                            | string          | 当前兼容错误字段。                                       |

以下顶层字段也属于该端点契约，推理开发者无需再查数采文档：

| 字段 | 类型/取值 | 含义 |
|---|---|---|
| `managed_process_alive` | boolean | 是否至少有一个相机缓存。 |
| `managed_pid` | 当前 `null` | 兼容字段。 |
| `camera_id` / `camera_name` / `camera_mode` | integer / string / `rgb` | `streams[0]` 的快捷字段；无相机时为 `null`、空字符串、空字符串。 |
| `shared_seq` | integer | 主相机帧序号；无帧为 `-1`。 |
| `shared_timestamp_ns` / `shared_age_ms` | integer | 主相机 host 单调时间与帧龄；无时间戳时分别为 `0`、`-1`。 |
| `requested_capture` | 当前 `null` | 兼容字段。 |
| `actual_capture` | object 或 `null` | 主相机尺寸、通道数、transport；子字段与 `streams[].actual_capture` 相同。 |
| `camera_name_map` | object | 相机 ID 字符串到相机名称的映射，例如 `{"0":"head"}`。 |
| `name_presets` | object[] | 预设角色列表；每项有 `name`、`role`。 |

顶层的 `camera_id`、`camera_name`、`camera_mode`、`shared_seq`、`shared_timestamp_ns`、
`shared_age_ms`、`requested_capture` 与 `actual_capture` 均对应 `streams[0]` 的快捷字段。

#### 1.1.4 `GET /camera/frame?camera_id=N`

功能：返回指定相机的最新单帧 JPEG 预览。

| 参数        | 必填 | 规则                                                             | 默认 |
| ----------- | ---- | ---------------------------------------------------------------- | ---- |
| `camera_id` | 否   | 十进制整数；`0`、`1`、`2` 分别为 head、left_wrist、right_wrist。 | `0`  |

调用示例：

```http
GET /camera/frame?camera_id=0
```

`camera_id` 可选，默认 0，必须是十进制整数；0/1/2 分别为 head、left_wrist、right_wrist。成功为 `200 image/jpeg` 和 JPEG 二进制；无帧为
`404` 响应：

```json
{"ok":false,"error":"camera_id=N has no frame"}
```

| 响应字段       | 类型/取值                                        | 含义                                   |
| -------------- | ------------------------------------------------ | -------------------------------------- |
| HTTP 状态码    | `200` / `404`                                    | 有帧时为 `200`；没有缓存帧时为 `404`。 |
| `Content-Type` | `image/jpeg` / `application/json; charset=utf-8` | 成功为 JPEG，失败为 JSON。             |
| 成功 body      | JPEG 二进制                                      | 当前最新单帧。                         |
| `ok`           | 固定 `false`，仅 404                             | 请求未取得图像。                       |
| `error`        | string，仅 404                                   | 无帧原因。                             |

`camera_id` 不是十进制整数时，当前服务没有定义 JSON 错误响应，连接可能直接失败。

#### 1.1.5 `GET /camera/stream?camera_id=N`

功能：兼容别名，返回指定相机的最新单帧 JPEG，不是视频流。

| 参数        | 必填 | 取值          | 默认 |
| ----------- | ---- | ------------- | ---- |
| `camera_id` | 否   | `0`、`1`、`2` | `0`  |

调用示例：

```http
GET /camera/stream?camera_id=1
```

成功响应不是 JSON：

```http
HTTP 200
Content-Type: image/jpeg

<JPEG binary>
```

无最新帧时的 JSON 响应：

```json
{"ok":false,"error":"camera_id=N has no frame"}
```

| 响应字段       | 类型/取值                                        | 含义                                   |
| -------------- | ------------------------------------------------ | -------------------------------------- |
| HTTP 状态码    | `200` / `404`                                    | 有帧时为 `200`；没有缓存帧时为 `404`。 |
| `Content-Type` | `image/jpeg` / `application/json; charset=utf-8` | 成功为 JPEG，失败为 JSON。             |
| 成功 body      | JPEG 二进制                                      | 当前最新单帧。                         |
| `ok`           | 固定 `false`，仅 404                             | 请求未取得图像。                       |
| `error`        | string，仅 404                                   | 无帧原因。                             |

当前只返回最新单帧，不是连续视频、MJPEG 或 WebRTC。
`camera_id` 必须是十进制整数；非整数参数没有定义 JSON 错误响应。

#### 1.1.6 `GET /command/start`

功能：将启动遥操作命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"start"}
```

| 响应字段  | 类型/取值              | 含义                                         |
| --------- | ---------------------- | -------------------------------------------- |
| `ok`      | 固定 `true`            | 当前 HTTP 请求已通过校验。                   |
| `queued`  | 固定 `true`            | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `start` | 已入队的命令名称。                           |

仅在
`/snapshot.teleop.started=true` 后，才可认为真机遥操作已启动。

#### 1.1.7 `GET /command/stop`

功能：将停止遥操作命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"stop"}
```

| 响应字段  | 类型/取值             | 含义                                         |
| --------- | --------------------- | -------------------------------------------- |
| `ok`      | 固定 `true`           | 当前 HTTP 请求已通过校验。                   |
| `queued`  | 固定 `true`           | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `stop` | 已入队的命令名称。                           |

通过 `teleop.stopping` 与最终
`teleop.started` 变化确认停止。

#### 1.1.8 `GET /command/home`

功能：将双臂回 home 命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"home"}
```

| 响应字段  | 类型/取值             | 含义                                         |
| --------- | --------------------- | -------------------------------------------- |
| `ok`      | 固定 `true`           | 当前 HTTP 请求已通过校验。                   |
| `queued`  | 固定 `true`           | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `home` | 已入队的命令名称。                           |

无独立完成字段，需要双臂反馈确认。

#### 1.1.9 `GET /command/recenter`

功能：将重置头部参考命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"recenter"}
```

| 响应字段  | 类型/取值                 | 含义                                         |
| --------- | ------------------------- | -------------------------------------------- |
| `ok`      | 固定 `true`               | 当前 HTTP 请求已通过校验。                   |
| `queued`  | 固定 `true`               | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `recenter` | 已入队的命令名称。                           |

实际能否执行受当前头参考模式限制。

### 1.2 在线推理 API

#### API 总表

| API                               | 用途                       |
| --------------------------------- | -------------------------- |
| `GET /inference/status`           | 在线推理及 provider 状态。 |
| `GET /inference/start?prompt=...` | 请求启动在线推理。         |
| `GET /inference/stop`             | 请求停止在线推理。         |
| `GET /ui/provider/hold`           | 切换至安全 hold。          |
| `GET /ui/provider/xr`             | 切回 XR 实时输入。         |

#### 1.2.1 `GET /inference/status`

功能：读取在线推理 provider 和运行状态。

```json
{
  "ok":true,
  "provider":{"active_provider":"hold","input_provider":"hold","last_error":"","last_reason":"ui_hold","real_replay":{"state":"idle","dataset_root":"","episode_name":"","episode_index":-1,"arm_source":"action","base_source":"none","speed_scale":1.0,"frame_index":-1,"error":"","reason":"","runtime_debug":{}},"online_inference":{"state":"idle","prompt":"","error":"","reason":"","debug":{},"runtime_debug":{}}},
  "online_inference":{
    "state":"idle",
    "prompt":"",
    "error":"",
    "reason":"",
    "debug":{},
    "runtime_debug":{}
  }
}
```

| 字段                                                                                                                     | 类型/取值                                                     | 含义                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `ok`                                                                                                                     | 固定 `true`                                                   | HTTP 状态读取成功。                                                                                     |
| `provider.active_provider`                                                                                               | `xr_live` / `hold` / `raw_replay` / `online_inference`        | 当前控制输入源。                                                                                        |
| `provider.input_provider`                                                                                                | string                                                        | 实际输入实现名。                                                                                        |
| `provider.last_error` / `provider.last_reason`                                                                           | string                                                        | 最近 provider 错误和状态原因。                                                                          |
| `provider.real_replay.state` / `dataset_root` / `episode_name` / `episode_index`                                         | string / string / string / integer                            | 真机回放状态和数据标识。                                                                                |
| `provider.real_replay.arm_source` / `base_source` / `speed_scale` / `frame_index` / `error` / `reason` / `runtime_debug` | string / string / number / integer / string / string / object | 真机回放来源、进度、错误、原因和诊断。                                                                  |
| `provider.online_inference`                                                                                              | object                                                        | 与顶层 `online_inference` 内容相同，含 `state`、`prompt`、`error`、`reason`、`debug`、`runtime_debug`。 |
| `online_inference`                                                                                                       | object                                                        | `provider.online_inference` 的便捷镜像。                                                                |
| `online_inference.state`                                                                                                 | `idle` / `running` / `stopped` / `error` / `disabled`         | 推理生命周期；`disabled` 表示状态对象缺失。                                                             |
| `online_inference.prompt`                                                                                                | string                                                        | 当前/最后一次启动传入的任务文本。                                                                       |
| `online_inference.error`                                                                                                 | string                                                        | 推理错误。`state=error` 时必须显示。                                                                    |
| `online_inference.reason`                                                                                                | string                                                        | 启动、停止或失败原因。                                                                                  |
| `online_inference.debug`                                                                                                 | object                                                        | provider 原始调试快照，内部字段由 provider 决定。                                                       |
| `online_inference.runtime_debug`                                                                                         | object                                                        | 主控制循环诊断，结构可扩展。                                                                            |

#### 1.2.2 `GET /inference/start?prompt=<text>`

功能：将在线推理启动请求放入主控制循环队列。

| 参数     | 必填 | 规则                                    |
| -------- | ---- | --------------------------------------- |
| `prompt` | 是   | 去首尾空白后非空；调用方负责 URL 编码。 |

调用示例：

```http
GET /inference/start?prompt=pick%20up%20the%20cube
```

成功响应：

```json
{"ok":true,"queued":true,"command":"start_online_inference"}
```

| 响应字段  | 类型/取值                               | 含义                                         |
| --------- | --------------------------------------- | -------------------------------------------- |
| `ok`      | 固定 `true`                             | 当前 HTTP 请求已通过校验。                   |
| `queued`  | 固定 `true`                             | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `start_online_inference` | 已入队的命令名称。                           |

| HTTP `400` 条件                    | `error` 语义                       |
| ---------------------------------- | ---------------------------------- |
| 录制 active 或 armed               | 先停止/取消数采，不能切 provider。 |
| `active_provider=raw_replay`       | 真机回放活动中，禁止推理。         |
| `active_provider=online_inference` | 推理已经运行。                     |
| provider 非 `hold`、`xr_live`      | 当前输入状态不允许启动。           |
| prompt 为空                        | 必须提供任务文本。                 |

缺少 head、left_wrist 或 right_wrist 相机不是 HTTP `400`：请求会先返回 `queued=true`，主控制循环随后将
`online_inference.state` 置为 `error`，并在 `error` 中说明缺失相机。

响应入队后，必须等 1.2.1 中 `online_inference.state=running` 才表示推理实际启动；
`state=error` 时读取 `error`、`reason`、`runtime_debug`。

#### 1.2.3 `GET /inference/stop`

功能：将在线推理停止请求放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"stop_online_inference"}
```

| 响应字段  | 类型/取值                              | 含义                                         |
| --------- | -------------------------------------- | -------------------------------------------- |
| `ok`      | 固定 `true`                            | 当前 HTTP 请求已通过校验。                   |
| `queued`  | 固定 `true`                            | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `stop_online_inference` | 已入队的命令名称。                           |

后续必须等待
`online_inference.state=stopped` 或 `error`，并确认 `provider.active_provider=hold`。

#### 1.2.4 `GET /ui/provider/hold`

功能：将输入 provider 切换到安全 hold 放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"set_provider_hold"}
```

| 响应字段  | 类型/取值                          | 含义                                         |
| --------- | ---------------------------------- | -------------------------------------------- |
| `ok`      | 固定 `true`                        | 当前 HTTP 请求已通过校验。                   |
| `queued`  | 固定 `true`                        | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `set_provider_hold` | 已入队的命令名称。                           |

录制 active/armed 时返回 HTTP `400`：

```json
{"ok":false,"error":"recording is active or armed; ..."}
```

成功后以
`provider.active_provider=hold` 确认。

#### 1.2.5 `GET /ui/provider/xr`

功能：将输入 provider 切换回 XR 实时输入放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"set_provider_xr"}
```

| 响应字段  | 类型/取值                        | 含义                                         |
| --------- | -------------------------------- | -------------------------------------------- |
| `ok`      | 固定 `true`                      | 当前 HTTP 请求已通过校验。                   |
| `queued`  | 固定 `true`                      | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `set_provider_xr` | 已入队的命令名称。                           |

成功后以
`provider.active_provider=xr_live` 确认。

### 1.3 推理与数采协同 API

#### API 总表

| API                     | 用途                                   |
| ----------------------- | -------------------------------------- |
| `GET /recording/status` | 推理前检查录制、对齐、校验与底盘故障。 |
| `GET /recording/start`  | 推理过程开始数采。                     |
| `GET /recording/stop`   | 停止并保存推理数采。                   |
| `GET /recording/cancel` | 取消推理数采。                         |

#### 1.3.1 `GET /recording/status`

功能：读取录制、相机对齐、校验和底盘安全状态。

推理 UI 用此端点判断当前是否允许切换 provider 或启动推理。推理决策相关响应片段：

```json
{
  "active": false,
  "phase": "idle",
  "last_alignment": {
    "waiting_for_first_frame": false,
    "pending_samples": 0,
    "record_start_monotonic_ns": null
  },
  "base": {
    "enabled": true,
    "receiver_alive": true,
    "odom_topic": "rt/odom",
    "height_topic": "rt/sportmodestate",
    "state_max_age_ms": 100.0,
    "action_max_age_ms": 100.0,
    "stop_confirmed": true,
    "control_fault": false,
    "fault_reason": ""
  }
}
```

| 字段                                               | 类型/取值                                     | 推理 UI 的处理要求                                                                              |
| -------------------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `active`                                           | boolean                                       | `true` 时录制正在写入或处于 armed，禁止切换 provider、启动推理。                                |
| `phase`                                            | `idle` / `armed` / `recording` / `validating` | 只有 `idle` 或 `validating` 才不存在当前录制切换风险；`armed`、`recording` 必须先 stop/cancel。 |
| `last_alignment.waiting_for_first_frame`           | boolean                                       | `true` 表示录制已请求但仍在等待所有相机首帧；等价于不安全的 armed 状态。                        |
| `last_alignment.pending_samples`                   | integer                                       | 已暂存、尚未落盘的对齐样本数。                                                                  |
| `last_alignment.record_start_monotonic_ns`         | integer 或 `null`                             | 当前录制请求起始单调时间；无录制时为 `null`。                                                   |
| `base.enabled`                                     | boolean                                       | 是否启用底盘数据采集。                                                                          |
| `base.receiver_alive`                              | boolean                                       | 底盘状态接收线程是否存活。`false` 时不能把底盘状态视为可信。                                    |
| `base.odom_topic` / `base.height_topic`            | string                                        | 当前底盘状态 DDS topic。                                                                        |
| `base.state_max_age_ms` / `base.action_max_age_ms` | number                                        | 底盘状态和动作可接受的最大对齐时间差。                                                          |
| `base.stop_confirmed`                              | boolean                                       | STOP 是否收到严格确认。                                                                         |
| `base.control_fault`                               | boolean                                       | `true` 表示底盘控制/停车已有明确故障，必须阻止继续自动执行。                                    |
| `base.fault_reason`                                | string                                        | 控制故障原因；非空必须向操作者显示。                                                            |

#### 1.3.2 `GET /recording/start`

功能：将开始录制命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"record_toggle"}
```

| 响应字段  | 类型/取值                      | 含义                                         |
| --------- | ------------------------------ | -------------------------------------------- |
| `ok`      | 固定 `true`                    | 当前 HTTP 请求已通过校验。                   |
| `queued`  | 固定 `true`                    | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `record_toggle` | 已入队的命令名称。                           |

推理数据真正开始写入的确认是
`phase=recording` 且 `frame_index` 增长；`phase=armed` 只是等待相机首帧。

#### 1.3.3 `GET /recording/stop`

功能：将停止并保存录制命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"record_toggle"}
```

| 响应字段  | 类型/取值                      | 含义                                         |
| --------- | ------------------------------ | -------------------------------------------- |
| `ok`      | 固定 `true`                    | 当前 HTTP 请求已通过校验。                   |
| `queued`  | 固定 `true`                    | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `record_toggle` | 已入队的命令名称。                           |

等待 `active=false`；该端点只提供 `last_validation`，当前不提供后台校验是否仍在运行的实时字段。

#### 1.3.4 `GET /recording/cancel`

功能：将取消当前或 armed 录制命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"record_cancel"}
```

| 响应字段  | 类型/取值                      | 含义                                         |
| --------- | ------------------------------ | -------------------------------------------- |
| `ok`      | 固定 `true`                    | 当前 HTTP 请求已通过校验。                   |
| `queued`  | 固定 `true`                    | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `record_cancel` | 已入队的命令名称。                           |

等待 `active=false` 后才可以切换
provider 或重新开始推理。

#### 推理安全调用时序

```text
1. GET /snapshot：确认 teleop.started=true、recording.active=false、phase!=armed
2. GET /ui/provider/hold：等待 provider.active_provider=hold
3. GET /inference/start?prompt=...
4. 等 online_inference.state=running
5. 需要记录则 GET /recording/start，等 phase=recording 和 frame_index 增长
6. GET /inference/stop，确认 provider=hold，再 stop 或 cancel recording
```
