# 数采平台开发 API

本文描述当前 `teleop/real/teleop_hand_and_arm.py --ui` 实际启动的
`TeleopUiServer`。默认地址为 `http://<robot-host>:8085`，所有当前端点均为 `GET`。

## 1. 宇树 XR Teleoperate

### 1.1 共同状态和控制 API

#### API 总表

| API                              | 响应类型 | 用途                                 |
| -------------------------------- | -------- | ------------------------------------ |
| `GET /`                          | HTML     | 现有 XR UI 页面。                    |
| `GET /index.html`                | HTML     | `/` 的别名。                         |
| `GET /assets/<path>`             | 静态文件 | 页面 CSS、JavaScript、vendor 文件。  |
| `GET /snapshot`                  | JSON     | 一次性完整运行状态。                 |
| `GET /events`                    | SSE      | 实时状态流。                         |
| `GET /camera/status`             | JSON     | 相机健康、时间戳和可用性。           |
| `GET /camera/frame?camera_id=N`  | JPEG     | 最新单帧图像。                       |
| `GET /camera/stream?camera_id=N` | JPEG     | `/camera/frame` 的别名，不是视频流。 |
| `GET /camera/realsense_list`     | JSON     | 兼容相机枚举接口。                   |
| `GET /camera/start`              | JSON     | 未实现，返回 501。                   |
| `GET /camera/stop`               | JSON     | 未实现，返回 501。                   |
| `GET /command/start`             | JSON     | 将启动遥操作命令入队。               |
| `GET /command/stop`              | JSON     | 将停止遥操作命令入队。               |
| `GET /command/home`              | JSON     | 将双臂回 home 命令入队。             |
| `GET /command/recenter`          | JSON     | 将重置头部参考命令入队。             |

共同命令响应：

```json
{"ok":true,"queued":true,"command":"start"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `true` | 当前 HTTP 请求已通过校验。 |
| `queued` | 固定 `true` | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `start` | 已入队的命令名称。 |

`queued=true` 只表示 HTTP 线程完成校验并放入 `UiCommandBus`，不是机器人动作已完成。
命令结果必须通过 `/snapshot` 或 `/events` 确认。业务拒绝通常是 HTTP `400`：

```json
{"ok":false,"error":"具体拒绝原因"}
```

#### 1.1.1 `GET /`

功能：返回现有 XR UI 页面。

| 项目           | 值                          |
| -------------- | --------------------------- |
| 成功状态码     | `200`                       |
| `Content-Type` | `text/html; charset=utf-8`  |
| body           | 现有 XR UI 的 HTML 字节流。 |

此端点不返回 JSON。

#### 1.1.2 `GET /index.html`

功能：返回现有 XR UI 页面。

与 `GET /` 完全相同：`200 text/html; charset=utf-8`，body 为 XR UI HTML，不返回 JSON。

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| HTTP 状态码 | 固定 `200` | 页面成功返回。 |
| `Content-Type` | 固定 `text/html; charset=utf-8` | HTML 页面类型。 |
| body | HTML 字节流 | 与 `GET /` 相同的 XR UI 页面。 |

#### 1.1.3 `GET /assets/<path>`

功能：返回页面所需的静态资源文件。

调用示例：

```http
GET /assets/app.css
```

| 项目                      | 值                                                            |
| ------------------------- | ------------------------------------------------------------- |
| 成功状态码                | `200`                                                         |
| `Content-Type`            | 按文件扩展名推断，例如 `text/css`、`application/javascript`。 |
| body                      | 静态文件原始字节。                                            |
| 文件缺失或越过 web 根目录 | `404`，无 JSON body。                                         |

#### 1.1.4 `GET /snapshot`

功能：读取一次完整运行状态，用于页面初始加载或 SSE 重连。

```json
{
  "schema":"data_collector/v1",
  "left":{"q_fb":null,"gripper_q_fb":null,"gripper_q_cmd":null,"stamp_fb_ns":null,"fk_flange_pose":null},
  "right":{"q_fb":null,"gripper_q_fb":null,"gripper_q_cmd":null,"stamp_fb_ns":null,"fk_flange_pose":null},
  "recording_active":false,
  "recording":{"active":false,"enabled":false,"phase":"idle","session_dir":"","root_dir":"","active_root_dir":"","fps":0.0,"frame_index":0,"error":"","last_alignment":{},"last_alert":{},"alert_seq":0,"last_validation":{},"base":{"enabled":false,"receiver_alive":false,"odom_topic":"","height_topic":"","state_max_age_ms":0.0,"action_max_age_ms":0.0,"stop_confirmed":true,"control_fault":false,"fault_reason":""}},
  "playback":{"state":"disabled","error":"playback is not implemented in xr_teleoperate UI"},
  "convert":{"ok":true,"state":"idle","phase":"idle","running":false,"message":"ready: LeRobot v2 raw exporter is available"},
  "provider":{"active_provider":"hold","input_provider":"hold","last_error":"","last_reason":"ui_hold","real_replay":{"state":"idle","dataset_root":"","episode_name":"","episode_index":-1,"arm_source":"action","base_source":"none","speed_scale":1.0,"frame_index":-1,"error":"","reason":"","runtime_debug":{}},"online_inference":{"state":"idle","prompt":"","protocol_profile":"","error":"","reason":"","debug":{},"runtime_debug":{}}},
  "updated_mono":123.45,
  "teleop":{"started":false,"ready":false,"stopping":false}
}
```

| 字段                                           | 类型/取值                                              | 含义                                             |
| ---------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------ |
| `schema`                                       | 固定 `data_collector/v1`                               | 当前页面 payload 标识，不等于公开版本化 API。    |
| `left` / `right`                               | object                                                 | 左右臂反馈对象。                                 |
| `*.q_fb`                                       | `number[7]` / `number[8]` / `null`                     | 无夹爪反馈时为 7 个机械臂关节；有夹爪反馈时第 8 项为夹爪反馈。 |
| `*.gripper_q_fb`                               | number 或 `null`                                       | 夹爪反馈位置。                                   |
| `*.gripper_q_cmd`                              | number 或 `null`                                       | 夹爪命令位置。                                   |
| `*.stamp_fb_ns`                                | integer 或 `null`                                      | 反馈单调纳秒时间。                               |
| `*.fk_flange_pose`                             | 当前为 `null`                                          | 真机 UI 当前未发布法兰 FK 位姿。                 |
| `recording_active`                             | boolean                                                | `recording.active` 的快捷字段。                  |
| `recording`                                    | object                                                 | 录制状态：`active`、`enabled`、`phase`、目录、采样、对齐、告警、最近校验与 `base`；不返回 `is_recording`、`validation_pending`、`validation_current_episode_dir`、`validation_queued_episode_dirs`。 |
| `playback`                                     | object                                                 | 固定网页回放占位：`state="disabled"`、`error="playback is not implemented in xr_teleoperate UI"`；网页回放实际状态只能从 `/playback/status` 读取。 |
| `convert`                                      | object                                                 | 固定导出占位：`ok=true`、`state="idle"`、`phase="idle"`、`running=false`；导出真实状态只能从 `/convert/status` 读取。 |
| `provider.active_provider`                     | `xr_live` / `hold` / `raw_replay` / `online_inference` | 当前输入来源。                                   |
| `provider.input_provider`                      | string                                                 | 输入实现名，例如 `hold`、`lerobot_offline`。     |
| `provider.last_error` / `provider.last_reason` | string                                                 | 最近 provider 错误和状态原因。                   |
| `provider.real_replay.state`                   | `idle` / `running` / `stopped` / `finished` / `error` | 真机回放生命周期。                               |
| `provider.real_replay.dataset_root` / `episode_name` / `episode_index` | string / string / integer | 数据根、episode 名和数字索引。 |
| `provider.real_replay.arm_source` / `base_source` | `action` / `state` / `fk_cmd_pose`；`none` / `action` | 机械臂和底盘动作来源。 |
| `provider.real_replay.speed_scale` / `frame_index` / `error` / `reason` / `runtime_debug` | number / integer / string / string / object | 真机回放进度、错误、原因和诊断。 |
| `provider.online_inference.state` / `prompt` / `protocol_profile` / `error` / `reason` | `idle` / `running` / `stopped` / `error`；string | 在线推理生命周期、任务文本、实际协议、错误和原因。 |
| `provider.online_inference.debug` / `runtime_debug` | object | provider 原始调试快照和主控制循环诊断。 |
| `updated_mono`                                 | number                                                 | 本机单调秒数，不能转换为 UTC。                   |
| `teleop.started`                               | boolean                                                | 遥操作是否已启动。                               |
| `teleop.ready`                                 | boolean                                                | 是否已具备接管条件。                             |
| `teleop.stopping`                              | boolean                                                | 是否正在停止。                                   |

#### 1.1.5 `GET /events`

功能：建立 SSE 连接，持续接收运行状态变化。

| 项目           | 值                                    |
| -------------- | ------------------------------------- |
| 成功状态码     | `200`                                 |
| `Content-Type` | `text/event-stream; charset=utf-8`    |
| 消息格式       | `data:<snapshot JSON>\n\n`            |
| 发送条件       | 状态版本变化时发送，默认最多约 5 Hz。 |

`data` 的 JSON 字段与 1.1.4 `/snapshot` 相同。每次新建 SSE 连接都会先发送当前缓存状态；
它不推送相机图片，不含 `event:`、`id:` 或历史补偿。客户端可额外请求 `/snapshot`，但不是重连的必要步骤。

#### 1.1.6 `GET /camera/status`

功能：读取当前相机缓存、帧龄和采集元数据。

```json
{
  "url":"/camera/frame?camera_id=0",
  "managed_process_alive":true,
  "managed_pid":null,
  "camera_id":0,
  "camera_name":"head",
  "camera_mode":"rgb",
  "shared_seq":42,
  "shared_timestamp_ns":123456789,
  "shared_age_ms":15,
  "requested_capture":null,
  "actual_capture":{"frame_width":640,"frame_height":480,"channels":3,"transport":"zmq_raw"},
  "active_camera_ids":[0,1,2],
  "streams":[],
  "camera_name_map":{"0":"head"},
  "name_presets":[],
  "last_error":""
}
```

| 字段                                        | 类型/取值                | 含义                                  |
| ------------------------------------------- | ------------------------ | ------------------------------------- |
| `url`                                       | string                   | 主相机预览地址；无相机时为空字符串。  |
| `managed_process_alive`                     | boolean                  | 是否有至少一个相机缓存。              |
| `managed_pid`                               | 当前 `null`              | 兼容字段。                            |
| `camera_id` / `camera_name` / `camera_mode` | integer / string / `rgb` | `streams[0]` 的快捷字段。             |
| `shared_seq`                                | integer                  | 主相机帧序号；无帧时 `-1`。           |
| `shared_timestamp_ns`                       | integer                  | host 收帧或采集的单调纳秒时间。       |
| `shared_age_ms`                             | integer                  | 当前帧龄；无时间戳时 `-1`。           |
| `requested_capture`                         | 当前 `null`              | 兼容字段。                            |
| `actual_capture`                            | object 或 `null`         | 宽、高、通道数及 transport。          |
| `active_camera_ids`                         | integer[]                | 当前可用相机编号。                    |
| `streams`                                   | object[]                 | 每路相机完整状态，见下表。            |
| `camera_name_map`                           | object                   | 编号字符串到名称映射。                |
| `name_presets`                              | object[]                 | 固定角色预设，每项有 `name`、`role`。 |
| `last_error`                                | 当前空字符串             | 兼容字段。                            |

`streams[]`：

| 字段                                                       | 类型/取值       | 含义                                  |
| ---------------------------------------------------------- | --------------- | ------------------------------------- |
| `streams[].camera_id` | `0` / `1` / `2` | `head`、`left_wrist`、`right_wrist`。 |
| `streams[].camera_name` / `streams[].camera_role` | string | `head`、`left_wrist`、`right_wrist`。 |
| `streams[].camera_mode` | 固定 `rgb` | 当前图像模式。 |
| `streams[].url` | string | `/camera/frame?camera_id=N`。 |
| `streams[].shared_seq` / `streams[].shared_timestamp_ns` / `streams[].shared_age_ms` | integer | 帧序号、时间、帧龄。 |
| `streams[].requested_capture` | `null` | 兼容字段。 |
| `streams[].actual_capture.frame_width` / `streams[].actual_capture.frame_height` / `streams[].actual_capture.channels` | integer | 图片尺寸与通道数。 |
| `streams[].actual_capture.transport` | string | 相机 transport 名。 |

#### 1.1.7 `GET /camera/frame?camera_id=N`

功能：返回指定相机的最新单帧 JPEG 预览。

| 参数        | 必填 | 取值          | 默认 |
| ----------- | ---- | ------------- | ---- |
| `camera_id` | 否   | 十进制整数：`0`、`1`、`2` 分别为 head、left_wrist、right_wrist | `0` |

调用示例：

```http
GET /camera/frame?camera_id=0
```

| 情况     | 状态码 | 响应                                                |
| -------- | ------ | --------------------------------------------------- |
| 有最新帧 | `200`  | `Content-Type: image/jpeg`，body 为 JPEG 二进制。   |
| 无最新帧 | `404` | JSON 错误响应，见下方。 |

`camera_id` 不是十进制整数时，当前服务没有定义 JSON 错误响应，连接可能直接失败；调用方必须只发送参数表中的整数。

无最新帧时的响应：

```json
{"ok":false,"error":"camera_id=N has no frame"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| HTTP 状态码 | `200` / `404` | 有帧时为 `200`；数值相机 ID 没有缓存帧时为 `404`。 |
| `Content-Type` | `image/jpeg` / `application/json; charset=utf-8` | 成功为 JPEG，失败为 JSON。 |
| 成功 body | JPEG 二进制 | 当前最新单帧。 |
| `ok` | 固定 `false`，仅 404 | 请求未取得图像。 |
| `error` | string，仅 404 | 无帧原因。 |

#### 1.1.8 `GET /camera/stream?camera_id=N`

功能：兼容别名，返回指定相机的最新单帧 JPEG，不是视频流。

| 参数 | 必填 | 取值 | 默认 |
|---|---|---|---|
| `camera_id` | 否 | 十进制整数：`0`、`1`、`2` 分别为 head、left_wrist、right_wrist | `0` |

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

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| HTTP 状态码 | `200` / `404` | 有帧时为 `200`；没有缓存帧时为 `404`。 |
| `Content-Type` | `image/jpeg` / `application/json; charset=utf-8` | 成功为 JPEG，失败为 JSON。 |
| 成功 body | JPEG 二进制 | 当前最新单帧。 |
| `ok` | 固定 `false`，仅 404 | 请求未取得图像。 |
| `error` | string，仅 404 | 无帧原因。 |

当前只是单帧 JPEG 别名，**不是**连续视频、MJPEG 或 WebRTC；需要预览刷新时由客户端周期性请求。
`camera_id` 不是十进制整数时，当前服务没有定义 JSON 错误响应，连接可能直接失败。

#### 1.1.9 `GET /camera/realsense_list`

功能：返回兼容相机枚举结果。

```json
{"devices":[]}
```

| 字段      | 类型       | 含义                                                     |
| --------- | ---------- | -------------------------------------------------------- |
| `devices` | 固定空数组 | 当前架构不通过此路由枚举 RealSense；不表示系统没有相机。 |

#### 1.1.10 `GET /camera/start`

功能：请求启动相机；当前接口未实现。

未实现，HTTP `501`：

```json
{"ok":false,"error":"/camera/start is not implemented in xr_teleoperate UI"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `false` | 当前路由不可用。 |
| `error` | string | 未实现原因。 |

#### 1.1.11 `GET /camera/stop`

功能：请求停止相机；当前接口未实现。

未实现，HTTP `501`：

```json
{"ok":false,"error":"/camera/stop is not implemented in xr_teleoperate UI"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `false` | 当前路由不可用。 |
| `error` | string | 未实现原因。 |

#### 1.1.12 `GET /command/start`

功能：将启动遥操作命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"start"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `true` | 当前 HTTP 请求已通过校验。 |
| `queued` | 固定 `true` | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `start` | 已入队的命令名称。 |

随后确认
`/snapshot.teleop.started=true`；未确认前不得认为遥操作已启动。

#### 1.1.13 `GET /command/stop`

功能：将停止遥操作命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"stop"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `true` | 当前 HTTP 请求已通过校验。 |
| `queued` | 固定 `true` | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `stop` | 已入队的命令名称。 |

随后检查
`/snapshot.teleop.stopping` 和最终 `started` 状态。

#### 1.1.14 `GET /command/home`

功能：将双臂回 home 命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"home"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `true` | 当前 HTTP 请求已通过校验。 |
| `queued` | 固定 `true` | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `home` | 已入队的命令名称。 |

当前没有独立 home 完成字段；结合
双臂反馈与运行日志确认。

#### 1.1.15 `GET /command/recenter`

功能：将重置头部参考命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"recenter"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `true` | 当前 HTTP 请求已通过校验。 |
| `queued` | 固定 `true` | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `recenter` | 已入队的命令名称。 |

是否能实际执行由当前头参考模式决定。

### 1.2 数采 API

#### API 总表

| API                                              | 用途                         |
| ------------------------------------------------ | ---------------------------- |
| `GET /recording/status`                          | 录制、对齐、校验和底盘状态。 |
| `GET /recording/start`                           | 开始或 armed 录制。          |
| `GET /recording/stop`                            | 停止并保存录制。             |
| `GET /recording/toggle`                          | 切换录制。                   |
| `GET /recording/cancel`                          | 取消当前或 armed 录制。      |
| `GET /recording/set_root_dir?root_dir=...`       | 切换根目录。                 |
| `GET /recording/episodes?root_dir=...&limit=...` | 列出 episode。               |
| `GET /recording/delete_episodes?...`             | 删除 episode。               |
| `GET /recording/set_fps`                         | 兼容占位，不修改 FPS。       |

#### 1.2.1 `GET /recording/status`

功能：读取录制、相机对齐、校验和底盘安全状态。

响应示例：

```json
{
  "is_recording": false,
  "active": false,
  "enabled": true,
  "phase": "idle",
  "session_dir": "",
  "root_dir": "/data/task",
  "active_root_dir": "/data/task",
  "fps": 30.0,
  "frame_index": 0,
  "error": "",
  "last_alignment": {
    "waiting_for_first_frame": false,
    "pending_samples": 0,
    "record_start_monotonic_ns": null
  },
  "last_alert": {},
  "alert_seq": 0,
  "validation_pending": false,
  "validation_current_episode_dir": "",
  "validation_queued_episode_dirs": [],
  "last_validation": {},
  "base": {
    "enabled": false,
    "receiver_alive": false,
    "odom_topic": "",
    "height_topic": "",
    "state_max_age_ms": 0.0,
    "action_max_age_ms": 0.0,
    "stop_confirmed": true,
    "control_fault": false,
    "fault_reason": ""
  }
}
```

| 字段                                       | 类型/取值                                     | 含义                                |
| ------------------------------------------ | --------------------------------------------- | ----------------------------------- |
| `is_recording` / `active`                | boolean                                       | 同值；正在写入，或已 armed 等待相机首帧。 |
| `enabled`                                  | boolean                                       | 是否以 `--record` 启动。            |
| `phase`                                    | `idle` / `armed` / `recording` / `validating` | 当前录制生命周期。                  |
| `session_dir`                              | string                                        | 当前 episode 目录；未开始为空。     |
| `root_dir` / `active_root_dir`             | string                                        | 当前数采根目录。                    |
| `fps` / `frame_index`                      | number / integer                              | 采样频率与已写入帧数。              |
| `error`                                    | string                                        | 当前录制错误，正常为空。            |
| `last_alignment.waiting_for_first_frame`   | boolean                                       | 是否等待 post-start 首帧。          |
| `last_alignment.pending_samples`           | integer                                       | 暂存且未落盘的对齐样本数。          |
| `last_alignment.record_start_monotonic_ns` | integer 或 `null`                             | 录制请求的单调起始时间。            |
| `last_alert` / `alert_seq`                 | object / integer                              | 当前默认 `{}`、`0`。                |
| `validation_pending`                       | boolean                                       | 后台校验是否正在执行或排队；`true` 时不能开始新录制、改目录或删除 episode。 |
| `validation_current_episode_dir`           | string                                        | 正在校验的 episode 目录；无任务时为空字符串。 |
| `validation_queued_episode_dirs`           | string[]                                      | 等待校验的 episode 目录。 |
| `last_validation`                          | object                                        | 最近 episode 的校验报告。           |
| `base`                                     | object                                        | 底盘数采与停车状态，见下表。        |

`base`：

| 字段                                     | 类型    | 含义                       |
| ---------------------------------------- | ------- | -------------------------- |
| `enabled`                                | boolean | 是否 `--record-base`。     |
| `receiver_alive`                         | boolean | 底盘状态接收线程是否存活。 |
| `odom_topic` / `height_topic`            | string  | DDS 状态 topic。           |
| `state_max_age_ms` / `action_max_age_ms` | number  | 对齐最大允许时间差。       |
| `stop_confirmed`                         | boolean | 底盘停止状态是否已锁存为安全；程序启动时默认 `true`，不能单独解释为本次 STOP 已收到确认。 |
| `control_fault`                          | boolean | 底盘控制/停车故障。        |
| `fault_reason`                           | string  | 故障原因；非空不可忽略。   |

#### 1.2.2 `GET /recording/start`

功能：将开始录制命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"record_toggle"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `true` | 当前 HTTP 请求已通过校验。 |
| `queued` | 固定 `true` | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `record_toggle` | 已入队的命令名称。 |

| HTTP `400` 条件           | `error` 语义              |
| ------------------------- | ------------------------- |
| `enabled=false`           | 录制未启用。              |
| `active=true`             | 已在录制。                |

入队后可能先变为 `phase=armed`；只有 `phase=recording` 或 `frame_index` 增长才表示开始写数据。

#### 1.2.3 `GET /recording/stop`

功能：将停止并保存录制命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"record_toggle"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `true` | 当前 HTTP 请求已通过校验。 |
| `queued` | 固定 `true` | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `record_toggle` | 已入队的命令名称。 |

如果当前不是录制状态则 HTTP `400`：

```json
{"ok":false,"error":"recording is not active"}
```

后续等 `active=false` 并按需等待 validation。

#### 1.2.4 `GET /recording/toggle`

功能：将录制切换命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"record_toggle"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `true` | 当前 HTTP 请求已通过校验。 |
| `queued` | 固定 `true` | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `record_toggle` | 已入队的命令名称。 |

该端点不在 HTTP 层检查当前
录制状态，必须通过 1.2.1 确认最终状态。

#### 1.2.5 `GET /recording/cancel`

功能：将取消当前或 armed 录制命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"record_cancel"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `true` | 当前 HTTP 请求已通过校验。 |
| `queued` | 固定 `true` | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `record_cancel` | 已入队的命令名称。 |

用于取消当前或 armed 数采；
完成后确认 `active=false`。

#### 1.2.6 `GET /recording/set_root_dir?root_dir=<path>`

功能：将新的录制根目录放入主控制循环队列。

| 参数       | 必填 | 规则             |
| ---------- | ---- | ---------------- |
| `root_dir` | 是   | 非空路径字符串。 |

调用示例：

```http
GET /recording/set_root_dir?root_dir=%2Fdata%2Fraw_task
```

成功响应：

```json
{"ok":true,"queued":true,"command":"set_record_root","root_dir":"/data/task"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` / `queued` | 固定 `true` | HTTP 校验通过且切换请求已入主控制循环队列。 |
| `command` | 固定 `set_record_root` | 已入队命令名称。 |
| `root_dir` | string | 请求切换到的根目录；不是已切换完成的确认。 |

录制 active/armed、真机回放中或录制未启用时返回 `400`。必须等待
`/recording/status.active_root_dir` 变为目标值。

#### 1.2.7 `GET /recording/episodes?root_dir=<path>&limit=<n>`

功能：列出指定录制根目录中的 episode。

| 参数       | 必填 | 规则                                             |
| ---------- | ---- | ------------------------------------------------ |
| `root_dir` | 否   | 缺省为 `active_root_dir`。不存在目录返回空列表。 |
| `limit`    | 否   | 非负整数；缺省或 0 不限量，非法时 `400`。        |

调用示例：

```http
GET /recording/episodes?root_dir=%2Fdata%2Fraw_task&limit=20
```

```json
{
  "ok":true,
  "root_dir":"/data/task",
  "episodes":[{
    "name":"episode_0001",
    "path":"/data/task/episode_0001",
    "frame_count":123,
    "duration_sec":4.1,
    "start_time":"2026-07-23 14:30:00",
    "end_time":"2026-07-23 14:30:00",
    "validation_level":"ok",
    "validation":{"level":"ok","errors":[],"warnings":[]},
    "cameras":[{"camera_id":0,"camera_name":"head","camera_mode":"rgb"}]
  }]
}
```

| `episodes[]` 字段                | 类型/取值                  | 含义                                            |
| -------------------------------- | -------------------------- | ----------------------------------------------- |
| `name` / `path`                  | string                     | episode 名和路径。                              |
| `frame_count` / `duration_sec`   | integer / number           | 最大相机帧数和估算时长。                        |
| `start_time` / `end_time`        | string                     | `data.json` 修改时间格式化的本地时间。          |
| `validation_level`               | `ok` / `warning` / `error` | 校验等级。                                      |
| `validation.level`               | `ok` / `warning` / `error` | 持久化校验等级。                                |
| `validation.status`              | string，可选               | 如 `missing`、`stale`。                         |
| `validation.errors` / `warnings` | string[]                   | 具体校验问题。                                  |
| `validation.structural` / `time_alignment` / `action_semantics` | object | 结构、时间对齐和动作语义校验的完整子报告。 |
| `validation.mobile_training` | object | 移动训练数据校验报告；无移动训练字段时 `status=not_applicable`。包含 `status`、`frame_count`、`present_frame_count`、`identity_extrinsic_assumption_frames`、`timing`、`limits`、`max_alignment_delta_ms`、`max_slam_tf_age_ms`、`max_map_speed_mps`、`max_map_yaw_rate_radps`、`issues`、`errors`、`warnings`。 |
| `cameras`                        | object[]                   | `camera_id`、`camera_name`、`camera_mode=rgb`。 |

#### 1.2.8 `GET /recording/delete_episodes?root_dir=<path>&episode=<name>`

功能：删除指定录制根目录中的一个或多个 episode。

| 参数       | 必填       | 规则                         |
| ---------- | ---------- | ---------------------------- |
| `root_dir` | 否         | 缺省为 active root。         |
| `episode`  | 是，可重复 | 必须为 `episode_` 后接一个或多个十进制数字，例如 `episode_1`、`episode_0001`。 |

调用示例：

```http
GET /recording/delete_episodes?root_dir=%2Fdata%2Fraw_task&episode=episode_0001&episode=episode_0002
```

成功响应：

```json
{"ok":true,"deleted":["episode_0001"],"root_dir":"/data/task"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `true` | 删除操作完成。 |
| `deleted` | string[] | 已实际删除的 episode 名称。 |
| `root_dir` | string | 实际解析后的删除根目录。 |

录制/armed、真机回放、目录不存在、目标不存在或正播放目标时会被拒绝（`400` 或 `404`）。该接口
直接删除文件目录。

#### 1.2.9 `GET /recording/set_fps`

功能：兼容占位接口；不会修改录制采样频率。

固定响应：

```json
{"ok":true,"fps":0.0}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `true` | 兼容请求已处理。 |
| `fps` | 固定 `0.0` | 占位值，不代表当前采样频率，也不会修改采样频率。 |

这是兼容占位，**不会**修改采样频率。

### 1.3 网页离线回放 API

#### API 总表

| API                                           | 用途               |
| --------------------------------------------- | ------------------ |
| `GET /playback/load?root_dir=...&episode=...` | 加载网页回放。     |
| `GET /playback/start` / `pause` / `stop`      | 控制网页回放。     |
| `GET /playback/seek?frame=N`                  | 跳转帧。           |
| `GET /playback/status`                        | 当前网页回放状态。 |
| `GET /playback/curves?max_points=N`           | 双臂曲线。         |
| `GET /playback/image?camera_id=N&frame=M`     | 指定图像。         |

网页回放只读 episode 数据和图片，不向真机下发动作。

#### 1.3.1 `GET /playback/load?root_dir=<path>&episode=<name>`

功能：加载指定 episode 供浏览器离线回放。

| 参数 | 必填 | 含义 |
|---|---|---|
| `root_dir` | 否 | episode 根目录；未传时使用当前录制根目录。 |
| `episode` | 是 | 要加载的 episode 目录名，例如 `episode_0001`。 |

调用示例：

```http
GET /playback/load?root_dir=%2Fdata%2Fraw_task&episode=episode_0001
```

成功响应：

```json
{
  "ok": true,
  "playback": {
    "enabled": true,
    "state": "paused",
    "episode_name": "episode_0001",
    "episode_dir": "/data/task/episode_0001",
    "frame_index": 0,
    "total_frames": 240,
    "current_time_sec": 0.0,
    "duration_sec": 8.0,
    "timestamp_ns": 123456789,
    "cameras": [{"camera_id": 0, "camera_name": "head", "camera_mode": "rgb"}]
  }
}
```

| 字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `true` | 请求成功。 |
| `playback.enabled` | boolean | 是否成功加载至少一帧。 |
| `playback.state` | `paused` 或 `empty` | 有数据时为 `paused`；episode 无帧时为 `empty`。 |
| `playback.episode_name` / `episode_dir` | string | 实际加载的 episode 名和目录。 |
| `playback.frame_index` / `total_frames` | integer | 初始帧下标和总样本数。 |
| `playback.current_time_sec` / `duration_sec` | number | 初始相对时间和总时长。 |
| `playback.timestamp_ns` | integer 或 `null` | 第 0 帧的 `sample_monotonic_ns`。 |
| `playback.cameras` | object[] | 可用相机；每项有 `camera_id`、`camera_name`、`camera_mode`。 |

缺少 `episode` 时为 HTTP `400`：

```json
{"ok":false,"error":"missing playback episode"}
```

目录不存在或 `data.json` 错误会抛出未转换的服务端异常；当前没有稳定的 HTTP 错误 JSON 或状态码，调用方必须处理请求失败。

#### 1.3.2 `GET /playback/status`

功能：读取浏览器离线回放当前状态。

直接返回当前回放状态：

```json
{
  "enabled": true,
  "state": "paused",
  "episode_name": "episode_0001",
  "episode_dir": "/data/task/episode_0001",
  "frame_index": 12,
  "total_frames": 240,
  "current_time_sec": 0.4,
  "duration_sec": 8.0,
  "timestamp_ns": 123456789,
  "cameras": [{"camera_id": 0, "camera_name": "head", "camera_mode": "rgb"}]
}
```

| 字段 | 类型/取值 | 含义 |
|---|---|---|
| `enabled` | boolean | 是否已加载至少一帧。 |
| `state` | `disabled` / `empty` / `paused` / `playing` | 当前网页回放状态。 |
| `episode_name` / `episode_dir` | string | 当前加载的数据。未加载时为空字符串。 |
| `frame_index` / `total_frames` | integer | 当前播放帧下标和总样本数。 |
| `current_time_sec` / `duration_sec` | number | 当前帧相对起始时刻的秒数与总时长。 |
| `timestamp_ns` | integer 或 `null` | 当前帧的 `sample_monotonic_ns`。 |
| `cameras` | object[] | 每项含 `camera_id`、`camera_name`、`camera_mode`。 |

#### 1.3.3 `GET /playback/start`

功能：开始浏览器离线回放。

成功后直接返回当前回放状态：

```json
{"enabled":true,"state":"playing","episode_name":"episode_0001","episode_dir":"/data/task/episode_0001","frame_index":12,"total_frames":240,"current_time_sec":0.4,"duration_sec":8.0,"timestamp_ns":123456789,"cameras":[{"camera_id":0,"camera_name":"head","camera_mode":"rgb"}]}
```

| 字段 | 类型/取值 | 含义 |
|---|---|---|
| `enabled` | 固定 `true` | 此接口成功时已有可播放数据。 |
| `state` | `playing` 或 `paused` | 多帧且未到最后一帧时为 `playing`；单帧 episode 或从最后一帧启动时会立即为 `paused`。 |
| `episode_name` / `episode_dir` | string | 当前播放的数据。 |
| `frame_index` / `total_frames` | integer | 当前帧和总帧数。 |
| `current_time_sec` / `duration_sec` | number | 当前播放进度与总时长。 |
| `timestamp_ns` | integer 或 `null` | 当前帧采样时间。 |
| `cameras` | object[] | episode 可用相机。 |

未加载 episode 时会抛出未转换的服务端异常；当前没有稳定的 HTTP 错误 JSON 或状态码，不能将请求失败当作空回放状态。

#### 1.3.4 `GET /playback/pause`

功能：暂停浏览器离线回放。

成功后直接返回当前回放状态：

```json
{"enabled":true,"state":"paused","episode_name":"episode_0001","episode_dir":"/data/task/episode_0001","frame_index":12,"total_frames":240,"current_time_sec":0.4,"duration_sec":8.0,"timestamp_ns":123456789,"cameras":[{"camera_id":0,"camera_name":"head","camera_mode":"rgb"}]}
```

| 字段 | 类型/取值 | 含义 |
|---|---|---|
| `enabled` | boolean | 是否已加载回放数据。 |
| `state` | `paused` 或 `disabled` | 有数据时暂停；没有数据时为 `disabled`。 |
| `episode_name` / `episode_dir` | string | 当前加载的数据。 |
| `frame_index` / `total_frames` | integer | 暂停所在帧和总帧数。 |
| `current_time_sec` / `duration_sec` | number | 暂停位置和总时长。 |
| `timestamp_ns` | integer 或 `null` | 暂停帧采样时间。 |
| `cameras` | object[] | episode 可用相机。 |

#### 1.3.5 `GET /playback/stop`

功能：停止浏览器离线回放并回到第 0 帧。

成功后直接返回回到第 0 帧的状态：

```json
{"enabled":true,"state":"paused","episode_name":"episode_0001","episode_dir":"/data/task/episode_0001","frame_index":0,"total_frames":240,"current_time_sec":0.0,"duration_sec":8.0,"timestamp_ns":123456789,"cameras":[{"camera_id":0,"camera_name":"head","camera_mode":"rgb"}]}
```

| 字段 | 类型/取值 | 含义 |
|---|---|---|
| `enabled` | boolean | 是否已加载回放数据。 |
| `state` | `paused` 或 `disabled` | 有数据时回到暂停；没有数据时禁用。 |
| `episode_name` / `episode_dir` | string | 当前加载的数据。 |
| `frame_index` | 固定 `0` | stop 后总是回到第一帧。 |
| `total_frames` | integer | 当前 episode 总帧数。 |
| `current_time_sec` | 固定 `0.0` | stop 后回到起点。 |
| `duration_sec` | number | episode 总时长。 |
| `timestamp_ns` | integer 或 `null` | 第 0 帧采样时间。 |
| `cameras` | object[] | episode 可用相机。 |

#### 1.3.6 `GET /playback/seek?frame=N`

功能：跳转浏览器离线回放到指定帧。

| 参数 | 必填 | 规则 |
|---|---|---|
| `frame` | 否 | 默认 0；非整数按 0；越界夹到 `[0,total_frames-1]`。 |

调用示例：

```http
GET /playback/seek?frame=100
```

成功后直接返回跳转后的状态：

```json
{"enabled":true,"state":"paused","episode_name":"episode_0001","episode_dir":"/data/task/episode_0001","frame_index":100,"total_frames":240,"current_time_sec":3.33,"duration_sec":8.0,"timestamp_ns":123456789,"cameras":[{"camera_id":0,"camera_name":"head","camera_mode":"rgb"}]}
```

| 字段 | 类型/取值 | 含义 |
|---|---|---|
| `enabled` | boolean | 是否已加载回放数据。 |
| `state` | `paused` / `playing` / `disabled` | seek 后的当前播放状态。 |
| `episode_name` / `episode_dir` | string | 当前加载的数据。 |
| `frame_index` | integer | 实际跳转到的帧，已应用越界夹取。 |
| `total_frames` | integer | 当前 episode 总帧数。 |
| `current_time_sec` / `duration_sec` | number | 跳转后的相对时间和总时长。 |
| `timestamp_ns` | integer 或 `null` | 跳转后当前帧采样时间。 |
| `cameras` | object[] | episode 可用相机。 |

#### 1.3.7 `GET /playback/curves?max_points=N`

功能：读取离线 episode 的双臂关节和夹爪曲线。

`max_points` 可选，缺省或非法时为 900。

调用示例：

```http
GET /playback/curves?max_points=900
```

响应示例：

```json
{
  "ok": true,
  "total_frames": 240,
  "sample_count": 120,
  "sample_stride": 2,
  "frame_indices": [0, 2, 4],
  "left": {
    "joint1": [0.1, 0.2, 0.3],
    "joint2": [0.0, 0.0, 0.0],
    "joint3": [0.0, 0.0, 0.0],
    "joint4": [0.0, 0.0, 0.0],
    "joint5": [0.0, 0.0, 0.0],
    "joint6": [0.0, 0.0, 0.0],
    "joint7": [0.0, 0.0, 0.0],
    "gripper_state": [0.0, null, 0.1]
  },
  "right": {
    "joint1": [0.0, 0.0, 0.0],
    "joint2": [0.0, 0.0, 0.0],
    "joint3": [0.0, 0.0, 0.0],
    "joint4": [0.0, 0.0, 0.0],
    "joint5": [0.0, 0.0, 0.0],
    "joint6": [0.0, 0.0, 0.0],
    "joint7": [0.0, 0.0, 0.0],
    "gripper_state": [0.0, 0.0, 0.0]
  }
}
```

| 字段                                              | 类型        | 含义                                              |
| ------------------------------------------------- | ----------- | ------------------------------------------------- |
| `ok`                                              | 固定 `true` | 请求成功。                                        |
| `total_frames` / `sample_count` / `sample_stride` | integer     | 原始帧数、抽样点数、抽样间隔。                    |
| `frame_indices`                                   | integer[]   | 曲线点对应的原始帧。                              |
| `left` / `right`                                  | object      | 均含 `joint1` 到 `joint7`、`gripper_state` 数组。 |

每个数组长度等于 `sample_count`；缺失值为 `null`。

#### 1.3.8 `GET /playback/image?camera_id=N&frame=M`

功能：返回离线 episode 指定相机和指定帧的图像。

| 参数 | 必填 | 规则 | 默认 |
|---|---|---|---|
| `camera_id` | 否 | 十进制整数；`0`、`1`、`2` 分别为 head、left_wrist、right_wrist。非整数按 `0`。 | `0` |
| `frame` | 否 | 十进制整数；越界夹取到 `[0,total_frames-1]`。非整数按 `0`。 | `0` |

调用示例：

```http
GET /playback/image?camera_id=0&frame=100
```

`camera_id` 和 `frame` 均可选，默认 0。成功为 `200`，body 是原始图像文件，`Content-Type`
由扩展名推断。未加载 episode、无帧、该相机无图或文件丢失时，当前服务没有定义稳定的 HTTP 错误 JSON 或状态码，客户端必须处理请求失败。

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| HTTP 状态码 | 成功时 `200` | 成功时返回原始图片文件。 |
| `Content-Type` | 按图片扩展名推断 | 通常为 `image/jpeg` 或 `image/png`。 |
| 成功 body | 图像二进制 | episode 中保存的原始图像文件。 |

### 1.4 真机轨迹回放 API

#### API 总表

| API                          | 用途                  |
| ---------------------------- | --------------------- |
| `GET /replay/real/status`    | 真机回放状态。        |
| `GET /replay/real/start?...` | 回放 episode 到真机。 |
| `GET /replay/real/stop`      | 停止真机回放。        |

#### 1.4.1 `GET /replay/real/status`

功能：读取真机轨迹回放 provider 的状态。

```json
{
  "ok":true,
  "provider":{
    "active_provider":"hold",
    "input_provider":"hold",
    "last_error":"",
    "last_reason":"ui_hold",
    "real_replay":{"state":"idle","dataset_root":"","episode_name":"","episode_index":-1,"arm_source":"action","base_source":"none","speed_scale":1.0,"frame_index":-1,"error":"","reason":"","runtime_debug":{}},
    "online_inference":{"state":"idle","prompt":"","error":"","reason":"","debug":{},"runtime_debug":{}}
  }
}
```

`provider` 顶层字段：

| 字段 | 类型/取值 | 含义 |
|---|---|---|
| `provider.active_provider` | `xr_live` / `hold` / `raw_replay` / `online_inference` | 当前控制输入源。 |
| `provider.input_provider` | string | 实际输入实现名。 |
| `provider.last_error` / `provider.last_reason` | string | 最近 provider 错误和状态原因。 |
| `provider.online_inference.state` / `prompt` / `error` / `reason` | string | 在线推理状态、任务文本、错误与原因。 |
| `provider.online_inference.debug` / `runtime_debug` | object | 在线推理调试快照和控制循环诊断。 |

`provider.real_replay`：

| 字段                                              | 类型/取值                                             | 含义                                         |
| ------------------------------------------------- | ----------------------------------------------------- | -------------------------------------------- |
| `state`                                           | `idle` / `running` / `stopped` / `finished` / `error` | 回放生命周期。                               |
| `dataset_root` / `episode_name` / `episode_index` | string / string / integer                             | 数据根、episode、数字索引；未开始索引 `-1`。 |
| `arm_source`                                      | `action` / `state` / `fk_cmd_pose`                    | 机械臂动作字段来源。                         |
| `base_source`                                     | `none` / `action`                                     | 底盘动作字段来源。                           |
| `speed_scale` / `frame_index`                     | number / integer                                      | 倍率与当前帧。                               |
| `error` / `reason`                                | string                                                | 错误与状态原因。                             |
| `runtime_debug`                                   | object                                                | 可扩展执行诊断。                             |

#### 1.4.2 `GET /replay/real/start`

功能：将指定 episode 的真机轨迹回放请求放入主控制循环队列。

| 参数          | 必填 | 取值/默认                                 |
| ------------- | ---- | ----------------------------------------- |
| `root_dir`    | 否   | active root。                             |
| `episode`     | 是   | 数字或 `episode_0245`。                   |
| `arm_source`  | 否   | `action`；还可为 `state`、`fk_cmd_pose`。 |
| `base_source` | 是   | `none` 或 `action`。                      |
| `speed_scale` | 否   | 正有限数，默认 `1.0`。                    |

调用示例：

```http
GET /replay/real/start?root_dir=%2Fdata%2Fraw_task&episode=episode_0001&arm_source=action&base_source=none&speed_scale=1.0
```

成功响应：

```json
{"ok":true,"queued":true,"command":"start_raw_replay"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `true` | 当前 HTTP 请求已通过校验。 |
| `queued` | 固定 `true` | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `start_raw_replay` | 已入队的命令名称。 |

录制 active/armed、遥操作未
启动、provider 不允许、参数非法时为 `400`。仅 `real_replay.state=running` 表示真正开始。
当 `base_source=action` 时，HTTP 层只校验取值；若进程未以 `--base-motion` 启动，或
`--base-controller` 不是 `g1d_agv`，请求仍会先返回 `queued=true`，随后主控制循环将
`provider.real_replay.state` 置为 `error`。

#### 1.4.3 `GET /replay/real/stop`

功能：将停止真机轨迹回放命令放入主控制循环队列。

成功响应：

```json
{"ok":true,"queued":true,"command":"stop_raw_replay"}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `true` | 当前 HTTP 请求已通过校验。 |
| `queued` | 固定 `true` | 命令已进入主控制循环队列，不代表动作已完成。 |
| `command` | string，当前为 `stop_raw_replay` | 已入队的命令名称。 |

后续以 1.4.1 确认终态。

### 1.5 数据导出 API

#### API 总表

| API                      | 用途                        |
| ------------------------ | --------------------------- |
| `GET /convert/status`    | 后台 LeRobot 导出任务状态。 |
| `GET /convert/start?...` | 启动后台 LeRobot v2 导出。  |

#### 1.5.1 `GET /convert/status`

功能：读取后台 LeRobot 导出任务状态。

此接口直接返回当前后台导出状态。不同 `state` 的字段不同，客户端必须先按 `state` 分支处理。

空闲状态：

```json
{
  "ok": true,
  "state": "idle",
  "phase": "idle",
  "running": false,
  "message": "ready: LeRobot v2 raw exporter is available",
  "processed_frames": 0,
  "total_frames": 0,
  "output_total_frames": 0,
  "append_base_frames": 0,
  "export_video": false,
  "export_fk": false,
  "export_verify": false,
  "export_verification": {}
}
```

运行状态：

```json
{
  "ok": true,
  "state": "running",
  "phase": "exporting",
  "running": true,
  "message": "export started: 2 episodes -> /data/lerobot/demo",
  "processed_frames": 0,
  "total_frames": 480,
  "output_total_frames": 0,
  "append_base_frames": 0,
  "total_episodes": 2,
  "processed_episodes": 0,
  "source_root": "/data/raw_task",
  "output_root": "/data/lerobot/demo",
  "dataset_name": "demo",
  "selected_episodes": ["episode_0001", "episode_0002"],
  "episode_frame_counts": {"episode_0001": 240, "episode_0002": 240},
  "episode_length_reference": {"method": "iqr", "sample_count": 2, "minimum_sample_count": 4},
  "episode_length_warnings": [],
  "format_version": "v2",
  "export_mode": "new",
  "export_video": false,
  "export_fk": false,
  "export_verify": true,
  "urdf_path": "/path/to/g1_d.urdf",
  "start_time_ns": 1780000000000000000,
  "end_time_ns": 0,
  "export_verification": {}
}
```

完成状态：

```json
{
  "ok": true,
  "state": "done",
  "phase": "done",
  "running": false,
  "message": "export done: 2 episodes, 480 frames",
  "processed_frames": 480,
  "total_frames": 480,
  "output_total_frames": 480,
  "append_base_frames": 0,
  "total_episodes": 2,
  "processed_episodes": 2,
  "source_root": "/data/raw_task",
  "output_root": "/data/lerobot/demo",
  "dataset_name": "demo",
  "selected_episodes": ["episode_0001", "episode_0002"],
  "episode_frame_counts": {"episode_0001": 240, "episode_0002": 240},
  "episode_length_reference": {"method": "iqr", "sample_count": 2, "minimum_sample_count": 4},
  "episode_length_warnings": [],
  "format_version": "v2",
  "export_mode": "new",
  "export_video": false,
  "export_fk": false,
  "export_verify": true,
  "urdf_path": "/path/to/g1_d.urdf",
  "start_time_ns": 1780000000000000000,
  "end_time_ns": 1780000001000000000,
  "export_verification": {"ok": true, "errors": [], "summary_path": "/data/lerobot/demo/export_summary.json", "samples": []},
  "export_summary": {}
}
```

请求被导出器同步拒绝时的失败状态：

```json
{
  "ok": false,
  "state": "error",
  "phase": "error",
  "running": false,
  "message": "export rejected",
  "error": "具体失败原因",
  "processed_frames": 0,
  "total_frames": 0,
  "output_total_frames": 0,
  "append_base_frames": 0,
  "export_verification": {},
  "end_time_ns": 1780000001000000000
}
```

后台任务开始后失败时，保留启动时的运行字段，并更新以下字段：

```json
{
  "ok": false,
  "state": "error",
  "phase": "error",
  "running": false,
  "message": "export failed",
  "error": "具体失败原因",
  "traceback": "Python traceback",
  "end_time_ns": 1780000001000000000
}
```

| 字段 | 存在状态 | 类型/取值 | 含义 |
|---|---|---|---|
| `ok` | 全部 | boolean | 当前导出是否成功或可运行。 |
| `state` / `phase` | 全部 | `idle` / `running` / `done` / `error` | 生命周期与阶段。 |
| `running` | 全部 | boolean | 后台导出线程是否运行。 |
| `message` | 全部 | string | 可直接显示的进度或失败说明。 |
| `error` | `error` | string | 同步拒绝或后台失败原因。 |
| `traceback` | 后台失败，或部分计划解析失败 | string | Python traceback；普通同步拒绝时没有该字段。 |
| `processed_frames` / `total_frames` / `output_total_frames` | 全部 | integer | 已处理、计划、输出帧数。 |
| `append_base_frames` | 全部 | integer | 当前兼容字段，固定为 0。 |
| `total_episodes` / `processed_episodes` | `running`、`done`、后台失败 | integer | 总 episode 和已导出数；同步拒绝时不存在。 |
| `source_root` / `output_root` / `dataset_name` | `running`、`done`、后台失败 | string | 输入、实际输出、数据集名；同步拒绝时不存在。 |
| `selected_episodes` / `episode_frame_counts` | `running`、`done`、后台失败 | string[] / object | 导出计划和逐集帧数；同步拒绝时不存在。 |
| `episode_length_reference` / `episode_length_warnings` | `running`、`done`、后台失败 | object / object[] | IQR 长度统计；同步拒绝时不存在。 |
| `format_version` / `export_mode` | `running`、`done`、后台失败 | `v2` / `new` 或 `replace` | 输出配置；同步拒绝时不存在。 |
| `export_video` / `export_fk` / `export_verify` | `idle`、`running`、`done`、后台失败 | boolean | 导出开关；同步拒绝时不存在。 |
| `urdf_path` | `running`、`done`、后台失败 | string | FK 使用的 URDF；同步拒绝时不存在。 |
| `start_time_ns` | `running`、`done`、后台失败 | integer | 导出开始的 wall-clock 纳秒时间；同步拒绝时不存在。 |
| `end_time_ns` | `running` 为 `0`；`done`、`error` 存在 | integer | 完成或失败时的结束时间；idle 不存在。 |
| `export_verification` | 全部 | object | idle、running、error 为空对象；仅 `done` 且 `export_verify=true` 时含 `ok`、`errors`、`summary_path`、`samples`。 |
| `export_summary` | `done` | object | 导出器原样返回的摘要，内部字段由导出器定义。 |

#### 1.5.2 `GET /convert/start`

功能：启动后台 LeRobot v2 导出任务。

| 参数                                           | 必填       | 取值/默认                               |
| ---------------------------------------------- | ---------- | --------------------------------------- |
| `source_root`                                  | 否         | active root。                           |
| `output_root`                                  | 是         | 输出父目录。                            |
| `dataset_name`                                 | 是         | 非空目录名；不能为 `.` 或 `..`，且不能含 `/` 或 `\\`。 |
| `default_task`                                 | 是         | 可用 `task` 代替。                      |
| `fps`                                          | 否         | 当前录制 FPS 或 30，必须正有限。        |
| `format_version`                               | 否         | 仅 `v2`。                               |
| `export_mode`                                  | 否         | `new` 或 `replace`；`append` 未实现。   |
| `export_video` / `export_fk` / `export_verify` | 否         | 布尔值 `0/1/true/false/yes/no/on/off`；默认依次为 `false`、`false`、`true`。 |
| `episode`                                      | 否，可重复 | 缺省导出全部。                          |
| `urdf_path`                                    | 否         | `export_fk=true` 时必须是存在文件。     |

调用示例：

```http
GET /convert/start?source_root=%2Fdata%2Fraw_task&output_root=%2Fdata%2Flerobot&dataset_name=demo_pick&default_task=pick%20up%20the%20cube&fps=30&export_video=false&export_fk=false&export_verify=true&episode=episode_0001&episode=episode_0002
```

启动成功立即返回运行状态：

```json
{
  "ok": true,
  "state": "running",
  "phase": "exporting",
  "running": true,
  "message": "export started: 2 episodes -> /data/lerobot/demo",
  "processed_frames": 0,
  "total_frames": 480,
  "output_total_frames": 0,
  "append_base_frames": 0,
  "total_episodes": 2,
  "processed_episodes": 0,
  "source_root": "/data/raw_task",
  "output_root": "/data/lerobot/demo",
  "dataset_name": "demo",
  "selected_episodes": ["episode_0001", "episode_0002"],
  "episode_frame_counts": {"episode_0001": 240, "episode_0002": 240},
  "episode_length_reference": {"method": "iqr", "sample_count": 2, "minimum_sample_count": 4},
  "episode_length_warnings": [],
  "format_version": "v2",
  "export_mode": "new",
  "export_video": false,
  "export_fk": false,
  "export_verify": true,
  "urdf_path": "/path/to/g1_d.urdf",
  "start_time_ns": 1780000000000000000,
  "end_time_ns": 0,
  "export_verification": {}
}
```

| 响应字段 | 类型/取值 | 含义 |
|---|---|---|
| `ok` | 固定 `true` | 导出任务已被接受并创建后台线程。 |
| `state` / `phase` | 固定 `running` / `exporting` | 返回时导出处于启动后的运行状态。 |
| `running` | 固定 `true` | 后台导出线程已启动。 |
| `message` | string | 已启动任务的进度说明。 |
| `processed_frames` / `total_frames` / `output_total_frames` | integer | 当前已处理、计划处理和已输出的帧数；刚启动时前两者为 0/计划总数。 |
| `append_base_frames` | 固定 `0` | 当前兼容统计字段。 |
| `total_episodes` / `processed_episodes` | integer | 计划导出的 episode 总数和已完成数；刚启动时后者为 0。 |
| `source_root` / `output_root` / `dataset_name` | string | 输入根目录、实际输出数据集目录、数据集名。 |
| `selected_episodes` | string[] | 本次任务实际选择的 episode 名称。 |
| `episode_frame_counts` | object | episode 名称到该 episode 样本帧数的映射。 |
| `episode_length_reference` / `episode_length_warnings` | object / object[] | episode 长度的 IQR 统计基准和异常列表。 |
| `format_version` | 固定 `v2` | 当前 LeRobot 格式版本。 |
| `export_mode` | `new` 或 `replace` | 输出目录策略。 |
| `export_video` / `export_fk` / `export_verify` | boolean | 视频、FK 和验证开关。 |
| `urdf_path` | string | FK 导出使用的 URDF 路径。 |
| `start_time_ns` | integer | 导出开始的 wall-clock 纳秒时间。 |
| `end_time_ns` | 固定 `0` | 导出未结束。 |
| `export_verification` | 固定空对象 | 导出完成前没有验证结果。 |

请求参数缺失或格式错误时，HTTP `400` 返回：

```json
{"ok":false,"error":"output_root is required"}
```

若参数通过 HTTP 层校验、但导出器拒绝请求，则 HTTP `400` 返回 `1.5.1` 的 `state=error` 结构。
启动成功后必须轮询 `GET /convert/status`，直到 `state=done` 或 `state=error`。
