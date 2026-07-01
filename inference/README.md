# inference

在线推理 / VLA 业务域。这里不属于 XR 遥操本体，而是作为策略源被 `teleop` 或 `data_pipeline/replay` 复用。

不要再拆 `clients/ protocols/ sessions/ transforms/` 这种空目录。当前文件少，直接按职责放平：

- `transport.py`：HTTP / TCP JSONL 传输和 action chunk 解析
- `pi05_protocol.py`：PI0.5 / pose payload 编解码
- `online_session.py`：observation 组包、调用策略、chunk 运行节奏
- `pose_transform.py`：pose7 / rot6d / matrix 转换和配置加载

`teleop/inference/*` 只是旧 import 兼容层。
