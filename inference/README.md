# inference

在线推理 / VLA 业务域。这里不属于 XR 遥操本体，而是作为策略源被 `teleop` 或 `data_pipeline/replay` 复用。

当前阶段先建立目录边界，旧实现仍保留在 `teleop/inference`，后续迁移：

- `protocols/`：pika / pi05 / JSONL / HTTP schema
- `clients/`：tcp/http transport
- `sessions/`：OnlineInferenceSession / action chunk buffer
- `transforms/`：ob/action transform 配置加载
