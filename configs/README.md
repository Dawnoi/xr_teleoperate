# configs

仓库级配置文件。

当前分组：

- `inference/`：在线推理 / VLA 的 pose transform 配置。
- `vive/`：VIVE 踏板 deadman 的通用 udev 权限模板；设备路径仍需按主机单独配置。

示例：

```bash
--online-inference-transform-config configs/inference/unitree_dual_arm_identity_transform.json
--online-inference-transform-config configs/inference/unitree_right_arm_identity_transform.json
```
