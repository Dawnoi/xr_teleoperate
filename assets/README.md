# assets

机器人模型和手部 retargeting 静态资源。

保留原则：

- `g1/`, `h1/`, `h1_2/`, `h2/`：由 `teleop/robot_control/robot_arm_ik.py` 按 arm 类型加载。
- `g1_d/`, `.generated/`, `g1_description/`：MuJoCo/G1D/Dex1 场景使用；`.generated` 中的场景会引用 `g1_description` 和 `g1_d` 下的 mesh。
- `unitree_hand/`, `inspire_hand/`, `brainco_hand/`：由 hand retargeting 配置加载。

已清理项：未被任何 URDF/XML/YAML 引用的孤立 mesh 文件。删除清单见：

```text
docs/refactor/unused_assets_removed.txt
```
