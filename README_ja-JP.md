# xr_teleoperate

## Current Status

This repository is now maintained as an **XR-Robotics-only** teleoperation path.

Removed from the active flow:
- `televuer`
- `teleimager`
- websocket / WebRTC / browser image pipeline

Current path:

```text
XR-Robotics SDK
-> XRRoboticsWrapper
-> TeleData
-> Arm IK
-> Arm Controller / Hand Controller
-> MuJoCo or real robot DDS
```

Default operation now uses a **calibrated head reference**:
- the headset translation is sampled once at startup
- normal head motion does not move the arms
- real robot path supports `c` to recenter

For the current operational details, please refer to:
- `README.md`
- `README_zh-CN.md`
- `AGENTS.md`
- `XR_Robotics_xr_teleoperate_集成上下文.md`
