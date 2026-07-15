# Episode Data Quality Validation Plan

## Scope and boundary

This document defines the validation contract for XR teleoperation episodes used for
training. It separates two different questions that must not be mixed:

1. **Episode-level validation after recording stops**: whether one episode is
   structurally complete, temporally consistent, and suitable to enter a training
   candidate set.
2. **Project-level pre-export validation**: whether selected episodes are anomalous
   relative to the selected project/batch distribution.

Frame-count and average-FPS outlier detection are project-level checks only. A
single finished episode has no reliable project distribution reference, so recording
self-check must not label it short or long.

This is a target design. Existing UI validation text and implementation must be
updated together when this design is implemented.

## Time basis

All recording alignment and validation use host-side monotonic timestamps only:

- `sample_monotonic_ns`
- `state.host_monotonic_ns`
- `action.host_monotonic_ns`
- `camera.*.host_recv_monotonic_ns` or `camera.*.host_monotonic_ns`

Remote sender timestamps and wall-clock timestamps are diagnostic metadata only;
they must not be used for cross-machine alignment decisions.

For sample `i`, record the timestamp of the source item actually selected from each
history buffer. Alignment error is the absolute difference to the sample anchor:

```text
state_error[i]  = abs(sample_time[i] - selected_state_time[i])
action_error[i] = abs(sample_time[i] - selected_action_time[i])
camera_error[camera][i] = abs(sample_time[i] - selected_camera_time[camera][i])
```

Do not compare numerical `state.qpos` and `action.qpos` values as a data-validity
criterion. Action is a control target/input and state is robot feedback; their
physical response delay and controller error make numerical equality neither
expected nor desirable as a validation rule.

## Recording-stop workflow

```text
stop recording
-> close writer and atomically finalize episode files
-> run one complete episode validation job
-> write episode_xxxx/validation.json
-> publish validation result to the UI and episode list
-> allow the next recording only after this job has completed
```

Validation must not run in the real-time teleoperation/DDS control loop. The control
loop remains available for hold and safety commands, while the UI reports that
episode validation is in progress and rejects a new recording request until it
finishes. Validation jobs are serialized; no two finished episodes are validated at
the same time.

## Episode-level complete validation

### Hard errors

The following conditions produce `error` and prevent the episode from being used by
the exporter:

- `data.json` cannot be parsed, is not finalized, or violates the expected schema.
- The episode has zero frames.
- Any enabled camera is missing for any sample. Coverage is exactly `N/N`; 95%
  coverage is not sufficient for training observations.
- A referenced image is missing, cannot be decoded, or has an inconsistent image
  shape/channel count within its camera stream.
- Required left/right arm state or action is absent, has a wrong dimension, or
  contains non-finite values.
- If gripper collection is enabled for the task, gripper state and action are both
  present and finite for every sample; otherwise it is an error.
- `sample_monotonic_ns` is missing, repeated, or moves backwards.
- The final file set does not match the episode manifest.

### Timing report and severity

For valid strictly increasing sample timestamps, define:

```text
dt[i] = sample_time[i] - sample_time[i - 1]
T = mean(dt)
average_fps = 1 / T
```

`T` is episode-specific. Do not use configured camera FPS, target export FPS, or a
fixed `28`/`30` FPS threshold as a data-validity threshold.

Every episode report includes:

- frame count, duration, and average FPS
- frame-interval median, P95, and maximum
- mean, P95, P99, and maximum alignment error for state, action, and every enabled
  camera

Default severity rules:

| Condition | Severity |
| --- | --- |
| Any timestamp repeated or moving backwards | error |
| One frame interval greater than `2.5T` | warning |
| At least `max(3, ceil(number_of_intervals * 1%))` intervals greater than `2.5T` | error |
| At least three consecutive intervals greater than `2.5T` | error |
| Any source alignment-error P95 greater than `0.5T` | warning |
| Alignment-error P99 greater than `T`, and at least `max(3, ceil(frame_count * 1%))` samples exceed `T` | error |
| At least three consecutive source alignment errors greater than `T` | error |

All thresholds, measured values, and offending-frame counts must be included in
`validation.json`. They must not be silently downgraded or hidden from the UI.

## Action semantics and anomalous-motion diagnostics

Timestamp alignment answers whether a sample selected temporally nearby source
records. It does not answer whether the recorded action sequence is physically
plausible or whether the robot response follows commands. Episode validation must
therefore also produce an action-semantic report.

For every joint-space arm action stream, report separately for each arm and joint:

- action position-step magnitude
- action velocity and acceleration peak, median, and P95
- gripper position-step magnitude and jump count
- fraction of state/action samples near each configured joint hard limit
- command-to-feedback response-lag distribution

Action velocity and acceleration are computed from the recorded action timestamps,
not from the configured target FPS. Given joint command `u` and its timestamp `t`:

```text
velocity[i] = (u[i] - u[i - 1]) / (t[i] - t[i - 1])
acceleration[i] = (velocity[i] - velocity[i - 1]) / (t[i] - t[i - 1])
```

Gripper jumps use the same timestamped position-step analysis. Joint-limit
proximity is evaluated only against explicit robot hard limits from the configured
robot model; it must never be inferred from observed dataset min/max values.

Command-to-feedback lag is not `state[i] - action[i]`. It is a response-time
diagnostic: for each sufficiently large command-motion event, find the later
feedback motion event with matching joint/direction, then report the host-monotonic
time difference. The report includes sample count, median, P95, maximum, unmatched
command-event count, and the configured matching window. This preserves the
distinction between normal controller phase delay and a dataset timestamp mismatch.

The default severity policy is:

| Condition | Severity |
| --- | --- |
| NaN/Inf, invalid dimension, or value beyond a configured hard joint limit | error |
| One/few excessive arm velocity, acceleration, or gripper jumps | warning |
| A large number of excessive jumps or rapid actions | error |
| A large fraction of samples near a hard limit | error |
| One/few unmatched or unusually delayed response events | warning |
| Persistent unmatched or excessively delayed response events | error |

The exact robot-dependent limits must come from explicit configuration and be stored
in `validation.json` with the measured statistics. They are not implicit fallback
defaults. "Large number", "large fraction", and "persistent" must likewise be
explicit numeric thresholds in configuration and in the report.

For pose-only action recordings, a joint-limit or joint-action velocity judgment is
not applicable because no joint command is recorded. The report must explicitly say
`not_applicable: joint-space action is absent`; it must not claim the check passed.
Pose-only recordings may separately report translation and rotation step/velocity
statistics, but these do not substitute for joint hard-limit validation. With
`--record-arm-repr both`, run the joint-space checks on the qpos action and report
the pose statistics separately.

## Project-level pre-export validation

Run this only after the user has selected the episodes to export. It does not redo
the expensive per-image episode validation. The exporter accepts only episodes with
a matching, successful `validation.json`; if source files changed after validation,
the validation report is stale and export must explicitly request revalidation.

The selected batch report includes:

- selected episode count and total frames
- each episode's frame count and average FPS
- batch average FPS
- frame-count median and IQR bounds
- average-FPS median and IQR bounds
- frame-count outliers (`short` / `long`)
- average-FPS distribution outliers

At least four valid selected episodes are required for distribution-based outlier
classification. With fewer than four, show an explicit "insufficient samples; skip
distribution outlier check" status and do not fabricate a warning.

Use IQR-based outlier detection for both frame count and average FPS:

```text
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
```

Outliers are `warning`, not an automatic export block. Per-episode `error` remains
an export block. The project-level FPS check compares only the selected batch
distribution; it does not compare against `expected_fps` or export metadata FPS.

## Required UI wording

The UI must make the scope of every result clear. The following wording replaces
ambiguous current text:

| UI location | Required wording/behavior |
| --- | --- |
| Recording self-check | "单条 episode 完整性与时间对齐检查" |
| Camera rule | "要求：每路已启用相机逐帧完整覆盖（N/N）"; do not display "覆盖率 >= 95%" |
| FPS item | "平均采样频率"; show measured average FPS, interval statistics, and `T`; do not label configured FPS as an expected pass/fail value |
| Timing item | "采样间隔质量"; show median/P95/max and long-interval warnings |
| Alignment item | "时间对齐误差"; separately show state, action, and each camera mean/P95/P99/max |
| State/action explanation | "不比较 state 与 action 的数值差；这里只检查它们各自与采样时刻的时间对齐" |
| Action semantics item | "动作语义与异常动作"; show joint action amplitude/velocity/acceleration, gripper jumps, joint-limit proximity, and command-to-feedback lag separately from timestamp alignment |
| Validation pending | "正在完成 episode 自检；完成前不能开始下一条录制" |
| Export page | "项目级分布预检" |
| Export frame warning | "同一导出批次按帧数 IQR 检测，仅提示，不阻断导出" |
| Export FPS warning | "同一导出批次按平均采样频率 IQR 检测，仅提示，不与目标 FPS 比较" |
| Fewer than four selected episodes | "样本不足，跳过帧数与平均采样频率的分布异常判断" |

Never present a warning as `ok`, and never silently fall back to a weaker check when
the required timestamps, image metadata, or validation report are absent. Missing
required evidence is an explicit error or a request for revalidation.

## Training-quality boundary

This validation establishes a lower bound: files are complete, signals are
time-aligned, and obvious temporal failures are visible. It does not prove that a
demonstration accomplishes the task, covers relevant visual states, or will train a
high-quality PI0.5 policy. Those questions require task-level replay review,
dataset composition analysis, and an exported-dataset loader smoke test.
