import * as THREE from "./vendor/three.module.js";

async function api(path, options = {}) {
  const resp = await fetch(path, { cache: "no-store" });
  const text = await resp.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch (_) { data = { error: text }; }
  if (!resp.ok || (data?.ok === false && !options.allowApplicationError)) throw new Error(data?.error || "request failed");
  return data;
}

function qs(values) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(values)) {
    if (v !== undefined && v !== null && String(v) !== "") q.set(k, String(v));
  }
  return q.toString();
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c] || c));
}

const state = {
  active: "record",
  snapshot: {},
  cameras: [],
  cameraStatus: {},
  realsenseDevices: [],
  episodes: [],
  convertStatus: {},
  inferenceStatus: {},
  exportConfig: {},
  exportSelected: new Set(),
  exportRangeInput: "",
  lastAlertSeq: 0,
  selectedPreviewCameraId: 0,
  liveCurveHistory: [],
  playbackCurves: null,
  playbackSelectedCameraId: null,
  playbackLastImageKey: "",
  playbackStatusTimer: 0,
  playbackEpisodeSearch: "",
  quickPlaybackQuery: "",
  quickPlaybackOpen: false,
  quickPlaybackActiveIndex: 0,
  realReplayCommandPending: "",
  lastConvertNoticeKey: "",
  cameraConfigOpen: false,
  cameraFpsHistory: {},
  lastCameraStatusRefreshMs: 0,
  cameraStatusError: "",
  inferenceTrajectoryHistory: [],
  lastInferenceTrajectorySampleNs: 0,
  lastInferenceRenderMs: 0,
  inferenceThreeState: new WeakMap(),
  shuttingDown: false,
  drawRaf: 0,
  lastCurveDrawMs: 0,
  lastSideStatusMs: 0,
};

function enrichCameraStatus(status) {
  const nowSec = performance.now() / 1000;
  const history = state.cameraFpsHistory || {};
  const streams = Array.isArray(status?.streams) ? status.streams : [];
  streams.forEach((stream) => {
    const cameraId = String(stream.camera_id ?? "");
    const seq = Number(stream.shared_seq);
    const actual = stream.actual_capture || {};
    let measuredFps = Number(actual.measured_fps || 0);
    const prev = history[cameraId];
    if ((!Number.isFinite(measuredFps) || measuredFps <= 0) && prev && Number.isFinite(seq) && seq > prev.seq) {
      const dt = nowSec - prev.t;
      if (dt > 0.05) measuredFps = (seq - prev.seq) / dt;
    }
    history[cameraId] = {
      seq: Number.isFinite(seq) ? seq : prev?.seq ?? -1,
      t: nowSec,
      fps: Number.isFinite(measuredFps) && measuredFps > 0 ? measuredFps : prev?.fps ?? 0,
    };
    stream.actual_capture = {
      ...actual,
      measured_fps: Number.isFinite(measuredFps) && measuredFps > 0 ? measuredFps : history[cameraId].fps,
    };
  });
  state.cameraFpsHistory = history;
  return status || {};
}

function runtimeRecordRoot() {
  const r = state.snapshot?.recording || {};
  return String(r.active_root_dir || r.root_dir || "").trim();
}

function effectiveRecordRoot() {
  const saved = String(localStorage.recordRoot || "").trim();
  const runtime = runtimeRecordRoot();
  if (!saved || saved === "~/data/record/") return runtime || saved || "~/data/record/";
  return saved;
}

function connectionIsOffline() {
  const text = String(el("connText")?.textContent || "").toLowerCase();
  return text === "offline" || text === "exiting" || text === "exited";
}

function expectedShutdownFetchError(error) {
  const message = String(error?.message || error || "");
  return !!state.shuttingDown && (
    message.includes("NetworkError") ||
    message.includes("Failed to fetch") ||
    message.includes("Load failed") ||
    message.includes("fetch")
  );
}


function tone(level) {
  const l = String(level || "unknown").toLowerCase();
  return l === "ok" || l === "running" || l === "connected" ? "ok" : l === "error" || l === "offline" ? "err" : l === "warning" ? "warn" : "";
}

function badge(level, label) {
  const l = String(level || "unknown");
  return `<span class="pill"><span class="dot ${tone(l)}"></span>${esc(label ?? l)}</span>`;
}

function metric(label, value, hint = "") {
  return `<div class="metric"><span>${esc(label)}</span><b>${esc(value)}</b>${hint ? `<small class="mini">${esc(hint)}</small>` : ""}</div>`;
}

function progress(done, total) {
  const d = Number(done || 0);
  const t = Number(total || 0);
  const pct = t > 0 ? Math.max(0, Math.min(100, (d / t) * 100)) : 0;
  return `<div class="progress" title="${d}/${t}"><i style="--pct:${pct.toFixed(1)}%"></i></div>`;
}

function empty(text) {
  return `<div class="empty-state"><div class="empty-icon">⌁</div><div>${esc(text)}</div></div>`;
}


function checkDot(ok) {
  return `<span class="dot ${ok === true ? "ok" : ok === false ? "err" : ""}"></span>`;
}

function validationPending(v) {
  const statuses = [v?.status, v?.level, v?.validation_status, v?.action_semantics?.status]
    .map((value) => String(value || "").toLowerCase());
  return v?.pending === true || statuses.includes("pending");
}

function reportStatus(value) {
  const status = String(value || "").toLowerCase();
  return ["ok", "warning", "error", "pending", "missing", "stale", "incomplete"].includes(status)
    ? status
    : "unknown";
}

function reportState(v) {
  if (!v || typeof v !== "object" || Object.keys(v).length === 0) {
    return { key: "missing", label: "报告缺失", detail: "validation report 尚未提供，不能判定 episode 可用性。" };
  }
  if (validationPending(v)) {
    return { key: "pending", label: "正在自检", detail: "正在完成 episode 自检；完成前不能开始下一条录制。" };
  }
  const explicit = reportStatus(v.status);
  if (["missing", "stale", "incomplete"].includes(explicit)) {
    const labels = { missing: "报告缺失", stale: "报告已过期", incomplete: "报告未完成" };
    return { key: explicit, label: labels[explicit], detail: "当前报告不能作为有效检查证据，请重新执行 episode 自检。" };
  }
  if (!v.checked_at_ns) {
    return { key: "missing", label: "报告未完成", detail: "报告缺少 checked_at_ns，不能判定任何检查为通过。" };
  }
  const level = reportStatus(v.level);
  if (["ok", "warning", "error"].includes(level)) {
    return { key: level, label: semanticStatusLabel(level), detail: "" };
  }
  return { key: "unknown", label: "未完成", detail: "报告缺少有效 level，不能显示通过。" };
}

function qualityClass(status) {
  return status === "ok" ? "ok" : status === "warning" || status === "pending" ? "warning" : "error";
}

function qualityBadge(state) {
  return badge(qualityClass(state.key), state.label);
}

function reportValue(value) {
  if (value === undefined || value === null || value === "") return "未提供";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "无";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function timeUnit(report) {
  return report?.timestamp_source === "host_monotonic_only" ? "ms" : "未知单位";
}

function timeValue(value, unit) {
  if (value === undefined || value === null || value === "") return "未提供";
  const number = Number(value);
  if (!Number.isFinite(number)) return "无效";
  return unit === "ms" ? `${(number / 1e6).toFixed(3)} ms` : `${number.toFixed(3)} ${unit}`;
}

const alignmentStatFields = [
  ["mean_error_ns", "平均误差"],
  ["p95_error_ns", "P95 误差"],
  ["p99_error_ns", "P99 误差"],
  ["max_error_ns", "最大误差"],
  ["above_warning_threshold_count", "超 0.5T"],
  ["above_error_threshold_count", "超 T"],
  ["longest_consecutive_error_run", "最长连续超 T"],
];

function timeAlignmentContract(v) {
  const report = v?.time_alignment;
  if (!report || typeof report !== "object" || Array.isArray(report)) {
    return { report: null, missing: ["time_alignment"] };
  }
  const missing = ["status", "errors", "warnings", "config", "enabled_cameras", "sample_interval", "action_support", "sources"]
    .filter((key) => !Object.prototype.hasOwnProperty.call(report, key));
  if (!Array.isArray(report.errors)) missing.push("errors(list)");
  if (!Array.isArray(report.warnings)) missing.push("warnings(list)");
  const actionSupport = report.action_support;
  if (!actionSupport || typeof actionSupport !== "object" || Array.isArray(actionSupport)) missing.push("action_support(object)");
  else {
    ["status", "linear_count", "exact_count", "nearest_fallback_count", "nearest_fallback_frame_indices", "support_nearest_abs_delta_p95_ns", "support_nearest_abs_delta_max_ns", "support_max_abs_delta_p95_ns", "support_max_abs_delta_max_ns", "support_span_p95_ns", "support_span_max_ns", "warning_frame_indices", "error_frame_indices"]
      .forEach((key) => { if (!Object.prototype.hasOwnProperty.call(actionSupport, key)) missing.push(`action_support.${key}`); });
    ["nearest_fallback_frame_indices", "warning_frame_indices", "error_frame_indices"].forEach((key) => {
      if (!Array.isArray(actionSupport[key])) missing.push(`action_support.${key}(list)`);
    });
  }
  const sample = report.sample_interval;
  if (!sample || typeof sample !== "object" || Array.isArray(sample)) missing.push("sample_interval(object)");
  else {
    ["status", "T_ns", "interval_median_ns", "interval_p95_ns", "interval_max_ns", "long_gap_count", "longest_consecutive_long_gap_run", "offending_frame_indices"]
      .forEach((key) => { if (!Object.prototype.hasOwnProperty.call(sample, key)) missing.push(`sample_interval.${key}`); });
    if (!Array.isArray(sample.offending_frame_indices)) missing.push("sample_interval.offending_frame_indices(list)");
  }
  const sources = report.sources;
  if (!sources || typeof sources !== "object" || Array.isArray(sources)) missing.push("sources(object)");
  else {
    ["state", "action"].forEach((key) => { if (!Object.prototype.hasOwnProperty.call(sources, key)) missing.push(`sources.${key}`); });
    if (!Array.isArray(report.enabled_cameras)) missing.push("enabled_cameras(list)");
    else report.enabled_cameras.forEach((camera) => {
      const sourceName = `camera.${camera}`;
      if (!Object.prototype.hasOwnProperty.call(sources, sourceName)) missing.push(`sources.${sourceName}`);
    });
    Object.entries(sources).forEach(([name, source]) => {
      if (!source || typeof source !== "object" || Array.isArray(source)) {
        missing.push(`sources.${name}`);
        return;
      }
      ["status", ...alignmentStatFields.map(([key]) => key)].forEach((key) => {
        if (!Object.prototype.hasOwnProperty.call(source, key)) missing.push(`sources.${name}.${key}`);
      });
      if (!Array.isArray(source.offending_frame_indices)) missing.push(`sources.${name}.offending_frame_indices(list)`);
      if (name.startsWith("camera.")) {
        ["camera_frame_reuse_count", "camera_frame_reuse_fraction", "longest_consecutive_camera_frame_reuse_run", "camera_frame_reuse_indices"]
          .forEach((key) => { if (!Object.prototype.hasOwnProperty.call(source, key)) missing.push(`sources.${name}.${key}`); });
        if (!Array.isArray(source.camera_frame_reuse_indices)) missing.push(`sources.${name}.camera_frame_reuse_indices(list)`);
      }
    });
  }
  return { report, missing: [...new Set(missing)] };
}

function timeAlignmentState(v) {
  const parent = reportState(v);
  if (["pending", "missing", "stale", "incomplete"].includes(parent.key)) {
    return { key: parent.key, label: parent.label, detail: parent.detail, contract: timeAlignmentContract(v) };
  }
  const contract = timeAlignmentContract(v);
  if (!contract.report) return { key: "missing", label: "报告缺失", detail: "缺少 v.time_alignment，不能判定时间对齐。", contract };
  if (contract.missing.length) {
    return { key: "error", label: "报告字段缺失", detail: `time_alignment 缺少：${contract.missing.join("、")}`, contract };
  }
  const status = reportStatus(contract.report.status);
  if (status === "pending") return { key: "pending", label: "正在自检", detail: "时间对齐报告仍在生成。", contract };
  if (status === "unknown") return { key: "error", label: "未完成", detail: "time_alignment.status 无效，不能显示通过。", contract };
  if (status === "ok" && contract.report.errors.length) return { key: "error", label: "错误", detail: "报告存在 errors，不能显示通过。", contract };
  if (status === "ok" && contract.report.warnings.length) return { key: "warning", label: "有警告", detail: "报告存在 warnings，不能显示通过。", contract };
  return { key: status, label: semanticStatusLabel(status), detail: "", contract };
}

function validationCardState(v) {
  const base = reportState(v);
  if (!["ok", "warning", "error"].includes(base.key)) return base;
  const alignment = timeAlignmentState(v);
  return alignment.key === "ok" ? base : alignment;
}

function alignmentIssues(report) {
  return [
    ...report.errors.map((issue) => `<li class="err-text">${esc(semanticIssueText(issue))}</li>`),
    ...report.warnings.map((issue) => `<li class="warn-text">${esc(semanticIssueText(issue))}</li>`),
  ].join("");
}

function frameIndexPreview(frames) {
  if (!Array.isArray(frames) || !frames.length) return "无";
  const preview = frames.slice(0, 12).join(", ");
  return frames.length > 12 ? `${frames.length} 帧：${preview} ...` : preview;
}

function timeAlignmentSourceHtml(name, source, unit) {
  const stats = alignmentStatFields.map(([key, label]) => `<div class="alignment-stat"><span>${esc(label)}</span><b>${esc(key.endsWith("count") || key === "longest_consecutive_error_run" ? reportValue(source[key]) : timeValue(source[key], unit))}</b></div>`).join("");
  const frames = source.offending_frame_indices;
  const reuseStats = name.startsWith("camera.")
    ? `<div class="alignment-stat"><span>相机帧复用</span><b>${esc(reportValue(source.camera_frame_reuse_count))}</b></div><div class="alignment-stat"><span>复用比例</span><b>${Number.isFinite(Number(source.camera_frame_reuse_fraction)) ? `${(Number(source.camera_frame_reuse_fraction) * 100).toFixed(1)}%` : "未提供"}</b></div><div class="alignment-stat"><span>连续复用</span><b>${esc(reportValue(source.longest_consecutive_camera_frame_reuse_run))}</b></div>`
    : "";
  const reuseText = name.startsWith("camera.") ? `<span class="mini">复用帧 ${esc(frameIndexPreview(source.camera_frame_reuse_indices))}</span>` : "";
  return `<div class="alignment-source"><div class="alignment-source-head"><b>${esc(name)}</b>${badge(reportStatus(source.status))}<span class="mini">异常帧 ${esc(frameIndexPreview(frames))}</span>${reuseText}</div><div class="alignment-stat-grid">${stats}${reuseStats}</div></div>`;
}

function actionSupportSummary(report, unit) {
  const support = report.action_support;
  const status = reportStatus(support.status);
  const state = { key: status, label: semanticStatusLabel(status), detail: "" };
  const stats = [
    ["linear_count", "线性插值"],
    ["exact_count", "原始动作精确命中"],
    ["nearest_fallback_count", "最近邻备用"],
    ["support_nearest_abs_delta_p95_ns", "P95 最近支撑距离"],
    ["support_nearest_abs_delta_max_ns", "最大最近支撑距离"],
    ["support_max_abs_delta_p95_ns", "P95 最远支撑距离"],
    ["support_max_abs_delta_max_ns", "最大最远支撑距离"],
    ["support_span_p95_ns", "P95 支撑跨度"],
    ["support_span_max_ns", "最大支撑跨度"],
  ].map(([key, label]) => `<div class="alignment-stat"><span>${esc(label)}</span><b>${esc(key.endsWith("count") ? reportValue(support[key]) : timeValue(support[key], unit))}</b></div>`).join("");
  return `<div class="quality-section"><div class="quality-section-head"><div><b>动作插值支撑质量</b><span class="mini">动作标签由 head anchor 前后的原始动作构造；支撑距离越小越可靠</span></div>${qualityBadge(state)}</div><div class="alignment-stat-grid">${stats}</div><p class="mini">最近邻备用帧：${esc(frameIndexPreview(support.nearest_fallback_frame_indices))}；支撑异常帧：${esc(frameIndexPreview([...new Set([...support.warning_frame_indices, ...support.error_frame_indices])]))}</p></div>`;
}

function timeAlignmentSummary(v) {
  const state = timeAlignmentState(v);
  const report = state.contract?.report;
  if (!report || state.contract.missing.length) {
    return `<div class="quality-section"><div class="quality-section-head"><div><b>时间对齐误差</b><span class="mini">state、action、camera 各自与采样时刻的误差</span></div>${qualityBadge(state)}</div><div class="validation-conclusion error"><b>${esc(state.label)}</b><span>${esc(state.detail)}</span></div></div>`;
  }
  const unit = timeUnit(report);
  const sample = report.sample_interval;
  const sampleMissing = !sample || typeof sample !== "object" || Array.isArray(sample)
    ? ["sample_interval"]
    : ["status", "T_ns", "interval_median_ns", "interval_p95_ns", "interval_max_ns", "long_gap_count", "longest_consecutive_long_gap_run", "offending_frame_indices"]
      .filter((key) => !Object.prototype.hasOwnProperty.call(sample, key));
  const sampleState = sampleMissing.length
    ? { key: "error", label: "字段缺失", detail: `sample_interval 缺少：${sampleMissing.join("、")}` }
    : { key: reportStatus(sample.status), label: semanticStatusLabel(reportStatus(sample.status)), detail: "" };
  const sampleText = sample && typeof sample === "object" && !Array.isArray(sample)
    ? `<div class="alignment-stat-grid">${[
        ["T_ns", "平均间隔 T"], ["interval_median_ns", "中位间隔"], ["interval_p95_ns", "P95 间隔"], ["interval_max_ns", "最大间隔"],
        ["long_gap_count", "超 2.5T 次数"], ["longest_consecutive_long_gap_run", "最长连续超 2.5T"],
      ].map(([key, label]) => `<div class="alignment-stat"><span>${label}</span><b>${esc(key.includes("count") || key.startsWith("longest_") ? reportValue(sample[key]) : timeValue(sample[key], unit))}</b></div>`).join("")}</div><p class="mini">异常帧索引：${esc(reportValue(sample.offending_frame_indices))}</p>`
    : `<div class="validation-conclusion error"><b>采样间隔字段缺失</b><span>sample_interval 不是对象，不能判定采样间隔质量。</span></div>`;
  const sources = Object.entries(report.sources).map(([name, source]) => timeAlignmentSourceHtml(name, source, unit)).join("");
  const issueHtml = alignmentIssues(report);
  return `<div class="quality-section"><div class="quality-section-head"><div><b>采样间隔质量</b><span class="mini">按 episode 自身采样间隔统计 · 单位：${esc(unit)}</span></div>${qualityBadge(sampleState)}</div>${sampleText}${sampleMissing.length ? `<div class="validation-conclusion error"><b>${esc(sampleState.label)}</b><span>${esc(sampleState.detail)}</span></div>` : ""}</div>${actionSupportSummary(report, unit)}<div class="quality-section"><div class="quality-section-head"><div><b>时间对齐误差</b><span class="mini">state、action、camera 各自与采样时刻的误差 · 单位：${esc(unit)}</span></div>${qualityBadge(state)}</div><div class="validation-conclusion ${qualityClass(state.key)}"><b>时间对齐：${esc(state.label)}</b><span>不比较 state 与 action 的数值差；这里只检查它们各自与采样时刻的时间对齐。</span></div><div class="alignment-source-grid">${sources}</div>${issueHtml ? `<ul class="validation-issues">${issueHtml}</ul>` : "<p class=\"mini\">errors：无；warnings：无。</p>"}</div>`;
}

function semanticField(source, keys) {
  if (!source || typeof source !== "object") return undefined;
  for (const key of keys) {
    if (Object.prototype.hasOwnProperty.call(source, key)) return source[key];
  }
  return undefined;
}

function semanticNumber(value, digits = 3) {
  if (value === undefined || value === null || value === "") return "未提供";
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "无效";
}

function semanticStatusClass(status) {
  const value = String(status || "unknown").toLowerCase();
  return ["ok", "warning", "error"].includes(value) ? value : "unknown";
}

function semanticStatusLabel(status) {
  const value = String(status || "unknown").toLowerCase();
  return {
    ok: "通过",
    warning: "有警告",
    error: "错误",
    not_applicable: "不适用",
    pending: "正在自检",
    unknown: "未完成",
  }[value] || "未完成";
}

function semanticStatText(stats, unit) {
  if (!stats || typeof stats !== "object") return `P95 未提供 · 峰值未提供${unit}`;
  const peak = semanticField(stats, ["peak", "max"]);
  return `P95 ${semanticNumber(stats.p95)}${unit} · 峰值 ${semanticNumber(peak)}${unit}`;
}

function semanticRatioText(report) {
  const near = semanticField(report, ["near_limit", "hard_limit_proximity"]);
  const pairedNear = semanticField(report, ["near_hard_limit"]);
  if (pairedNear && typeof pairedNear === "object") {
    const actionRatio = semanticField(pairedNear.action, ["near_ratio", "ratio", "fraction"]);
    const stateRatio = semanticField(pairedNear.state, ["near_ratio", "ratio", "fraction"]);
    const formatRatio = (value) => value === undefined ? "未提供" : `${semanticNumber(Number(value) * 100, 1)}%`;
    return `action ${formatRatio(actionRatio)} · state ${formatRatio(stateRatio)}`;
  }
  const ratio = semanticField(report, ["near_limit_ratio", "near_limit_fraction"])
    ?? semanticField(near, ["ratio", "fraction"]);
  const count = semanticField(report, ["near_limit_count"])
    ?? semanticField(near, ["count"]);
  const ratioText = ratio === undefined ? "比例未提供" : `${semanticNumber(Number(ratio) * 100, 1)}%`;
  const countText = count === undefined ? "样本数未提供" : `${semanticNumber(count, 0)} 个样本`;
  return `${ratioText} · ${countText}`;
}

function semanticIssueText(issue) {
  if (typeof issue === "string") return issue;
  if (issue && typeof issue === "object") {
    const code = semanticField(issue, ["code"]);
    const message = semanticField(issue, ["message", "reason", "detail"]);
    const scope = [semanticField(issue, ["side"]), semanticField(issue, ["joint"])]
      .filter((value) => value !== undefined && value !== null && String(value) !== "")
      .join(" / ");
    return [code, scope, message].filter((value) => value !== undefined && value !== null && String(value) !== "").join(" · ") || "报告提供了未命名异常";
  }
  return String(issue ?? "报告提供了空异常项");
}

function semanticThresholdText(key, value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "无效";
  if (key.endsWith("_ratio")) return `${(number * 100).toFixed(1)}%`;
  if (key.endsWith("_rad_s2")) return `${number.toFixed(3)} rad/s²`;
  if (key.endsWith("_rad_s")) return `${number.toFixed(3)} rad/s`;
  if (key.endsWith("_m_s")) return `${number.toFixed(3)} m/s`;
  if (key.endsWith("_rad")) return `${number.toFixed(3)} rad`;
  if (key.endsWith("_m")) return `${number.toFixed(3)} m`;
  if (key.endsWith("_s")) return `${number.toFixed(3)} s`;
  return Number.isInteger(number) ? String(number) : number.toFixed(3);
}

const semanticThresholdLabels = {
  near_limit_margin_rad: "硬限位近邻裕量",
  near_limit_warning_ratio: "近限位警告比例",
  near_limit_error_ratio: "近限位错误比例",
  action_step_warning_rad: "动作步长警告阈值",
  action_step_error_rad: "动作步长错误阈值",
  action_velocity_warning_rad_s: "动作速度警告阈值",
  action_velocity_error_rad_s: "动作速度错误阈值",
  action_acceleration_warning_rad_s2: "动作加速度警告阈值",
  action_acceleration_error_rad_s2: "动作加速度错误阈值",
  motion_error_count: "快速动作错误次数",
  gripper_jump_warning_rad: "夹爪跳变警告阈值",
  gripper_jump_error_rad: "夹爪跳变错误阈值",
  gripper_jump_error_count: "夹爪跳变错误次数",
  action_event_min_step_rad: "动作事件最小步长",
  feedback_event_min_step_rad: "反馈事件最小步长",
  response_window_s: "响应匹配窗口",
  response_lag_warning_s: "响应滞后警告阈值",
  response_lag_error_s: "响应滞后错误阈值",
  response_lag_error_count: "响应滞后错误次数",
  response_unmatched_warning_count: "未匹配警告次数",
  response_unmatched_error_count: "未匹配错误次数",
};

function actionSemanticsBadge(v) {
  const state = reportState(v);
  if (!["ok", "warning", "error"].includes(state.key)) return qualityBadge(state);
  const semantics = v?.action_semantics;
  if (!semantics || typeof semantics !== "object") return badge("error", "报告缺失");
  return badge(semantics.status || "unknown", semanticStatusLabel(semantics.status));
}

function actionSemanticsSummary(v) {
  const validation = reportState(v);
  if (validation.key === "pending") {
    return `<div class="action-semantic-section"><div class="validation-conclusion warning"><b>正在完成 episode 自检</b><span>动作语义报告尚未生成；完成前不能开始下一条录制。</span></div></div>`;
  }
  if (["missing", "stale", "incomplete", "unknown"].includes(validation.key)) {
    return `<div class="action-semantic-section"><div class="validation-conclusion error"><b>${esc(validation.label)}</b><span>动作语义不能基于缺失或过期的 validation report 判定。</span></div></div>`;
  }
  const semantics = v?.action_semantics;
  if (!semantics || typeof semantics !== "object") {
    return `<div class="action-semantic-section"><div class="validation-conclusion error"><b>动作语义报告缺失</b><span>当前只能显示完整性与时间对齐结果，动作异常不能判定；请重新执行 episode 自检。</span></div></div>`;
  }

  const status = String(semantics.status || "unknown").toLowerCase();
  const jointSpace = semantics.joint_space || {};
  const jointStatus = String(jointSpace.status || "unknown").toLowerCase();
  const jointReason = semanticField(jointSpace, ["reason", "not_applicable_reason"]);
  const qposText = jointStatus === "not_applicable"
    ? `未记录 qpos 动作，因此无法进行关节空间动作步长、速度、加速度和硬限位检查。${jointReason && jointReason !== "joint-space action is absent" ? ` 报告理由：${jointReason}` : ""}`
    : `状态：${semanticStatusLabel(jointStatus)}${jointStatus === "unknown" ? "；报告未提供 qpos 检查状态，不能判定。" : ""}`;
  const poseSpace = semantics.pose_space;
  const poseText = poseSpace && typeof poseSpace === "object"
    ? `位姿空间：${semanticStatusLabel(poseSpace.status)}`
    : "位姿空间：报告未提供";
  const armReports = jointSpace.arms && typeof jointSpace.arms === "object" ? jointSpace.arms : {};
  const responseArms = semantics.response_lag?.arms && typeof semantics.response_lag.arms === "object" ? semantics.response_lag.arms : {};
  const sides = ["left", "right"];
  const armHtml = sides.map((side) => {
    const arm = armReports[side] && typeof armReports[side] === "object" ? armReports[side] : {};
    const joints = arm.joints && typeof arm.joints === "object" ? arm.joints : {};
    const responseJoints = responseArms[side]?.joints && typeof responseArms[side].joints === "object" ? responseArms[side].joints : {};
    const jointNames = [...new Set([...Object.keys(joints), ...Object.keys(responseJoints)])];
    const rows = jointNames.length ? jointNames.map((jointName) => {
      const report = joints[jointName] && typeof joints[jointName] === "object" ? joints[jointName] : {};
      const lag = responseJoints[jointName] && typeof responseJoints[jointName] === "object" ? responseJoints[jointName] : {};
      const matched = semanticField(lag, ["matched_count", "sample_count"]);
      const unmatched = semanticField(lag, ["unmatched_count", "unmatched_command_event_count"]);
      const lagText = `已匹配 ${semanticNumber(matched, 0)} · 未匹配 ${semanticNumber(unmatched, 0)} · median ${semanticNumber(lag.median_s)}s · P95 ${semanticNumber(semanticField(lag, ["p95_s", "percentile_95_s"]))}s · max ${semanticNumber(lag.max_s)}s`;
      return `<tr><td><b>${esc(jointName)}</b><small class="mini">${esc(semanticStatusLabel(report.status))}</small></td><td>${esc(semanticStatText(report.action_step_rad, " rad"))}</td><td>${esc(semanticStatText(report.action_velocity_rad_s, " rad/s"))}</td><td>${esc(semanticStatText(report.action_acceleration_rad_s2, " rad/s²"))}</td><td>${esc(semanticRatioText(report))}</td><td>${esc(lagText)}</td></tr>`;
    }).join("") : `<tr><td colspan="6" class="mini">${jointStatus === "not_applicable" ? "qpos 检查不适用：未记录关节动作。" : "报告未提供该侧关节统计，不能判定。"}</td></tr>`;
    return `<div class="action-arm-card"><div class="action-block-head"><b>${side === "left" ? "左臂" : "右臂"}</b><span class="mini">每关节统计</span></div><div class="table-wrap action-table-wrap"><table><thead><tr><th>关节</th><th>动作步长<br><small>P95 / 峰值</small></th><th>速度<br><small>P95 / 峰值</small></th><th>加速度<br><small>P95 / 峰值</small></th><th>靠近硬限位</th><th>响应滞后</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
  }).join("");

  const gripperReports = semantics.grippers || jointSpace.grippers || {};
  const gripperRows = sides.map((side) => {
    const arm = armReports[side] || {};
    const report = gripperReports[side] || arm.gripper || {};
    const jumpStats = semanticField(report, ["action_step_rad", "jump_magnitude_rad", "step_rad"]);
    const jumpCount = semanticField(report, ["jump_count", "excessive_jump_count"])
      ?? semanticField(report.jump_stats, ["count", "jump_count"]);
    return `<tr><td><b>${side === "left" ? "左夹爪" : "右夹爪"}</b></td><td>${esc(semanticNumber(jumpCount, 0))}</td><td>${esc(semanticStatText(jumpStats, " rad"))}</td></tr>`;
  }).join("");

  const thresholds = semantics.config?.thresholds && typeof semantics.config.thresholds === "object" ? semantics.config.thresholds : {};
  const thresholdEntries = Object.entries(thresholds);
  const thresholdHtml = thresholdEntries.length
    ? thresholdEntries.map(([key, value]) => `<div><span>${esc(semanticThresholdLabels[key] || key)}</span><b>${esc(semanticThresholdText(key, value))}</b></div>`).join("")
    : `<p class="mini">报告未提供 config.thresholds，不能解释本次检查使用的阈值。</p>`;
  const errors = Array.isArray(semantics.errors) ? semantics.errors : null;
  const warnings = Array.isArray(semantics.warnings) ? semantics.warnings : null;
  const issueHtml = [
    ...(errors === null ? [["error", "报告未提供 errors 字段"]] : errors.map((issue) => ["error", semanticIssueText(issue)])),
    ...(warnings === null ? [["warning", "报告未提供 warnings 字段"]] : warnings.map((issue) => ["warning", semanticIssueText(issue)])),
  ].map(([level, issue]) => `<li class="${level === "error" ? "err-text" : "warn-text"}">${esc(issue)}</li>`).join("");
  return `<div class="action-semantic-section">
    <div class="validation-conclusion ${semanticStatusClass(status)}"><b>动作语义检查：${esc(semanticStatusLabel(status))}</b><span>${esc(qposText)} ${esc(poseText)}</span></div>
    <div class="action-block"><div class="action-block-head"><b>左右臂动作统计</b><span class="mini">步长、速度、加速度 P95/峰值与硬限位比例</span></div><div class="action-arm-grid">${armHtml}</div></div>
    <div class="action-block"><div class="action-block-head"><b>夹爪跳变</b><span class="mini">跳变次数与步长统计</span></div><div class="table-wrap action-table-wrap"><table><thead><tr><th>对象</th><th>跳变次数</th><th>跳变步长</th></tr></thead><tbody>${gripperRows}</tbody></table></div></div>
    <div class="action-block"><div class="action-block-head"><b>配置阈值</b><span class="mini">来自 validation report 的 config.thresholds</span></div><div class="action-thresholds">${thresholdHtml}</div></div>
    <div class="action-block"><div class="action-block-head"><b>errors / warnings</b><span class="mini">动作语义报告原始异常</span></div>${issueHtml ? `<ul class="validation-issues">${issueHtml}</ul>` : "<p class=\"mini\">errors：无；warnings：无。</p>"}</div>
  </div>`;
}

function validationSummary(v) {
  if (validationPending(v)) return `<div class="validation-summary"><div class="validation-conclusion warning"><b>正在完成 episode 自检</b><span>完成前不能开始下一条录制。</span></div></div>`;
  const persistedState = reportState(v);
  if (["missing", "stale", "incomplete"].includes(persistedState.key)) {
    return `<div class="validation-summary quality-section"><div class="quality-section-head"><div><b>结构完整性</b><span class="mini">关节、夹爪、manifest 相机路径和 host receive 时间戳</span></div>${qualityBadge(persistedState)}</div><div class="validation-conclusion error"><b>${esc(persistedState.label)}</b><span>${esc(persistedState.detail)}</span></div></div>`;
  }
  if (!v || !v.checked_at_ns) return `<div class="validation-summary quality-section"><div class="validation-conclusion error"><b>报告未完成</b><span>validation report 缺少 checked_at_ns，不能判定 episode 通过。</span></div></div>`;
  const checks = v.checks || {};
  const cov = checks.camera_coverage || {};
  const manifest = checks.camera_manifest || {};
  const robot = checks.robot_state || {};
  const fps = checks.fps || {};
  const errors = Array.isArray(v.errors) ? v.errors : [];
  const warnings = (Array.isArray(v.warnings) ? v.warnings : []).filter((x) => {
    const text = String(x || "").toLowerCase();
    return !text.includes("robot-camera skew") && !text.includes("camera target delta");
  });
  const cameraCounts = cov.counts || {};
  const frameCount = Number(v.frame_count || checks.frame_count?.count || 0);
  const cameraEntries = Object.entries(cameraCounts);
  const enabledCameras = Array.isArray(manifest.enabled_cameras) ? manifest.enabled_cameras : [];
  const cameraText = enabledCameras.length
    ? `${enabledCameras.join("、")}：每帧必需`
    : "manifest 缺失，不能确认相机完整性";
  const robotCounts = robot.counts || {};
  const hasRobotCounts = Object.keys(robotCounts).length > 0;
  const robotText = hasRobotCounts
    ? ["left", "right"].map((side) => {
        const it = robotCounts[side] || {};
        return `${side} q8 ${Number(it.valid_q8 || 0)}/${frameCount}, gripper ${Number(it.valid_gripper || 0)}/${frameCount}`;
      }).join(" · ")
    : "旧 episode 未记录细分统计；未发现 q_fb 缺失错误";
  const robotOk = robot.ok !== undefined
    ? !!robot.ok
    : !errors.some((x) => String(x).includes(".q_fb"));
  const cameraOk = manifest.ok === true;
  const issueHtml = [...errors.map((x) => ["error", x]), ...warnings.map((x) => ["warning", x])]
    .slice(0, 8)
    .map(([level, text]) => `<li class="${level === "error" ? "err-text" : "warn-text"}">${esc(text)}</li>`)
    .join("");
  return `<div class="validation-summary quality-section">
    <div class="quality-section-head"><div><b>结构完整性</b><span class="mini">关节、夹爪、manifest 相机路径和 host receive 时间戳</span></div>${qualityBadge(reportState(v))}</div>
    <div class="validation-title"><b>${esc(v.episode_name || "episode")}</b>${badge(v.level || "unknown")}</div>
    <div class="validation-conclusion ${v.level || "unknown"}">
      <b>${v.level === "ok" ? "结论：本 episode 数据可用" : v.level === "warning" ? "结论：本 episode 可用但存在风险" : "结论：本 episode 存在数据错误"}</b>
      <span>支撑：关节/夹爪 ${robotOk ? "完整" : "异常"}；相机 ${cameraOk ? "全部覆盖达标" : "存在缺失或覆盖不足"}。</span>
    </div>
    <div class="validation-grid">
      <div>${checkDot(checks.frame_count?.ok)}<span>帧数</span><b>${frameCount}</b></div>
      <div>${checkDot(fps.ok)}<span>观测采样率</span><b>${Number(v.observed_fps || fps.observed || 0).toFixed(1)} fps</b><small>只作结构信息；间隔质量由下方独立判定</small></div>
      <div>${checkDot(robotOk)}<span>关节/夹爪</span><b>${esc(robotText)}</b><small>要求：左右臂每帧 7 关节 + gripper 均为有限数</small></div>
      <div>${checkDot(cameraOk)}<span>相机完整性</span><b>${esc(cameraText)}</b><small>要求：manifest 中每路相机每帧均有路径、文件和 host receive 时间戳</small></div>
    </div>
    ${issueHtml ? `<ul class="validation-issues">${issueHtml}</ul>` : `<p class="mini">未发现错误或警告，数据完整性检查通过。</p>`}
    ${timeAlignmentSummary(v)}
  </div>`;
}

function renderRecord() {
  const r = state.snapshot?.recording || {};
  const v = { ...(r.last_validation || {}), pending: !!r.validation_pending };
  const teleop = state.snapshot?.teleop || {};
  const isRecording = !!r.active;
  const recordEnabled = r.enabled !== false;
  const disabled = isRecording ? "disabled" : "";
  const fixedRuntimeDisabled = "disabled";
  const loadedRoot = String(r.active_root_dir || effectiveRecordRoot());
  const activeIds = state.cameraStatus?.active_camera_ids || [];
  const rsDevices = [...(state.realsenseDevices || [])].sort((a, b) => String(a.serial || "").localeCompare(String(b.serial || "")));
  const streamItems = state.cameraStatus?.streams || [];
  const rsRows = rsDevices.map((c, i) => {
    const serial = String(c.serial || "");
    let cid = i;
    try {
      const saved = JSON.parse(localStorage.cameraSerialIds || "{}");
      if (Number.isFinite(Number(saved[serial]))) cid = Number(saved[serial]);
    } catch (_) {}
    const label = c.name || `RealSense ${serial.slice(-4)}`;
    return `<tr>
      <td><b>cam${cid}</b></td><td>${esc(serial || "-")}</td><td>${esc(label)}</td><td>${esc(c.usb || "-")}</td>
      <td class="row"><button ${fixedRuntimeDisabled}>程序启动时配置</button><button class="secondary" ${fixedRuntimeDisabled}>只读</button></td>
    </tr>`;
  }).join("");
  const runtimeRows = streamItems.map((s) => {
    const actual = s.actual_capture || {};
    const source = actual.transport || "runtime";
    return `<tr>
      <td><b>cam${s.camera_id}</b></td><td>${esc(source)}</td><td>${esc(s.camera_name || "-")}</td><td>${esc(source === "zmq_raw" ? "ZMQ" : "CLI")}</td>
      <td class="row"><button ${fixedRuntimeDisabled}>运行中</button><button class="secondary" ${fixedRuntimeDisabled}>预览只读</button></td>
    </tr>`;
  }).join("");
  const rows = rsRows || runtimeRows;
  const streams = streamItems.map((s) => {
    const actual = s.actual_capture || {};
    const req = s.requested_capture || {};
    const transportText = actual.transport ? `transport ${actual.transport}` : "runtime source";
    const profileText = actual.width && actual.height ? `${actual.width}×${actual.height} @ ${Number(actual.fps || 0).toFixed(0)}fps` : transportText;
    const frameText = actual.frame_width && actual.frame_height
      ? `${actual.frame_width}×${actual.frame_height} @ ${Number(actual.measured_fps || 0).toFixed(1)}fps`
      : "等待首帧";
    const reqText = req.width && req.height ? `${req.width}×${req.height} @ ${Number(req.fps || 0).toFixed(0)}fps` : "-";
    return `<tr><td><b>cam${s.camera_id}</b></td><td>${esc(s.camera_name)}</td><td>${esc(s.camera_mode)}</td><td><b>${esc(frameText)}</b><br><small class="mini">profile ${esc(profileText)}</small></td><td>${esc(reqText)}</td><td>${s.shared_age_ms}ms</td></tr>`;
  }).join("");
  const previews = streamItems.map((s) => {
    const cid = Number(s.camera_id || 0);
    const actual = s.actual_capture || {};
    const frameText = actual.frame_width && actual.frame_height
      ? `${actual.frame_width}×${actual.frame_height} · ${Number(actual.measured_fps || 0).toFixed(1)}fps`
      : "等待首帧";
    return `<div class="preview-tile"><div class="preview-title"><b>${esc(s.camera_name || `cam${cid}`)}</b><span>${esc(s.camera_mode || "rgb")} · ${esc(frameText)} · ${s.shared_age_ms ?? "-"}ms</span></div><div class="preview"><span class="hint">等待 cam${cid}...</span><img data-live-camera="${cid}" data-frame-src="/camera/frame?camera_id=${cid}" style="display:none"></div></div>`;
  }).join("");
  return `<div class="grid">
    <div class="card hero full"><div class="card-head"><div><div class="eyebrow">XR Teleoperate</div><div class="headline">${r.active ? "正在采集" : teleop.started ? "遥操运行中" : "等待接管"}</div><div class="subline">${esc(isRecording && r.session_dir ? r.session_dir : `采集目录：${loadedRoot}`)}</div></div>${badge(r.active ? "running" : teleop.started ? "connected" : "idle", r.active ? "REC" : teleop.started ? "START" : "IDLE")}</div>
      <div class="form record-prep-form"><label>采集 FPS<input id="recordFps" value="${esc(localStorage.recordFps || r.fps || 30)}" disabled></label><div><p class="mini">这里沿用参考 collector 的布局；实际 FPS、相机和目录来自启动参数，网页只发控制意图。</p><p class="row"><button onclick="startTeleop()" ${teleop.started ? "disabled" : ""}>开始接管</button><button class="secondary" onclick="homeTeleop()" ${teleop.started ? "" : "disabled"}>回到 Home</button><button class="secondary" onclick="recenterTeleop()" ${teleop.started ? "" : "disabled"}>重置头参考</button><button class="danger" onclick="stopTeleop()">停止退出</button></p><p class="row"><button onclick="startRec()" ${r.active || r.validation_pending || !teleop.started || !recordEnabled ? "disabled" : ""}>● 开始录制</button><button class="danger" onclick="stopRec()" ${r.active ? "" : "disabled"}>停止并保存</button><button class="secondary" onclick="cancelRec()" ${r.active ? "" : "disabled"}>取消录制</button>${recordEnabled ? "" : `<span class="pill"><span class="dot warn"></span>启动时未加 --record</span>`}</p></div></div>
      ${isRecording ? `<p class="mini">录制已进入 ${esc(r.phase || "recording")}；相机帧、机器人状态和 action 仍由主循环按 monotonic 时间对齐。</p>` : ""}
      <div class="prep-panels"><div><div class="toolbar"><div><h2>相机来源</h2><p class="mini">当前仓库相机由 CLI 参数打开；网页不启动、不停止相机。</p></div><button class="secondary" onclick="refreshAll()">刷新</button></div><div class="table-wrap"><table><thead><tr><th>Stream ID</th><th>来源</th><th>名称</th><th>链路</th><th>状态</th></tr></thead><tbody>${rows || `<tr><td colspan="5">${empty("尚未收到运行时相机帧；请检查启动参数、ZMQ sender 和网络链路")}</td></tr>`}</tbody></table></div></div><div><div class="toolbar"><div><h2>运行中流</h2><p class="mini">${activeIds.length} active${state.cameraStatusError ? ` · ${esc(state.cameraStatusError)}` : ""}</p></div></div><div class="table-wrap"><table><thead><tr><th>ID</th><th>名称</th><th>模式</th><th>实际输出</th><th>请求参数</th><th>帧年龄</th></tr></thead><tbody>${streams || `<tr><td colspan="6">${empty("暂无运行流；检查启动参数是否启用本地或 ZMQ 相机")}</td></tr>`}</tbody></table></div></div></div>
    </div>
    <div class="card full record-validation-card"><div class="card-head"><h2>单条 episode 完整性与时间对齐检查</h2>${qualityBadge(validationCardState(v))}</div>${validationSummary(v)}</div>
    <div class="card full record-action-semantics-card"><div class="card-head"><h2>动作语义与异常动作</h2>${actionSemanticsBadge(v)}</div>${actionSemanticsSummary(v)}</div>
    <div class="card full"><div class="card-head"><div><h2>多相机实时预览</h2><p class="mini">预览只读取当前进程已有 latest frame；不重启相机，不参与录制写盘。</p></div><span class="pill">${streamItems.length} live</span></div><div class="preview-grid layout-placeholder">${previews || empty("暂无可预览相机；请检查启动参数和相机链路")}</div></div>
    <div class="card full"><div class="card-head"><div><h2>实时数据曲线</h2><p class="mini">四图布局：左/右臂 J1-J7 与左/右夹爪分别显示，不再用线型区分。</p></div><span class="pill">4 charts</span></div><div class="curve-quad"><div class="curve-panel"><div class="curve-title"><b>Left J1-J7</b><span>q_fb</span></div><canvas id="liveLeftJointCurveCanvas" height="220"></canvas><div class="curve-legend" id="liveLeftJointLegend"></div></div><div class="curve-panel"><div class="curve-title"><b>Right J1-J7</b><span>q_fb</span></div><canvas id="liveRightJointCurveCanvas" height="220"></canvas><div class="curve-legend" id="liveRightJointLegend"></div></div><div class="curve-panel"><div class="curve-title"><b>Left Gripper</b><span>q_fb</span></div><canvas id="liveLeftGripperCurveCanvas" height="180"></canvas><div class="curve-legend" id="liveLeftGripperLegend"></div></div><div class="curve-panel"><div class="curve-title"><b>Right Gripper</b><span>q_fb</span></div><canvas id="liveRightGripperCurveCanvas" height="180"></canvas><div class="curve-legend" id="liveRightGripperLegend"></div></div></div></div>
    <div class="modal ${state.cameraConfigOpen ? "show" : ""}" onclick="if(event.target===this) closeCameraConfig()"><div class="modal-card"><div class="card-head"><div><h2>相机启动参数说明</h2><p class="mini">当前仓库相机在 Python 进程启动时打开，网页只读取 latest frame。</p></div><button class="secondary" onclick="closeCameraConfig()">关闭</button></div><div class="form compact-form"><label>宽<input id="camWidth" value="${esc(localStorage.camWidth || "1280")}" disabled></label><label>高<input id="camHeight" value="${esc(localStorage.camHeight || "720")}" disabled></label><label>FPS<input id="camFps" value="${esc(localStorage.camFps || "30")}" disabled></label><label>模式<select id="camMode" disabled><option value="rgb" ${(localStorage.camMode || "rgb") === "rgb" ? "selected" : ""}>RGB</option><option value="depth" ${localStorage.camMode === "depth" ? "selected" : ""}>Depth</option></select></label><label>名称<input id="camName" placeholder="head / left_wrist / right_wrist" value="${esc(localStorage.camName || "")}" disabled></label><label>角色<input id="camRole" value="${esc(localStorage.camRole || "runtime")}" disabled></label></div><p class="mini">请用 --head-camera-id / --left-camera-id / --right-camera-id 或 --head-zmq-endpoint / --left-zmq-endpoint / --right-zmq-endpoint 配置。这样录制线程和 UI 预览共享同一个相机 source，不会因为网页操作重启相机而破坏对齐。</p></div></div>
  </div>`;
}

function displayedPlayback() {
  const playback = state.snapshot?.playback || {};
  const provider = state.snapshot?.provider || {};
  const realReplay = provider.real_replay || {};
  const frameIndex = Number(realReplay.frame_index);
  if (
    provider.active_provider === "raw_replay"
    && String(realReplay.episode_name || "") === String(playback.episode_name || "")
    && Number.isFinite(frameIndex)
    && frameIndex >= 0
  ) {
    return { ...playback, frame_index: frameIndex, state: "real_replay" };
  }
  return playback;
}

function quickPlaybackMatches() {
  const query = String(state.quickPlaybackQuery || "").trim().toLowerCase();
  return (state.episodes || []).filter((episode) => {
    const name = String(episode?.name || "");
    return name && (!query || name.toLowerCase().includes(query));
  });
}

function quickPlaybackOptionsHtml() {
  const matches = quickPlaybackMatches();
  if (!matches.length) return `<div class="quick-episode-empty">没有匹配的 episode</div>`;
  const active = Math.max(0, Math.min(Number(state.quickPlaybackActiveIndex || 0), matches.length - 1));
  return matches.map((episode, index) => `<button type="button" class="quick-episode-option ${index === active ? "active" : ""}" onclick='selectQuickPlaybackEpisode(${JSON.stringify(String(episode.name))})'>${esc(episode.name)}</button>`).join("");
}

function renderQuickPlaybackOptions() {
  const menu = document.getElementById("quickPlaybackOptions");
  const list = document.getElementById("quickPlaybackOptionList");
  if (!menu || !list) return;
  list.innerHTML = quickPlaybackOptionsHtml();
  menu.classList.toggle("hidden", !state.quickPlaybackOpen);
  const toggle = document.getElementById("quickPlaybackToggle");
  if (toggle) toggle.setAttribute("aria-expanded", String(state.quickPlaybackOpen));
}

function updateQuickPlaybackQuery(value) {
  state.quickPlaybackQuery = String(value || "");
  state.quickPlaybackOpen = true;
  state.quickPlaybackActiveIndex = 0;
  renderQuickPlaybackOptions();
}

function openQuickPlaybackOptions() {
  state.quickPlaybackOpen = true;
  state.quickPlaybackActiveIndex = 0;
  renderQuickPlaybackOptions();
}

function toggleQuickPlaybackOptions() {
  state.quickPlaybackOpen = !state.quickPlaybackOpen;
  state.quickPlaybackActiveIndex = 0;
  renderQuickPlaybackOptions();
  if (state.quickPlaybackOpen) document.getElementById("quickPlaybackFilter")?.focus();
}

function closeQuickPlaybackOptions() {
  state.quickPlaybackOpen = false;
  renderQuickPlaybackOptions();
}

function selectQuickPlaybackEpisode(name) {
  const episodeName = String(name || "");
  if (!episodeName) return;
  state.quickPlaybackQuery = "";
  state.quickPlaybackOpen = false;
  state.quickPlaybackActiveIndex = 0;
  const toggle = document.getElementById("quickPlaybackToggle");
  if (toggle) toggle.firstChild.textContent = episodeName;
  renderQuickPlaybackOptions();
  loadPlayback(episodeName);
}

function handleQuickPlaybackKey(event) {
  const matches = quickPlaybackMatches();
  if (event.key === "Escape") {
    event.preventDefault();
    closeQuickPlaybackOptions();
    return;
  }
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    state.quickPlaybackOpen = true;
    if (matches.length) {
      const delta = event.key === "ArrowDown" ? 1 : -1;
      state.quickPlaybackActiveIndex = (Number(state.quickPlaybackActiveIndex || 0) + delta + matches.length) % matches.length;
    }
    renderQuickPlaybackOptions();
    return;
  }
  if (event.key === "Enter") {
    event.preventDefault();
    const exact = matches.find((episode) => String(episode.name) === String(state.quickPlaybackQuery || ""));
    const selected = exact || matches[Math.max(0, Math.min(Number(state.quickPlaybackActiveIndex || 0), matches.length - 1))];
    if (!selected) {
      showBanner("没有可加载的 episode。", "error");
      return;
    }
    selectQuickPlaybackEpisode(selected.name);
  }
}

function renderPlayback() {
  const p = displayedPlayback();
  const r = state.snapshot?.recording || {};
  const teleop = state.snapshot?.teleop || {};
  const provider = state.snapshot?.provider || {};
  const realReplay = provider.real_replay || {};
  const realReplayCommandPending = String(state.realReplayCommandPending || "");
  const realReplayRunning = provider.active_provider === "raw_replay" || realReplay.state === "running" || realReplayCommandPending === "start";
  const canStartRealReplay = !!p.episode_name && !!teleop.started && !r.active && provider.active_provider === "hold" && !realReplayRunning && !realReplayCommandPending;
  const realReplaySpeed = String(localStorage.realReplaySpeedScale || realReplay.speed_scale || "1.0");
  const realReplayArmSource = String(localStorage.realReplayArmSource || realReplay.arm_source || "action");
  const replayTrace = realReplay.runtime_debug?.execution_trace || {};
  const pendingReplayTrace = replayTrace.current || null;
  const latestReplayTrace = replayTrace.latest || null;
  const trace = latestReplayTrace || pendingReplayTrace || {};
  const traceState = pendingReplayTrace
    ? `pending #${pendingReplayTrace.seq ?? "-"} · showing #${latestReplayTrace?.seq ?? "-"}`
    : latestReplayTrace ? `${latestReplayTrace.status || "completed"} #${latestReplayTrace.seq ?? "-"}` : "waiting";
  const traceMs = (key) => Number.isFinite(Number(trace[key])) ? `${Number(trace[key]).toFixed(1)} ms` : "-";
  const query = String(state.playbackEpisodeSearch || "").trim().toLowerCase();
  const filteredEpisodes = (state.episodes || []).filter((e) => {
    const name = String(e.name || "").toLowerCase();
    const level = String(e.validation_level || "").toLowerCase();
    const time = String(e.start_time || e.end_time || "").toLowerCase();
    return !query || name.includes(query) || level.includes(query) || time.includes(query);
  });
  const captureTime = (e) => String(e.start_time || e.end_time || "").trim() || "-";
  const rows = filteredEpisodes.map((e) => `<tr>
    <td><input data-delete-ep type="checkbox" value="${esc(e.name)}" style="width:auto"></td><td><b>${esc(e.name)}</b></td><td>${esc(captureTime(e))}</td><td>${badge(e.validation_level)}</td><td>${e.frame_count || 0}</td><td>${Number(e.duration_sec || 0).toFixed(1)}s</td>
    <td><button onclick='loadPlayback(${JSON.stringify(e.name)})' ${realReplayRunning ? "disabled" : ""}>加载</button><button class="secondary" onclick='deleteEpisodes([${JSON.stringify(e.name)}])'>删除</button></td>
  </tr>`).join("");
  const cameraCards = (p.cameras || []).map((c) => {
    const cid = Number(c.camera_id || 0);
    return `<div class="preview-tile playback-camera-tile" data-playback-tile="${cid}"><div class="preview-title"><b>${esc(c.camera_name || `cam${cid}`)}</b><span>id=${cid}</span></div><div class="preview"><span class="hint" data-playback-hint="${cid}">等待 cam${cid}...</span><img data-playback-camera="${cid}" style="display:none"></div></div>`;
  }).join("");
  const maxFrame = Math.max(0, Number(p.total_frames || 0) - 1);
  const selectedEpisode = (state.episodes || []).find((episode) => String(episode.name || "") === String(p.episode_name || ""));
  const selectedEpisodeMeta = selectedEpisode
    ? `${Number(selectedEpisode.frame_count || 0)} 帧 · ${Number(selectedEpisode.duration_sec || 0).toFixed(1)}s · 自检 ${String(selectedEpisode.validation_level || "unknown")}`
    : "加载 episode 后显示帧数、时长和自检结果";
  return `<div class="grid">
    <div class="playback-top-grid">
    <div class="card hero playback-control-card"><div class="card-head"><div><div class="eyebrow">Episode Playback</div><div class="headline">${esc(p.episode_name || "选择一个 episode")}</div><div class="subline">${esc(selectedEpisodeMeta)}</div></div>${badge(p.state || "idle")}</div>
      <div class="quick-episode-picker"><label for="quickPlaybackToggle">快速选择 episode</label><div class="quick-episode-combobox"><button id="quickPlaybackToggle" type="button" class="secondary quick-episode-toggle" aria-expanded="${state.quickPlaybackOpen}" onclick="toggleQuickPlaybackOptions()" ${realReplayRunning ? "disabled" : ""}><span>${esc(p.episode_name || "选择 episode")}</span><span aria-hidden="true">&#9662;</span></button><div id="quickPlaybackOptions" class="quick-episode-options ${state.quickPlaybackOpen ? "" : "hidden"}"><input id="quickPlaybackFilter" type="search" autocomplete="off" value="${esc(state.quickPlaybackQuery)}" placeholder="过滤 episode" oninput="updateQuickPlaybackQuery(this.value)" onkeydown="handleQuickPlaybackKey(event)"><div id="quickPlaybackOptionList">${quickPlaybackOptionsHtml()}</div></div></div></div>
      <div class="metric-grid"><div class="metric"><span>Frame</span><b id="playbackMetricFrame">${p.frame_index || 0}/${p.total_frames || 0}</b></div><div class="metric"><span>Time</span><b id="playbackMetricTime">${Number(p.current_time_sec || 0).toFixed(2)}s</b></div><div class="metric"><span>Provider</span><b>${esc(provider.active_provider || "-")}</b></div><div class="metric"><span>Real Replay</span><b>${esc(realReplay.state || "idle")}</b><small>${esc(realReplay.episode_name || "")} ${Number(realReplay.frame_index || -1) >= 0 ? `#${Number(realReplay.frame_index)}` : ""}</small></div></div>
      <p class="row"><button onclick="playbackStart()" ${realReplayRunning ? "disabled" : ""}>播放画面</button><button class="secondary" onclick="playbackPause()" ${realReplayRunning ? "disabled" : ""}>暂停画面</button><button class="secondary" onclick="playbackStop()" ${realReplayRunning ? "disabled" : ""}>停止画面</button><button class="secondary" onclick="loadPlaybackCurves()">重载曲线</button></p>
      <div class="form compact-form real-replay-controls"><label>arm_source<select id="realReplayArmSource" onchange="localStorage.realReplayArmSource=this.value"><option value="action" ${realReplayArmSource === "action" ? "selected" : ""}>action</option><option value="state" ${realReplayArmSource === "state" ? "selected" : ""}>state</option><option value="fk_cmd_pose" ${realReplayArmSource === "fk_cmd_pose" ? "selected" : ""}>fk_cmd_pose</option></select></label><label>speed_scale<input id="realReplaySpeed" value="${esc(realReplaySpeed)}" onchange="localStorage.realReplaySpeedScale=this.value"></label><div><p class="mini">切到回放页后 provider 会进入 HOLD；开始真机回放前必须未录制、已启动 teleop、且已加载 episode。</p><p class="row"><button class="danger" onclick="startRealReplay()" ${canStartRealReplay ? "" : "disabled"}>开始真机回放</button><button class="secondary" onclick="stopRealReplay()" ${realReplayRunning && realReplayCommandPending !== "stop" ? "" : "disabled"}>停止真机回放</button>${realReplayCommandPending ? `<span class="pill"><span class="dot warn"></span>${esc(realReplayCommandPending === "start" ? "启动指令待确认" : "停止指令待确认")}</span>` : ""}${realReplay.error ? `<span class="pill"><span class="dot err"></span>${esc(realReplay.error)}</span>` : ""}</p></div></div>
      <div class="playback-controls"><input id="playbackSeek" type="range" min="0" max="${maxFrame}" value="${Number(p.frame_index || 0)}" oninput="previewPlaybackSeek(this.value)" onchange="seekPlayback(this.value)" ${realReplayRunning ? "disabled" : ""}><span id="playbackTimeLabel" class="mini">frame ${Number(p.frame_index || 0)} / ${Number(p.total_frames || 0)}</span></div></div>
    <div class="card playback-trace-card"><div class="card-head"><div><h2>真机回放执行 trace</h2><p class="mini">以 raw episode 当前帧进入控制循环为起点，测量 DDS 发布和状态线程首次检测到关节运动；不包含网页图片传输。</p></div><span id="playbackTraceBadge">${badge(pendingReplayTrace ? "running" : trace.status || "idle", traceState)}</span></div><div class="metric-grid"><div class="metric"><span>帧到 DDS 发布</span><b id="playbackTraceRecvToPub">${traceMs("recv_to_pub_ms")}</b></div><div class="metric"><span>DDS 到线程反馈运动</span><b id="playbackTracePubToThread">${traceMs("pub_to_exec_thread_ms")}</b></div><div class="metric"><span>帧到线程反馈运动</span><b id="playbackTraceRecvToThread">${traceMs("recv_to_exec_thread_ms")}</b></div><div class="metric"><span>控制线程排队</span><b id="playbackTraceEnqueueToPublish">${traceMs("enqueue_to_publish_ms")}</b></div><div class="metric"><span>DDS 写入</span><b id="playbackTraceDdsWrite">${traceMs("dds_write_ms")}</b></div><div class="metric"><span>触发关节差</span><b id="playbackTraceJointDelta">${Number(trace.q_delta_thread_trigger || 0).toFixed(4)} rad</b><small class="mini" id="playbackTraceFrame">episode frame ${Number(trace.raw_replay_frame_index ?? -1)}</small></div></div></div></div>
    <div class="card full"><div class="card-head"><div><h2>全部相机同步回放</h2><p class="mini" id="playbackFrameMeta">真机回放运行时，图像和曲线游标跟随真机实际进入控制循环的 episode frame。</p></div><span class="pill">${(p.cameras || []).length} cameras</span></div><div class="preview-grid playback-camera-grid layout-placeholder">${cameraCards || empty("当前 episode 没有相机图像；加载 episode 后这里保留相机回放占位")}</div></div>
    <div class="card full"><div class="card-head"><div><h2>夹爪状态复现对比</h2><p class="mini">蓝线为 raw replay 当前 frame 进入控制循环前读到的真机 DDS 反馈 state；橙线为该 frame 的 episode recorded state。两者比较回放状态是否复现采集时真机状态。</p></div><span class="pill">2 charts</span></div><div class="curve-quad"><div class="curve-panel"><div class="curve-title"><b>Left Gripper</b><span>raw replay</span></div><canvas id="liveLeftGripperCurveCanvas" height="180"></canvas><div class="curve-legend" id="liveLeftGripperLegend"></div></div><div class="curve-panel"><div class="curve-title"><b>Right Gripper</b><span>raw replay</span></div><canvas id="liveRightGripperCurveCanvas" height="180"></canvas><div class="curve-legend" id="liveRightGripperLegend"></div></div></div></div>
    <div class="card full"><div class="card-head"><div><h2>机械臂轨迹同步回放</h2><p class="mini" id="playbackCurveMeta">左右臂 J1-J7 来自 episode 回放数据，游标与历史图像同步。</p></div><span class="pill">2 charts</span></div><div class="curve-quad"><div class="curve-panel"><div class="curve-title"><b>Left J1-J7</b><span>playback</span></div><canvas id="playbackLeftJointCurveCanvas" height="220"></canvas><div class="curve-legend" id="playbackLeftJointLegend"></div></div><div class="curve-panel"><div class="curve-title"><b>Right J1-J7</b><span>playback</span></div><canvas id="playbackRightJointCurveCanvas" height="220"></canvas><div class="curve-legend" id="playbackRightJointLegend"></div></div></div></div>
    <div class="card full episodes-card"><div class="toolbar"><div><h2>Episodes</h2><p class="mini">来自当前 Root：${esc(effectiveRecordRoot())} · ${filteredEpisodes.length}/${(state.episodes || []).length}</p></div><div class="row episode-search"><input id="playbackEpisodeSearch" placeholder="搜索 episode / 时间 / validation" value="${esc(state.playbackEpisodeSearch || "")}" oninput="setPlaybackSearch(this.value)"><button class="secondary" onclick="refreshEpisodes().then(render)">刷新</button><button class="danger" onclick="deleteSelectedEpisodes()">删除所选</button></div></div><div class="table-wrap episode-table-wrap"><table><thead><tr><th></th><th>Name</th><th>采集时间</th><th>Validation</th><th>Frames</th><th>Duration</th><th>操作</th></tr></thead><tbody>${rows || `<tr><td colspan="7">${empty(query ? "没有匹配的 episode" : "暂无 episode")}</td></tr>`}</tbody></table></div></div>
  </div>`;
}

function renderInference() {
  const provider = state.snapshot?.provider || {};
  const inference = provider.online_inference || state.inferenceStatus?.online_inference || {};
  const debug = inference.debug || {};
  const active = provider.active_provider === "online_inference";
  const streams = state.cameraStatus?.streams || [];
  const requiredNames = ["head", "left_wrist", "right_wrist"];
  const availableNames = new Set(streams.map((stream) => String(stream.camera_name || "")));
  const missingNames = requiredNames.filter((name) => !availableNames.has(name));
  const prompt = String(localStorage.inferencePrompt || inference.prompt || "");
  const phase = String(debug.status || inference.state || "idle");
  const latestObservation = debug.last_observation || {};
  const latestAction = debug.last_action || {};
  const postActionDelay = debug.post_action_delay || {};
  const runtimeDebug = inference.runtime_debug || {};
  const latency = runtimeDebug.latency || {};
  const safety = runtimeDebug.safety || {};
  const feedback = safety.provider_feedback || {};
  const feedbackText = feedback.reason || feedback.error || (Object.keys(feedback).length ? JSON.stringify(feedback) : "-");
  const executionTrace = runtimeDebug.execution_trace || {};
  const pendingTrace = executionTrace.current || null;
  const latestTrace = executionTrace.latest || null;
  const trace = pendingTrace || latestTrace || {};
  const traceState = pendingTrace ? `pending #${pendingTrace.seq ?? "-"}` : latestTrace ? `${latestTrace.status || "completed"} #${latestTrace.seq ?? "-"}` : "waiting";
  const chunkIndex = Number(latestAction.chunk_index ?? latestAction.online_chunk_index ?? -1);
  const chunkSize = Number(latestAction.chunk_size ?? latestAction.online_chunk_size ?? 0);
  const ms = (value) => Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)} ms` : "-";
  const traceMs = (key) => ms(trace[key]);
  const inferencePreviews = requiredNames.map((name) => {
    const stream = streams.find((item) => String(item.camera_name || "") === name);
    const cameraId = Number(stream?.camera_id);
    const ready = Number.isFinite(cameraId);
    return `<div class="preview-tile"><div class="preview-title"><b>${esc(name)}</b><span>${ready ? `${stream.shared_age_ms ?? "-"}ms` : "missing"}</span></div><div class="preview"><span class="hint" data-inference-camera-hint="${ready ? cameraId : ""}">${ready ? `等待 ${esc(name)}...` : "相机未就绪"}</span>${ready ? `<img data-inference-camera="${cameraId}" data-frame-src="/camera/frame?camera_id=${cameraId}" style="display:none">` : ""}</div></div>`;
  }).join("");
  return `<div class="grid">
    <div class="card hero full"><div class="card-head"><div><div class="eyebrow">HTTP pi0.5</div><div class="headline">真机推理</div><div class="subline">当前 provider=${esc(provider.active_provider || "hold")} · ${esc(phase)}</div></div>${badge(active ? "running" : inference.state || "idle", active ? "RUNNING" : inference.state || "IDLE")}</div>
      <div class="form"><label class="full">任务描述 / prompt<input id="inferencePrompt" value="${esc(prompt)}" placeholder="例如：pick up the cube" oninput="localStorage.inferencePrompt=this.value"></label></div>
      <p class="row"><button class="danger" onclick="startInference()" ${active || missingNames.length ? "disabled" : ""}>启动真机推理</button><button class="secondary" onclick="stopInference()" ${active ? "" : "disabled"}>停止并 HOLD</button><button class="secondary" onclick="restoreXrInput()" ${active ? "disabled" : ""}>恢复 XR</button></p>
      ${missingNames.length ? `<p class="err-text">缺少推理相机：${esc(missingNames.join(", "))}</p>` : ""}
      ${inference.error ? `<p class="err-text">${esc(inference.error)}</p>` : ""}</div>
    <div class="card"><div class="card-head"><h2>会话状态</h2>${badge(phase)}</div><div class="kv"><span class="muted">阶段</span><span>${esc(phase)}</span><span class="muted">协议</span><span>${esc(debug.protocol_profile || "pi05_dual_arm_20d")}</span><span class="muted">手臂</span><span>${esc(debug.arm_side || "both")}</span><span class="muted">动作块</span><span>${chunkIndex >= 0 ? `${chunkIndex + 1}/${chunkSize || "?"}` : "-"}</span>${phase === "post_action_delay" ? `<span class="muted">延迟剩余</span><span>${Number(postActionDelay.remaining_ms ?? 0).toFixed(1)} ms</span>` : ""}</div></div>
    <div class="card"><div class="card-head"><h2>最近推理数据</h2><span class="mini">HTTP 18027</span></div><div class="kv"><span class="muted">观测序号</span><span>${esc(latestObservation.observation_seq ?? "-")}</span><span class="muted">动作块序号</span><span>${esc(latestAction.chunk_seq ?? "-")}</span><span class="muted">实际发送 prompt</span><span>${esc(latestObservation.prompt || "-")}</span><span class="muted">错误</span><span class="${inference.error ? "err-text" : ""}">${esc(inference.error || debug.error || "-")}</span></div></div>
    <div class="card full"><div class="card-head"><div><h2>推理闭环延迟</h2><p class="mini">HTTP 从 observation 发出到 action 收到；其余是同一轮主控制循环的实测耗时。</p></div><span class="pill">${esc(runtimeDebug.updated_monotonic_ns ? "live" : "waiting")}</span></div><div class="metric-grid"><div class="metric"><span>HTTP 往返</span><b>${ms(latency.http_roundtrip_ms)}</b></div><div class="metric"><span>读取输入</span><b>${ms(latency.tele_fetch_ms)}</b></div><div class="metric"><span>IK</span><b>${ms(latency.ik_ms)}</b></div><div class="metric"><span>安全限幅</span><b>${ms(latency.safety_ms)}</b></div><div class="metric"><span>重力补偿</span><b>${ms(latency.gravity_ms)}</b></div><div class="metric"><span>反馈上报</span><b>${ms(latency.provider_feedback_ms)}</b></div><div class="metric"><span>目标下发</span><b>${ms(latency.target_submit_ms)}</b><small class="mini">仅表示 ctrl_dual_arm 调用完成</small></div></div></div>
    <div class="card full"><div class="card-head"><div><h2>执行 trace</h2><p class="mini">DDS 发布后的反馈运动由低层状态线程按 q/dq 阈值检测，不表示已到达最终目标。</p></div>${badge(pendingTrace ? "running" : trace.status || "idle", traceState)}</div><div class="metric-grid"><div class="metric"><span>上传到 DDS 发布</span><b>${traceMs("online_obs_send_to_publish_ms")}</b></div><div class="metric"><span>action 到 DDS 发布</span><b>${traceMs("online_action_recv_to_publish_ms")}</b></div><div class="metric"><span>DDS 发布到线程反馈运动</span><b>${traceMs("pub_to_exec_thread_ms")}</b></div><div class="metric"><span>上传到线程反馈运动</span><b>${traceMs("online_obs_send_to_exec_thread_ms")}</b></div><div class="metric"><span>控制线程排队</span><b>${traceMs("enqueue_to_publish_ms")}</b></div><div class="metric"><span>DDS 写入</span><b>${traceMs("dds_write_ms")}</b></div><div class="metric"><span>线程触发关节差</span><b>${Number(trace.q_delta_thread_trigger || 0).toFixed(4)} rad</b><small class="mini">dq ${Number(trace.dq_peak_thread_trigger || 0).toFixed(4)} rad/s</small></div></div></div>
    <div class="card full"><div class="card-head"><div><h2>安全与下发</h2><p class="mini">出现 workspace、速度或 IK 拒绝时，控制链会反馈 provider 并保持当前姿态。</p></div>${badge(feedback.fatal ? "error" : "ok", feedback.fatal ? "REJECTED" : "CLEAR")}</div><div class="metric-grid three"><div class="metric"><span>安全反馈</span><b class="${feedback.fatal ? "err-text" : ""}">${esc(feedbackText)}</b></div><div class="metric"><span>目标关节最大差</span><b>${Number(safety.command_delta_max_abs || 0).toFixed(4)} rad</b><small class="mini">L2 ${Number(safety.command_delta_l2 || 0).toFixed(4)} rad</small></div><div class="metric"><span>目标提交</span><b>${safety.target_submitted ? "已调用" : "未提交"}</b><small class="mini">不是 DDS 执行确认</small></div></div></div>
    <div class="card full"><div class="card-head"><div><h2>双臂 TCP 3D 轨迹</h2><p class="mini">目标来自模型 action pose；反馈来自当前机械臂关节 FK。拖拽旋转，滚轮缩放。</p></div><span class="pill">目标 / 反馈</span></div><div class="inference-3d-wrap"><canvas id="inferenceDualTcpCanvas" class="inference-3d-canvas"></canvas><p id="inferenceWebglError" class="err-text hidden">浏览器不支持 WebGL，无法渲染双臂 TCP 3D 轨迹。</p></div></div>
    <div class="card full"><div class="card-head"><div><h2>双臂位置与姿态轨迹</h2><p class="mini">XYZ 与 Rot6D 都同时显示目标和反馈；旋转误差为二者相对旋转角。</p></div><span class="pill">最近 180 点</span></div><div class="curve-quad"><div class="curve-panel"><div class="curve-title"><b>Left wrist XYZ</b><span>target / feedback</span></div><canvas id="inferenceLeftWristCurveCanvas" height="220"></canvas><div class="curve-legend" id="inferenceLeftWristLegend"></div></div><div class="curve-panel"><div class="curve-title"><b>Right wrist XYZ</b><span>target / feedback</span></div><canvas id="inferenceRightWristCurveCanvas" height="220"></canvas><div class="curve-legend" id="inferenceRightWristLegend"></div></div><div class="curve-panel"><div class="curve-title"><b>Left Rot6D / rotation error</b><span>target / feedback</span></div><canvas id="inferenceLeftRot6dCanvas" height="220"></canvas><canvas id="inferenceLeftRotationErrorCanvas" height="160"></canvas><div class="curve-legend" id="inferenceLeftRot6dLegend"></div></div><div class="curve-panel"><div class="curve-title"><b>Right Rot6D / rotation error</b><span>target / feedback</span></div><canvas id="inferenceRightRot6dCanvas" height="220"></canvas><canvas id="inferenceRightRotationErrorCanvas" height="160"></canvas><div class="curve-legend" id="inferenceRightRot6dLegend"></div></div></div></div>
    <div class="card full"><div class="card-head"><div><h2>推理相机预览</h2><p class="mini">仅预览现有 latest frame，不重启或修改任何相机。</p></div><span class="pill">RGB</span></div><div class="preview-grid">${inferencePreviews}</div></div>
    <div class="card full"><div class="card-head"><div><h2>推理相机</h2><p class="mini">启动推理前必须同时具备 head、left_wrist、right_wrist 三路运行时相机。</p></div><span class="pill">${streams.length}/3 streams</span></div><div class="table-wrap"><table><thead><tr><th>名称</th><th>状态</th><th>帧年龄</th><th>实际输出</th></tr></thead><tbody>${requiredNames.map((name) => { const stream = streams.find((item) => String(item.camera_name || "") === name); const actual = stream?.actual_capture || {}; return `<tr><td><b>${esc(name)}</b></td><td>${stream ? "ready" : "missing"}</td><td>${stream ? `${stream.shared_age_ms ?? "-"}ms` : "-"}</td><td>${stream ? esc(`${actual.frame_width || "?"}x${actual.frame_height || "?"} @ ${Number(actual.measured_fps || 0).toFixed(1)}fps`) : "-"}</td></tr>`; }).join("")}</tbody></table></div></div>
  </div>`;
}

function renderExport() {
  const DEFAULT_URDF = "/home/luopengcheng/Programs/xr_teleoperate/assets/g1/g1_body29_hand14.urdf";
  const c = state.convertStatus || {};
  const cfg = state.exportConfig || {};
  const outputRoot = String(cfg.outputRoot ?? localStorage.exportOut ?? "~/data/export/");
  const datasetName = String(cfg.datasetName ?? localStorage.exportName ?? "nero_dataset");
  const urdfPath = String(cfg.urdfPath ?? localStorage.exportUrdf ?? DEFAULT_URDF);
  const task = String(cfg.task ?? localStorage.exportTask ?? "");
  const formatVersion = String(cfg.formatVersion ?? localStorage.exportFormat ?? "v2");
  const mode = String(cfg.mode ?? localStorage.exportMode ?? "new");
  const exportVideo = cfg.exportVideo !== undefined ? !!cfg.exportVideo : localStorage.exportVideo === "1";
  const exportFk = cfg.exportFk !== undefined ? !!cfg.exportFk : localStorage.exportFk === "1";
  const exportVerify = cfg.exportVerify !== undefined ? !!cfg.exportVerify : localStorage.exportVerify === "1";
  const selected = state.exportSelected instanceof Set ? state.exportSelected : new Set();
  const availableNames = new Set((state.episodes || []).map((e) => String(e.name || "")));
  const selectedCount = [...selected].filter((name) => availableNames.has(String(name))).length;
  const processedFrames = Math.max(0, Number(c.processed_frames || 0));
  const totalFrames = Math.max(0, Number(c.total_frames || 0));
  const outputTotalFrames = Math.max(0, Number(c.output_total_frames || 0));
  const appendBaseFrames = Math.max(0, Number(c.append_base_frames || 0));
  const progressText = appendBaseFrames > 0
    ? `本次 ${processedFrames}/${totalFrames} · 数据集 ${outputTotalFrames || appendBaseFrames + processedFrames}`
    : `${processedFrames}/${totalFrames}`;
  const verify = c.export_verification || {};
  const verifySamples = Array.isArray(verify.samples) ? verify.samples : [];
  const verifyRows = verifySamples.map((s) => `<tr><td>${esc(s.episode || "-")}</td><td>${Number(s.frame_index || 0)}</td><td>${s.state_match ? "OK" : "FAIL"}</td><td>${s.action_match ? "OK" : "FAIL"}</td><td>${Number(s.state_max_abs_diff || 0).toExponential(2)}</td><td><code>${esc((s.raw_state_head || []).map((x) => Number(x).toFixed(3)).join(", "))}</code></td><td><code>${esc((s.export_state_head || []).map((x) => Number(x).toFixed(3)).join(", "))}</code></td></tr>`).join("");
  const verifyErrors = Array.isArray(verify.errors) ? verify.errors : [];
  const showVerifyCard = !!exportVerify || !!c.export_verify;
  const verifyHtml = showVerifyCard ? `<div class="card full"><div class="card-head"><div><h2>导出抽样验证</h2><p class="mini">默认关闭。开启后导出完成自动抽查最多 2 帧：原始 frames.jsonl 的 q_fb 是否等于 parquet 的 observation.state/action。</p></div><span class="pill ${verify.ok === false ? "err" : verify.ok ? "ok" : ""}">${verify.ok === undefined ? "等待导出" : verify.ok ? "验证通过" : "验证失败"}</span></div>${verifyRows ? `<div class="table-wrap"><table><thead><tr><th>Episode</th><th>Frame</th><th>State</th><th>Action</th><th>MaxDiff</th><th>Raw head</th><th>Export head</th></tr></thead><tbody>${verifyRows}</tbody></table></div>` : empty("开启验证后，导出结束会自动显示抽样明细。")}${verifyErrors.length ? `<ul>${verifyErrors.map((e) => `<li class="err-text">${esc(e)}</li>`).join("")}</ul>` : ""}</div>` : "";
  const episodeLengthWarnings = Array.isArray(c.episode_length_warnings) ? c.episode_length_warnings : [];
  const episodeLengthReference = c.episode_length_reference || {};
  const episodeLengthWarningHtml = episodeLengthWarnings.length ? `<div class="card full"><div class="card-head"><div><h2>Episode 时长预警</h2><p class="mini">同一批次按帧数 IQR 检测；仅提示，不阻断导出。</p></div>${badge("warning", `${episodeLengthWarnings.length} 个异常`)}</div><p class="mini">中位数 ${Number(episodeLengthReference.median_frame_count || 0).toFixed(0)} 帧 · 异常短 &lt; ${Number(episodeLengthReference.iqr_lower_bound || 0).toFixed(1)} · 异常长 &gt; ${Number(episodeLengthReference.iqr_upper_bound || 0).toFixed(1)}</p><ul>${episodeLengthWarnings.map((item) => `<li class="warn-text">${item.kind === "short" ? "异常短" : "异常长"}：${esc(item.episode)}（${Number(item.frame_count || 0)} 帧）</li>`).join("")}</ul></div>` : "";
  const episodeChecks = (state.episodes || []).map((e) => {
    const name = String(e.name || "");
    const checked = selected.has(name) ? "checked" : "";
    const level = String(e.validation_level || "unknown");
    return `<label class="episode-option ${checked ? "selected" : ""}"><input data-ep type="checkbox" value="${esc(name)}" ${checked} onchange="toggleExportEpisode(${JSON.stringify(name)}, this.checked)"><span class="fake-check"></span><span class="episode-meta"><b>${esc(name)}</b><small>${e.frame_count || 0} frames · ${esc(level)} · ${Number(e.duration_sec || 0).toFixed(1)}s</small></span></label>`;
  }).join("");
  return `<div class="grid">
    <div class="card hero wide"><div class="card-head"><div><div class="eyebrow">LeRobot Export</div><div class="headline">导出所选 Episodes</div><div class="subline">调用 xr_teleoperate raw_to_lerobot_v2；LeRobot v2 固定生成 mp4，视频开关用于导出后逐帧校验。</div></div><span class="pill">${esc(c.phase || "idle")}</span></div>
      <div class="form"><label>output_root<input id="exportOut" value="${esc(outputRoot)}" oninput="setExportConfig('outputRoot', this.value)" onchange="setExportConfig('outputRoot', this.value)"></label><label>dataset_name<input id="exportName" value="${esc(datasetName)}" oninput="setExportConfig('datasetName', this.value)" onchange="setExportConfig('datasetName', this.value)"></label><label>urdf_path<input id="exportUrdf" value="${esc(urdfPath)}" placeholder=".../nero_dual_arm_with_gripper_generated.urdf" oninput="setExportConfig('urdfPath', this.value)" onchange="setExportConfig('urdfPath', this.value)"></label><label class="full">批量任务描述 / task<input id="exportTask" value="${esc(task)}" placeholder="例如：pick red block to tray" oninput="setExportConfig('task', this.value)" onchange="setExportConfig('task', this.value)"></label><label>format<select id="exportFormat" onchange="setExportConfig('formatVersion', this.value)"><option value="v2" ${formatVersion === "v2" ? "selected" : ""}>LeRobot v2</option><option value="v3" ${formatVersion === "v3" ? "selected" : ""}>LeRobot v3</option><option value="umi_zarr" ${formatVersion === "umi_zarr" ? "selected" : ""}>UMI zarr.zip</option></select></label><label>mode<select id="exportMode" onchange="setExportConfig('mode', this.value)"><option value="new" ${mode === "new" ? "selected" : ""}>新建</option><option value="append" ${mode === "append" ? "selected" : ""}>追加到已有数据集</option><option value="replace" ${mode === "replace" ? "selected" : ""}>安全替换</option></select></label><label class="row"><input id="exportVideo" type="checkbox" style="width:auto" ${exportVideo ? "checked" : ""} onchange="setExportConfig('exportVideo', this.checked)">校验 mp4 帧数</label><label class="row"><input id="exportFk" type="checkbox" style="width:auto" ${exportFk ? "checked" : ""} onchange="setExportConfig('exportFk', this.checked)">导出 FK(xyz+rpy)</label><label class="row"><input id="exportVerify" type="checkbox" style="width:auto" ${exportVerify ? "checked" : ""} onchange="setExportConfig('exportVerify', this.checked)">导出后抽样验证</label></div>
      <p class="row"><button id="exportStartBtn" onclick="startExport()">导出所选 ${selectedCount ? `(${selectedCount})` : ""}</button><span class="mini">当前实现支持 LeRobot v2 的 raw episode 导出；append、v3、UMI zarr、自定义 URDF FK 会明确报错，不做静默降级。</span></p></div>
    <div class="card"><div class="card-head"><h2>进度</h2><span class="mini">${esc(progressText)}</span></div>${progress(processedFrames, totalFrames)}<div class="kv" style="margin-top:14px"><span class="muted">状态</span><span>${esc(c.phase || "idle")}</span><span class="muted">消息</span><span>${esc(c.message || "-")}</span>${c.error ? `<span class="muted">错误</span><span class="err-text">${esc(c.error)}</span>` : ""}</div></div>
    ${episodeLengthWarningHtml}
    ${verifyHtml}
    <div class="card full"><div class="toolbar"><div><h2>选择 Episode</h2><p class="mini">支持批量范围，例如 <code>1-50, 50-100</code>；按 episode 编号匹配 episode_000001 这种名称。</p></div><div class="row"><span class="pill" id="exportSelectedCount">${selectedCount} selected</span><button class="secondary" onclick="refreshEpisodes().then(render)">刷新</button></div></div>
      <div class="selection-tools"><label class="range-input">批量范围<input id="exportRange" value="${esc(state.exportRangeInput || "")}" oninput="setExportRangeInput(this.value)" placeholder="1-50, 50-100 或 episode_000001-episode_000050"></label><button onclick="selectExportRange('replace')">按范围选择</button><button class="secondary" onclick="selectAllExportEpisodes()">全选</button><button class="secondary" onclick="invertExportSelection()">反选</button><button class="secondary" onclick="clearExportSelection()">清空</button></div>
      <p class="mini">提示：范围是闭区间；如果你的 episode 从 0 开始，请输入 0-49。选择状态会在导出页刷新时保留。</p>
      <div class="episode-list">${episodeChecks || empty("请先刷新 Episodes；如果列表为空，请检查 source_root 是否正确")}</div></div>
  </div>`;
}


const tabs = [
  ["record", "录制", "实时预览 / Episode", "●"],
  ["inference", "推理", "HTTP pi0.5 真机执行", "⌁"],
  ["export", "导出", "LeRobot 可选", "↗"],
  ["playback", "回放", "本地 episode 检查", "▶"],
];
const titles = {
  record: ["遥操录制工作台", "网页只发控制意图；相机、录制和对齐仍由当前 teleop 主循环负责。"],
  playback: ["回放检查", "网页只做本地 episode 图片和曲线检查；不会向真机下发回放动作。"],
  inference: ["真机推理", "HTTP pi0.5 动作经现有安全链和 DDS 下发；页面不直接控制真机。"],
  export: ["LeRobot 导出", "UI 已接入 raw_to_lerobot_v2；选择 episode 后导出为 LeRobot v2 数据集。"],
};
const root = document.getElementById("app");

root.innerHTML = `<div class="app"><aside class="side"><div class="brand"><div class="brand-mark">XR</div><div><div class="brand-title">xr_teleoperate</div><div class="brand-sub">UI control bridge</div></div></div><div class="side-status" id="sideStatus"></div><div class="side-root"><label>Record Root<input id="globalRecordRoot" value="${localStorage.recordRoot || "~/data/record/"}" onkeydown="if(event.key==='Enter'){loadGlobalRecordRoot()}" onchange="loadGlobalRecordRoot()"></label><p class="row"><button class="secondary" onclick="loadGlobalRecordRoot()">切换 Root</button></p><p class="mini">输入包含 episode_xxxx 的目标目录；切换仅允许在录制、自检和真机回放均停止时执行。</p></div><div class="nav" id="nav"></div><div class="side-foot">Reference collector layout · xr runtime</div></aside><main class="main"><div class="top"><div><h1 id="topTitle">遥操录制工作台</h1><p id="topSub">connecting...</p></div><div class="top-actions"><span class="pill"><span id="connDot" class="dot"></span><span id="connText">connecting</span></span><span class="pill"><span id="recDot" class="dot"></span><span id="recText">record idle</span></span><button class="secondary" id="refreshBtn">刷新</button></div></div><div id="alert" class="banner"></div><section id="content"></section></main></div>`;

function el(id) { return document.getElementById(id); }
function input(id) { return (document.getElementById(id))?.value || ""; }
function checked(id) { return !!(document.getElementById(id))?.checked; }
function showBanner(message, level= "warning") {
  const box = el("alert");
  box.className = `banner show ${level}`;
  box.textContent = message;
}

function renderNav() {
  el("nav").innerHTML = tabs.map(([id, name, desc, ico]) => `<button class="${state.active === id ? "active" : ""}" data-tab="${id}"><span class="nav-ico">${ico}</span><span><span class="nav-title">${name}</span><span class="nav-desc">${desc}</span></span></button>`).join("");
  el("nav").querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => setTab((b).dataset.tab)));
}

function renderSideStatus() {
  const r = state.snapshot?.recording || {};
  const provider = state.snapshot?.provider || {};
  const cams = state.cameraStatus?.active_camera_ids || [];
  const val = r.last_validation?.level || "unknown";
  el("sideStatus").innerHTML = `<div class="side-card"><div class="eyebrow">Status</div><div class="side-kv"><div class="side-row"><span>连接</span><b>${el("connText")?.textContent || "-"}</b></div><div class="side-row"><span>录制</span><b>${r.active ? "ON" : "OFF"}</b></div><div class="side-row"><span>输入</span><b>${esc(provider.active_provider || "-")}</b></div><div class="side-row"><span>帧</span><b>${r.frame_index || 0}</b></div><div class="side-row"><span>相机</span><b>${cams.length}</b></div><div class="side-row"><span>自检</span><b>${val}</b></div></div></div>`;
}

function renderHeader() {
  const [title, sub] = titles[state.active] || titles.record;
  el("topTitle").textContent = title;
  el("topSub").textContent = sub;
}

function render() {
  if (state.active === "export") {
    syncExportConfigFromDom();
    pruneExportSelectionToAvailable();
  }
  renderHeader();
  renderNav();
  const nowMs = performance.now();
  if (nowMs - Number(state.lastSideStatusMs || 0) > 500) {
    state.lastSideStatusMs = nowMs;
    renderSideStatus();
  }
  const content = el("content");
  content.innerHTML = state.active === "record" ? renderRecord()
    : state.active === "playback" ? renderPlayback()
    : state.active === "inference" ? renderInference()
    : renderExport();
  if (state.active === "inference") state.lastInferenceRenderMs = performance.now();
  updatePreviewLoop();
  renderCurveLegends();
  scheduleCurveDraw(true);
  syncPlaybackPanel(displayedPlayback());
  managePlaybackPolling();
}

function setTab(id) {
  if (state.active === "export") syncExportConfigFromDom();
  if (state.snapshot?.provider?.active_provider === "online_inference" && id !== "inference") {
    showBanner("推理运行中，请先停止并 HOLD", "error");
    return;
  }
  const recording = state.snapshot?.recording || {};
  const alignment = recording.last_alignment || {};
  if (id === "playback" && (recording.active || recording.phase === "armed" || alignment.waiting_for_first_frame)) {
    showBanner("录制 active/armed 时不能切到回放 HOLD；请先停止保存或取消录制。", "error");
    return;
  }
  state.active = id;
  render();
  if (id === "playback") {
    api("/ui/provider/hold")
      .then(() => refreshAll())
      .catch((e) => showBanner(`切换 HOLD 失败：${e?.message || e}`, "error"));
  } else if (id === "record" && state.snapshot?.provider?.active_provider !== "online_inference") {
    api("/ui/provider/xr")
      .then(() => refreshAll())
      .catch((e) => showBanner(`恢复 XR 输入失败：${e?.message || e}`, "error"));
  }
}

function applySnapshot(payload) {
  const prev = state.snapshot || {};
  const incoming = payload || {};
  const incomingRecording = incoming.recording || {};
  const incomingValidation = incomingRecording.last_validation || {};
  const prevValidation = prev.recording?.last_validation || {};
  if (
    incomingRecording &&
    !incomingRecording.active &&
    prevValidation.checked_at_ns &&
    !incomingValidation.checked_at_ns
  ) {
    incoming.recording = { ...incomingRecording, last_validation: prevValidation };
  }
  const incomingPlayback = incoming.playback || {};
  const prevPlayback = prev.playback || {};
  if (
    prevPlayback.enabled &&
    prevPlayback.episode_name &&
    (!incomingPlayback.enabled || incomingPlayback.state === "disabled")
  ) {
    incoming.playback = prevPlayback;
  }
  state.snapshot = incoming;
  if (state.realReplayCommandPending && realReplayCommandResolved(incoming.provider || {}, state.realReplayCommandPending)) {
    state.realReplayCommandPending = "";
  }
  const r = state.snapshot.recording || {};
  const prevR = prev.recording || {};
  pushLiveCurveSample(state.snapshot);
  pushInferenceTrajectorySample(state.snapshot);
  el("connDot").className = "dot ok";
  el("connText").textContent = "connected";
  el("recDot").className = "dot " + (r.active ? "ok" : "");
  el("recText").textContent = r.active ? `recording #${r.frame_index || 0}` : "record idle";
  renderSideStatus();
  const alert = r.last_alert || {};
  if (alert.seq && alert.seq !== state.lastAlertSeq) {
    state.lastAlertSeq = alert.seq;
    showBanner(`${alert.level || "warning"} · ${alert.code || ""} · ${alert.message || ""}`, alert.level === "error" ? "error" : "warning");
  }
  const changedRecording = !!r.active !== !!prevR.active;
  const changedValidation = (r.last_validation?.checked_at_ns || 0) !== (prevR.last_validation?.checked_at_ns || 0);
  if (prevR.active && !r.active) window.setTimeout(() => refreshAll(), 1200);
  const teleop = state.snapshot.teleop || {};
  const prevTeleop = prev.teleop || {};
  const changedTeleop = (
    !!teleop.started !== !!prevTeleop.started ||
    !!teleop.ready !== !!prevTeleop.ready ||
    !!teleop.stopping !== !!prevTeleop.stopping
  );
  const replay = state.snapshot.provider?.real_replay || {};
  const prevReplay = prev.provider?.real_replay || {};
  const replayTrace = replay.runtime_debug?.execution_trace || {};
  const prevReplayTrace = prevReplay.runtime_debug?.execution_trace || {};
  const changedPlaybackProviderState = (
    state.snapshot.provider?.active_provider !== prev.provider?.active_provider ||
    replay.state !== prevReplay.state ||
    replay.error !== prevReplay.error ||
    replay.reason !== prevReplay.reason
  );
  const changedPlaybackRuntime = (
    changedPlaybackProviderState ||
    replay.frame_index !== prevReplay.frame_index ||
    replayTrace.current?.seq !== prevReplayTrace.current?.seq ||
    replayTrace.latest?.seq !== prevReplayTrace.latest?.seq ||
    replay.runtime_debug?.gripper_state_current?.frame_index !== prevReplay.runtime_debug?.gripper_state_current?.frame_index
  );
  const inference = state.snapshot.provider?.online_inference || {};
  const prevInference = prev.provider?.online_inference || {};
  const inferenceDebug = inference.debug || {};
  const prevInferenceDebug = prevInference.debug || {};
  const inferenceRuntimeDebug = inference.runtime_debug || {};
  const prevInferenceRuntimeDebug = prevInference.runtime_debug || {};
  const inferenceAction = inferenceDebug.last_action || {};
  const prevInferenceAction = prevInferenceDebug.last_action || {};
  const inferenceObservation = inferenceDebug.last_observation || {};
  const prevInferenceObservation = prevInferenceDebug.last_observation || {};
  const changedInference = (
    state.snapshot.provider?.active_provider !== prev.provider?.active_provider ||
    inference.state !== prevInference.state ||
    inference.error !== prevInference.error ||
    inferenceDebug.status !== prevInferenceDebug.status ||
    inferenceObservation.observation_seq !== prevInferenceObservation.observation_seq ||
    inferenceAction.chunk_seq !== prevInferenceAction.chunk_seq ||
    inferenceAction.chunk_index !== prevInferenceAction.chunk_index ||
    inferenceAction.chunk_size !== prevInferenceAction.chunk_size
  );
  const changedInferenceRuntime = inferenceRuntimeDebug.updated_monotonic_ns !== prevInferenceRuntimeDebug.updated_monotonic_ns;
  const shouldRenderInference = state.active === "inference" && (
    changedInference || (changedInferenceRuntime && performance.now() - Number(state.lastInferenceRenderMs || 0) >= 150)
  );
  const shouldRenderPlayback = state.active === "playback" && changedPlaybackProviderState;
  if ((changedRecording || changedValidation || changedTeleop || shouldRenderInference || shouldRenderPlayback) && document.activeElement?.tagName !== "INPUT") render();
  if (state.active === "playback") {
    syncPlaybackPanel(displayedPlayback());
    syncPlaybackTrace();
  }
  scheduleCurveDraw(false);
}

function connectSse() {
  const es = new EventSource("/events");
  es.onopen = () => { el("connDot").className = "dot ok"; el("connText").textContent = "connected"; };
  es.onerror = () => {
    el("connDot").className = "dot err";
    el("connText").textContent = state.shuttingDown ? "exited" : "offline";
    renderSideStatus();
  };
  es.onmessage = (e) => { try { applySnapshot(JSON.parse(e.data)); } catch (_) {} };
}

async function refreshEpisodes() {
  const rootDir = effectiveRecordRoot();
  const box = document.getElementById("globalRecordRoot");
  if (box && box.value !== rootDir) box.value = rootDir;
  try {
    state.episodes = (await api("/recording/episodes?" + qs({ root_dir: rootDir, limit: 0 }))).episodes || [];
  } catch (e) {
    if (expectedShutdownFetchError(e) || connectionIsOffline()) return;
    showBanner(`Episode 列表刷新失败：${e?.message || e}`, "error");
    state.episodes = [];
  }
  pruneExportSelectionToAvailable();
}

function loadGlobalRecordRoot() {
  const node = document.getElementById("globalRecordRoot");
  setGlobalRecordRoot(node?.value || "");
}

async function setGlobalRecordRoot(value) {
  const rootDir = String(value || "").trim() || "~/data/record/";
  try {
    await api("/recording/set_root_dir?" + qs({ root_dir: rootDir }));
    localStorage.recordRoot = rootDir;
    const box = document.getElementById("globalRecordRoot");
    if (box && box.value !== rootDir) box.value = rootDir;
    showBanner(`已请求切换采集 Root：${rootDir}；控制循环完成切换后可开始录制。`, "warning");
  } catch (e) {
    showBanner(`切换采集 Root 失败：${e?.message || e}`, "error");
    return;
  }
  await refreshEpisodes();
  if (state.active !== "record" || !state.snapshot?.recording?.active) render();
}

async function refreshAll() {
  if (state.shuttingDown) return;
  try { await refreshCameraStatusOnly(false); } catch (_) {}
  try { state.realsenseDevices = (await api("/camera/realsense_list")).devices || []; } catch (_) { state.realsenseDevices = []; }
  const recordingStatus = await api("/recording/status");
  state.snapshot = { ...(state.snapshot || {}), recording: recordingStatus };
  await refreshEpisodes();
  try { state.convertStatus = await api("/convert/status", { allowApplicationError: true }); } catch (_) {}
  state.inferenceStatus = await api("/inference/status");
  render();
}

async function refreshCameraStatusOnly(shouldRender = true) {
  state.cameraStatus = enrichCameraStatus(await api("/camera/status"));
  state.cameraStatusError = "";
  if (shouldRender && state.active === "record" && document.activeElement?.tagName !== "INPUT") render();
}

function pushLiveCurveSample(snapshot) {
  const left = snapshot?.left?.q_fb;
  const right = snapshot?.right?.q_fb;
  if (!Array.isArray(left) && !Array.isArray(right)) return;
  const hist = Array.isArray(state.liveCurveHistory) ? state.liveCurveHistory : [];
  const last = hist[hist.length - 1];
  const now = performance.now() / 1000;
  if (last && now - Number(last.t || 0) < 0.12) return;
  hist.push({
    t: now,
    left: Array.isArray(left) ? left.slice(0, 8).map(Number) : null,
    right: Array.isArray(right) ? right.slice(0, 8).map(Number) : null,
    leftGripperFeedback: finiteOrNull(snapshot?.left?.gripper_q_fb),
    rightGripperFeedback: finiteOrNull(snapshot?.right?.gripper_q_fb),
    leftGripperCommand: finiteOrNull(snapshot?.left?.gripper_q_cmd),
    rightGripperCommand: finiteOrNull(snapshot?.right?.gripper_q_cmd),
  });
  while (hist.length > 240) hist.shift();
  state.liveCurveHistory = hist;
}

function finiteOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function pushInferenceTrajectorySample(snapshot) {
  const runtimeDebug = snapshot?.provider?.online_inference?.runtime_debug || {};
  const trajectory = runtimeDebug.trajectory || {};
  const sampleNs = Number(trajectory.sample_monotonic_ns || runtimeDebug.updated_monotonic_ns || 0);
  if (!Number.isFinite(sampleNs) || sampleNs <= 0 || sampleNs === state.lastInferenceTrajectorySampleNs) return;
  const poseKeys = ["left_target", "left_feedback", "right_target", "right_feedback"];
  if (!poseKeys.every((key) => {
    const pose = trajectory[key] || {};
    return Array.isArray(pose.xyz) && pose.xyz.length === 3 && pose.xyz.every((value) => Number.isFinite(Number(value)))
      && Array.isArray(pose.rot6d) && pose.rot6d.length === 6 && pose.rot6d.every((value) => Number.isFinite(Number(value)));
  })) return;
  const hist = Array.isArray(state.inferenceTrajectoryHistory) ? state.inferenceTrajectoryHistory : [];
  hist.push({
    t: sampleNs,
    leftTarget: trajectory.left_target.xyz.map(Number),
    leftFeedback: trajectory.left_feedback.xyz.map(Number),
    rightTarget: trajectory.right_target.xyz.map(Number),
    rightFeedback: trajectory.right_feedback.xyz.map(Number),
    leftTargetRot6d: trajectory.left_target.rot6d.map(Number),
    leftFeedbackRot6d: trajectory.left_feedback.rot6d.map(Number),
    rightTargetRot6d: trajectory.right_target.rot6d.map(Number),
    rightFeedbackRot6d: trajectory.right_feedback.rot6d.map(Number),
  });
  while (hist.length > 180) hist.shift();
  state.inferenceTrajectoryHistory = hist;
  state.lastInferenceTrajectorySampleNs = sampleNs;
}

const JOINT_NAMES = ["j1","j2","j3","j4","j5","j6","j7"];
const JOINT_KEYS = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"];
const JOINT_COLORS = ["#007aff","#5856d6","#34c759","#ff9f0a","#ff3b30","#00c7be","#af52de"];
const GRIPPER_COLORS = { left: "#007aff", right: "#ff3b30", feedback: "#007aff", command: "#ff9f0a" };
const INFERENCE_AXIS_COLORS = ["#007aff", "#34c759", "#ff9f0a"];
const INFERENCE_FEEDBACK_COLORS = ["#6baeff", "#70d78b", "#ffc163"];
const ROT6D_COLORS = ["#ff3b30", "#ff9f0a", "#ffd60a", "#34c759", "#007aff", "#af52de"];
const DEFAULT_URDF = "/home/luopengcheng/Programs/xr_teleoperate/assets/g1_d/g1_d.urdf";

function readExportConfig() {
  const cfg = state.exportConfig || {};
  return {
    outputRoot: String(cfg.outputRoot ?? localStorage.exportOut ?? "~/data/export/"),
    datasetName: String(cfg.datasetName ?? localStorage.exportName ?? "nero_dataset"),
    urdfPath: String(cfg.urdfPath ?? localStorage.exportUrdf ?? DEFAULT_URDF),
    task: String(cfg.task ?? localStorage.exportTask ?? ""),
    formatVersion: String(cfg.formatVersion ?? localStorage.exportFormat ?? "v2"),
    mode: String(cfg.mode ?? localStorage.exportMode ?? "new"),
    exportVideo: cfg.exportVideo !== undefined ? !!cfg.exportVideo : localStorage.exportVideo === "1",
    exportFk: cfg.exportFk !== undefined ? !!cfg.exportFk : localStorage.exportFk === "1",
    exportVerify: cfg.exportVerify !== undefined ? !!cfg.exportVerify : localStorage.exportVerify === "1",
  };
}

function setExportConfig(key, value) {
  if (!state.exportConfig) state.exportConfig = {};
  const normalized = (key === "exportVideo" || key === "exportFk" || key === "exportVerify") ? !!value : String(value ?? "");
  state.exportConfig[key] = normalized;
  const storageKey = {
    outputRoot: "exportOut",
    datasetName: "exportName",
    urdfPath: "exportUrdf",
    task: "exportTask",
    formatVersion: "exportFormat",
    mode: "exportMode",
    exportVideo: "exportVideo",
    exportFk: "exportFk",
    exportVerify: "exportVerify",
  };
  const k = storageKey[key];
  if (k) localStorage.setItem(k, (key === "exportVideo" || key === "exportFk" || key === "exportVerify") ? (normalized ? "1" : "0") : String(normalized));
}

function syncExportConfigFromDom() {
  const textFields = [
    ["exportOut", "outputRoot"],
    ["exportName", "datasetName"],
    ["exportUrdf", "urdfPath"],
    ["exportTask", "task"],
    ["exportFormat", "formatVersion"],
    ["exportMode", "mode"],
  ];
  for (const [id, key] of textFields) {
    const node = document.getElementById(id);
    if (node) setExportConfig(key, node.value);
  }
  const video = document.getElementById("exportVideo");
  if (video) setExportConfig("exportVideo", video.checked);
  const fk = document.getElementById("exportFk");
  if (fk) setExportConfig("exportFk", fk.checked);
  const verify = document.getElementById("exportVerify");
  if (verify) setExportConfig("exportVerify", verify.checked);
}

function renderCurveLegends() {
  const jointHtml = JOINT_NAMES.map((n, i) => `<span><i style="background:${JOINT_COLORS[i]}"></i>${n}</span>`).join("");
  const liveGripHtml = `<span><i style="background:${GRIPPER_COLORS.feedback}"></i>当前反馈 q</span><span><i style="background:${GRIPPER_COLORS.command}"></i>已下发目标 q</span>`;
  const replayGripHtml = `<span><i style="background:${GRIPPER_COLORS.feedback}"></i>当前真机反馈 state</span><span><i style="background:${GRIPPER_COLORS.command}"></i>episode recorded state</span>`;
  const leftGripHtml = `<span><i style="background:${GRIPPER_COLORS.left}"></i>left gripper</span>`;
  const rightGripHtml = `<span><i style="background:${GRIPPER_COLORS.right}"></i>right gripper</span>`;
  const wristHtml = ["X", "Y", "Z"].map((axis, index) => `<span><i style="background:${INFERENCE_AXIS_COLORS[index]}"></i>${axis} target</span><span><i style="background:${INFERENCE_FEEDBACK_COLORS[index]}"></i>${axis} feedback</span>`).join("");
  const rot6dHtml = ROT6D_COLORS.map((color, index) => `<span><i style="background:${color}"></i>r${index + 1}</span>`).join("") + `<span><i style="background:#111827"></i>rotation error</span>`;
  const items = [
    ["liveLeftJointLegend", jointHtml], ["liveRightJointLegend", jointHtml],
    ["playbackLeftJointLegend", jointHtml], ["playbackRightJointLegend", jointHtml],
    ["liveLeftGripperLegend", state.active === "playback" ? replayGripHtml : liveGripHtml], ["playbackLeftGripperLegend", leftGripHtml],
    ["liveRightGripperLegend", state.active === "playback" ? replayGripHtml : liveGripHtml], ["playbackRightGripperLegend", rightGripHtml],
    ["inferenceLeftWristLegend", wristHtml], ["inferenceRightWristLegend", wristHtml],
    ["inferenceLeftRot6dLegend", rot6dHtml], ["inferenceRightRot6dLegend", rot6dHtml],
  ];
  for (const [id, html] of items) {
    const node = document.getElementById(id);
    if (node && node.innerHTML !== html) node.innerHTML = html;
  }
}

function scheduleCurveDraw(force = false) {
  const now = performance.now();
  if (!force && now - Number(state.lastCurveDrawMs || 0) < 120) return;
  if (state.drawRaf) return;
  state.drawRaf = requestAnimationFrame(() => {
    state.drawRaf = 0;
    state.lastCurveDrawMs = performance.now();
    if (state.active === "record") drawLiveCurvesDual();
    if (state.active === "playback") drawPlaybackCurves(displayedPlayback());
    if (state.active === "inference") drawInferenceTrajectoryCurves();
    if (state.active === "inference") drawInferenceDualTcp3D();
  });
}

function canvasMetrics(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const parent = canvas.parentElement;
  const rectWidth = Math.floor(parent?.getBoundingClientRect().width || canvas.getBoundingClientRect().width || 680);
  const parentWidth = Math.max(0, rectWidth - 4);
  const cssW = Math.min(Math.max(parentWidth, 320), 860);
  let cssH = Number(canvas.dataset.cssHeight || 0);
  if (!Number.isFinite(cssH) || cssH <= 0) {
    const declared = Number(canvas.getAttribute("height") || 0);
    cssH = Number.isFinite(declared) && declared > 0 && declared <= 320
      ? declared
      : canvas.id.toLowerCase().includes("gripper") ? 180 : 220;
    canvas.dataset.cssHeight = String(cssH);
  }
  canvas.style.width = "100%";
  canvas.style.maxWidth = "100%";
  canvas.style.height = `${cssH}px`;
  const w = Math.floor(cssW * dpr);
  const h = Math.floor(cssH * dpr);
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  return { dpr, w, h };
}

function drawAxes(ctx, w, h, dpr, title) {
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  const padL = 36 * dpr, padR = 10 * dpr, padT = 18 * dpr, padB = 24 * dpr;
  ctx.strokeStyle = "rgba(60,60,67,.14)";
  ctx.lineWidth = 1 * dpr;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (h - padT - padB) * i / 4;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
  }
  ctx.fillStyle = "rgba(29,29,31,.74)";
  ctx.font = `${11 * dpr}px sans-serif`;
  ctx.fillText(title, padL, 12 * dpr);
  return { padL, padR, padT, padB };
}

function drawSeriesCanvas(id, title, series, cursorRatio= null) {
  const canvas = document.getElementById(id);
  if (!canvas) return;
  const { dpr, w, h } = canvasMetrics(canvas);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { padL, padR, padT, padB } = drawAxes(ctx, w, h, dpr, title);
  const vals = series.flatMap(s => s.values).filter((v) => typeof v === "number" && Number.isFinite(v));
  if (series.length === 0 || vals.length === 0 || Math.max(...series.map(s => s.values.length)) < 2) {
    ctx.fillStyle = "rgba(110,110,115,.90)";
    ctx.fillText("等待曲线数据...", padL, h / 2);
    return;
  }
  let min = Math.min(...vals), max = Math.max(...vals);
  if (!Number.isFinite(min) || !Number.isFinite(max) || Math.abs(max-min) < 1e-6) { min = -1; max = 1; }
  const span = max - min; min -= span * .08; max += span * .08;
  const n = Math.max(...series.map(s => s.values.length));
  const xAt = (i) => padL + (w - padL - padR) * i / Math.max(1, n - 1);
  const yAt = (v) => padT + (h - padT - padB) * (1 - (v - min) / (max - min));
  series.forEach((s) => {
    ctx.strokeStyle = s.color;
    ctx.globalAlpha = 0.84;
    ctx.lineWidth = 1.25 * dpr;
    ctx.setLineDash([]);
    ctx.beginPath(); let started = false;
    s.values.forEach((v, i) => {
      if (typeof v !== "number" || !Number.isFinite(v)) { started = false; return; }
      const x = xAt(i), y = yAt(v);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    if (started) ctx.stroke();
  });
  ctx.setLineDash([]); ctx.globalAlpha = 1;
  if (cursorRatio !== null && Number.isFinite(cursorRatio)) {
    const cx = padL + (w - padL - padR) * Math.max(0, Math.min(1, cursorRatio));
    ctx.strokeStyle = "rgba(255,59,48,.72)";
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(cx,padT); ctx.lineTo(cx,h-padB); ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.fillStyle = "rgba(110,110,115,.95)";
  ctx.fillText(max.toFixed(2), 2*dpr, padT + 4*dpr);
  ctx.fillText(min.toFixed(2), 2*dpr, h - padB);
}

function drawLiveCurvesDual() {
  const hist = state.liveCurveHistory || [];
  const leftJointSeries = JOINT_NAMES.map((name, j) => ({ name, color:JOINT_COLORS[j], values:hist.map((x) =>x?.left?.[j] ?? null) }));
  const rightJointSeries = JOINT_NAMES.map((name, j) => ({ name, color:JOINT_COLORS[j], values:hist.map((x) =>x?.right?.[j] ?? null) }));
  drawSeriesCanvas("liveLeftJointCurveCanvas", "Left J1-J7 feedback", leftJointSeries);
  drawSeriesCanvas("liveRightJointCurveCanvas", "Right J1-J7 feedback", rightJointSeries);
  drawSeriesCanvas("liveLeftGripperCurveCanvas", "Left gripper q", [
    { name:"当前反馈 q", color:GRIPPER_COLORS.feedback, values:hist.map((x) =>x?.leftGripperFeedback ?? null) },
    { name:"已下发目标 q", color:GRIPPER_COLORS.command, values:hist.map((x) =>x?.leftGripperCommand ?? null) },
  ]);
  drawSeriesCanvas("liveRightGripperCurveCanvas", "Right gripper q", [
    { name:"当前反馈 q", color:GRIPPER_COLORS.feedback, values:hist.map((x) =>x?.rightGripperFeedback ?? null) },
    { name:"已下发目标 q", color:GRIPPER_COLORS.command, values:hist.map((x) =>x?.rightGripperCommand ?? null) },
  ]);
}

function drawInferenceTrajectoryCurves() {
  const hist = state.inferenceTrajectoryHistory || [];
  const buildSeries = (targetKey, feedbackKey) => [0, 1, 2].flatMap((index) => [
    { name: `XYZ`.charAt(index) + " target", color: INFERENCE_AXIS_COLORS[index], values: hist.map((sample) => sample[targetKey]?.[index] ?? null) },
    { name: `XYZ`.charAt(index) + " feedback", color: INFERENCE_FEEDBACK_COLORS[index], values: hist.map((sample) => sample[feedbackKey]?.[index] ?? null) },
  ]);
  drawSeriesCanvas("inferenceLeftWristCurveCanvas", "Left wrist position (m)", buildSeries("leftTarget", "leftFeedback"));
  drawSeriesCanvas("inferenceRightWristCurveCanvas", "Right wrist position (m)", buildSeries("rightTarget", "rightFeedback"));
  const buildRot6d = (targetKey, feedbackKey) => ROT6D_COLORS.map((color, index) => ({
    name: `r${index + 1}`,
    color,
    values: hist.map((sample) => sample[targetKey]?.[index] ?? null),
  })).concat(ROT6D_COLORS.map((color, index) => ({
    name: `r${index + 1} feedback`,
    color: `${color}88`,
    values: hist.map((sample) => sample[feedbackKey]?.[index] ?? null),
  })));
  const rotationErrors = (targetKey, feedbackKey) => hist.map((sample) => rotationErrorDegrees(sample[targetKey], sample[feedbackKey]));
  drawSeriesCanvas("inferenceLeftRot6dCanvas", "Left Rot6D target / feedback", buildRot6d("leftTargetRot6d", "leftFeedbackRot6d"));
  drawSeriesCanvas("inferenceRightRot6dCanvas", "Right Rot6D target / feedback", buildRot6d("rightTargetRot6d", "rightFeedbackRot6d"));
  drawSeriesCanvas("inferenceLeftRotationErrorCanvas", "Left target-feedback rotation error (deg)", [{ name: "rotation error", color: "#111827", values: rotationErrors("leftTargetRot6d", "leftFeedbackRot6d") }]);
  drawSeriesCanvas("inferenceRightRotationErrorCanvas", "Right target-feedback rotation error (deg)", [{ name: "rotation error", color: "#111827", values: rotationErrors("rightTargetRot6d", "rightFeedbackRot6d") }]);
}

function normalizeVector3(values) {
  if (!Array.isArray(values) || values.length !== 3) return null;
  const vector = values.map(Number);
  const norm = Math.hypot(...vector);
  if (!Number.isFinite(norm) || norm < 1e-8) return null;
  return vector.map((value) => value / norm);
}

function crossVector3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function rot6dBasis(rot6d) {
  if (!Array.isArray(rot6d) || rot6d.length !== 6) return null;
  const c1 = normalizeVector3(rot6d.slice(0, 3));
  if (!c1) return null;
  const c2Raw = rot6d.slice(3, 6).map(Number);
  const projection = c1.reduce((sum, value, index) => sum + value * c2Raw[index], 0);
  const c2 = normalizeVector3(c2Raw.map((value, index) => value - c1[index] * projection));
  if (!c2) return null;
  const c3 = normalizeVector3(crossVector3(c1, c2));
  return c3 ? [c1, c2, c3] : null;
}

function rotationErrorDegrees(targetRot6d, feedbackRot6d) {
  const target = rot6dBasis(targetRot6d);
  const feedback = rot6dBasis(feedbackRot6d);
  if (!target || !feedback) return null;
  let trace = 0;
  for (let row = 0; row < 3; row += 1) {
    for (let col = 0; col < 3; col += 1) trace += target[col][row] * feedback[col][row];
  }
  const cosine = Math.max(-1, Math.min(1, (trace - 1) / 2));
  return Math.acos(cosine) * 180 / Math.PI;
}

function attachInferenceThreeControls(canvas, threeState) {
  if (threeState.controlsAttached) return;
  threeState.controlsAttached = true;
  canvas.addEventListener("pointerdown", (event) => {
    threeState.dragging = true;
    threeState.lastX = event.clientX;
    threeState.lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointerup", (event) => {
    threeState.dragging = false;
    canvas.releasePointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointercancel", () => { threeState.dragging = false; });
  canvas.addEventListener("pointermove", (event) => {
    if (!threeState.dragging) return;
    threeState.yaw += (event.clientX - threeState.lastX) * 0.008;
    threeState.pitch = Math.max(-1.35, Math.min(1.15, threeState.pitch + (event.clientY - threeState.lastY) * 0.008));
    threeState.lastX = event.clientX;
    threeState.lastY = event.clientY;
    drawInferenceDualTcp3D();
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    threeState.zoom = Math.max(0.45, Math.min(2.6, threeState.zoom * (event.deltaY > 0 ? 1.1 : 0.9)));
    drawInferenceDualTcp3D();
  }, { passive: false });
}

function disposeInferenceThreeRoot(root) {
  while (root.children.length) {
    const child = root.children.pop();
    child.traverse((item) => {
      item.geometry?.dispose?.();
      if (Array.isArray(item.material)) item.material.forEach((material) => material.dispose?.());
      else item.material?.dispose?.();
    });
  }
}

function drawInferenceDualTcp3D() {
  const canvas = document.getElementById("inferenceDualTcpCanvas");
  const error = document.getElementById("inferenceWebglError");
  if (!canvas) return;
  const probe = document.createElement("canvas").getContext("webgl2") || document.createElement("canvas").getContext("webgl");
  if (!probe) {
    if (error) error.classList.remove("hidden");
    return;
  }
  if (error) error.classList.add("hidden");
  let threeState = state.inferenceThreeState.get(canvas);
  if (!threeState) {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.001, 100);
    const root = new THREE.Group();
    scene.add(new THREE.AmbientLight(0xffffff, 1));
    scene.add(root);
    threeState = { renderer, scene, camera, root, yaw: 2.3, pitch: -0.55, zoom: 1, dragging: false, lastX: 0, lastY: 0, controlsAttached: false };
    state.inferenceThreeState.set(canvas, threeState);
    attachInferenceThreeControls(canvas, threeState);
  }

  const rect = canvas.getBoundingClientRect();
  const width = Math.max(360, Math.floor(rect.width));
  const height = Math.max(300, Math.floor(rect.height));
  threeState.renderer.setSize(width, height, false);
  threeState.camera.aspect = width / height;
  disposeInferenceThreeRoot(threeState.root);

  const history = state.inferenceTrajectoryHistory || [];
  const paths = [
    { key: "leftTarget", rotKey: "leftTargetRot6d", color: 0x007aff, opacity: 1, dashed: false },
    { key: "leftFeedback", rotKey: "leftFeedbackRot6d", color: 0x6baeff, opacity: 0.72, dashed: true },
    { key: "rightTarget", rotKey: "rightTargetRot6d", color: 0xff3b30, opacity: 1, dashed: false },
    { key: "rightFeedback", rotKey: "rightFeedbackRot6d", color: 0xffa09b, opacity: 0.72, dashed: true },
  ].map((entry) => ({ ...entry, points: history.map((sample) => sample[entry.key]).filter((point) => Array.isArray(point) && point.length === 3) }));
  const allPoints = paths.flatMap((entry) => entry.points).map((point) => new THREE.Vector3(point[0], point[1], point[2]));
  if (!allPoints.length) {
    threeState.renderer.render(threeState.scene, threeState.camera);
    return;
  }
  const box = new THREE.Box3().setFromPoints(allPoints);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z, 0.05);
  const distance = span * 3.1 * threeState.zoom;
  const cosPitch = Math.cos(threeState.pitch);
  threeState.camera.position.set(
    center.x + distance * cosPitch * Math.cos(threeState.yaw),
    center.y + distance * cosPitch * Math.sin(threeState.yaw),
    center.z + distance * Math.sin(threeState.pitch),
  );
  threeState.camera.lookAt(center);
  threeState.camera.updateProjectionMatrix();

  const grid = new THREE.GridHelper(Math.max(span * 1.5, 0.2), 8, 0x94a3b8, 0xd1d5db);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(center.x, center.y, box.min.z);
  threeState.root.add(grid);
  paths.forEach((entry) => {
    const points = entry.points.map((point) => new THREE.Vector3(point[0], point[1], point[2]));
    if (points.length < 2) return;
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = entry.dashed
      ? new THREE.LineDashedMaterial({ color: entry.color, transparent: true, opacity: entry.opacity, dashSize: span * 0.035, gapSize: span * 0.02 })
      : new THREE.LineBasicMaterial({ color: entry.color, transparent: true, opacity: entry.opacity });
    const line = new THREE.Line(geometry, material);
    if (entry.dashed) line.computeLineDistances();
    threeState.root.add(line);
    const latest = points[points.length - 1];
    const marker = new THREE.Mesh(new THREE.SphereGeometry(span * 0.018, 14, 10), new THREE.MeshBasicMaterial({ color: entry.color }));
    marker.position.copy(latest);
    threeState.root.add(marker);
  });
  const latest = history[history.length - 1];
  [
    [latest?.leftTarget, latest?.leftTargetRot6d],
    [latest?.rightTarget, latest?.rightTargetRot6d],
  ].forEach(([position, rotation]) => {
    const basis = rot6dBasis(rotation);
    if (!Array.isArray(position) || !basis) return;
    const axes = new THREE.AxesHelper(Math.max(span * 0.1, 0.025));
    axes.position.set(position[0], position[1], position[2]);
    axes.setRotationFromMatrix(new THREE.Matrix4().makeBasis(
      new THREE.Vector3(...basis[0]), new THREE.Vector3(...basis[1]), new THREE.Vector3(...basis[2]),
    ));
    threeState.root.add(axes);
  });
  threeState.renderer.render(threeState.scene, threeState.camera);
}

function updatePreviewLoop() {
  if (state.active !== "record" && state.active !== "inference") return;
  document.querySelectorAll("img[data-live-camera], img[data-inference-camera]").forEach((img) => {
    const cameraId = img.dataset.liveCamera ?? img.dataset.inferenceCamera ?? "0";
    const base = img.dataset.frameSrc || `/camera/frame?camera_id=${cameraId}`;
    const sep = base.includes("?") ? "&" : "?";
    const url = `${base}${sep}_=${Math.floor(performance.now() / 200)}`;
    const hint = img.parentElement?.querySelector(".hint") || null;
    img.onload = () => { img.style.display = "block"; if (hint) hint.textContent = ""; };
    img.onerror = () => { img.style.display = "none"; if (hint) hint.textContent = `cam${cameraId} 暂无图像`; };
    if (hint && !img.getAttribute("src")) hint.textContent = `等待 cam${cameraId}...`;
    if (img.getAttribute("src") !== url) img.src = url;
  });
}

async function queueCommand(path, message) {
  const result = await api(path);
  showBanner(message || `指令已进入控制循环：${result.command || path}`, "warning");
  await refreshAll();
  return result;
}

async function startTeleop() {
  try { await queueCommand("/command/start", "开始接管指令已进入控制循环"); } catch (e) { showBanner(`开始接管失败：${e?.message || e}`, "error"); }
}
async function stopTeleop() {
  try {
    state.shuttingDown = true;
    await api("/command/stop");
    el("connText").textContent = "exiting";
    showBanner("停止退出指令已进入控制循环；teleop 进程退出后页面会断开连接", "warning");
    renderSideStatus();
  } catch (e) {
    if (expectedShutdownFetchError(e)) {
      el("connText").textContent = "exited";
      showBanner("teleop 进程已退出，页面连接已断开", "warning");
      renderSideStatus();
      return;
    }
    state.shuttingDown = false;
    showBanner(`停止失败：${e?.message || e}`, "error");
  }
}
async function homeTeleop() {
  try { await queueCommand("/command/home", "Home 指令已进入控制循环"); } catch (e) { showBanner(`Home 失败：${e?.message || e}`, "error"); }
}
async function recenterTeleop() {
  try { await queueCommand("/command/recenter", "重置头参考指令已进入控制循环"); } catch (e) { showBanner(`重置头参考失败：${e?.message || e}`, "error"); }
}
function startInference() {
  const prompt = String(input("inferencePrompt") || "").trim();
  if (!prompt) {
    showBanner("请填写任务描述。", "error");
    return;
  }
  localStorage.inferencePrompt = prompt;
  api("/inference/start?" + qs({ prompt })).then(() => refreshAll()).then(() => {
    showBanner("真机推理启动指令已进入控制循环", "warning");
    render();
  });
}
function stopInference() {
  api("/inference/stop").then(() => refreshAll()).then(() => {
    showBanner("停止推理指令已进入控制循环，机器人将保持当前姿态。", "warning");
    render();
  });
}
function restoreXrInput() {
  api("/ui/provider/xr").then(() => refreshAll()).then(() => {
    showBanner("XR 输入恢复指令已进入控制循环", "warning");
    render();
  });
}

async function startCam(id) {
  showBanner(`当前 UI 不启动相机。请在启动 teleop 进程时配置 camera_id=${id} 对应的 CLI 参数。`, "error");
}
function cameraIdForSerial(serial, fallback) {
  const key = String(serial || "").trim();
  let map = {};
  try { map = JSON.parse(localStorage.cameraSerialIds || "{}") || {}; } catch (_) { map = {}; }
  if (key && Number.isFinite(Number(map[key]))) return Math.round(Number(map[key]));
  const used = new Set(Object.values(map).map((v) => Math.round(Number(v))).filter((v) => Number.isFinite(v) && v >= 0));
  let next = Math.max(0, Math.round(Number(fallback || 0)));
  while (used.has(next)) next += 1;
  if (key) {
    map[key] = next;
    localStorage.cameraSerialIds = JSON.stringify(map);
  }
  return next;
}
async function startRsCam(serial, cameraId) {
  showBanner(`当前 UI 不启动 RealSense ${serial || ""}。请在启动 teleop 进程时配置相机参数。`, "error");
}
async function stopCam(id) { showBanner(`当前 UI 不停止相机 cam${id}。相机生命周期由 teleop 进程管理。`, "error"); }
function selectPreviewCamera(id) { state.selectedPreviewCameraId = id; render(); }
function openCameraConfig() { state.cameraConfigOpen = true; render(); }
function closeCameraConfig() { state.cameraConfigOpen = false; render(); }
function setPlaybackSearch(value) {
  state.playbackEpisodeSearch = value;
  render();
  requestAnimationFrame(() => {
    const box = document.getElementById("playbackEpisodeSearch");
    if (box) {
      box.focus();
      const n = box.value.length;
      try { box.setSelectionRange(n, n); } catch (_) {}
    }
  });
}
function setExportRangeInput(value) { state.exportRangeInput = value; }

async function setRecordFps(value) {
  try {
    const fps = String(value || "30").trim();
    localStorage.recordFps = fps;
    await api("/recording/set_fps?" + qs({ fps }));
  } catch (e) { showBanner(`设置采集 FPS 失败：${e?.message || e}`, "error"); }
}

async function startRec() {
  try {
    await api("/recording/start");
    showBanner("开始录制指令已进入控制循环；真正开始时间由主循环和相机首帧对齐决定", "warning");
    await refreshAll();
  } catch (e) { showBanner(`开始录制失败：${e?.message || e}`, "error"); }
}
async function stopRec() { try { const result = await api("/recording/stop"); if (result.last_validation) state.snapshot.recording = { ...(state.snapshot.recording || {}), last_validation: result.last_validation }; showBanner("停止并保存指令已进入控制循环", "warning"); await refreshAll(); } catch (e) { showBanner(`停止录制失败：${e?.message || e}`, "error"); } }
async function cancelRec() { try { await api("/recording/cancel"); showBanner("取消录制指令已进入控制循环", "warning"); await refreshAll(); } catch (e) { showBanner(`取消录制失败：${e?.message || e}`, "error"); } }
async function startExport() {
  try {
    syncExportSelectionFromDom();
    syncExportConfigFromDom();
    const cfg = readExportConfig();
    const selected = getSelectedExportEpisodes();
    const sourceRoot = String(effectiveRecordRoot()).trim();
    const outputRoot = cfg.outputRoot.trim();
    const datasetName = cfg.datasetName.trim();
    const exportTask = cfg.task.trim();
    if (!selected.length) throw new Error("未选择 episode。请勾选 episode，或输入范围如 1-50 后点击“按范围选择”。");
    if (!sourceRoot) throw new Error("source_root 为空。请填写包含 episodes/ 的采集根目录。");
    if (!outputRoot) throw new Error("output_root 为空。请填写 LeRobot 数据集输出根目录。");
    if (!datasetName) throw new Error("dataset_name 为空。请填写导出的数据集名称。");
    const urdfPath = cfg.urdfPath.trim() || DEFAULT_URDF;
    if (!exportTask) throw new Error("导出任务描述为空。请在导出页填写本批次 episode 的统一 task。");
    const q = new URLSearchParams({ source_root: sourceRoot, output_root: outputRoot, dataset_name: datasetName, urdf_path: urdfPath, format_version: cfg.formatVersion || "v2", export_mode: cfg.mode || "new", export_video: cfg.exportVideo ? "1" : "0", export_fk: cfg.exportFk ? "1" : "0", export_verify: cfg.exportVerify ? "1" : "0", default_task: exportTask });
    selected.forEach((name) => q.append("episode", name));
    setExportConfig("outputRoot", outputRoot);
    setExportConfig("datasetName", datasetName);
    setExportConfig("urdfPath", urdfPath);
    setExportConfig("task", exportTask);
    showBanner(`开始导出 ${selected.length} 个 episode：${selected.slice(0, 3).join(", ")}${selected.length > 3 ? " ..." : ""}`, "warning");
    state.lastConvertNoticeKey = "";
    state.convertStatus = await api("/convert/start?" + q.toString());
    const lengthWarnings = state.convertStatus.episode_length_warnings || [];
    if (lengthWarnings.length) {
      showBanner(`导出已开始，但发现 ${lengthWarnings.length} 个异常短/长 episode；请查看导出页预警卡。`, "warning");
    }
    render();
    pollConvert();
  } catch (e) { showBanner(`导出失败：${e?.message || e}`, "error"); }
}
async function pollConvert() {
  try {
    state.convertStatus = await api("/convert/status", { allowApplicationError: true });
    render();
    const c = state.convertStatus || {};
    if (c.running) {
      window.setTimeout(pollConvert, 900);
      return;
    }
    const key = `${c.phase || ""}:${c.ok}:${c.error || ""}:${c.end_time_ns || ""}`;
    if (key && key !== state.lastConvertNoticeKey) {
      state.lastConvertNoticeKey = key;
      if (c.ok === false || c.phase === "error") {
        showBanner(`导出失败：${c.error || c.message || "unknown error"}`, "error");
      } else if (c.phase === "done") {
        const v = c.export_verification || {};
        if (c.export_verify) showBanner(v.ok === false ? "导出完成，但抽样验证失败，请查看验证卡片" : "导出完成，抽样验证通过", v.ok === false ? "error" : "warning");
        else showBanner("导出完成", "warning");
      }
    }
  } catch (e) {
    showBanner(`导出状态获取失败：${e?.message || e}`, "error");
  }
}

function ensureExportSet() {
  if (!(state.exportSelected instanceof Set)) state.exportSelected = new Set();
  return state.exportSelected;
}
function pruneExportSelectionToAvailable() {
  const set = ensureExportSet();
  const available = new Set((state.episodes || []).map((e) => String(e.name || "")));
  for (const name of [...set]) if (!available.has(String(name))) set.delete(name);
}
function syncExportSelectionFromDom() {
  const set = ensureExportSet();
  document.querySelectorAll("[data-ep]").forEach((x) => {
    if (x.checked) set.add(x.value); else set.delete(x.value);
  });
  updateExportSelectionUi();
}
function getSelectedExportEpisodes() {
  const available = new Set((state.episodes || []).map((e) => String(e.name || "")));
  return [...ensureExportSet()].filter((name) => available.has(name)).sort();
}
function updateExportSelectionUi() {
  const count = getSelectedExportEpisodes().length;
  const pill = document.getElementById("exportSelectedCount");
  if (pill) pill.textContent = `${count} selected`;
  const btn = document.getElementById("exportStartBtn");
  if (btn) btn.textContent = `导出所选 ${count ? `(${count})` : ""}`;
}
function toggleExportEpisode(name, isChecked) {
  const set = ensureExportSet();
  if (isChecked) set.add(name); else set.delete(name);
  const box = document.querySelector(`[data-ep][value="${CSS.escape(name)}"]`);
  if (box) box.checked = isChecked;
  if (box) box.closest(".episode-option")?.classList.toggle("selected", isChecked);
  updateExportSelectionUi();
}
function episodeNo(name) {
  const m = String(name || "").match(/(\d+)$/);
  return m ? Number(m[1]) : null;
}
function parseExportRange(expr) {
  const episodes = (state.episodes || []).map((e) => String(e.name || "")).filter(Boolean);
  const byNo = new Map();
  for (const name of episodes) {
    const n = episodeNo(name);
    if (n === null || !Number.isFinite(n)) continue;
    const arr = byNo.get(n) || [];
    arr.push(name);
    byNo.set(n, arr);
  }
  const tokens = String(expr || "").split(/[,\n，、;；]+/).map((x) => x.trim()).filter(Boolean);
  if (!tokens.length) throw new Error("范围为空。示例：1-50, 50-100。");
  const matched = new Set();
  const bad = [];
  for (const token of tokens) {
    const m = token.match(/^(?:episode_)?0*(\d+)(?:\s*[-~至]\s*(?:episode_)?0*(\d+))?$/i);
    if (!m) { bad.push(token); continue; }
    let a = Number(m[1]); let b = m[2] === undefined ? a : Number(m[2]);
    if (!Number.isFinite(a) || !Number.isFinite(b)) { bad.push(token); continue; }
    if (a > b) [a, b] = [b, a];
    for (let n = a; n <= b; n += 1) for (const name of byNo.get(n) || []) matched.add(name);
  }
  if (bad.length) throw new Error(`范围格式无法识别：${bad.join(", ")}。请使用 1-50, 50-100 或 episode_000001-episode_000050。`);
  if (!matched.size) {
    const nums = [...byNo.keys()].sort((a, b) => a - b);
    const scope = nums.length ? `${nums[0]}-${nums[nums.length - 1]}` : "无 episode";
    throw new Error(`范围未匹配到 episode。当前可选编号范围：${scope}；如果 episode 从 0 开始，请输入 0-49。`);
  }
  return [...matched].sort();
}
function selectExportRange(_mode = "replace") {
  try {
    syncExportSelectionFromDom();
    syncExportConfigFromDom();
    state.exportRangeInput = input("exportRange");
    const names = parseExportRange(input("exportRange"));
    const set = ensureExportSet();
    set.clear();
    names.forEach((name) => set.add(name));
    showBanner(`已选择 ${names.length} 个 episode，当前共 ${set.size} 个`, "warning");
    render();
  } catch (e) { showBanner(`选择范围失败：${e?.message || e}`, "error"); }
}
function selectAllExportEpisodes() {
  syncExportConfigFromDom();
  const set = ensureExportSet();
  (state.episodes || []).forEach((e) => e?.name && set.add(String(e.name)));
  render();
}
function clearExportSelection() { syncExportConfigFromDom(); ensureExportSet().clear(); render(); }
function invertExportSelection() {
  syncExportConfigFromDom();
  const set = ensureExportSet();
  (state.episodes || []).forEach((e) => {
    const name = String(e.name || "");
    if (!name) return;
    if (set.has(name)) set.delete(name); else set.add(name);
  });
  render();
}

async function loadPlayback(name) {
  try {
    await api("/playback/load?" + qs({ root_dir: effectiveRecordRoot(), episode: name }));
    state.playbackCurves = null;
    state.playbackLastImageKey = "";
    await pollPlaybackStatusOnce({ renderPanel: true });
    await loadPlaybackCurves();
    syncPlaybackPanel(displayedPlayback());
  } catch (e) { showBanner(`加载回放失败：${e?.message || e}`, "error"); }
}
async function deleteEpisodes(names) {
  try {
    const selected = (names || []).filter(Boolean);
    if (!selected.length) throw new Error("先选择 episode");
    if (!window.confirm(`确认删除 ${selected.length} 个 episode？该操作不可撤销。`)) return;
    const q = new URLSearchParams({ root_dir: effectiveRecordRoot() });
    selected.forEach((name) => q.append("episode", name));
    await api("/recording/delete_episodes?" + q.toString());
    showBanner(`已删除 ${selected.length} 个 episode`, "warning");
    await refreshEpisodes();
    render();
  } catch (e) { showBanner(`删除 episode 失败：${e?.message || e}`, "error"); }
}
function deleteSelectedEpisodes() {
  const selected = [...document.querySelectorAll("[data-delete-ep]:checked")].map((x) => x.value);
  deleteEpisodes(selected);
}
async function playbackStart() {
  await api("/playback/start");
  showBanner("开始播放 episode 图像和曲线；不会触发真机动作。", "warning");
  await pollPlaybackStatusOnce();
  managePlaybackPolling();
}
async function playbackPause() { await api("/playback/pause"); await pollPlaybackStatusOnce(); }
async function playbackStop() { await api("/playback/stop"); state.playbackLastImageKey = ""; await pollPlaybackStatusOnce(); }
function realReplayCommandResolved(provider, command) {
  const activeProvider = String(provider?.active_provider || "");
  const replayState = String(provider?.real_replay?.state || "");
  if (command === "start") {
    return activeProvider === "raw_replay" || replayState === "running" || replayState === "finished" || replayState === "error";
  }
  if (command === "stop") {
    return activeProvider !== "raw_replay" && replayState !== "running";
  }
  return false;
}

function refreshRealReplayStatus() {
  return api("/replay/real/status").then((result) => {
    const provider = result?.provider;
    if (!provider || typeof provider !== "object") throw new Error("真机回放状态响应缺少 provider");
    state.snapshot = { ...(state.snapshot || {}), provider };
    return provider;
  });
}

function reconcileRealReplayStatus(command, attempt = 0) {
  if (state.realReplayCommandPending !== command) return;
  refreshRealReplayStatus()
    .then((provider) => {
      if (state.realReplayCommandPending !== command) return;
      if (realReplayCommandResolved(provider, command)) {
        state.realReplayCommandPending = "";
        render();
        return;
      }
      if (attempt >= 30) {
        state.realReplayCommandPending = "";
        render();
        showBanner(`真机回放${command === "start" ? "启动" : "停止"}指令 3 秒内未被控制循环确认，请检查后端日志。`, "error");
        return;
      }
      window.setTimeout(() => reconcileRealReplayStatus(command, attempt + 1), 100);
    })
    .catch((e) => {
      if (state.realReplayCommandPending !== command) return;
      state.realReplayCommandPending = "";
      render();
      showBanner(`读取真机回放状态失败：${e?.message || e}`, "error");
    });
}

function startRealReplay() {
  const p = state.snapshot?.playback || {};
  const provider = state.snapshot?.provider || {};
  if (!p.episode_name) {
    showBanner("请先加载一个 episode，再开始真机回放。", "error");
    return;
  }
  if (provider.active_provider !== "hold") {
    showBanner("真机回放要求 provider=HOLD；请先切到回放页等待状态更新。", "error");
    return;
  }
  const speedNode = document.getElementById("realReplaySpeed");
  const sourceNode = document.getElementById("realReplayArmSource");
  const speedScale = String(speedNode?.value || localStorage.realReplaySpeedScale || "1.0").trim();
  const armSource = String(sourceNode?.value || localStorage.realReplayArmSource || "action").trim();
  localStorage.realReplaySpeedScale = speedScale;
  localStorage.realReplayArmSource = armSource;
  api("/replay/real/start?" + qs({
    root_dir: effectiveRecordRoot(),
    episode: p.episode_name,
    arm_source: armSource,
    speed_scale: speedScale,
  }))
    .then(() => {
      state.realReplayCommandPending = "start";
      showBanner(`真机回放启动指令已排队：${p.episode_name}`, "warning");
      render();
      reconcileRealReplayStatus("start");
    })
    .catch((e) => showBanner(`真机回放启动失败：${e?.message || e}`, "error"));
}
function stopRealReplay() {
  api("/replay/real/stop")
    .then(() => {
      state.realReplayCommandPending = "stop";
      showBanner("停止真机回放指令已排队，机器人将保持当前姿态。", "warning");
      render();
      reconcileRealReplayStatus("stop");
    })
    .catch((e) => showBanner(`停止真机回放失败：${e?.message || e}`, "error"));
}
async function seekPlayback(frame) {
  await api("/playback/seek?" + qs({ frame: Math.round(Number(frame || 0)) }));
  state.playbackLastImageKey = "";
  await pollPlaybackStatusOnce();
}
function previewPlaybackSeek(frame) {
  const p = { ...(state.snapshot?.playback || {}), frame_index: Math.round(Number(frame || 0)) };
  syncPlaybackPanel(p, { skipImage: true });
  drawPlaybackCurves(p);
}
function setPlaybackCamera(cameraId) {
  // Kept for backward compatibility with older static pages; playback now shows
  // all cameras by default, so no single-camera selection is needed.
  syncPlaybackPanel(displayedPlayback());
}
function previewPlaybackImage(cameraId) { setPlaybackCamera(cameraId); }

async function loadPlaybackCurves() {
  const meta = document.getElementById("playbackCurveMeta");
  if (meta) meta.textContent = "正在加载 q_fb 曲线...";
  try {
    state.playbackCurves = await api("/playback/curves?" + qs({ arm: "both", max_points: 900 }));
    if (meta) meta.textContent = `曲线样本 ${state.playbackCurves.sample_count || 0}/${state.playbackCurves.total_frames || 0} · stride=${state.playbackCurves.sample_stride || 1}`;
    renderCurveLegends();
    scheduleCurveDraw(true);
  } catch (e) {
    state.playbackCurves = null;
    if (meta) meta.textContent = `曲线加载失败：${e?.message || e}`;
  }
}

async function pollPlaybackStatusOnce(options= {}) {
  try {
    const p = await api("/playback/status");
    state.snapshot = { ...(state.snapshot || {}), playback: p };
    if (options.renderPanel) render();
    syncPlaybackPanel(displayedPlayback());
    scheduleCurveDraw(false);
  } catch (_) {}
}

function managePlaybackPolling() {
  if (state.playbackStatusTimer) {
    window.clearInterval(state.playbackStatusTimer);
    state.playbackStatusTimer = 0;
  }
  if (state.active !== "playback") return;
  state.playbackStatusTimer = window.setInterval(() => {
    if (state.active !== "playback" || document.hidden) return;
    const p = state.snapshot?.playback || {};
    if (p.enabled || p.state === "playing") pollPlaybackStatusOnce();
  }, 100);
}

function syncPlaybackPanel(playback, options= {}) {
  if (state.active !== "playback") return;
  const p = playback || {};
  const total = Math.max(0, Number(p.total_frames || 0));
  const idx = Math.max(0, Math.min(Math.max(0, total - 1), Number(p.frame_index || 0)));
  const seek = document.getElementById("playbackSeek");
  if (seek && document.activeElement !== seek) {
    seek.max = String(Math.max(0, total - 1));
    seek.value = String(Math.round(idx));
  }
  const label = document.getElementById("playbackTimeLabel");
  if (label) label.textContent = total > 0 ? `frame ${Math.round(idx) + 1}/${total} · ${Number(p.current_time_sec || 0).toFixed(2)}s / ${Number(p.duration_sec || 0).toFixed(2)}s · ${p.state || "idle"}` : "未加载 episode";
  const metricFrame = document.getElementById("playbackMetricFrame");
  if (metricFrame) metricFrame.textContent = total > 0 ? `${Math.round(idx) + 1}/${total}` : "0/0";
  const metricTime = document.getElementById("playbackMetricTime");
  if (metricTime) metricTime.textContent = `${Number(p.current_time_sec || 0).toFixed(2)}s`;
  const metricDuration = document.getElementById("playbackMetricDuration");
  if (metricDuration) metricDuration.textContent = `${Number(p.duration_sec || 0).toFixed(2)}s`;
  const metricCameras = document.getElementById("playbackMetricCameras");
  if (metricCameras) metricCameras.textContent = String((p.cameras || []).length);
  const meta = document.getElementById("playbackFrameMeta");
  if (meta) meta.textContent = total > 0 ? `frame ${Math.round(idx) + 1}/${total}` : "single frame";
  if (!options.skipImage) syncPlaybackImages(p, Math.round(idx));
  scheduleCurveDraw(false);
}

function syncPlaybackTrace() {
  if (state.active !== "playback") return;
  const replay = state.snapshot?.provider?.real_replay || {};
  const executionTrace = replay.runtime_debug?.execution_trace || {};
  const pending = executionTrace.current || null;
  const latest = executionTrace.latest || null;
  const trace = latest || pending || {};
  const traceState = pending
    ? `pending #${pending.seq ?? "-"} · showing #${latest?.seq ?? "-"}`
    : latest ? `${latest.status || "completed"} #${latest.seq ?? "-"}` : "waiting";
  const badgeNode = el("playbackTraceBadge");
  if (badgeNode) badgeNode.innerHTML = badge(pending ? "running" : trace.status || "idle", traceState);
  const traceMetricIds = {
    playbackTraceRecvToPub: "recv_to_pub_ms",
    playbackTracePubToThread: "pub_to_exec_thread_ms",
    playbackTraceRecvToThread: "recv_to_exec_thread_ms",
    playbackTraceEnqueueToPublish: "enqueue_to_publish_ms",
    playbackTraceDdsWrite: "dds_write_ms",
  };
  for (const [id, key] of Object.entries(traceMetricIds)) {
    const node = el(id);
    if (node) node.textContent = Number.isFinite(Number(trace[key])) ? `${Number(trace[key]).toFixed(1)} ms` : "-";
  }
  const jointDelta = el("playbackTraceJointDelta");
  if (jointDelta) jointDelta.textContent = `${Number(trace.q_delta_thread_trigger || 0).toFixed(4)} rad`;
  const frame = el("playbackTraceFrame");
  if (frame) frame.textContent = `episode frame ${Number(trace.raw_replay_frame_index ?? -1)}`;
}

function syncPlaybackImages(playback, frameIndex) {
  const p = playback || {};
  const cams = Array.isArray(p.cameras) ? p.cameras : [];
  const imgs = [...document.querySelectorAll("img[data-playback-camera]")];
  if (!imgs.length) return;
  if (!Number(p.total_frames || 0) || !cams.length) {
    imgs.forEach((img) => { img.style.display = "none"; img.removeAttribute("src"); });
    state.playbackLastImageKey = "";
    return;
  }
  const ts = p.timestamp_ns || frameIndex;
  imgs.forEach((img) => {
    const cid = Math.round(Number(img.dataset.playbackCamera || 0));
    const key = `${cid}|${frameIndex}|${ts}`;
    if (img.dataset.lastKey === key) return;
    img.dataset.pendingKey = key;
    const hint = document.querySelector(`[data-playback-hint="${cid}"]`);
    const url = `/playback/image?${qs({ camera_id: cid, frame: frameIndex, _ts: ts })}`;
    const pre = new Image();
    pre.onload = () => {
      if (img.dataset.pendingKey !== key) return;
      img.dataset.lastKey = key;
      img.src = url;
      img.style.display = "block";
      if (hint) hint.textContent = "";
    };
    pre.onerror = () => {
      if (img.dataset.pendingKey !== key) return;
      img.style.display = "none";
      if (hint) hint.textContent = `图像加载失败：cam${cid} frame=${frameIndex}`;
    };
    pre.src = url;
  });
}

function drawPlaybackCurves(playback) {
  const data = state.playbackCurves;
  const frames = data?.frame_indices || [];
  const frameIdx = Math.round(Number(playback?.frame_index || 0));
  let cursorRatio = null;
  if (Array.isArray(frames) && frames.length > 1) {
    let best = 0, bestD = Infinity;
    frames.forEach((f, i) => { const d = Math.abs(Number(f) - frameIdx); if (d < bestD) { bestD = d; best = i; } });
    cursorRatio = best / Math.max(1, frames.length - 1);
  }
  if (!data?.ok || !Array.isArray(frames) || frames.length < 2) {
    drawSeriesCanvas("playbackLeftJointCurveCanvas", "Left J1-J7 playback", [], null);
    drawSeriesCanvas("playbackRightJointCurveCanvas", "Right J1-J7 playback", [], null);
    drawRawReplayGripperStateCurves();
    return;
  }
  const leftJointSeries = JOINT_KEYS.map((key, j) => ({ name: JOINT_NAMES[j], color: JOINT_COLORS[j], values: data.left?.[key] || [] }));
  const rightJointSeries = JOINT_KEYS.map((key, j) => ({ name: JOINT_NAMES[j], color: JOINT_COLORS[j], values: data.right?.[key] || [] }));
  drawSeriesCanvas("playbackLeftJointCurveCanvas", "Left J1-J7 playback", leftJointSeries, cursorRatio);
  drawSeriesCanvas("playbackRightJointCurveCanvas", "Right J1-J7 playback", rightJointSeries, cursorRatio);
  drawRawReplayGripperStateCurves();
}

function drawRawReplayGripperStateCurves() {
  const replay = state.snapshot?.provider?.real_replay || {};
  const history = replay.runtime_debug?.gripper_state_history || [];
  const currentFrame = Number(replay.runtime_debug?.gripper_state_current?.frame_index ?? replay.frame_index ?? -1);
  const titleSuffix = Number.isFinite(currentFrame) && currentFrame >= 0 ? ` · frame ${currentFrame}` : "";
  const leftSeries = [
    { name: "当前真机反馈 state", color: GRIPPER_COLORS.feedback, values: history.map((item) => item?.left_feedback_q ?? null) },
    { name: "recorded state q", color: GRIPPER_COLORS.command, values: history.map((item) => item?.left_recorded_state_q ?? null) },
  ];
  const rightSeries = [
    { name: "当前真机反馈 state", color: GRIPPER_COLORS.feedback, values: history.map((item) => item?.right_feedback_q ?? null) },
    { name: "recorded state q", color: GRIPPER_COLORS.command, values: history.map((item) => item?.right_recorded_state_q ?? null) },
  ];
  drawSeriesCanvas("liveLeftGripperCurveCanvas", `Left gripper state${titleSuffix}`, leftSeries, history.length > 1 ? 1 : null);
  drawSeriesCanvas("liveRightGripperCurveCanvas", `Right gripper state${titleSuffix}`, rightSeries, history.length > 1 ? 1 : null);
}

Object.assign(window, { refreshAll, refreshEpisodes, render, setRecordFps, setGlobalRecordRoot, loadGlobalRecordRoot, startTeleop, stopTeleop, homeTeleop, recenterTeleop, startInference, stopInference, restoreXrInput, startCam, startRsCam, stopCam, selectPreviewCamera, openQuickPlaybackOptions, closeQuickPlaybackOptions, toggleQuickPlaybackOptions, updateQuickPlaybackQuery, selectQuickPlaybackEpisode, handleQuickPlaybackKey, openCameraConfig, closeCameraConfig, setPlaybackSearch, setExportConfig, setExportRangeInput, startRec, stopRec, cancelRec, startExport, toggleExportEpisode, selectExportRange, selectAllExportEpisodes, clearExportSelection, invertExportSelection, loadPlayback, deleteEpisodes, deleteSelectedEpisodes, playbackStart, playbackPause, playbackStop, startRealReplay, stopRealReplay, seekPlayback, previewPlaybackSeek, setPlaybackCamera, previewPlaybackImage, loadPlaybackCurves });
el("refreshBtn").addEventListener("click", refreshAll);
state.active = "record"; render(); connectSse(); refreshAll();
window.setInterval(() => {
  if (!document.hidden) {
    updatePreviewLoop();
    const now = performance.now();
    if (now - Number(state.lastCameraStatusRefreshMs || 0) > 1000) {
      state.lastCameraStatusRefreshMs = now;
      refreshCameraStatusOnly(true).then(null, (e) => {
        state.cameraStatusError = e?.message || "camera status offline";
      });
    }
  }
}, 200);
