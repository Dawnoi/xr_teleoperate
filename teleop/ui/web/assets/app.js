import * as THREE from "./vendor/three.module.js";

async function api(path) {
  const resp = await fetch(path, { cache: "no-store" });
  const text = await resp.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch (_) { data = { error: text }; }
  if (!resp.ok || data?.ok === false) throw new Error(data?.error || "request failed");
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

function validationSummary(v) {
  if (!v || !v.checked_at_ns) return empty("停止 episode 后会在这里显示自检小结");
  const checks = v.checks || {};
  const cov = checks.camera_coverage || {};
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
  const cameraText = cameraEntries.length
    ? cameraEntries.map(([cid, count]) => `cam${cid} ${count}/${frameCount}`).join(" · ")
    : "无相机帧统计";
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
  const cameraOk = cov.ok !== undefined ? !!cov.ok : !errors.some((x) => String(x).includes("camera") || String(x).includes("image"));
  const issueHtml = [...errors.map((x) => ["error", x]), ...warnings.map((x) => ["warning", x])]
    .slice(0, 8)
    .map(([level, text]) => `<li class="${level === "error" ? "err-text" : "warn-text"}">${esc(text)}</li>`)
    .join("");
  return `<div class="validation-summary">
    <div class="validation-title"><b>${esc(v.episode_name || "episode")}</b>${badge(v.level || "unknown")}</div>
    <div class="validation-conclusion ${v.level || "unknown"}">
      <b>${v.level === "ok" ? "结论：本 episode 数据可用" : v.level === "warning" ? "结论：本 episode 可用但存在风险" : "结论：本 episode 存在数据错误"}</b>
      <span>支撑：关节/夹爪 ${robotOk ? "完整" : "异常"}；相机 ${cameraOk ? "全部覆盖达标" : "存在缺失或覆盖不足"}。</span>
    </div>
    <div class="validation-grid">
      <div>${checkDot(checks.frame_count?.ok)}<span>帧数</span><b>${frameCount}</b></div>
      <div>${checkDot(fps.ok)}<span>采样</span><b>${Number(v.observed_fps || fps.observed || 0).toFixed(1)} fps</b><small>设置 ${Number(v.expected_fps || fps.expected || 0).toFixed(1)}</small></div>
      <div>${checkDot(robotOk)}<span>关节/夹爪</span><b>${esc(robotText)}</b><small>要求：左右臂每帧 7 关节 + gripper 均为有限数</small></div>
      <div>${checkDot(cameraOk)}<span>相机完整性</span><b>${esc(cameraText)}</b><small>要求：每路相机覆盖率 ≥ 95%，图片非空</small></div>
    </div>
    ${issueHtml ? `<ul class="validation-issues">${issueHtml}</ul>` : `<p class="mini">未发现错误或警告，数据完整性检查通过。</p>`}
  </div>`;
}

function renderRecord() {
  const r = state.snapshot?.recording || {};
  const v = r.last_validation || {};
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
      <div class="form record-prep-form"><label>采集 FPS<input id="recordFps" value="${esc(localStorage.recordFps || r.fps || 30)}" disabled></label><div><p class="mini">这里沿用参考 collector 的布局；实际 FPS、相机和目录来自启动参数，网页只发控制意图。</p><p class="row"><button onclick="startTeleop()" ${teleop.started ? "disabled" : ""}>开始接管</button><button class="secondary" onclick="homeTeleop()" ${teleop.started ? "" : "disabled"}>回到 Home</button><button class="secondary" onclick="recenterTeleop()" ${teleop.started ? "" : "disabled"}>重置头参考</button><button class="danger" onclick="stopTeleop()">停止退出</button></p><p class="row"><button onclick="startRec()" ${r.active || !teleop.started || !recordEnabled ? "disabled" : ""}>● 开始录制</button><button class="danger" onclick="stopRec()" ${r.active ? "" : "disabled"}>停止并保存</button><button class="secondary" onclick="cancelRec()" ${r.active ? "" : "disabled"}>取消录制</button>${recordEnabled ? "" : `<span class="pill"><span class="dot warn"></span>启动时未加 --record</span>`}</p></div></div>
      ${isRecording ? `<p class="mini">录制已进入 ${esc(r.phase || "recording")}；相机帧、机器人状态和 action 仍由主循环按 monotonic 时间对齐。</p>` : ""}
      <div class="prep-panels"><div><div class="toolbar"><div><h2>相机来源</h2><p class="mini">当前仓库相机由 CLI 参数打开；网页不启动、不停止相机。</p></div><button class="secondary" onclick="refreshAll()">刷新</button></div><div class="table-wrap"><table><thead><tr><th>Stream ID</th><th>来源</th><th>名称</th><th>链路</th><th>状态</th></tr></thead><tbody>${rows || `<tr><td colspan="5">${empty("尚未收到运行时相机帧；请检查启动参数、ZMQ sender 和网络链路")}</td></tr>`}</tbody></table></div></div><div><div class="toolbar"><div><h2>运行中流</h2><p class="mini">${activeIds.length} active${state.cameraStatusError ? ` · ${esc(state.cameraStatusError)}` : ""}</p></div></div><div class="table-wrap"><table><thead><tr><th>ID</th><th>名称</th><th>模式</th><th>实际输出</th><th>请求参数</th><th>帧年龄</th></tr></thead><tbody>${streams || `<tr><td colspan="6">${empty("暂无运行流；检查启动参数是否启用本地或 ZMQ 相机")}</td></tr>`}</tbody></table></div></div></div>
    </div>
    <div class="card full record-validation-card"><div class="card-head"><h2>采集自检小结</h2>${badge(v.level)}</div>${validationSummary(v)}</div>
    <div class="card full"><div class="card-head"><div><h2>多相机实时预览</h2><p class="mini">预览只读取当前进程已有 latest frame；不重启相机，不参与录制写盘。</p></div><span class="pill">${streamItems.length} live</span></div><div class="preview-grid layout-placeholder">${previews || empty("暂无可预览相机；请检查启动参数和相机链路")}</div></div>
    <div class="card full"><div class="card-head"><div><h2>实时数据曲线</h2><p class="mini">四图布局：左/右臂 J1-J7 与左/右夹爪分别显示，不再用线型区分。</p></div><span class="pill">4 charts</span></div><div class="curve-quad"><div class="curve-panel"><div class="curve-title"><b>Left J1-J7</b><span>q_fb</span></div><canvas id="liveLeftJointCurveCanvas" height="220"></canvas><div class="curve-legend" id="liveLeftJointLegend"></div></div><div class="curve-panel"><div class="curve-title"><b>Right J1-J7</b><span>q_fb</span></div><canvas id="liveRightJointCurveCanvas" height="220"></canvas><div class="curve-legend" id="liveRightJointLegend"></div></div><div class="curve-panel"><div class="curve-title"><b>Left Gripper</b><span>q_fb</span></div><canvas id="liveLeftGripperCurveCanvas" height="180"></canvas><div class="curve-legend" id="liveLeftGripperLegend"></div></div><div class="curve-panel"><div class="curve-title"><b>Right Gripper</b><span>q_fb</span></div><canvas id="liveRightGripperCurveCanvas" height="180"></canvas><div class="curve-legend" id="liveRightGripperLegend"></div></div></div></div>
    <div class="modal ${state.cameraConfigOpen ? "show" : ""}" onclick="if(event.target===this) closeCameraConfig()"><div class="modal-card"><div class="card-head"><div><h2>相机启动参数说明</h2><p class="mini">当前仓库相机在 Python 进程启动时打开，网页只读取 latest frame。</p></div><button class="secondary" onclick="closeCameraConfig()">关闭</button></div><div class="form compact-form"><label>宽<input id="camWidth" value="${esc(localStorage.camWidth || "1280")}" disabled></label><label>高<input id="camHeight" value="${esc(localStorage.camHeight || "720")}" disabled></label><label>FPS<input id="camFps" value="${esc(localStorage.camFps || "30")}" disabled></label><label>模式<select id="camMode" disabled><option value="rgb" ${(localStorage.camMode || "rgb") === "rgb" ? "selected" : ""}>RGB</option><option value="depth" ${localStorage.camMode === "depth" ? "selected" : ""}>Depth</option></select></label><label>名称<input id="camName" placeholder="head / left_wrist / right_wrist" value="${esc(localStorage.camName || "")}" disabled></label><label>角色<input id="camRole" value="${esc(localStorage.camRole || "runtime")}" disabled></label></div><p class="mini">请用 --head-camera-id / --left-camera-id / --right-camera-id 或 --head-zmq-endpoint / --left-zmq-endpoint / --right-zmq-endpoint 配置。这样录制线程和 UI 预览共享同一个相机 source，不会因为网页操作重启相机而破坏对齐。</p></div></div>
  </div>`;
}

function renderPlayback() {
  const p = state.snapshot?.playback || {};
  const r = state.snapshot?.recording || {};
  const teleop = state.snapshot?.teleop || {};
  const provider = state.snapshot?.provider || {};
  const realReplay = provider.real_replay || {};
  const realReplayRunning = provider.active_provider === "raw_replay" || realReplay.state === "running";
  const canStartRealReplay = !!p.episode_name && !!teleop.started && !r.active && provider.active_provider === "hold" && !realReplayRunning;
  const realReplaySpeed = String(localStorage.realReplaySpeedScale || realReplay.speed_scale || "1.0");
  const realReplayArmSource = String(localStorage.realReplayArmSource || realReplay.arm_source || "action");
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
    <td><button onclick='loadPlayback(${JSON.stringify(e.name)})'>加载</button><button class="secondary" onclick='deleteEpisodes([${JSON.stringify(e.name)}])'>删除</button></td>
  </tr>`).join("");
  const cameraCards = (p.cameras || []).map((c) => {
    const cid = Number(c.camera_id || 0);
    return `<div class="preview-tile playback-camera-tile" data-playback-tile="${cid}"><div class="preview-title"><b>${esc(c.camera_name || `cam${cid}`)}</b><span>id=${cid}</span></div><div class="preview"><span class="hint" data-playback-hint="${cid}">等待 cam${cid}...</span><img data-playback-camera="${cid}" style="display:none"></div></div>`;
  }).join("");
  const maxFrame = Math.max(0, Number(p.total_frames || 0) - 1);
  return `<div class="grid">
    <div class="card hero wide"><div class="card-head"><div><div class="eyebrow">Episode Playback</div><div class="headline">${esc(p.episode_name || "选择一个 episode")}</div><div class="subline">网页可播放图片/曲线；真机回放会切换到 raw offline provider 并下发动作。</div></div>${badge(p.state || "idle")}</div>
      <div class="metric-grid"><div class="metric"><span>Frame</span><b id="playbackMetricFrame">${p.frame_index || 0}/${p.total_frames || 0}</b></div><div class="metric"><span>Time</span><b id="playbackMetricTime">${Number(p.current_time_sec || 0).toFixed(2)}s</b></div><div class="metric"><span>Provider</span><b>${esc(provider.active_provider || "-")}</b></div><div class="metric"><span>Real Replay</span><b>${esc(realReplay.state || "idle")}</b><small>${esc(realReplay.episode_name || "")} ${Number(realReplay.frame_index || -1) >= 0 ? `#${Number(realReplay.frame_index)}` : ""}</small></div></div>
      <p class="row"><button onclick="playbackStart()">播放画面</button><button class="secondary" onclick="playbackPause()">暂停画面</button><button class="secondary" onclick="playbackStop()">停止画面</button><button class="secondary" onclick="loadPlaybackCurves()">重载曲线</button></p>
      <div class="form compact-form real-replay-controls"><label>arm_source<select id="realReplayArmSource" onchange="localStorage.realReplayArmSource=this.value"><option value="action" ${realReplayArmSource === "action" ? "selected" : ""}>action</option><option value="state" ${realReplayArmSource === "state" ? "selected" : ""}>state</option><option value="fk_cmd_pose" ${realReplayArmSource === "fk_cmd_pose" ? "selected" : ""}>fk_cmd_pose</option></select></label><label>speed_scale<input id="realReplaySpeed" value="${esc(realReplaySpeed)}" onchange="localStorage.realReplaySpeedScale=this.value"></label><div><p class="mini">切到回放页后 provider 会进入 HOLD；开始真机回放前必须未录制、已启动 teleop、且已加载 episode。</p><p class="row"><button class="danger" onclick="startRealReplay()" ${canStartRealReplay ? "" : "disabled"}>开始真机回放</button><button class="secondary" onclick="stopRealReplay()" ${realReplayRunning ? "" : "disabled"}>停止真机回放</button>${realReplay.error ? `<span class="pill"><span class="dot err"></span>${esc(realReplay.error)}</span>` : ""}</p></div></div>
      <div class="playback-controls"><input id="playbackSeek" type="range" min="0" max="${maxFrame}" value="${Number(p.frame_index || 0)}" oninput="previewPlaybackSeek(this.value)" onchange="seekPlayback(this.value)"><span id="playbackTimeLabel" class="mini">frame ${Number(p.frame_index || 0)} / ${Number(p.total_frames || 0)}</span></div></div>
    <div class="card full"><div class="card-head"><div><h2>全部相机同步回放</h2><p class="mini" id="playbackFrameMeta">加载 episode 后默认显示当前帧的全部相机；这不是机器人运动回放。</p></div><span class="pill">${(p.cameras || []).length} cameras</span></div><div class="preview-grid playback-camera-grid layout-placeholder">${cameraCards || empty("当前 episode 没有相机图像；加载 episode 后这里保留相机回放占位")}</div></div>
    <div class="card full"><div class="card-head"><div><h2>轨迹曲线同步回放</h2><p class="mini" id="playbackCurveMeta">四图布局：左/右臂 J1-J7 与左/右夹爪分别显示，游标与图像同步。</p></div><span class="pill">4 charts</span></div><div class="curve-quad"><div class="curve-panel"><div class="curve-title"><b>Left J1-J7</b><span>playback</span></div><canvas id="playbackLeftJointCurveCanvas" height="220"></canvas><div class="curve-legend" id="playbackLeftJointLegend"></div></div><div class="curve-panel"><div class="curve-title"><b>Right J1-J7</b><span>playback</span></div><canvas id="playbackRightJointCurveCanvas" height="220"></canvas><div class="curve-legend" id="playbackRightJointLegend"></div></div><div class="curve-panel"><div class="curve-title"><b>Left Gripper</b><span>playback</span></div><canvas id="playbackLeftGripperCurveCanvas" height="180"></canvas><div class="curve-legend" id="playbackLeftGripperLegend"></div></div><div class="curve-panel"><div class="curve-title"><b>Right Gripper</b><span>playback</span></div><canvas id="playbackRightGripperCurveCanvas" height="180"></canvas><div class="curve-legend" id="playbackRightGripperLegend"></div></div></div></div>
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
  const trace = latestTrace || pendingTrace || {};
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
    ${verifyHtml}
    <div class="card full"><div class="toolbar"><div><h2>选择 Episode</h2><p class="mini">支持批量范围，例如 <code>1-50, 50-100</code>；按 episode 编号匹配 episode_000001 这种名称。</p></div><div class="row"><span class="pill" id="exportSelectedCount">${selectedCount} selected</span><button class="secondary" onclick="refreshEpisodes().then(render)">刷新</button></div></div>
      <div class="selection-tools"><label class="range-input">批量范围<input id="exportRange" value="${esc(state.exportRangeInput || "")}" oninput="setExportRangeInput(this.value)" placeholder="1-50, 50-100 或 episode_000001-episode_000050"></label><button onclick="selectExportRange('replace')">按范围选择</button><button class="secondary" onclick="selectAllExportEpisodes()">全选</button><button class="secondary" onclick="invertExportSelection()">反选</button><button class="secondary" onclick="clearExportSelection()">清空</button></div>
      <p class="mini">提示：范围是闭区间；如果你的 episode 从 0 开始，请输入 0-49。选择状态会在导出页刷新时保留。</p>
      <div class="episode-list">${episodeChecks || empty("请先刷新 Episodes；如果列表为空，请检查 source_root 是否正确")}</div></div>
  </div>`;
}


const tabs = [
  ["record", "录制", "实时预览 / Episode", "●"],
  ["playback", "回放", "本地 episode 检查", "▶"],
  ["inference", "推理", "HTTP pi0.5 真机执行", "⌁"],
  ["export", "导出", "LeRobot 可选", "↗"],
];
const titles = {
  record: ["遥操录制工作台", "网页只发控制意图；相机、录制和对齐仍由当前 teleop 主循环负责。"],
  playback: ["回放检查", "网页只做本地 episode 图片和曲线检查；不会向真机下发回放动作。"],
  inference: ["真机推理", "HTTP pi0.5 动作经现有安全链和 DDS 下发；页面不直接控制真机。"],
  export: ["LeRobot 导出", "UI 已接入 raw_to_lerobot_v2；选择 episode 后导出为 LeRobot v2 数据集。"],
};
const root = document.getElementById("app");

root.innerHTML = `<div class="app"><aside class="side"><div class="brand"><div class="brand-mark">XR</div><div><div class="brand-title">xr_teleoperate</div><div class="brand-sub">UI control bridge</div></div></div><div class="side-status" id="sideStatus"></div><div class="side-root"><label>Record Root<input id="globalRecordRoot" value="${localStorage.recordRoot || "~/data/record/"}" disabled></label><p class="row"><button class="secondary" onclick="loadGlobalRecordRoot()">刷新列表</button></p><p class="mini">录制目录由启动参数 --task-dir/--task-name 决定；网页不修改当前进程配置。</p></div><div class="nav" id="nav"></div><div class="side-foot">Reference collector layout · xr runtime</div></aside><main class="main"><div class="top"><div><h1 id="topTitle">遥操录制工作台</h1><p id="topSub">connecting...</p></div><div class="top-actions"><span class="pill"><span id="connDot" class="dot"></span><span id="connText">connecting</span></span><span class="pill"><span id="recDot" class="dot"></span><span id="recText">record idle</span></span><button class="secondary" id="refreshBtn">刷新</button></div></div><div id="alert" class="banner"></div><section id="content"></section></main></div>`;

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
  syncPlaybackPanel(state.snapshot?.playback || {});
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
  if ((changedRecording || changedValidation || changedTeleop || shouldRenderInference) && document.activeElement?.tagName !== "INPUT") render();
  if (state.active === "playback") syncPlaybackPanel(state.snapshot.playback || {});
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
    state.episodes = (await api("/recording/episodes?" + qs({ root_dir: rootDir }))).episodes || [];
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
  localStorage.recordRoot = rootDir;
  const box = document.getElementById("globalRecordRoot");
  if (box && box.value !== rootDir) box.value = rootDir;
  try { await api("/recording/set_root_dir?" + qs({ root_dir: rootDir })); } catch (_) {}
  await refreshEpisodes();
  showBanner(`已加载 Root：${rootDir}`, "warning");
  if (state.active !== "record" || !state.snapshot?.recording?.active) render();
}

async function refreshAll() {
  if (state.shuttingDown) return;
  try { await refreshCameraStatusOnly(false); } catch (_) {}
  try { state.realsenseDevices = (await api("/camera/realsense_list")).devices || []; } catch (_) { state.realsenseDevices = []; }
  const recordingStatus = await api("/recording/status");
  state.snapshot = { ...(state.snapshot || {}), recording: recordingStatus };
  await refreshEpisodes();
  try { state.convertStatus = await api("/convert/status"); } catch (_) {}
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
  });
  while (hist.length > 240) hist.shift();
  state.liveCurveHistory = hist;
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
const GRIPPER_COLORS = { left: "#007aff", right: "#ff3b30" };
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
  const leftGripHtml = `<span><i style="background:${GRIPPER_COLORS.left}"></i>left gripper</span>`;
  const rightGripHtml = `<span><i style="background:${GRIPPER_COLORS.right}"></i>right gripper</span>`;
  const wristHtml = ["X", "Y", "Z"].map((axis, index) => `<span><i style="background:${INFERENCE_AXIS_COLORS[index]}"></i>${axis} target</span><span><i style="background:${INFERENCE_FEEDBACK_COLORS[index]}"></i>${axis} feedback</span>`).join("");
  const rot6dHtml = ROT6D_COLORS.map((color, index) => `<span><i style="background:${color}"></i>r${index + 1}</span>`).join("") + `<span><i style="background:#111827"></i>rotation error</span>`;
  const items = [
    ["liveLeftJointLegend", jointHtml], ["liveRightJointLegend", jointHtml],
    ["playbackLeftJointLegend", jointHtml], ["playbackRightJointLegend", jointHtml],
    ["liveLeftGripperLegend", leftGripHtml], ["playbackLeftGripperLegend", leftGripHtml],
    ["liveRightGripperLegend", rightGripHtml], ["playbackRightGripperLegend", rightGripHtml],
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
    if (state.active === "playback") drawPlaybackCurves(state.snapshot?.playback || {});
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
  drawSeriesCanvas("liveLeftGripperCurveCanvas", "Left gripper feedback", [{ name:"left gripper", color:GRIPPER_COLORS.left, values:hist.map((x) =>x?.left?.[7] ?? null) }]);
  drawSeriesCanvas("liveRightGripperCurveCanvas", "Right gripper feedback", [{ name:"right gripper", color:GRIPPER_COLORS.right, values:hist.map((x) =>x?.right?.[7] ?? null) }]);
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
    await api("/convert/start?" + q.toString());
    pollConvert();
  } catch (e) { showBanner(`导出失败：${e?.message || e}`, "error"); }
}
async function pollConvert() {
  try {
    state.convertStatus = await api("/convert/status");
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
    syncPlaybackPanel(state.snapshot?.playback || {});
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
      showBanner(`真机回放已进入控制循环：${p.episode_name}`, "warning");
      return refreshAll();
    })
    .then(() => render())
    .catch((e) => showBanner(`真机回放启动失败：${e?.message || e}`, "error"));
}
function stopRealReplay() {
  api("/replay/real/stop")
    .then(() => {
      showBanner("停止真机回放指令已进入控制循环，机器人将保持当前姿态。", "warning");
      return refreshAll();
    })
    .then(() => render())
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
  syncPlaybackPanel(state.snapshot?.playback || {});
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
    syncPlaybackPanel(p);
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
    drawSeriesCanvas("playbackLeftGripperCurveCanvas", "Left gripper playback", [], null);
    drawSeriesCanvas("playbackRightGripperCurveCanvas", "Right gripper playback", [], null);
    return;
  }
  const leftJointSeries = JOINT_KEYS.map((key, j) => ({ name: JOINT_NAMES[j], color: JOINT_COLORS[j], values: data.left?.[key] || [] }));
  const rightJointSeries = JOINT_KEYS.map((key, j) => ({ name: JOINT_NAMES[j], color: JOINT_COLORS[j], values: data.right?.[key] || [] }));
  drawSeriesCanvas("playbackLeftJointCurveCanvas", "Left J1-J7 playback", leftJointSeries, cursorRatio);
  drawSeriesCanvas("playbackRightJointCurveCanvas", "Right J1-J7 playback", rightJointSeries, cursorRatio);
  drawSeriesCanvas("playbackLeftGripperCurveCanvas", "Left gripper playback", [{ name:"left gripper", color:GRIPPER_COLORS.left, values:data.left?.gripper || [] }], cursorRatio);
  drawSeriesCanvas("playbackRightGripperCurveCanvas", "Right gripper playback", [{ name:"right gripper", color:GRIPPER_COLORS.right, values:data.right?.gripper || [] }], cursorRatio);
}

Object.assign(window, { refreshAll, refreshEpisodes, render, setRecordFps, setGlobalRecordRoot, loadGlobalRecordRoot, startTeleop, stopTeleop, homeTeleop, recenterTeleop, startInference, stopInference, restoreXrInput, startCam, startRsCam, stopCam, selectPreviewCamera, openCameraConfig, closeCameraConfig, setPlaybackSearch, setExportConfig, setExportRangeInput, startRec, stopRec, cancelRec, startExport, toggleExportEpisode, selectExportRange, selectAllExportEpisodes, clearExportSelection, invertExportSelection, loadPlayback, deleteEpisodes, deleteSelectedEpisodes, playbackStart, playbackPause, playbackStop, startRealReplay, stopRealReplay, seekPlayback, previewPlaybackSeek, setPlaybackCamera, previewPlaybackImage, loadPlaybackCurves });
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
