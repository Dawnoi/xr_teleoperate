const EXTENSION_ID = "xr.teleoperation";
const PROMPT_DRAFT_STORAGE_KEY = "xr.teleoperation.inference.prompt";
const ACTIONS = {
  resetHeadReference: "head_reference.reset",
  hold: "authority.hold",
  xr: "authority.xr",
  home: "home",
  cancelRecording: "record.cancel",
  startInference: "inference.start",
  stopInference: "inference.stop",
};

// The extension host publishes snapshots frequently.  Drafts must outlive a
// panel remount, while the DOM itself stays mounted through snapshot updates.
let promptDraft = sessionStorage.getItem(PROMPT_DRAFT_STORAGE_KEY) || "";
let selectedProfile = "mobile_pelvis_planar22";

function objectOrEmpty(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function stateFromSnapshot(snapshot) {
  const extensionState = snapshot?.data?.extensions?.[EXTENSION_ID];
  if (extensionState === undefined || extensionState === null || typeof extensionState !== "object") {
    return { error: "XR 扩展状态尚未由 Provider 发布。" };
  }
  return extensionState;
}

function recordingFromSnapshot(snapshot) {
  const collect = objectOrEmpty(snapshot?.data?.collect);
  return objectOrEmpty(collect.recording ?? collect.status);
}

function recordingIsActive(recording) {
  return recording.active === true || recording.is_recording === true;
}

function authorityLabel(authority) {
  if (authority === "hold") return "保持";
  if (authority === "xr" || authority === "xr_live") return "XR 接管";
  if (authority === "online_inference") return "在线推理";
  if (authority === "offline_replay") return "离线回放";
  return "未知";
}

function profileEntries(state) {
  const profiles = state.inference?.profiles;
  if (!Array.isArray(profiles)) return [];
  return profiles.filter((profile) => profile && typeof profile === "object" && typeof profile.id === "string");
}

function createControlButton(label, action, invoke) {
  const button = document.createElement("button");
  button.type = "button";
  button.addEventListener("click", () => invoke(action));
  button.dataset.action = action;
  button.dataset.label = label;
  return button;
}

function mountControls(api, pageMode) {
  if (!api || typeof api.snapshot !== "function" || typeof api.subscribe !== "function" || typeof api.invoke !== "function") {
    throw new TypeError("Nero Extension DOM API must provide snapshot, subscribe, and invoke.");
  }
  let latestSnapshot = api.snapshot();
  let pendingAction = "";
  let actionError = "";
  let profileSignature = "";
  const host = document.createElement("section");
  host.dataset.xrTeleoperation = pageMode;
  const shadow = host.attachShadow({ mode: "open" });
  const style = document.createElement("style");
  style.textContent = `
    :host { display:block; border-top:1px solid rgba(0,0,0,.09); padding-top:14px; }
    .xr { display:grid; gap:12px; color:#1c1c1e; font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif; }
    .head { display:flex; align-items:baseline; justify-content:space-between; gap:12px; }
    h3 { margin:0; font-size:14px; } .status { color:#636366; font-size:12px; }
    label { display:grid; gap:6px; color:#636366; font-size:12px; }
    textarea, select { width:100%; box-sizing:border-box; border:1px solid rgba(0,0,0,.12); border-radius:10px; background:#fff; color:#1c1c1e; padding:9px 10px; font:inherit; }
    textarea { min-height:72px; resize:vertical; } textarea:focus, select:focus { outline:2px solid rgba(0,122,255,.25); border-color:#007aff; }
    .row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
    button { border:1px solid transparent; border-radius:10px; padding:8px 12px; background:#007aff; color:#fff; cursor:pointer; font:600 13px/1.25 inherit; }
    button.secondary { background:rgba(118,118,128,.10); color:#1c1c1e; border-color:rgba(0,0,0,.09); }
    button.danger { background:#ff3b30; } button:disabled { cursor:not-allowed; opacity:.48; }
    .detail { margin:0; color:#636366; font-size:12px; overflow-wrap:anywhere; } .error { color:#d70015; }
    .inference { display:grid; gap:10px; } .controls { display:grid; gap:8px; }
    @media (max-width:680px) { .row button { flex:1 1 auto; } }
  `;
  const root = document.createElement("div");
  root.className = "xr";
  const heading = document.createElement("div");
  heading.className = "head";
  const title = document.createElement("h3");
  title.textContent = pageMode === "vla" ? "XR 在线推理" : "XR 控制";
  const authority = document.createElement("span");
  authority.className = "status";
  heading.append(title, authority);
  root.append(heading);

  const inferenceSection = document.createElement("section");
  inferenceSection.className = "inference";
  const promptLabel = document.createElement("label");
  promptLabel.textContent = "任务 Prompt";
  const promptInput = document.createElement("textarea");
  promptInput.rows = 3;
  promptInput.maxLength = 2048;
  promptInput.placeholder = "例如：把桌面上的杯子移动到右侧托盘";
  promptInput.value = promptDraft;
  promptInput.addEventListener("input", () => {
    promptDraft = promptInput.value;
    sessionStorage.setItem(PROMPT_DRAFT_STORAGE_KEY, promptDraft);
    update();
  });
  promptLabel.append(promptInput);
  const profileLabel = document.createElement("label");
  profileLabel.textContent = "移动操作协议";
  const profileInput = document.createElement("select");
  profileInput.addEventListener("change", () => {
    selectedProfile = profileInput.value;
    update();
  });
  profileLabel.append(profileInput);
  const profileStatus = document.createElement("p");
  profileStatus.className = "detail";
  const inferenceControls = document.createElement("div");
  inferenceControls.className = "row";
  const startInference = createControlButton("启动推理", ACTIONS.startInference, invoke);
  const stopInference = createControlButton("停止推理", ACTIONS.stopInference, invoke);
  inferenceControls.append(startInference, stopInference);
  const effectiveInference = document.createElement("p");
  effectiveInference.className = "detail";
  inferenceSection.append(promptLabel, profileLabel, profileStatus, inferenceControls, effectiveInference);
  if (pageMode === "vla") root.append(inferenceSection);

  const controls = document.createElement("section");
  controls.className = "controls";
  const controlDetails = document.createElement("p");
  controlDetails.className = "detail";
  const controlButtons = document.createElement("div");
  controlButtons.className = "row";
  const resetHeadReference = createControlButton("重置头参考", ACTIONS.resetHeadReference, invoke);
  const hold = createControlButton("保持当前姿态", ACTIONS.hold, invoke);
  hold.className = "secondary";
  const xr = createControlButton("恢复 XR 接管", ACTIONS.xr, invoke);
  xr.className = "secondary";
  const home = createControlButton("回零", ACTIONS.home, invoke);
  home.className = "secondary";
  const cancelRecording = createControlButton("取消当前录制", ACTIONS.cancelRecording, invoke);
  cancelRecording.className = "danger";
  controlButtons.append(resetHeadReference, hold, xr, home, cancelRecording);
  controls.append(controlDetails, controlButtons);
  root.append(controls);
  const error = document.createElement("p");
  error.className = "detail error";
  error.hidden = true;
  error.role = "alert";
  root.append(error);
  shadow.append(style, root);

  function invoke(action, extra = {}) {
    const actionExtra = action === ACTIONS.startInference
      ? { prompt: promptDraft.trim(), protocol_profile: selectedProfile }
      : extra;
    pendingAction = action;
    actionError = "";
    update();
    api.invoke(action, actionExtra).then(
      () => {
        pendingAction = "";
        update();
      },
      (requestError) => {
        pendingAction = "";
        actionError = requestError instanceof Error ? requestError.message : String(requestError);
        update();
      },
    );
  }

  function updateProfiles(profiles) {
    const signature = JSON.stringify(profiles.map((profile) => [profile.id, profile.label, profile.available, profile.reason]));
    if (signature === profileSignature) return;
    profileSignature = signature;
    profileInput.replaceChildren();
    for (const profile of profiles) {
      const option = document.createElement("option");
      option.value = profile.id;
      option.textContent = profile.available === true ? profile.label : `${profile.label}（不可用）`;
      option.disabled = profile.available !== true;
      profileInput.append(option);
    }
    if (!profiles.some((profile) => profile.id === selectedProfile)) selectedProfile = profiles[0]?.id || "";
    profileInput.value = selectedProfile;
  }

  function setButton(button, action, label, disabled) {
    button.textContent = pendingAction === action ? `${label}请求中...` : label;
    button.disabled = pendingAction !== "" || disabled;
  }

  function update() {
    const state = stateFromSnapshot(latestSnapshot);
    const currentRecording = recordingFromSnapshot(latestSnapshot);
    const recordingActive = recordingIsActive(currentRecording);
    const teleopReady = state.teleop?.started === true && state.teleop?.ready === true && state.teleop?.stopping !== true;
    const authorityValue = String(state.authority ?? "unknown");
    const hasStateError = state.error !== undefined;
    authority.textContent = `控制权：${authorityLabel(authorityValue)}`;
    controlDetails.textContent = `头参考：${state.head_reference?.mode ?? "未知"} · 遥操作：${teleopReady ? "已就绪" : "未就绪"}${recordingActive ? " · 录制中" : ""}`;

    if (pageMode === "vla") {
      const profiles = profileEntries(state);
      updateProfiles(profiles);
      const selected = profiles.find((profile) => profile.id === selectedProfile);
      profileInput.disabled = profiles.length === 0 || pendingAction !== "";
      if (!selected) profileStatus.textContent = "推理协议状态尚未加载，不能发起启动请求。";
      else if (selected.available !== true) profileStatus.textContent = `当前不可用：${String(selected.reason || "XR 运行条件不满足")}`;
      else profileStatus.textContent = `已选择：${String(selected.label)}。${String(selected.reason || "")}`;
      const inference = objectOrEmpty(state.inference);
      effectiveInference.textContent = `生效状态：${String(inference.state || "idle")} · ${String(inference.protocol_profile || "未启动")} · ${String(inference.prompt || "未提交 prompt")}`;
      setButton(startInference, ACTIONS.startInference, "启动推理", !teleopReady || recordingActive || promptDraft.trim().length === 0 || selected?.available !== true);
      setButton(stopInference, ACTIONS.stopInference, "停止推理", authorityValue !== "online_inference");
    }

    setButton(resetHeadReference, ACTIONS.resetHeadReference, "重置头参考", hasStateError || !teleopReady || state.head_reference?.recenter_supported !== true);
    setButton(hold, ACTIONS.hold, "保持当前姿态", hasStateError || !teleopReady || recordingActive || authorityValue === "hold");
    setButton(xr, ACTIONS.xr, "恢复 XR 接管", hasStateError || !teleopReady || recordingActive || authorityValue === "xr" || authorityValue === "xr_live");
    setButton(home, ACTIONS.home, "回零", hasStateError || !teleopReady);
    setButton(cancelRecording, ACTIONS.cancelRecording, "取消当前录制", hasStateError || pageMode !== "collector" || !recordingActive);
    error.textContent = actionError === "" ? "" : `操作失败：${actionError}`;
    error.hidden = actionError === "";
  }

  const unsubscribe = api.subscribe((snapshot) => {
    latestSnapshot = snapshot;
    update();
  });
  return { host, dispose: () => unsubscribe() };
}

function mountVlaControls(api) {
  const target = document.querySelector("#run .run-actions");
  if (!(target instanceof HTMLElement)) throw new Error("XR VLA direct layout requires #run .run-actions");
  const existing = document.querySelector('[data-xr-teleoperation="vla"]');
  if (existing instanceof HTMLElement) existing.remove();
  const controls = mountControls(api, "vla");
  target.append(controls.host);
  return () => {
    controls.dispose();
    controls.host.remove();
  };
}

function mountCollectorControls(api) {
  const content = document.getElementById("content");
  if (!(content instanceof HTMLElement)) throw new Error("XR Collector direct layout requires #content");
  const controls = mountControls(api, "collector");
  const attach = () => {
    const target = content.querySelector(".card.hero");
    if (target instanceof HTMLElement && !controls.host.isConnected) target.append(controls.host);
  };
  attach();
  const observer = new MutationObserver(attach);
  observer.observe(content, { childList: true, subtree: true });
  return () => {
    observer.disconnect();
    controls.dispose();
    controls.host.remove();
  };
}

export function mount(context) {
  if (!context || !context.descriptor || typeof context.registerDomMount !== "function") {
    throw new TypeError("Nero Extension SDK must provide descriptor and registerDomMount.");
  }
  context.registerDomMount({
    id: EXTENSION_ID,
    mount(api) {
      if (document.getElementById("run")) return mountVlaControls(api);
      return mountCollectorControls(api);
    },
  });
}
