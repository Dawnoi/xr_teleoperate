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

function paragraph(container) {
  const element = document.createElement("p");
  container.append(element);
  return element;
}

export function mount(context) {
  if (!context || !context.descriptor || typeof context.registerPanel !== "function") {
    throw new TypeError("Nero Extension SDK must provide descriptor and registerPanel.");
  }

  context.registerPanel({
    id: EXTENSION_ID,
    title: context.descriptor.title,
    mount(container, api) {
      if (!api || typeof api.snapshot !== "function" || typeof api.subscribe !== "function" || typeof api.invoke !== "function") {
        throw new TypeError("Nero Extension panel API must provide snapshot, subscribe, and invoke.");
      }
      let latestSnapshot = api.snapshot();
      let pendingAction = "";
      let actionError = "";
      let profileSignature = "";

      const authority = paragraph(container);
      const headReference = paragraph(container);
      const teleop = paragraph(container);
      const recording = paragraph(container);
      const inferenceSection = document.createElement("section");
      inferenceSection.className = "extension-inference";
      const inferenceHeading = document.createElement("h3");
      inferenceHeading.textContent = "XR 在线推理任务";
      const inferenceHint = paragraph(inferenceSection);
      inferenceHint.textContent = "Prompt 与协议只在本次启动请求中传入；控制循环完成校验后才会接管控制权。";
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
      const profileStatus = paragraph(inferenceSection);
      const inferenceControls = document.createElement("p");
      inferenceControls.className = "row";
      const startInference = document.createElement("button");
      startInference.type = "button";
      startInference.addEventListener("click", () => invoke(ACTIONS.startInference, {
        prompt: promptDraft.trim(),
        protocol_profile: selectedProfile,
      }));
      const stopInference = document.createElement("button");
      stopInference.type = "button";
      stopInference.addEventListener("click", () => invoke(ACTIONS.stopInference));
      inferenceControls.append(startInference, stopInference);
      const effectiveInference = paragraph(inferenceSection);
      inferenceSection.append(inferenceHeading, promptLabel, profileLabel, profileStatus, inferenceControls, effectiveInference);
      container.append(inferenceSection);

      const controls = document.createElement("p");
      controls.className = "row";
      const resetHeadReference = document.createElement("button");
      resetHeadReference.type = "button";
      resetHeadReference.addEventListener("click", () => invoke(ACTIONS.resetHeadReference));
      const hold = document.createElement("button");
      hold.type = "button";
      hold.addEventListener("click", () => invoke(ACTIONS.hold));
      const xr = document.createElement("button");
      xr.type = "button";
      xr.addEventListener("click", () => invoke(ACTIONS.xr));
      const home = document.createElement("button");
      home.type = "button";
      home.addEventListener("click", () => invoke(ACTIONS.home));
      const cancelRecording = document.createElement("button");
      cancelRecording.type = "button";
      cancelRecording.addEventListener("click", () => invoke(ACTIONS.cancelRecording));
      controls.append(resetHeadReference, hold, xr, home, cancelRecording);
      container.append(controls);
      const error = paragraph(container);
      error.role = "alert";

      function invoke(action, extra = {}) {
        pendingAction = action;
        actionError = "";
        update();
        api.invoke(action, extra).then(
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
        if (!profiles.some((profile) => profile.id === selectedProfile)) {
          selectedProfile = profiles[0]?.id || "";
        }
        profileInput.value = selectedProfile;
      }

      function setButton(button, action, label, disabled) {
        button.textContent = pendingAction === action ? `${label}请求中...` : label;
        button.disabled = pendingAction !== "" || disabled;
      }

      function update() {
        const state = stateFromSnapshot(latestSnapshot);
        const mode = latestSnapshot?.mode;
        const currentRecording = recordingFromSnapshot(latestSnapshot);
        const recordingActive = recordingIsActive(currentRecording);
        const teleopReady = state.teleop?.started === true && state.teleop?.ready === true && state.teleop?.stopping !== true;
        const authorityValue = String(state.authority ?? "unknown");
        const hasStateError = state.error !== undefined;
        authority.textContent = `当前控制权：${authorityLabel(authorityValue)}`;
        headReference.textContent = `头参考模式：${state.head_reference?.mode ?? "未知"}`;
        teleop.textContent = `遥操作状态：${teleopReady ? "已就绪" : "未就绪或正在停止"}`;
        recording.textContent = `录制状态：${mode === "collector" ? (recordingActive ? "录制中或正在收尾" : "未录制") : "VLA 页面不提供录制控制"}`;

        const profiles = profileEntries(state);
        updateProfiles(profiles);
        const selected = profiles.find((profile) => profile.id === selectedProfile);
        inferenceSection.hidden = mode !== "vla";
        profileInput.disabled = profiles.length === 0 || pendingAction !== "";
        if (!selected) {
          profileStatus.textContent = "推理协议状态尚未加载，不能发起启动请求。";
        } else if (selected.available !== true) {
          profileStatus.textContent = `当前不可用：${String(selected.reason || "XR 运行条件不满足")}`;
        } else {
          profileStatus.textContent = `已选择：${String(selected.label)}。${String(selected.reason || "")}`;
        }
        const inference = objectOrEmpty(state.inference);
        effectiveInference.textContent = `生效状态：${String(inference.state || "idle")} · ${String(inference.protocol_profile || "未启动")} · ${String(inference.prompt || "未提交 prompt")}`;
        const inferenceReady = state.teleop?.ready === true && state.teleop?.stopping !== true;
        setButton(
          startInference,
          ACTIONS.startInference,
          "启动推理",
          mode !== "vla" || !inferenceReady || recordingActive || promptDraft.trim().length === 0 || selected?.available !== true,
        );
        setButton(stopInference, ACTIONS.stopInference, "停止推理", mode !== "vla" || authorityValue !== "online_inference");

        setButton(resetHeadReference, ACTIONS.resetHeadReference, "重置头参考", hasStateError || !teleopReady || state.head_reference?.recenter_supported !== true);
        setButton(hold, ACTIONS.hold, "保持当前姿态", hasStateError || !teleopReady || recordingActive || authorityValue === "hold");
        setButton(xr, ACTIONS.xr, "恢复 XR 接管", hasStateError || !teleopReady || recordingActive || authorityValue === "xr" || authorityValue === "xr_live");
        setButton(home, ACTIONS.home, "回零", hasStateError || !teleopReady);
        setButton(cancelRecording, ACTIONS.cancelRecording, "取消当前录制", hasStateError || mode !== "collector" || !recordingActive);
        error.textContent = actionError === "" ? "" : `操作失败：${actionError}`;
        error.hidden = actionError === "";
      }

      update();
      return api.subscribe((snapshot) => {
        latestSnapshot = snapshot;
        update();
      });
    },
  });
}
