const commandRoot = document.getElementById("commands");
const previewRoot = document.getElementById("previews");
const snapshotRoot = document.getElementById("snapshot");
const runtimeState = document.getElementById("runtime-state");
const backendName = document.getElementById("backend-name");
const commandTemplate = document.getElementById("command-template");

let capabilitiesSignature = "";

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.message || payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function inputFor(field) {
  const input = document.createElement("input");
  input.name = field;
  input.placeholder = field;
  input.required = true;
  return input;
}

function renderCommands(capabilities) {
  commandRoot.replaceChildren();
  for (const capability of capabilities.commands) {
    const form = commandTemplate.content.firstElementChild.cloneNode(true);
    const label = form.querySelector("label");
    const button = form.querySelector("button");
    const fields = form.querySelector(".fields");
    const result = form.querySelector(".command-result");
    label.textContent = capability.type;
    button.textContent = "执行";
    button.disabled = !capability.enabled;
    result.textContent = capability.enabled ? "" : capability.disabled_reason;
    const required = capability.parameter_schema?.required || [];
    for (const field of required) fields.append(inputFor(field));
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const payload = Object.fromEntries(new FormData(form).entries());
      button.disabled = true;
      result.textContent = "提交中";
      api("/api/v1/intents", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: crypto.randomUUID(), type: capability.type, payload }),
        })
        .then((command) => {
          updateCommandResult(command.command_id, result, button, capability.enabled);
        });
    });
    commandRoot.append(form);
  }
}

function updateCommandResult(commandId, result, button, enabled) {
  api(`/api/v1/commands/${encodeURIComponent(commandId)}`).then((outcome) => {
    result.textContent = outcome.message || outcome.status;
    if (outcome.status === "queued" || outcome.status === "running") {
      window.setTimeout(() => updateCommandResult(commandId, result, button, enabled), 150);
      return;
    }
    button.disabled = !enabled;
  });
}

function renderPreviews(cameras) {
  previewRoot.replaceChildren();
  const streams = cameras?.streams || [];
  for (const stream of streams) {
    const figure = document.createElement("figure");
    figure.className = "preview";
    const image = document.createElement("img");
    image.alt = stream.camera_name;
    image.src = `/api/v1/previews/${encodeURIComponent(stream.camera_name)}`;
    const caption = document.createElement("figcaption");
    caption.textContent = `${stream.camera_name}  ${stream.shared_age_ms ?? "-"} ms`;
    figure.append(image, caption);
    previewRoot.append(figure);
  }
}

function renderSnapshot(snapshot) {
  runtimeState.textContent = snapshot.runtime?.state || "unknown";
  snapshotRoot.textContent = JSON.stringify(snapshot, null, 2);
  renderPreviews(snapshot.cameras);
  api("/api/v1/capabilities").then((capabilities) => {
    const signature = JSON.stringify(capabilities.commands);
    if (signature !== capabilitiesSignature) {
      capabilitiesSignature = signature;
      renderCommands(capabilities);
    }
  });
}

async function boot() {
  const capabilities = await api("/api/v1/capabilities");
  backendName.textContent = capabilities.backend_name;
  capabilitiesSignature = JSON.stringify(capabilities.commands);
  renderCommands(capabilities);
  renderSnapshot(await api("/api/v1/snapshot"));
  const events = new EventSource("/api/v1/events");
  events.onmessage = (event) => renderSnapshot(JSON.parse(event.data));
}

boot();
