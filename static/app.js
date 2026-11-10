"use strict";

const $ = (selector) => document.querySelector(selector);
const failureNames = {
  "Machine failure": "Machine failure",
  TWF: "Tool wear failure",
  HDF: "Heat dissipation failure",
  PWF: "Power failure",
  OSF: "Overstrain failure",
  RNF: "Random failure"
};
let latestAnalysis = null;
let toastTimer = null;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[character]));
}

function showToast(message, isError = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 3900);
}

function updateClock() {
  $("#clock").textContent = new Intl.DateTimeFormat(undefined, {
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false
  }).format(new Date());
}

async function requestJson(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error("Cannot reach the local API. Start the FastAPI server and try again.");
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("The API returned an unreadable response.");
  }
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "Request failed.";
    throw new Error(detail);
  }
  return data;
}

function updateHealth(health) {
  const modelsReady = Boolean(health.models && health.models.ready);
  $("#connectionDot").classList.toggle("ready", modelsReady);
  $("#connectionDot").classList.toggle("error", !modelsReady);
  $("#connectionText").textContent = modelsReady ? "Prediction engine ready" : "Trained models missing";
  $("#engineBadge").textContent = modelsReady ? "AI ENGINE ONLINE" : "MODELS REQUIRED";
  $(".model-badge").classList.toggle("ready", modelsReady);
  $("#documentCount").textContent = String(health.documents.count);
  const vectorIndex = health.documents.index || {};
  $("#documentCaption").textContent = vectorIndex.ready
    ? `${vectorIndex.chunks} embedded passages`
    : "Local vector index unavailable";
  $("#vectorStatus").textContent = vectorIndex.ready
    ? `CHROMA READY · ${vectorIndex.chunks} CHUNKS · LOCAL EMBEDDINGS`
    : "CHROMA INDEX NOT READY";
  $(".vector-status-dot").classList.toggle("ready", Boolean(vectorIndex.ready));
  $("#llmStatus").textContent = health.llm.configured ? "ONLINE" : "OFFLINE";
  $("#llmCaption").textContent = health.llm.configured
    ? `${health.llm.provider} · ${health.llm.model}`
    : "Set GROQ_API_KEY in your local environment";
  $("#llmFoot").classList.toggle("green-dot", health.llm.configured);
  $("#llmFoot").classList.toggle("amber-dot", !health.llm.configured);
  renderDocuments(health.documents.items || []);
  if (!modelsReady) {
    $("#formError").textContent = "Run python train.py to generate the six trained models before analyzing sensor data.";
  }
}

async function loadHealth() {
  try {
    updateHealth(await requestJson("/api/health"));
  } catch (error) {
    $("#connectionDot").classList.add("error");
    $("#connectionText").textContent = "API unavailable";
    $("#engineBadge").textContent = "API OFFLINE";
    $("#formError").textContent = error.message;
  }
}

function getSensorValues() {
  const form = $("#sensorForm");
  if (!form.reportValidity()) return null;
  const data = new FormData(form);
  return {
    machine_type: String(data.get("machine_type")),
    air_temperature_k: Number(data.get("air_temperature_k")),
    process_temperature_k: Number(data.get("process_temperature_k")),
    rotational_speed_rpm: Number(data.get("rotational_speed_rpm")),
    torque_nm: Number(data.get("torque_nm")),
    tool_wear_min: Number(data.get("tool_wear_min"))
  };
}

function updateGauge(prediction) {
  const probability = Math.max(0, Math.min(1, prediction.machine_failure_probability));
  const percent = Math.round(probability * 100);
  $("#gaugeValue").textContent = `${percent}%`;
  $("#riskValue").textContent = `${percent}%`;
  const progress = $("#gaugeProgress");
  progress.style.strokeDashoffset = String(258 * (1 - probability));
  $("#gaugeNeedle").style.transform = `rotate(${-90 + probability * 180}deg)`;
  const primary = prediction.predictions.find((item) => item.target === "Machine failure");
  $("#thresholdValue").textContent = `${Math.round(primary.threshold * 100)}%`;
  const riskTag = $("#riskTag");
  riskTag.className = "risk-tag";
  if (primary.predicted) {
    const high = probability >= 0.65;
    riskTag.classList.add(high ? "high" : "elevated");
    riskTag.textContent = high ? "HIGH RISK" : "FAILURE FLAG";
    $("#riskCaption").textContent = "Model estimates elevated failure probability";
  } else {
    riskTag.classList.add("low");
    riskTag.textContent = "BELOW THRESHOLD";
    $("#riskCaption").textContent = probability >= 0.25
      ? "Below model threshold · continued monitoring advised"
      : "Below the selected model decision threshold";
  }
}

function renderModes(predictions) {
  $("#modeCount").textContent = `${predictions.length} TARGETS · MODEL SCORES`;
  $("#failureModes").innerHTML = predictions.map((item) => {
    const percent = Math.round(item.probability * 100);
    const active = item.predicted;
    return `<article class="mode-card">
      <div class="mode-card-top"><span class="mode-name">${escapeHtml(failureNames[item.target] || item.target)}</span>
      <span class="mode-status ${active ? "alert" : ""}">${active ? "FLAGGED" : "CLEAR"}</span></div>
      <div class="mode-number">${percent}<small>%</small></div>
      <div class="mode-track"><div class="mode-fill ${active ? "alert" : ""}" style="width:${percent}%"></div></div>
      <div class="mode-meta"><span>THRESHOLD ${Math.round(item.threshold * 100)}%</span><span>${escapeHtml(item.model)}</span></div>
    </article>`;
  }).join("");
}

function renderExplanations(explanations) {
  const targets = Object.keys(explanations || {});
  if (!targets.length) {
    $("#explanation").innerHTML = '<div class="empty-state compact">No feature explanations returned.</div>';
    return;
  }
  const defaultTarget = targets.includes("Machine failure") ? "Machine failure" : targets[0];
  $("#explanation").innerHTML = `
    <div class="explanation-toolbar"><span>Contributions for</span>
      <select id="explanationTarget" aria-label="Select target explanation">
        ${targets.map((target) => `<option value="${escapeHtml(target)}" ${target === defaultTarget ? "selected" : ""}>${escapeHtml(failureNames[target] || target)}</option>`).join("")}
      </select>
    </div><div id="shapRows"></div>
    <div class="explanation-note">SHAP explains the model output; it does not establish physical causation. Red bars increase the model output, green bars decrease it.</div>`;
  $("#explanationTarget").addEventListener("change", (event) => {
    renderShapRows(explanations[event.target.value] || []);
  });
  renderShapRows(explanations[defaultTarget] || []);
}

function renderShapRows(items) {
  if (!items.length) {
    $("#shapRows").innerHTML = '<div class="empty-state compact">No feature contribution values available.</div>';
    return;
  }
  const maximum = Math.max(...items.map((item) => Math.abs(item.shap_value)), 0.000001);
  $("#shapRows").innerHTML = `<div class="shap-list">${items.map((item) => {
    const positive = item.shap_value >= 0;
    const width = Math.max(3, (Math.abs(item.shap_value) / maximum) * 48);
    return `<div class="shap-row">
      <span class="shap-feature" title="${escapeHtml(item.feature)}">${escapeHtml(item.feature)}</span>
      <span class="shap-bar-track"><span class="shap-bar ${positive ? "positive" : "negative"}" style="width:${width}%"></span></span>
      <span class="shap-value">${item.shap_value > 0 ? "+" : ""}${item.shap_value.toFixed(3)}</span>
    </div>`;
  }).join("")}</div>`;
}

function markdownToSafeHtml(markdown) {
  const escaped = escapeHtml(markdown);
  return escaped.split(/\n{2,}/).map((paragraph) => {
    const formatted = paragraph
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>")
      .replace(/\n/g, "<br>");
    return `<p>${formatted}</p>`;
  }).join("");
}

function renderDiagnosis(result) {
  $("#diagnosisSection").classList.remove("hidden");
  const verdict = String(result.faithfulness.status || "not_verified").toLowerCase();
  const badge = $("#faithfulnessBadge");
  badge.className = "faithfulness-badge";
  if (verdict === "verified") badge.classList.add("verified");
  else if (verdict === "partially_verified") badge.classList.add("partial");
  else badge.classList.add("unverified");
  badge.textContent = verdict.replaceAll("_", " ").toUpperCase();

  const actions = Array.isArray(result.diagnosis.recommended_actions)
    ? result.diagnosis.recommended_actions
    : [];
  const gaps = Array.isArray(result.diagnosis.evidence_gaps)
    ? result.diagnosis.evidence_gaps
    : [];
  const supported = Array.isArray(result.faithfulness.supported_claims)
    ? result.faithfulness.supported_claims
    : [];
  const unsupported = Array.isArray(result.faithfulness.unsupported_claims)
    ? result.faithfulness.unsupported_claims
    : [];
  $("#diagnosisBody").innerHTML = `
    <div class="verification-note">AI decision support only. The faithfulness check is advisory, not a guarantee; confirm all procedures with qualified maintenance personnel.</div>
    <div class="diagnosis-text">${markdownToSafeHtml(result.diagnosis.diagnosis_markdown || "No diagnosis text was returned.")}</div>
    <div>
      <div class="action-list"><h4>RECOMMENDED NEXT STEPS</h4>
        ${actions.length ? `<ul>${actions.map((action) => `<li>${escapeHtml(action)}</li>`).join("")}</ul>` : '<span class="document-empty">No action list returned.</span>'}
      </div>
      ${gaps.length ? `<div class="evidence-gaps"><strong>Evidence gaps</strong><br>${gaps.map(escapeHtml).join("<br>")}</div>` : ""}
    </div>
    <div class="faithfulness-summary"><strong>${supported.length} supported claim(s)</strong> · <strong>${unsupported.length} unsupported claim(s)</strong> · ${escapeHtml(result.faithfulness.summary || "")}</div>`;

  $("#sourcesList").innerHTML = result.sources && result.sources.length
    ? `<div class="source-heading">RETRIEVED SOURCES · ${result.sources.length}</div>${result.sources.map((source) => `
      <article class="source-card"><strong>[${escapeHtml(source.id)}] ${escapeHtml(source.source)} · PAGE ${escapeHtml(source.page)}</strong>
      <p>${escapeHtml(source.text)}</p></article>`).join("")}`
    : '<div class="document-empty">No source passages retrieved. Upload a maintenance manual to ground guidance in evidence.</div>';
  $("#diagnosisSection").scrollIntoView({ behavior: "smooth", block: "start" });
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  if (busy) {
    button.dataset.original = button.innerHTML;
    button.innerHTML = `<span class="loading-spinner"></span> ${escapeHtml(label)}`;
  } else if (button.dataset.original) {
    button.innerHTML = button.dataset.original;
  }
}

async function runAnalysis(mode) {
  const sensor = getSensorValues();
  if (!sensor) return;
  const button = mode === "diagnose" ? $("#diagnoseButton") : $("#predictButton");
  const other = mode === "diagnose" ? $("#predictButton") : $("#diagnoseButton");
  $("#formError").textContent = "";
  setBusy(button, true, mode === "diagnose" ? "Retrieving & verifying" : "Calculating SHAP");
  other.disabled = true;
  try {
    const result = await requestJson(`/api/${mode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(sensor)
    });
    latestAnalysis = result;
    updateGauge(result);
    renderModes(result.predictions);
    renderExplanations(result.shap_explanations);
    if (mode === "diagnose") {
      renderDiagnosis(result);
      showToast("Analysis, document retrieval, and faithfulness check completed.");
    } else {
      $("#diagnosisSection").classList.add("hidden");
      showToast("Predictions and SHAP explanations updated.");
    }
  } catch (error) {
    $("#formError").textContent = error.message;
    showToast(error.message, true);
  } finally {
    setBusy(button, false);
    other.disabled = false;
  }
}

function renderDocuments(documents) {
  const list = $("#documentList");
  if (!documents.length) {
    list.innerHTML = '<div class="document-empty">The configured PDF was not found or could not be indexed. Check the server logs and project manual path.</div>';
    return;
  }
  list.innerHTML = documents.map((document) => `
    <div class="document-chip"><span class="doc-type">${escapeHtml(document.type)}</span>
      <span>${escapeHtml(document.filename)}</span>
      <span class="document-local">LOCAL · CHROMA</span>
    </div>`).join("");
}

async function replaceManual(file) {
  if (!file) return;
  const extension = file.name.split(".").pop().toLowerCase();
  if (!["pdf", "docx"].includes(extension)) {
    showToast("Choose a PDF or DOCX manual.", true);
    return;
  }
  if (file.size === 0 || file.size > 20 * 1024 * 1024) {
    showToast("Manual must be non-empty and no larger than 20 MB.", true);
    return;
  }

  const button = $("#manualUploadButton");
  button.disabled = true;
  const original = button.innerHTML;
  button.innerHTML = '<span class="loading-spinner"></span> Indexing manual…';
  const reader = new FileReader();
  reader.onerror = () => {
    showToast("Could not read the selected manual.", true);
    button.disabled = false;
    button.innerHTML = original;
  };
  reader.onload = async () => {
    try {
      const base64 = String(reader.result).split(",", 2)[1];
      const result = await requestJson("/api/manual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: file.name, content: base64 })
      });
      $("#documentCount").textContent = "1";
      $("#documentCaption").textContent = `${result.chunks} embedded passages`;
      $("#vectorStatus").textContent = `CHROMA READY · ${result.chunks} CHUNKS · LOCAL EMBEDDINGS`;
      $(".vector-status-dot").classList.add("ready");
      renderDocuments([{ filename: result.filename, type: extension }]);
      const message = result.cleanup_warning || (result.replaced
        ? `${result.filename} indexed. The previous manual was replaced.`
        : `${result.filename} is already the active manual.`);
      showToast(message, Boolean(result.cleanup_warning));
    } catch (error) {
      showToast(error.message, true);
    } finally {
      $("#manualInput").value = "";
      button.disabled = false;
      button.innerHTML = original;
    }
  };
  reader.readAsDataURL(file);
}

$("#predictButton").addEventListener("click", () => runAnalysis("predict"));
$("#diagnoseButton").addEventListener("click", () => runAnalysis("diagnose"));
$("#manualUploadButton").addEventListener("click", () => $("#manualInput").click());
$("#manualInput").addEventListener("change", (event) => replaceManual(event.target.files[0]));
$("#sensorForm").addEventListener("submit", (event) => event.preventDefault());
updateClock();
setInterval(updateClock, 1000);
loadHealth();
