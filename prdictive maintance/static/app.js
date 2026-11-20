"use strict";

// Helper utility
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => document.querySelectorAll(selector);

// Clean failure titles
const failureDetails = {
  "Machine failure": { name: "Overall Machine Failure" },
  HDF: { name: "Heat Dissipation (HDF)" },
  PWF: { name: "Power Failure (PWF)" },
  OSF: { name: "Overstrain Failure (OSF)" },
  TWF: { name: "Tool Wear (TWF)" },
  RNF: { name: "Random Failure (RNF)" }
};

// Preset scenarios for instant testing and academic demonstrations
const presetScenarios = {
  normal: {
    name: "Normal Operation (Safe)",
    machine_type: "M",
    air_temperature_k: 298.5,
    process_temperature_k: 308.8,
    rotational_speed_rpm: 1550,
    torque_nm: 38.0,
    tool_wear_min: 45
  },
  hdf: {
    name: "Heat Dissipation (HDF) Risk",
    machine_type: "L",
    air_temperature_k: 302.5,
    process_temperature_k: 310.5,
    rotational_speed_rpm: 1320,
    torque_nm: 52.0,
    tool_wear_min: 110
  },
  pwf: {
    name: "Power Overload (PWF) Risk",
    machine_type: "L",
    air_temperature_k: 298.0,
    process_temperature_k: 308.0,
    rotational_speed_rpm: 2850,
    torque_nm: 65.0,
    tool_wear_min: 80
  },
  osf: {
    name: "Overstrain (OSF) Risk",
    machine_type: "L",
    air_temperature_k: 299.0,
    process_temperature_k: 309.0,
    rotational_speed_rpm: 1350,
    torque_nm: 68.0,
    tool_wear_min: 215
  },
  twf: {
    name: "Tool Wear (TWF) Warning",
    machine_type: "H",
    air_temperature_k: 298.0,
    process_temperature_k: 308.0,
    rotational_speed_rpm: 1500,
    torque_nm: 42.0,
    tool_wear_min: 245
  }
};

let latestAnalysis = null;
let toastTimer = null;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[character]));
}

// Format probability cleanly so near-zero values are recognizable
function formatProbability(prob) {
  const p = Math.max(0, Math.min(1, Number(prob) || 0));
  const percent = p * 100;
  if (percent === 0) return "0%";
  if (percent < 0.01) return "< 0.01%";
  if (percent < 0.1) return "< 0.1%";
  if (percent < 1.0) return percent.toFixed(2) + "%";
  if (percent < 10.0) return percent.toFixed(1) + "%";
  return Math.round(percent) + "%";
}

function showToast(message, isError = false) {
  const toast = $("#toast");
  const msgEl = $("#toastMessage");
  if (!toast || !msgEl) return;
  msgEl.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 3800);
}

// API client wrapper
async function requestJson(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error("Unable to connect to backend server.");
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("Received an unreadable response from the server.");
  }
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "Request execution failed.";
    throw new Error(detail);
  }
  return data;
}

// Navigation between views
function setupNavigation() {
  const navLinks = $$(".nav-item-link");
  const views = $$(".app-view");

  function switchView(targetId) {
    const cleanId = targetId.replace("#", "") || "overview";
    
    navLinks.forEach((link) => {
      const viewAttr = link.getAttribute("data-view");
      link.classList.toggle("active", viewAttr === cleanId);
    });

    views.forEach((view) => {
      const isTarget = view.id === `view-${cleanId}`;
      view.classList.toggle("active", isTarget);
    });

    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const targetView = link.getAttribute("data-view");
      window.location.hash = targetView;
      switchView(targetView);
    });
  });

  window.addEventListener("hashchange", () => {
    const currentHash = window.location.hash.replace("#", "") || "overview";
    switchView(currentHash);
  });

  if (window.location.hash) {
    switchView(window.location.hash.replace("#", ""));
  }
}

// Populate preset scenarios
function applyPreset(presetKey) {
  const preset = presetScenarios[presetKey];
  if (!preset) return;

  const form = $("#sensorForm");
  if (!form) return;

  $$(".preset-btn[data-scenario]").forEach((btn) => {
    btn.classList.toggle("active", btn.getAttribute("data-scenario") === presetKey);
  });

  form.elements["machine_type"].value = preset.machine_type;
  form.elements["air_temperature_k"].value = preset.air_temperature_k;
  form.elements["process_temperature_k"].value = preset.process_temperature_k;
  form.elements["rotational_speed_rpm"].value = preset.rotational_speed_rpm;
  form.elements["torque_nm"].value = preset.torque_nm;
  form.elements["tool_wear_min"].value = preset.tool_wear_min;

  syncTelemetryDisplays();
  showToast(`Loaded ${preset.name}. Click 'Predict Machine Health' to analyze.`);
}

function syncTelemetryDisplays() {
  const form = $("#sensorForm");
  if (!form) return;

  const air = $("#telemetryAirTemp");
  const proc = $("#telemetryProcTemp");
  const speed = $("#telemetrySpeed");
  const torque = $("#telemetryTorque");

  if (air) air.textContent = form.elements["air_temperature_k"].value;
  if (proc) proc.textContent = form.elements["process_temperature_k"].value;
  if (speed) speed.textContent = form.elements["rotational_speed_rpm"].value;
  if (torque) torque.textContent = form.elements["torque_nm"].value;
}

// Form reader
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

// System Health Update
function updateHealth(health) {
  const modelsReady = Boolean(health.models && health.models.ready);
  
  const connDot = $("#connectionDot");
  const connText = $("#connectionText");
  const badgeDot = $("#engineBadgeDot");
  const badgeText = $("#engineBadge");

  if (connDot) connDot.className = "status-dot " + (modelsReady ? "ready" : "error");
  if (connText) connText.textContent = modelsReady ? "Engine Ready" : "Models Missing";
  if (badgeDot) badgeDot.className = "status-dot " + (modelsReady ? "ready" : "error");
  if (badgeText) badgeText.textContent = modelsReady ? "ONLINE" : "REQUIRED";

  const docCount = $("#summaryDocCount");
  const docStatus = $("#summaryDocStatus");
  const vectorBadge = $("#vectorStatusBadge");
  const chunkCountEl = $("#knowledgeChunkCount");

  const docs = health.documents || {};
  const vectorIndex = docs.index || {};

  if (docCount) docCount.textContent = String(docs.count || 0);
  if (docStatus) {
    docStatus.className = "status-pill " + (vectorIndex.ready ? "normal" : "neutral");
    docStatus.textContent = vectorIndex.ready ? "Active" : "Offline";
  }
  if (vectorBadge) {
    vectorBadge.className = "status-pill " + (vectorIndex.ready ? "normal" : "caution");
    vectorBadge.textContent = vectorIndex.ready ? "Chroma DB Active" : "Not Ready";
  }
  if (chunkCountEl) {
    chunkCountEl.textContent = `${vectorIndex.chunks || 0} sections`;
  }

  const llmStatus = $("#summaryLlmStatus");
  const llmBadge = $("#summaryLlmBadge");
  const llm = health.llm || {};

  if (llmStatus) llmStatus.textContent = llm.configured ? "Ready" : "Offline";
  if (llmBadge) {
    llmBadge.className = "status-pill " + (llm.configured ? "normal" : "caution");
    llmBadge.textContent = llm.configured ? "Active" : "Key Needed";
  }

  renderDocuments(docs.items || []);

  if (!modelsReady) {
    const formErr = $("#formError");
    if (formErr) formErr.textContent = "Run python train.py first to generate model weights.";
  }
}

async function loadHealth() {
  try {
    const health = await requestJson("/api/health");
    updateHealth(health);
  } catch (error) {
    const connDot = $("#connectionDot");
    const connText = $("#connectionText");
    if (connDot) connDot.className = "status-dot error";
    if (connText) connText.textContent = "API Offline";
  }
}

// Render Results: Risk Gauge & Health Summary
function updateRiskGauge(prediction) {
  const probability = Math.max(0, Math.min(1, prediction.machine_failure_probability));
  const formattedProbability = formatProbability(probability);

  const sumVal = $("#summaryRiskVal");
  const sumBadge = $("#summaryRiskBadge");
  if (sumVal) sumVal.textContent = formattedProbability;

  const gaugeNumber = $("#gaugeNumber");
  const gaugeProgress = $("#gaugeProgress");
  const gaugeNeedle = $("#gaugeNeedle");
  const riskStatusPill = $("#riskStatusPill");
  const thresholdValEl = $("#thresholdValue");

  if (gaugeNumber) gaugeNumber.textContent = formattedProbability;
  
  if (gaugeProgress) {
    const fillRatio = Math.max(0.02, probability);
    gaugeProgress.style.strokeDashoffset = String(258 * (1 - fillRatio));
  }
  if (gaugeNeedle) {
    gaugeNeedle.style.transform = `rotate(${-90 + probability * 180}deg)`;
  }

  const primary = prediction.predictions.find((p) => p.target === "Machine failure") || {};
  const threshold = primary.threshold || 0.61;
  const thresholdPercent = Math.round(threshold * 100);

  if (thresholdValEl) thresholdValEl.textContent = `${thresholdPercent}%`;

  if (primary.predicted) {
    const isHigh = probability >= 0.65;
    const badgeText = isHigh ? "High Risk Flagged" : "Elevated Risk Flagged";
    const badgeClass = isHigh ? "status-pill alert" : "status-pill caution";

    if (sumBadge) {
      sumBadge.className = badgeClass;
      sumBadge.textContent = badgeText;
    }
    if (riskStatusPill) {
      riskStatusPill.className = badgeClass;
      riskStatusPill.textContent = badgeText;
    }
  } else {
    const isMonitor = probability >= 0.25;
    const badgeText = isMonitor ? "Monitor (Elevated)" : "Normal (Safe)";
    const badgeClass = isMonitor ? "status-pill caution" : "status-pill normal";

    if (sumBadge) {
      sumBadge.className = badgeClass;
      sumBadge.textContent = badgeText;
    }
    if (riskStatusPill) {
      riskStatusPill.className = badgeClass;
      riskStatusPill.textContent = badgeText;
    }
  }
}

// Render Failure Mode Cards (Clean & Streamlined)
function renderFailureModes(predictions) {
  const container = $("#failureModesContainer");
  if (!container) return;

  const countBadge = $("#failureTargetsCountBadge");
  if (countBadge) countBadge.textContent = `${predictions.length} TARGETS`;

  container.innerHTML = predictions.map((item) => {
    const formattedProb = formatProbability(item.probability);
    const thresholdPercent = Math.round(item.threshold * 100);
    const isAlert = item.predicted;
    const details = failureDetails[item.target] || { name: item.target };
    const barWidth = Math.max(item.probability > 0 ? 3 : 1, Math.min(100, item.probability * 100));

    return `
      <article class="failure-type-card ${isAlert ? "flagged" : ""}" style="padding: 16px;">
        <div class="failure-card-top" style="margin-bottom: 8px;">
          <h4 class="failure-card-title" style="font-size: 15px;">${escapeHtml(details.name)}</h4>
          <span class="status-pill ${isAlert ? "alert" : "normal"}">
            ${isAlert ? "FLAGGED" : "CLEAR"}
          </span>
        </div>

        <div class="failure-card-prob-row" style="margin-bottom: 8px;">
          <span class="failure-card-prob-val" style="font-size: 22px;">${formattedProb}</span>
          <span class="failure-card-threshold" style="font-size: 12px;">Cutoff: ${thresholdPercent}%</span>
        </div>

        <div class="failure-progress-track" style="margin-bottom: 0;">
          <div class="failure-progress-bar ${isAlert ? "alert" : ""}" style="width: ${barWidth}%;"></div>
        </div>
      </article>
    `;
  }).join("");
}

// Render SHAP Feature Contributions
function renderShapExplanations(explanations) {
  const container = $("#shapContainer");
  const targetSelect = $("#shapTargetSelect");
  if (!container || !targetSelect) return;

  const targets = Object.keys(explanations || {});
  if (!targets.length) {
    container.innerHTML = '<div class="empty-state-card">No feature explanation data.</div>';
    return;
  }

  const currentTarget = targetSelect.value || "Machine failure";
  renderShapRowsForTarget(explanations[currentTarget] || explanations[targets[0]] || []);

  targetSelect.onchange = (e) => {
    renderShapRowsForTarget(explanations[e.target.value] || []);
  };
}

function renderShapRowsForTarget(items) {
  const container = $("#shapContainer");
  if (!container) return;

  if (!items.length) {
    container.innerHTML = '<div class="empty-state-card">No records available.</div>';
    return;
  }

  const maxVal = Math.max(...items.map((it) => Math.abs(it.shap_value)), 0.0001);

  container.innerHTML = `
    <div class="shap-rows-container">
      ${items.map((item) => {
        const isPositive = item.shap_value >= 0;
        const widthPercent = Math.min(100, Math.max(8, (Math.abs(item.shap_value) / maxVal) * 100));
        const valFormatted = (item.shap_value > 0 ? "+" : "") + item.shap_value.toFixed(3);

        return `
          <div class="shap-feature-row">
            <span class="shap-feature-name">${escapeHtml(item.feature.replace(/_/g, " "))}</span>
            <div class="shap-bar-container">
              <div class="shap-bar-fill ${isPositive ? "positive" : "negative"}" style="width: ${widthPercent}%;"></div>
            </div>
            <span class="shap-val-text ${isPositive ? "positive" : "negative"}">${valFormatted}</span>
            <span class="shap-impact-badge ${isPositive ? "positive" : "negative"}">
              ${isPositive ? "Increases Risk" : "Lowers Risk"}
            </span>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

// Markdown parser with headers, lists, tags, and formatting
function formatMarkdown(text) {
  if (!text) return "<p>No text provided.</p>";
  
  // Escape raw HTML first
  let html = escapeHtml(text);

  // Headers (###, ##, #)
  html = html.replace(/^### (.*$)/gim, '<h4 style="color: #67e8f9; margin: 16px 0 8px; font-size: 16px; font-weight: 700;">$1</h4>');
  html = html.replace(/^## (.*$)/gim, '<h3 style="color: #a5f3fc; margin: 18px 0 10px; font-size: 18px; font-weight: 700;">$1</h3>');
  html = html.replace(/^# (.*$)/gim, '<h2 style="color: #e0f2fe; margin: 20px 0 12px; font-size: 20px; font-weight: 800;">$1</h2>');

  // Horizontal rules
  html = html.replace(/^---$/gim, '<hr style="border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 14px 0;">');

  // Source markers like [S1], [S2]
  html = html.replace(/\[(S\d+)\]/g, '<span style="display: inline-block; padding: 1px 6px; background-color: rgba(6,182,212,0.15); border: 1px solid rgba(6,182,212,0.4); border-radius: 4px; color: #38bdf8; font-family: monospace; font-size: 12px; font-weight: 600;">$1</span>');

  // Bold & Italic
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong style="color: #ffffff;">$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Lists (- item or * item)
  html = html.replace(/^\s*[-*]\s+(.*)$/gim, '<li style="margin-bottom: 6px; line-height: 1.55;">$1</li>');
  html = html.replace(/(<li.*<\/li>\n?)+/g, '<ul style="margin: 8px 0 14px; padding-left: 20px; list-style-type: disc;">$&</ul>');

  // Line breaks to paragraphs
  const paragraphs = html.split(/\n{2,}/).map(p => {
    p = p.trim();
    if (!p) return "";
    if (p.startsWith("<h") || p.startsWith("<ul") || p.startsWith("<hr")) return p;
    return `<p style="margin-bottom: 12px; line-height: 1.6;">${p.replace(/\n/g, "<br>")}</p>`;
  });

  return paragraphs.join("");
}

// Render AI Diagnosis
function renderAiDiagnosis(result) {
  const section = $("#aiDiagnosisSection");
  if (!section) return;

  section.style.display = "block";

  const faithfulness = result.faithfulness || {};
  const status = String(faithfulness.status || "not_verified").toLowerCase();

  const badge = $("#faithfulnessBadge");
  if (badge) {
    if (status === "verified") {
      badge.className = "status-pill normal";
      badge.textContent = "VERIFIED EVIDENCE";
    } else if (status === "partially_verified") {
      badge.className = "status-pill caution";
      badge.textContent = "PARTIALLY VERIFIED";
    } else {
      badge.className = "status-pill alert";
      badge.textContent = "UNVERIFIED";
    }
  }

  const bodyText = $("#diagnosisBodyText");
  if (bodyText) {
    const diag = result.diagnosis || {};
    bodyText.innerHTML = formatMarkdown(diag.diagnosis_markdown || "No diagnosis assessment.");
  }

  const actionsContainer = $("#actionsListContainer");
  if (actionsContainer) {
    const actions = (result.diagnosis && result.diagnosis.recommended_actions) || [];
    actionsContainer.innerHTML = actions.length
      ? `<ul>${actions.map((act) => `<li>${escapeHtml(act)}</li>`).join("")}</ul>`
      : '<p style="color: var(--text-muted); font-size: 13px;">No actions listed.</p>';
  }

  const gapsBox = $("#evidenceGapsBox");
  if (gapsBox) {
    const gaps = (result.diagnosis && result.diagnosis.evidence_gaps) || [];
    if (gaps.length) {
      gapsBox.style.display = "block";
      gapsBox.innerHTML = `<strong>Identified Documentation Gaps:</strong><br>${gaps.map(escapeHtml).join("<br>")}`;
    } else {
      gapsBox.style.display = "none";
    }
  }

  const summaryLine = $("#faithfulnessSummaryLine");
  if (summaryLine) {
    const supported = (faithfulness.supported_claims || []).length;
    const unsupported = (faithfulness.unsupported_claims || []).length;
    summaryLine.innerHTML = `<strong>Faithfulness Check:</strong> ${supported} supported statements &middot; ${unsupported} sensor-specific statements &middot; ${escapeHtml(faithfulness.summary || "")}`;
  }

  const sourcesGrid = $("#retrievedSourcesGrid");
  if (sourcesGrid) {
    const sources = result.sources || [];
    if (sources.length) {
      sourcesGrid.innerHTML = sources.map((src) => `
        <div class="source-passage-card">
          <div class="source-card-header">
            <span>[${escapeHtml(src.id)}] ${escapeHtml(src.source)}</span>
            <span>Page ${escapeHtml(src.page)}</span>
          </div>
          <p class="source-card-text">${escapeHtml(src.text)}</p>
        </div>
      `).join("");
    } else {
      sourcesGrid.innerHTML = '<p style="color: var(--text-muted); font-size: 13px; grid-column: 1 / -1;">No sources retrieved.</p>';
    }
  }

  section.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Render Document Chips
function renderDocuments(documents) {
  const container = $("#documentChipsContainer");
  if (!container) return;

  if (!documents.length) {
    container.innerHTML = '<p style="color: var(--text-muted); font-size: 13px;">No documents indexed.</p>';
    return;
  }

  container.innerHTML = documents.map((doc) => `
    <div style="display: flex; justify-content: space-between; align-items: center; background-color: rgba(255, 255, 255, 0.04); border: 1px solid var(--border-subtle); padding: 10px 14px; border-radius: var(--radius-md);">
      <span style="font-size: 14px; font-weight: 600; color: var(--text-main);">${escapeHtml(doc.filename)}</span>
      <span class="status-pill normal">${escapeHtml(doc.type.toUpperCase())}</span>
    </div>
  `).join("");
}

// Button Busy State
function setButtonBusy(button, busy, busyText) {
  if (!button) return;
  button.disabled = busy;
  if (busy) {
    button.dataset.originalHtml = button.innerHTML;
    button.innerHTML = `<span class="spinner"></span> <span>${escapeHtml(busyText)}</span>`;
  } else if (button.dataset.originalHtml) {
    button.innerHTML = button.dataset.originalHtml;
  }
}

// Main execution caller
async function runAnalysis(mode) {
  const sensor = getSensorValues();
  if (!sensor) return;

  const predictBtn = $("#predictButton");
  const diagnoseBtn = $("#diagnoseButton");
  const targetBtn = mode === "diagnose" ? diagnoseBtn : predictBtn;
  const otherBtn = mode === "diagnose" ? predictBtn : diagnoseBtn;
  const resultsSec = $("#resultsSection");

  const formErr = $("#formError");
  if (formErr) formErr.textContent = "";

  setButtonBusy(targetBtn, true, mode === "diagnose" ? "Diagnosing..." : "Evaluating...");
  if (otherBtn) otherBtn.disabled = true;
  if (resultsSec) resultsSec.classList.add("results-loading");

  try {
    const result = await requestJson(`/api/${mode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(sensor)
    });

    latestAnalysis = result;
    updateRiskGauge(result);
    renderFailureModes(result.predictions);
    renderShapExplanations(result.shap_explanations);

    if (mode === "diagnose") {
      renderAiDiagnosis(result);
      showToast("AI diagnosis complete");
    } else {
      const diagSec = $("#aiDiagnosisSection");
      if (diagSec) diagSec.style.display = "none";
      showToast("Predictions updated");
    }
  } catch (error) {
    if (formErr) formErr.textContent = error.message;
    showToast(error.message, true);
  } finally {
    setButtonBusy(targetBtn, false);
    if (otherBtn) otherBtn.disabled = false;
    if (resultsSec) resultsSec.classList.remove("results-loading");
  }
}

// Manual Upload
async function handleManualUpload(file) {
  if (!file) return;

  const ext = file.name.split(".").pop().toLowerCase();
  if (!["pdf", "docx"].includes(ext)) {
    showToast("Please choose a valid PDF or DOCX file.", true);
    return;
  }

  if (file.size === 0 || file.size > 20 * 1024 * 1024) {
    showToast("Document must be under 20 MB.", true);
    return;
  }

  const uploadBtn = $("#manualUploadButton");
  setButtonBusy(uploadBtn, true, "Embedding...");

  const reader = new FileReader();
  reader.onerror = () => {
    showToast("Failed to read file.", true);
    setButtonBusy(uploadBtn, false);
  };

  reader.onload = async () => {
    try {
      const base64Data = String(reader.result).split(",", 2)[1];
      const result = await requestJson("/api/manual", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: file.name, content: base64Data })
      });

      const docCount = $("#summaryDocCount");
      const chunkCountEl = $("#knowledgeChunkCount");
      if (docCount) docCount.textContent = "1";
      if (chunkCountEl) chunkCountEl.textContent = `${result.chunks} sections`;

      renderDocuments([{ filename: result.filename, type: ext }]);
      showToast(result.replaced ? `${result.filename} indexed.` : `${result.filename} active.`);
    } catch (err) {
      showToast(err.message, true);
    } finally {
      $("#manualInput").value = "";
      setButtonBusy(uploadBtn, false);
    }
  };

  reader.readAsDataURL(file);
}

// Event Listeners Initialization
function initApp() {
  setupNavigation();

  const predictBtn = $("#predictButton");
  const diagnoseBtn = $("#diagnoseButton");
  if (predictBtn) predictBtn.addEventListener("click", () => runAnalysis("predict"));
  if (diagnoseBtn) diagnoseBtn.addEventListener("click", () => runAnalysis("diagnose"));

  const form = $("#sensorForm");
  if (form) {
    form.addEventListener("submit", (e) => e.preventDefault());
    form.addEventListener("input", () => {
      syncTelemetryDisplays();
      $$(".preset-btn[data-scenario]").forEach((btn) => btn.classList.remove("active"));
    });
  }

  $$(".preset-btn[data-scenario]").forEach((btn) => {
    btn.addEventListener("click", () => {
      applyPreset(btn.getAttribute("data-scenario"));
    });
  });

  const dropzone = $("#manualDropzone");
  const manualInput = $("#manualInput");
  const manualUploadBtn = $("#manualUploadButton");

  if (manualUploadBtn && manualInput) {
    manualUploadBtn.addEventListener("click", () => manualInput.click());
  }

  if (manualInput) {
    manualInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleManualUpload(e.target.files[0]);
      }
    });
  }

  if (dropzone) {
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "var(--teal-primary)";
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.style.borderColor = "";
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "";
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleManualUpload(e.dataTransfer.files[0]);
      }
    });
  }

  loadHealth();
  syncTelemetryDisplays();
  runAnalysis("predict");
}

document.addEventListener("DOMContentLoaded", initApp);
