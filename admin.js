"use strict";

const REFRESH_INTERVAL_MS = 10_000;
const ADMIN_KEY = "hackathon-secret";
const numberFormatter = new Intl.NumberFormat("en-US");
const timeFormatter = new Intl.DateTimeFormat(undefined, {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
});
const dateFormatter = new Intl.DateTimeFormat(undefined, {
  month: "short",
  day: "2-digit",
});

const refs = {
  connectionStatus: document.getElementById("connection-status"),
  connectionText: document.getElementById("connection-text"),
  lastUpdated: document.getElementById("last-updated"),
  refreshButton: document.getElementById("refresh-button"),
  refreshIcon: document.getElementById("refresh-icon"),
  errorBanner: document.getElementById("error-banner"),
  errorText: document.getElementById("error-text"),
  metricTotal: document.getElementById("metric-total"),
  metricTotalSub: document.getElementById("metric-total-sub"),
  metricPii: document.getElementById("metric-pii"),
  metricPiiSub: document.getElementById("metric-pii-sub"),
  metricBlocked: document.getElementById("metric-blocked"),
  metricBlockedSub: document.getElementById("metric-blocked-sub"),
  legendPassed: document.getElementById("legend-passed"),
  legendBlocked: document.getElementById("legend-blocked"),
  lockdownPanel: document.getElementById("lockdown-panel"),
  lockdownButton: document.getElementById("lockdown-button"),
  lockdownButtonLabel: document.getElementById("lockdown-button-label"),
  lockdownStatus: document.getElementById("lockdown-status"),
  lockdownDot: document.getElementById("lockdown-dot"),
  logBody: document.getElementById("threat-log-body"),
  logCount: document.getElementById("log-count"),
  footerCount: document.getElementById("footer-count"),
};

let decisionChart = null;
let piiChart = null;
let refreshing = false;
let lockdownActive = false;
let lockdownBusy = false;
let latestChartState = { passed: 0, blocked: 0, piiTypes: {} };

function toCount(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? Math.round(parsed) : 0;
}

function formatCount(value) {
  return numberFormatter.format(toCount(value));
}

function pluralize(count, singular, plural = `${singular}s`) {
  return count === 1 ? singular : plural;
}

function setConnection(state, message) {
  refs.connectionStatus.dataset.state = state;
  refs.connectionText.textContent = message;
}

function showError(message) {
  refs.errorText.textContent = message;
  refs.errorBanner.hidden = false;
  setConnection("error", "Audit stream degraded");
}

function hideError() {
  refs.errorBanner.hidden = true;
}

function renderLockdown(active) {
  lockdownActive = Boolean(active);
  const state = String(lockdownActive);
  refs.lockdownPanel.dataset.active = state;
  refs.lockdownButton.dataset.active = state;
  refs.lockdownButton.setAttribute("aria-pressed", state);
  refs.lockdownStatus.dataset.active = state;
  refs.lockdownDot.dataset.active = state;
  refs.lockdownButtonLabel.textContent = lockdownActive
    ? "DEACTIVATE LOCKDOWN"
    : "ACTIVATE LOCKDOWN";
  refs.lockdownStatus.textContent = lockdownActive
    ? "LOCKDOWN ACTIVE — all screened traffic blocked"
    : "Traffic flowing normally";
  document.body.classList.toggle("lockdown-active", lockdownActive);
}

async function toggleLockdown() {
  if (lockdownBusy) return;
  lockdownBusy = true;
  refs.lockdownButton.disabled = true;

  try {
    const response = await fetch("/api/lockdown", {
      method: "POST",
      cache: "no-store",
      headers: {
        Accept: "application/json",
        "X-Admin-Key": ADMIN_KEY,
      },
    });

    let data = {};
    try {
      data = await response.json();
    } catch (parseError) {
      data = {};
    }

    if (!response.ok) {
      throw new Error(data.error || `Lockdown request failed (HTTP ${response.status})`);
    }
    if (typeof data.lockdown !== "boolean") {
      throw new Error("Lockdown endpoint returned an invalid response");
    }

    renderLockdown(data.lockdown);
    hideError();
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown lockdown error";
    showError(`Lockdown control unavailable: ${message}`);
  } finally {
    lockdownBusy = false;
    refs.lockdownButton.disabled = false;
    void loadAnalytics({ silent: true });
  }
}

function flashValue(element) {
  element.classList.remove("value-flash");
  void element.offsetWidth;
  element.classList.add("value-flash");
}

function setMetric(element, value) {
  const next = formatCount(value);
  if (element.textContent !== next) {
    element.textContent = next;
    flashValue(element);
  }
}

function chartTooltip() {
  return {
    backgroundColor: "rgba(7, 13, 24, 0.96)",
    borderColor: "rgba(71, 85, 105, 0.65)",
    borderWidth: 1,
    titleColor: "#f8fafc",
    bodyColor: "#cbd5e1",
    padding: 11,
    cornerRadius: 8,
    displayColors: true,
    boxPadding: 4,
    titleFont: { family: "Inter", size: 11, weight: "600" },
    bodyFont: { family: "Inter", size: 11 },
  };
}
const centerLabelPlugin = {
  id: "centerLabel",
  beforeDatasetsDraw(chart) {
    const data = chart.data.datasets[0]?.data ?? [];
    const total = data.reduce((sum, value) => sum + Number(value || 0), 0);
    if (total > 0) return;

    const { ctx, chartArea } = chart;
    if (!chartArea) return;
    const x = (chartArea.left + chartArea.right) / 2;
    const y = (chartArea.top + chartArea.bottom) / 2;
    const radius = Math.min(
      chartArea.right - chartArea.left,
      chartArea.bottom - chartArea.top,
    ) / 2 - 23;

    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.strokeStyle = "#172033";
    ctx.lineWidth = 20;
    ctx.stroke();
    ctx.restore();
  },
  afterDraw(chart) {
    const data = chart.data.datasets[0]?.data ?? [];
    const total = data.reduce((sum, value) => sum + Number(value || 0), 0);
    const { ctx, chartArea } = chart;
    if (!chartArea) return;

    const x = (chartArea.left + chartArea.right) / 2;
    const y = (chartArea.top + chartArea.bottom) / 2;

    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#f8fafc";
    ctx.font = "600 29px Inter, sans-serif";
    ctx.fillText(numberFormatter.format(total), x, y - 7);
    ctx.fillStyle = "#64748b";
    ctx.font = "600 9px Inter, sans-serif";
    ctx.fillText("SCREENED", x, y + 17);
    ctx.restore();
  },
};

function createDecisionChart() {
  if (!window.Chart) return;
  const canvas = document.getElementById("decision-chart");
  if (!canvas) return;

  decisionChart = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: ["Passed", "Blocked"],
      datasets: [
        {
          data: [0, 0],
          backgroundColor: ["#34d399", "#fb7185"],
          hoverBackgroundColor: ["#6ee7b7", "#fda4af"],
          borderWidth: 0,
          borderRadius: 6,
          spacing: 3,
          hoverOffset: 5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "72%",
      animation: { duration: 650 },
      layout: { padding: 12 },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...chartTooltip(),
          callbacks: {
            label(context) {
              const values = context.dataset.data.map((value) => Number(value || 0));
              const total = values.reduce((sum, value) => sum + value, 0);
              const value = Number(context.raw || 0);
              const percentage = total ? Math.round((value / total) * 100) : 0;
              return ` ${context.label}: ${formatCount(value)} (${percentage}%)`;
            },
          },
        },
      },
    },
    plugins: [centerLabelPlugin],
  });
}

function createPiiChart() {
  if (!window.Chart) return;
  const canvas = document.getElementById("pii-chart");
  if (!canvas) return;

  piiChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels: ["SSN", "Email", "Credit Card"],
      datasets: [
        {
          label: "Entities redacted",
          data: [0, 0, 0],
          backgroundColor: [
            "rgba(56, 189, 248, 0.82)",
            "rgba(251, 191, 36, 0.82)",
            "rgba(167, 139, 250, 0.82)",
          ],
          hoverBackgroundColor: ["#7dd3fc", "#fcd34d", "#c4b5fd"],
          borderWidth: 0,
          borderRadius: 8,
          borderSkipped: false,
          maxBarThickness: 58,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 650 },
      layout: { padding: { top: 12, right: 8, left: 0 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...chartTooltip(),
          callbacks: {
            label(context) {
              const count = toCount(context.raw);
              return ` ${formatCount(count)} ${pluralize(count, "entity", "entities")} redacted`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          border: { display: false },
          ticks: {
            color: "#94a3b8",
            font: { family: "Inter", size: 11, weight: "500" },
            padding: 6,
          },
        },
        y: {
          beginAtZero: true,
          grace: "15%",
          border: { display: false },
          grid: { color: "rgba(148, 163, 184, 0.08)", drawTicks: false },
          ticks: {
            color: "#64748b",
            precision: 0,
            padding: 8,
            font: { family: "JetBrains Mono", size: 10 },
          },
        },
      },
    },
  });
}

function syncChartsWithState() {
  if (decisionChart) {
    decisionChart.data.datasets[0].data = [
      latestChartState.passed,
      latestChartState.blocked,
    ];
    decisionChart.update();
  }
  if (piiChart) {
    piiChart.data.datasets[0].data = [
      toCount(latestChartState.piiTypes.ssn),
      toCount(latestChartState.piiTypes.email),
      toCount(latestChartState.piiTypes.credit_card),
    ];
    piiChart.update();
  }
}

function updateCharts(passed, blocked, piiTypes) {
  latestChartState = {
    passed: toCount(passed),
    blocked: toCount(blocked),
    piiTypes: piiTypes && typeof piiTypes === "object" ? piiTypes : {},
  };
  syncChartsWithState();
}

const PII_ROWS = [
  ["ssn", "SSN", "audit-chip-ssn"],
  ["email", "Email", "audit-chip-email"],
  ["credit_card", "Card", "audit-chip-card"],
];

function createCell(className, content) {
  const cell = document.createElement("td");
  cell.className = className;
  if (content) cell.appendChild(content);
  return cell;
}

function createChip(label, variant = "") {
  const chip = document.createElement("span");
  chip.className = `audit-chip ${variant}`.trim();
  chip.textContent = label;
  return chip;
}

function createTriggerPill(word) {
  const pill = document.createElement("span");
  pill.className = "audit-trigger-pill";
  pill.textContent = word;
  pill.title = `Classifier trigger: ${word}`;
  pill.setAttribute("role", "listitem");
  return pill;
}

function createTriggerCell(event, blocked) {
  const wrap = document.createElement("div");
  wrap.className = "flex max-w-[250px] flex-wrap gap-1.5";

  if (!blocked) {
    const empty = document.createElement("span");
    empty.className = "audit-trigger-empty";
    empty.textContent = "-";
    empty.title = "No AI trigger words for this benign prompt";
    empty.setAttribute("aria-label", "No AI trigger words");
    wrap.appendChild(empty);
    return wrap;
  }

  const words = Array.isArray(event.suspicious_words)
    ? event.suspicious_words
      .map((word) => String(word).trim())
      .filter(Boolean)
      .slice(0, 5)
    : [];

  wrap.setAttribute("role", "list");
  if (words.length > 0) {
    wrap.setAttribute("aria-label", `AI trigger words: ${words.join(", ")}`);
    for (const word of words) {
      wrap.appendChild(createTriggerPill(word));
    }
  } else {
    const fallback = createTriggerPill("classifier signal");
    fallback.title = "Classifier block without a keyword attribution";
    wrap.appendChild(fallback);
  }
  return wrap;
}

function createSignalCell(event, blocked) {
  const wrap = document.createElement("div");
  wrap.className = "w-[160px]";

  if (!blocked) {
    wrap.appendChild(createChip("✓ ML cleared", "audit-chip-pass"));
    const note = document.createElement("p");
    note.className = "mt-1.5 font-mono text-[9px] text-slate-600";
    note.textContent = "No injection pattern";
    wrap.appendChild(note);
    return wrap;
  }

  const label = document.createElement("div");
  label.className = "flex items-center gap-1.5";
  const dot = document.createElement("span");
  dot.className = "h-1.5 w-1.5 rounded-full bg-rose-400 shadow-[0_0_7px_rgba(251,113,133,0.8)]";
  dot.setAttribute("aria-hidden", "true");
  const labelText = document.createElement("span");
  labelText.className = "font-mono text-[8px] font-semibold uppercase tracking-[0.14em] text-rose-300";
  labelText.textContent = "Injection detected";
  label.append(dot, labelText);
  wrap.appendChild(label);

  const confidence = Number(event.confidence);
  if (Number.isFinite(confidence)) {
    const percentage = Math.min(
      100,
      Math.max(0, confidence <= 1 ? confidence * 100 : confidence),
    );
    const meta = document.createElement("div");
    meta.className = "mt-1.5 flex items-baseline justify-between gap-2 font-mono text-[9px]";
    const metaLabel = document.createElement("span");
    metaLabel.className = "uppercase tracking-[0.08em] text-slate-600";
    metaLabel.textContent = "Confidence";
    const metaValue = document.createElement("span");
    metaValue.className = "text-rose-200";
    metaValue.textContent = `${percentage.toFixed(2)}%`;
    meta.append(metaLabel, metaValue);

    const track = document.createElement("div");
    track.className = "audit-confidence-track mt-1";
    track.setAttribute("aria-hidden", "true");
    const fill = document.createElement("div");
    fill.className = "audit-confidence-fill";
    fill.style.width = `${percentage}%`;
    track.appendChild(fill);

    wrap.append(meta, track);
  } else {
    const note = document.createElement("p");
    note.className = "mt-1.5 font-mono text-[9px] text-slate-600";
    note.textContent = "Classifier block";
    wrap.appendChild(note);
  }

  return wrap;
}

function createEmptyLogRow() {
  const row = document.createElement("tr");
  row.id = "threat-log-empty";

  const cell = document.createElement("td");
  cell.colSpan = 6;
  cell.className = "px-5 py-14";

  const wrapper = document.createElement("div");
  wrapper.className = "flex flex-col items-center justify-center text-center";

  const mark = document.createElement("span");
  mark.className = "mb-3 grid h-11 w-11 place-items-center rounded-xl border border-slate-800 bg-slate-900 text-slate-500";
  mark.textContent = "—";

  const title = document.createElement("p");
  title.className = "text-xs font-medium text-slate-400";
  title.textContent = "Waiting for security events";

  const hint = document.createElement("p");
  hint.className = "mt-1 text-[11px] text-slate-600";
  hint.textContent = "Send a prompt from Secure Chat and it will appear here.";

  wrapper.append(mark, title, hint);
  cell.appendChild(wrapper);
  row.appendChild(cell);
  return row;
}

function renderLog(events, totalEvents) {
  const items = Array.isArray(events) ? events : [];
  refs.logBody.replaceChildren();

  if (items.length === 0) {
    refs.logBody.appendChild(createEmptyLogRow());
    refs.logCount.textContent = "0 events";
    return;
  }

  const fragment = document.createDocumentFragment();

  for (const event of items) {
    const blocked = event.decision === "blocked";
    const row = document.createElement("tr");
    row.className = "audit-row";
    row.dataset.decision = blocked ? "blocked" : "passed";

    const timestamp = new Date(event.timestamp);
    const hasTimestamp = !Number.isNaN(timestamp.getTime());
    const timeWrap = document.createElement("div");
    const time = document.createElement("p");
    time.className = "font-mono text-[10px] text-slate-300";
    time.textContent = hasTimestamp ? timeFormatter.format(timestamp) : "—";
    const date = document.createElement("p");
    date.className = "mt-0.5 font-mono text-[9px] text-slate-600";
    date.textContent = hasTimestamp ? dateFormatter.format(timestamp) : "unknown";
    timeWrap.append(time, date);
    row.appendChild(createCell("px-5 py-3.5 align-top", timeWrap));

    const verdict = document.createElement("span");
    verdict.className = `audit-badge ${blocked ? "audit-badge-blocked" : "audit-badge-passed"}`;
    verdict.textContent = blocked ? "BLOCKED" : "PASSED";
    row.appendChild(createCell("px-4 py-3.5 align-top", verdict));

    const piiCounts = event.pii_counts && typeof event.pii_counts === "object"
      ? event.pii_counts
      : {};
    const piiWrap = document.createElement("div");
    piiWrap.className = "flex max-w-[170px] flex-wrap gap-1";
    let piiTotal = 0;
    for (const [key, label, variant] of PII_ROWS) {
      const count = toCount(piiCounts[key]);
      if (count > 0) {
        piiTotal += count;
        piiWrap.appendChild(createChip(`${label} ×${count}`, variant));
      }
    }
    if (piiTotal === 0) {
      const none = document.createElement("span");
      none.className = "text-[10px] text-slate-600";
      none.textContent = "—";
      piiWrap.appendChild(none);
    }
    row.appendChild(createCell("px-4 py-3.5 align-top", piiWrap));

    row.appendChild(
      createCell("px-4 py-3.5 align-top", createTriggerCell(event, blocked)),
    );
    row.appendChild(
      createCell("px-4 py-3.5 align-top", createSignalCell(event, blocked)),
    );

    const preview = String(event.preview || "").trim() || "No preview available";
    const previewText = document.createElement("p");
    previewText.className = "text-[11px] leading-relaxed text-slate-400";
    previewText.style.maxWidth = "380px";
    previewText.textContent = preview;
    previewText.title = preview;
    row.appendChild(createCell("px-5 py-3.5 align-top", previewText));

    fragment.appendChild(row);
  }

  refs.logBody.appendChild(fragment);
  const total = toCount(totalEvents);
  refs.logCount.textContent = total > items.length
    ? `${items.length} of ${formatCount(total)} events`
    : `${formatCount(items.length)} ${pluralize(items.length, "event")}`;
}

function renderAnalytics(data) {
  if (typeof data.lockdown === "boolean" && !lockdownBusy) {
    renderLockdown(data.lockdown);
  }

  const total = toCount(data.total_requests);
  const blocked = toCount(data.blocked ?? data.injection_blocked);
  const passed = data.passed == null
    ? Math.max(total - blocked, 0)
    : toCount(data.passed);
  const piiTypes = data.pii_types && typeof data.pii_types === "object"
    ? data.pii_types
    : {};
  const piiTotal = data.pii_redacted == null
    ? Object.values(piiTypes).reduce((sum, value) => sum + toCount(value), 0)
    : toCount(data.pii_redacted);
  const piiRequests = toCount(data.pii_requests);
  const computedRate = total ? (blocked / total) * 100 : 0;
  const blockRate = Number.isFinite(Number(data.block_rate))
    ? Number(data.block_rate)
    : computedRate;

  setMetric(refs.metricTotal, total);
  setMetric(refs.metricPii, piiTotal);
  setMetric(refs.metricBlocked, blocked);

  refs.metricTotalSub.textContent =
    `${formatCount(passed)} passed · ${formatCount(blocked)} blocked`;
  refs.metricPiiSub.textContent = piiTotal === 0
    ? "No sensitive entities found"
    : `Across ${formatCount(piiRequests)} ${pluralize(piiRequests, "prompt")}`;
  refs.metricBlockedSub.textContent = `${blockRate.toFixed(1)}% of screened traffic`;

  refs.legendPassed.textContent = formatCount(passed);
  refs.legendBlocked.textContent = formatCount(blocked);
  refs.footerCount.textContent = formatCount(total);

  updateCharts(passed, blocked, piiTypes);
  renderLog(data.recent_events, total);

  const now = new Date();
  const generatedRaw = new Date(data.generated_at || now);
  const generated = Number.isNaN(generatedRaw.getTime()) ? now : generatedRaw;
  refs.lastUpdated.textContent = timeFormatter.format(generated);
  refs.lastUpdated.title = generated.toISOString();
}

let chartLoadFailed = false;

async function loadAnalytics({ silent = false } = {}) {
  if (refreshing) return;
  refreshing = true;
  refs.refreshButton.disabled = true;
  refs.refreshIcon.classList.add("spin");
  if (!silent) setConnection("loading", "Syncing audit stream");

  try {
    const response = await fetch("/api/analytics", {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`Analytics request failed (HTTP ${response.status})`);
    }
    const data = await response.json();
    if (!data || typeof data !== "object") {
      throw new Error("Analytics returned an invalid response");
    }

    renderAnalytics(data);
    setConnection("live", "Audit stream live");
    if (chartLoadFailed) {
      showError("Chart.js could not be loaded. Metrics and logs are still current.");
    } else {
      hideError();
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown analytics error";
    showError(message);
  } finally {
    refreshing = false;
    refs.refreshButton.disabled = false;
    refs.refreshIcon.classList.remove("spin");
  }
}

function initializeCharts() {
  if (!window.Chart) return false;
  if (decisionChart && piiChart) return true;

  Chart.defaults.color = "#94a3b8";
  Chart.defaults.font.family = "Inter, ui-sans-serif, system-ui, sans-serif";
  Chart.defaults.borderColor = "rgba(148, 163, 184, 0.08)";
  if (!decisionChart) createDecisionChart();
  if (!piiChart) createPiiChart();
  syncChartsWithState();
  chartLoadFailed = false;
  return Boolean(decisionChart && piiChart);
}

function waitForCharts() {
  let attempts = 0;
  const timer = window.setInterval(() => {
    if (initializeCharts()) {
      window.clearInterval(timer);
      return;
    }
    attempts += 1;
    if (attempts >= 60) {
      window.clearInterval(timer);
      chartLoadFailed = true;
      showError("Chart.js could not be loaded. Metrics and logs are still current.");
    }
  }, 100);
}

function init() {
  if (!initializeCharts()) waitForCharts();

  refs.refreshButton.addEventListener("click", () => {
    void loadAnalytics();
  });

  refs.lockdownButton.addEventListener("click", () => {
    void toggleLockdown();
  });

  renderLockdown(lockdownActive);
  void loadAnalytics();
  window.setInterval(() => {
    if (!document.hidden) void loadAnalytics({ silent: true });
  }, REFRESH_INTERVAL_MS);

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) void loadAnalytics({ silent: true });
  });
}

init();

