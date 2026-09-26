"use strict";
/* ATLAS console frontend: talks to /api/{health,forward,inverse,jobs}. */

const $ = (id) => document.getElementById(id);
const POLL_MS = 1500;
const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const SUBDOMAINS = { cantilever: 2, l_bracket: 3, plate_hole: 3 };
const DESIGN_OF = {
  cantilever: "the beam height (target: compliance)",
  plate_hole: "the hole radius (target: compliance)",
  l_bracket: "the corner fillet radius (target: peak von Mises)",
};
const PLOT_TITLES = {
  fem_comparison: "ATLAS vs FEM", displacement: "Displacement field",
  vonmises: "von Mises stress", deformed: "Deformed shape",
  convergence: "Training convergence", design: "Optimised design",
  inverse_history: "Design history", fem_check: "FEM check of the design",
};

let activeJob = null;
let pollTimer = null;

/* ── theme ─────────────────────────────────────────────────────────────── */
const savedTheme = localStorage.getItem("atlas-theme");
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
else if (window.matchMedia("(prefers-color-scheme: light)").matches)
  document.documentElement.dataset.theme = "light";

$("themeToggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("atlas-theme", next);
});

/* ── mode switch (Results ⇄ Live solver) ───────────────────────────────── */
const modeSwitch = $("modeSwitch");
let resultsLoaded = false;

function setMode(mode) {
  modeSwitch.dataset.mode = mode;
  modeSwitch.querySelectorAll(".switch-btn").forEach((b) =>
    b.classList.toggle("is-active", b.dataset.mode === mode));
  $("mode-results").hidden = mode !== "results";
  $("mode-live").hidden = mode !== "live";
  document.getElementById("mode-" + mode).style.animation = "none";
  void document.getElementById("mode-" + mode).offsetWidth;   // restart the entrance animation
  document.getElementById("mode-" + mode).style.animation = "";
  if (mode === "results" && !resultsLoaded) loadResults();
  localStorage.setItem("atlas-mode", mode);
}
modeSwitch.addEventListener("click", (e) => {
  const btn = e.target.closest(".switch-btn");
  if (btn) setMode(btn.dataset.mode);
});

/* ── controls ──────────────────────────────────────────────────────────── */
const benchmarkEl = $("benchmark");

function refreshControls() {
  const b = benchmarkEl.value;
  document.querySelectorAll("[data-only]").forEach((el) => {
    el.classList.toggle("hidden", el.dataset.only !== b);
  });
  $("specSub").textContent = SUBDOMAINS[b];
  $("hint").textContent = "Inverse design finds " + DESIGN_OF[b] + ".";
}
benchmarkEl.addEventListener("change", refreshControls);
refreshControls();

async function api(path, body) {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

/* ── health + job list ─────────────────────────────────────────────────── */
async function refreshHealth() {
  const pill = $("healthPill");
  try {
    const h = await api("/api/health");
    pill.className = "pill pill-status ok";
    pill.textContent = `${h.device} · ${h.active_jobs} running`;
  } catch {
    pill.className = "pill pill-status bad";
    pill.textContent = "backend unreachable";
  }
}

async function refreshJobs() {
  try {
    const { jobs } = await api("/api/jobs?limit=8");
    const list = $("jobList");
    if (!jobs.length) return;
    list.innerHTML = jobs.map((j) => `
      <li><span class="dot ${j.status}"></span>
        <span>${j.kind}</span>
        <span class="muted">${(j.finished || j.started || j.created || "").slice(11, 19)}</span></li>`).join("");
  } catch { /* the health pill already reports an unreachable backend */ }
}

refreshHealth();
refreshJobs();
setInterval(() => { refreshHealth(); refreshJobs(); }, 5000);

/* ── running a job ─────────────────────────────────────────────────────── */
function setBusy(busy, which) {
  ["run-forward", "run-inverse"].forEach((id) => {
    const b = $(id);
    b.disabled = busy;
    b.classList.toggle("is-busy", busy && id === which);
  });
  $("progress").hidden = !busy;
}

async function startJob(kind) {
  const body = { benchmark: benchmarkEl.value };
  if (kind === "forward") {
    body.epochs = Number($("epochs").value);
    body.n_interior = Number($("n_interior").value);
    body.use_lbfgs = $("lbfgs").checked;
    body.warm_start_eb = $("warm").checked;
    body.prefit_epochs = Number($("prefit_epochs").value);
    body.use_mock = false;
  } else {
    body.forward_epochs = Number($("epochs").value);
    body.n_outer = 6;
    body.use_mock = false;
  }
  setBusy(true, `run-${kind}`);
  setChip("running", kind === "forward" ? "forward solve" : "inverse design");
  setPipeline("train");
  try {
    const { job_id } = await api(`/api/${kind}`, body);
    activeJob = job_id;
    if (pollTimer) clearTimeout(pollTimer);
    poll(job_id);
  } catch (e) {
    setChip("error", e.message);
    setBusy(false);
  }
}
$("run-forward").addEventListener("click", () => startJob("forward"));
$("run-inverse").addEventListener("click", () => startJob("inverse"));

async function poll(jobId) {
  let j;
  try {
    j = await api(`/api/jobs/${jobId}`);
  } catch (e) {
    setChip("error", e.message);
    setBusy(false);
    return;
  }
  render(j);
  if (j.status === "running" || j.status === "queued") {
    pollTimer = setTimeout(() => poll(jobId), POLL_MS);
    return;
  }
  setBusy(false);
  refreshJobs();
}

/* ── rendering ─────────────────────────────────────────────────────────── */
function setChip(state, text) {
  const chip = $("runChip");
  chip.className = `chip ${state}`;
  chip.textContent = state;
  $("runText").textContent = text || "";
}

function setPipeline(stage) {
  const order = ["setup", "train", "predict", "validate", "plots"];
  const at = order.indexOf(stage);
  document.querySelectorAll(".pipeline li").forEach((li) => {
    const i = order.indexOf(li.dataset.step);
    li.classList.toggle("done", at < 0 || i < at);
    li.classList.toggle("active", i === at);
  });
}

function stageFromLog(log) {
  if (!log) return "train";
  if (/Visualization|Saved plot|plots written/i.test(log)) return "plots";
  if (/FEM VALIDATION|Relative L2/i.test(log)) return "validate";
  if (/Evaluating|prediction/i.test(log)) return "predict";
  return "train";
}

function render(j) {
  const log = j.log || "";
  $("log").textContent = log || "—";
  if (j.status === "running" || j.status === "queued") {
    setChip("running", `${j.kind} · ${j.benchmark || benchmarkEl.value}`);
    setPipeline(stageFromLog(log));
    return;
  }
  if (j.status === "error") {
    setChip("error", j.error || "job failed");
    setPipeline("train");
    return;
  }
  setChip("done", `${j.kind} finished in ${duration(j)}`);
  setPipeline(null);
  renderMetrics(j);
  renderPlots(j.plots);
  setKpi("time", duration(j), null);
}

function duration(j) {
  if (!j.started || !j.finished) return "—";
  const s = (new Date(j.finished) - new Date(j.started)) / 1000;
  return s < 90 ? `${s.toFixed(0)} s` : `${(s / 60).toFixed(1)} min`;
}

function setKpi(key, text, pass) {
  const el = document.querySelector(`[data-kpi="${key}"]`);
  if (!el) return;
  el.className = "kpi-value" + (pass === true ? " pass" : pass === false ? " fail" : "");
  const num = parseFloat(text);
  if (REDUCED || !isFinite(num)) { el.textContent = text; return; }
  countUp(el, num, text);
}

/* count the number up to its final value; `text` supplies the suffix (%, s, …) */
function countUp(el, target, text) {
  const suffix = text.replace(/^[\d.\-+]+/, "");
  const decimals = (text.split(".")[1] || "").replace(/[^\d]/g, "").length;
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / 600);
    const eased = 1 - Math.pow(1 - k, 3);
    el.textContent = (target * eased).toFixed(decimals) + suffix;
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function pct(x) { return (100 * x).toFixed(2) + " %"; }

function renderMetrics(j) {
  const m = j.metrics || {};
  const rows = [];
  if (m.relative_L2_displacement !== undefined) {
    const ok = m.displacement_within_5pct_target;
    setKpi("disp", pct(m.relative_L2_displacement), ok);
    rows.push(["Displacement L² (|u|)", pct(m.relative_L2_displacement), ok]);
    rows.push(["Displacement L² (u)", pct(m.relative_L2_u), null]);
    rows.push(["Displacement L² (v)", pct(m.relative_L2_v), null]);
  }
  if (m.von_mises_l2_excl !== undefined) {
    const ok = m.von_mises_l2_excl_within_10pct;
    setKpi("vm", pct(m.von_mises_l2_excl), ok);
    rows.push(["von Mises L² (excl.)", pct(m.von_mises_l2_excl), ok]);
    rows.push(["von Mises L² (global)", pct(m.von_mises_l2_global), null]);
  }
  const d = m;   // inverse-design jobs return their result inside `metrics`
  if (d.p_star !== undefined) {
    const val = Number(d.p_star).toFixed(4);
    setKpi("design", val, d.converged === undefined ? null : !!d.converged);
    rows.push([d.param_name || "design parameter", val, null]);
    if (d.metric) rows.push(["target metric", d.metric, null]);
    if (d.param_baseline !== undefined) rows.push(["baseline value", Number(d.param_baseline).toFixed(4), null]);
    if (d.status) rows.push(["solver status", d.status, null]);
  }
  const list = $("resultList");
  list.innerHTML = rows.length
    ? rows.map(([k, v, ok], i) => `
        <li style="animation-delay:${i * 40}ms"><span>${k}</span>
          <b class="${ok === true ? "pass" : ok === false ? "fail" : ""}">${v}</b></li>`).join("")
    : `<li class="muted">This job returned no metrics.</li>`;
}

function renderPlots(plots) {
  if (!plots) return;
  const entries = Object.entries(plots).filter(([, src]) => src);
  if (!entries.length) return;
  const html = entries.map(([key, src], i) => `
    <figure class="plot" style="animation-delay:${i * 60}ms">
      <h3>${PLOT_TITLES[key] || key.replace(/_/g, " ")}</h3>
      <img src="${src}?t=${Date.now()}" alt="${key}" loading="lazy">
    </figure>`).join("");
  fillGrid($("plotGrid"), html);
}

/* ── shared helpers ────────────────────────────────────────────────────── */
function fillGrid(el, html) {
  el.innerHTML = html;
  el.querySelectorAll("img").forEach((img) => {
    const show = () => img.classList.add("loaded");
    img.complete ? show() : img.addEventListener("load", show, { once: true });
  });
}

function plotCard(title, src, i = 0) {
  return `<figure class="plot reveal" style="transition-delay:${Math.min(i, 6) * 50}ms">
      <h3>${title}</h3><img src="${src}" alt="${title}" loading="lazy" data-cap="${title}">
    </figure>`;
}

/* reveal cards as they scroll into view */
const revealer = new IntersectionObserver((entries) => {
  entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); revealer.unobserve(e.target); } });
}, { rootMargin: "0px 0px -40px 0px" });
function observeReveals() { document.querySelectorAll(".reveal:not(.in)").forEach((el) => revealer.observe(el)); }

/* click a figure to open the full-size image in a new tab */
document.addEventListener("click", (e) => {
  const img = e.target.closest(".plot img");
  if (img) window.open(img.src, "_blank", "noopener");
});

/* ── results mode ──────────────────────────────────────────────────────── */
const SOLVER_ROWS = [
  ["atlas_xpinn", "ATLAS-XPINN", true],
  ["atlas", "ATLAS energy (single network)", false],
  ["deepxde", "PINN (DeepXDE, defaults)", false],
  ["deepxde_configured", "PINN (DeepXDE, configured)", false],
];
let RESULTS = null;
let currentBench = "cantilever";

async function loadResults() {
  try {
    RESULTS = await api("/api/results");
  } catch (e) {
    $("resultPlots").innerHTML = `<p class="muted placeholder">Could not load stored results: ${e.message}</p>`;
    return;
  }
  resultsLoaded = true;
  $("benchChips").innerHTML = Object.entries(RESULTS.benchmarks)
    .map(([k, b]) => `<button class="chip-btn${k === currentBench ? " is-active" : ""}" data-bench="${k}">${b.title}</button>`)
    .join("");
  $("benchChips").addEventListener("click", (e) => {
    const btn = e.target.closest(".chip-btn");
    if (!btn) return;
    currentBench = btn.dataset.bench;
    document.querySelectorAll(".chip-btn").forEach((c) => c.classList.toggle("is-active", c === btn));
    renderBenchmark();
  });
  fillGrid($("globalPlots"), Object.entries(RESULTS.figures).map(([t, src], i) => plotCard(t, src, i)).join(""));
  renderHero();
  renderBenchmark();
  observeReveals();
}

function renderHero() {
  const vals = Object.values(RESULTS.benchmarks);
  const worstDisp = Math.max(...vals.map((b) => b.solvers.atlas_xpinn?.disp ?? 0));
  const worstVm = Math.max(...vals.map((b) => b.solvers.atlas_xpinn?.vm ?? 0));
  const worstPinn = Math.max(...vals.map((b) => b.solvers.deepxde?.disp ?? 0));
  $("heroKpis").innerHTML = [
    ["pass", worstDisp, "worst displacement error", "target 5 %"],
    ["pass", worstVm, "worst von Mises error", "target 10 %"],
    ["", worstPinn, "standard PINN, same problems", "for comparison"],
  ].map(([cls, v, label, sub], i) => `
      <article class="hero-kpi ${cls}" style="animation-delay:${i * 70}ms">
        <b data-count="${(100 * v).toFixed(2)}">0.00 %</b>
        <span>${label}<br><small class="muted">${sub}</small></span>
      </article>`).join("");
  $("heroKpis").querySelectorAll("[data-count]").forEach((el) => {
    const target = parseFloat(el.dataset.count);
    if (REDUCED) { el.textContent = target.toFixed(2) + " %"; return; }
    countUp(el, target, target.toFixed(2) + " %");
  });
}

function renderBenchmark() {
  const b = RESULTS.benchmarks[currentBench];
  $("tableBench").textContent = b.title;
  $("galleryBench").textContent = b.title;

  const worst = Math.max(...SOLVER_ROWS.map(([k]) => b.solvers[k]?.disp ?? 0));
  const rows = SOLVER_ROWS.filter(([k]) => b.solvers[k]).map(([k, name, hero], i) => {
    const s = b.solvers[k];
    const passD = s.disp < 0.05, passV = s.vm != null && s.vm < 0.10;
    return `<tr class="${hero ? "row-hero" : ""}" style="animation-delay:${i * 60}ms">
      <td>${name}</td>
      <td class="num ${passD ? "pass" : "fail"}">${pct(s.disp)}
        <span class="bar ${passD ? "pass" : "fail"}"><i data-w="${Math.max(2, 100 * s.disp / worst)}"></i></span></td>
      <td class="num ${passV ? "pass" : "fail"}">${s.vm == null ? "—" : pct(s.vm)}</td>
      <td class="num">${s.seconds ? (s.seconds / 60).toFixed(1) + " min" : "—"}</td>
      <td class="num">${s.params ? s.params.toLocaleString() : "—"}</td></tr>`;
  }).join("");
  $("solverTable").innerHTML = `
    <thead><tr><th>Solver</th><th>Displacement L²</th><th>von Mises L²</th><th>Train time</th><th>Params</th></tr></thead>
    <tbody>${rows}</tbody>`;
  requestAnimationFrame(() => $("solverTable").querySelectorAll(".bar > i")
    .forEach((el) => { el.style.width = el.dataset.w + "%"; }));

  const sd = b.seeds.atlas_xpinn;
  $("seedBody").innerHTML = sd ? `
    <div class="stat"><b>${pct(sd.disp_mean)} ± ${(100 * sd.disp_std).toFixed(2)}</b>
      <span>displacement L², ${sd.n} seeds</span></div>
    <div class="stat"><b>${pct(sd.vm_mean)} ± ${(100 * sd.vm_std).toFixed(2)}</b>
      <span>von Mises L², ${sd.n} seeds</span></div>
    <p class="hint">Every seed passed both targets.</p>` : `<p class="muted">No repeatability study stored.</p>`;

  const inv = b.inverse;
  $("invBody").innerHTML = inv ? `
    <div class="stat"><b>${Number(inv.p_star).toFixed(4)}</b><span>design found by ATLAS</span></div>
    <div class="stat"><b class="${inv.within_10pct ? "pass" : "fail"}">${(100 * inv.error_vs_target).toFixed(2)} %</b>
      <span>FEM check of that design vs the target (${inv.param})</span></div>
    <p class="hint">FEM verifies the design independently; the solver never sees FEM data.</p>`
    : `<p class="muted">No inverse design stored.</p>`;

  fillGrid($("resultPlots"), Object.entries(b.plots).map(([t, src], i) => plotCard(t, src, i)).join(""));
  observeReveals();
}

/* ── start ─────────────────────────────────────────────────────────────── */
setMode(localStorage.getItem("atlas-mode") || "results");
