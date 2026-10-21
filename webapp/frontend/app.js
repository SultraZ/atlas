"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch {} },
};
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
};

const pctNum = (x) => { const p = x * 100; return p < 0.1 ? p.toFixed(3) : p < 10 ? p.toFixed(2) : p.toFixed(1); };
const pct = (x) => (x == null ? "—" : `${pctNum(x)} %`);
const spct = (x) => (x == null ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(2)} %`);
const meanStd = (m, s) => { const p = m * 100; const d = p < 1 ? 3 : 2; return `${p.toFixed(d)} ± ${(s * 100).toFixed(d)} %`; };
const mins = (s) => (s == null ? "—" : `${(s / 60).toFixed(1)} min`);
const dur = (s) => { if (s == null) return "—"; const m = Math.floor(s / 60); const r = Math.round(s - 60 * m); return m ? `${m} min ${r} s` : `${r} s`; };
const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
const sci = (x) => { const n = Math.floor(Math.log10(Math.abs(x))); const exp = String(n).replace("-", "⁻").replace(/\d/g, (d) => SUP[+d]); return `${(x / 10 ** n).toFixed(2)} × 10${exp}`; };
const gpa = (e) => `${+(e / 1e9).toFixed(1)} GPa`;
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
const word = (n) => WORDS[n] ?? String(n);
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
const secondsBetween = (a, b) => (a && b ? (new Date(b) - new Date(a)) / 1000 : null);

const ICON = {
  pass: (s = 14) => `<svg class="icon" width="${s}" height="${s}" viewBox="0 0 16 16" role="img" aria-label="Meets target"><circle cx="8" cy="8" r="7.5" fill="var(--pass)"/><path d="M4.8 8.2l2.1 2.1 4.3-4.6" stroke="var(--on-icon)" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  fail: (s = 14) => `<svg class="icon" width="${s}" height="${s}" viewBox="0 0 16 16" role="img" aria-label="Misses target"><circle cx="8" cy="8" r="7.5" fill="var(--fail)"/><path d="M5.6 5.6l4.8 4.8M10.4 5.6l-4.8 4.8" stroke="var(--on-icon)" stroke-width="1.8" stroke-linecap="round"/></svg>`,
  mixed: (s = 14) => `<svg class="icon" width="${s}" height="${s}" viewBox="0 0 16 16" role="img" aria-label="Mixed result"><circle cx="8" cy="8" r="7.5" fill="var(--mixed)"/><path d="M4.8 8h6.4" stroke="var(--on-icon)" stroke-width="1.8" stroke-linecap="round"/></svg>`,
  check: `<svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true"><path d="M2.5 6.2l2.2 2.2 4.8-5" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  open: `<svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M6 3h7v7M13 3L4 12" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
};
const passIcon = (ok, s) => (ok ? ICON.pass(s) : ICON.fail(s));

const simple = () => document.documentElement.getAttribute("data-level") !== "expert";
const S = (plain, expert) => (simple() ? plain : expert);

const GLOSSARY = {
  displacement: ["Bending error (displacement)", "How far each point of the part moves under the load, compared with the trusted answer over the whole part. 0.25 % means the predicted movement is off by about 0.25 % overall."],
  vonmises: ["Stress error (von Mises)", "Von Mises stress rolls all the pulling, squeezing and shearing inside the material into one number. Engineers compare it with the material's strength: where it is highest, the part would start to give way first."],
  fem: ["FEM, the trusted answer", "The finite element method is the standard engineering simulation: it cuts the part into many tiny triangles and solves the physics on each. Here FEniCS, an open-source FEM program, gives the reference answer every method is scored against."],
  pinn: ["Standard PINN", "A physics-informed neural network: a network trained so its output obeys the equations of physics, with no example answers. The standard version is hard to train on problems like these; here it is off by 30 to 100 %."],
  xpinn: ["ATLAS-XPINN", "Splits the part into regions and gives each its own small network. The pieces are joined so they fit together at the borders. It helps most where two materials meet, because each material gets its own network."],
  energy: ["Energy method", "A loaded part settles into the shape that stores the least energy. ATLAS trains its network to find that shape, which is more reliable than forcing the equations point by point, as a standard PINN does."],
  seeds: ["Repeat runs", "Training starts from random numbers. Running five times from different random starts (seeds 42 to 46) shows the result is not luck."],
  matched: ["Same size", "A single network with as many adjustable numbers as ATLAS-XPINN, so a better result cannot just come from ATLAS-XPINN being bigger."],
  target: ["The goals", "The project's accuracy targets: bending error under 5 % and stress error under 10 %, both against the trusted answer."],
  excl: ["Points left out", "At sharp inside corners and where a support meets an edge, the exact stress is infinite, so no method can match it there. A tiny disc around those points is left out of the stress error."],
  compliance: ["Compliance", "How much the part gives under its load: the work the load does as the part deforms. Lower compliance means a stiffer part."],
  inverse: ["Inverse design", "A normal simulation answers \"how does this shape behave?\". Inverse design asks the reverse, \"which shape behaves the way I want?\", and adjusts one dimension until the goal is met."],
  interface: ["Material boundary", "Where two bonded materials meet. They stretch by different amounts, so the strain jumps at the boundary. A single smooth network blurs that jump."],
  E: ["Young's modulus, E", "How stiff a material is: how hard it is to stretch. Steel is about 210 GPa, aluminium 70, concrete 30 and epoxy 3.5."],
  nu: ["Poisson's ratio, ν", "How much a material narrows when you stretch it. Most metals are about 0.3. It can range from 0 to 0.5."],
  training: ["Training", "The network starts from random numbers and adjusts them thousands of times until the energy stored in the part stops going down. A shorter run adjusts fewer times, so its errors are larger: on the plate with a hole, 20 % of the training gave 5.9 % bending error instead of 0.75 %, just over the 5 % goal."],
  device: ["GPU vs CPU compute", "A graphics card (CUDA GPU) computes thousands of network weights at once and finishes runs several times faster. Switching to CPU runs everything on your processor instead — handy when keeping a laptop GPU cool, running without a GPU, or comparing GPU and CPU times."],
  solver: ["Solver architecture", "ATLAS-XPINN splits the part into regions (one network per region or material) joined smoothly at their borders. Single-network ATLAS trains one network over the whole part."],
};

const tip = (key) => (GLOSSARY[key] ? `<button type="button" class="tip" data-tip="${key}" aria-label="What is ${esc(GLOSSARY[key][0])}?" aria-expanded="false">?</button>` : "");

const MATERIALS = [
  ["steel", "Steel", 210, 0.30], ["titanium", "Titanium", 115, 0.34], ["copper", "Copper", 120, 0.34],
  ["aluminium", "Aluminium", 70, 0.33], ["glass", "Glass", 70, 0.22], ["magnesium", "Magnesium", 45, 0.29],
  ["concrete", "Concrete", 30, 0.20], ["epoxy", "Epoxy", 3.5, 0.35], ["nylon", "Nylon", 2.7, 0.39],
];

const SIMPLE_SUB = {
  cantilever: "beam fixed at one end", l_bracket: "L-shape with a sharp inside corner", plate_hole: "plate with a hole, pulled",
  steel_insert: "steel insert in aluminium", bimaterial: "soft insert in steel", filled_slot: "epoxy-filled slot in steel",
};
const SIMPLE_LABEL = { sigma: "Pulling stress", a: "Slot half-width, along the pull", b: "Slot half-height, across the pull", E: "Stiffness (Young's modulus)" };
const FULL = { epochs: 2000, lbfgs: 2000 };
const MAX_LBFGS = 4 * FULL.lbfgs;
const LBFGS_UNIT = 400;
const ADAM_SHARE = 0.38;
const RUN_STEPS = [
  { epochs: 400, lbfgs: 400, label: "20 %" }, { epochs: 800, lbfgs: 800, label: "40 %" },
  { epochs: 1200, lbfgs: 1200, label: "60 %" }, { epochs: 1600, lbfgs: 1600, label: "80 %" },
  { epochs: 2000, lbfgs: 2000, label: "Full" },
];
const FULL_STEP = RUN_STEPS.length - 1;
const snapLbfgs = (n) => (n <= 0 ? 0 : Math.min(MAX_LBFGS, Math.max(100, Math.round(n / 100) * 100)));
const stepOf = (b) => RUN_STEPS.findIndex((r) => r.epochs === b.epochs && r.lbfgs === b.lbfgs);

const SOLVER_DEFAULT = {
  formulation: "energy_xpinn",
  hidden_layers: 4,
  neurons: 64,
  lr: 0.002,
  precision: "float64",
  seed: 42,
  warm_start: 1000,
  al_stages: 3,
  gamma_u: 30,
  gamma_t: 0.1,
  inv_metric: {},
  inv_ratio: {},
  forward_epochs: 2000,
  forward_lbfgs: 2000,
  n_outer: 12,
};

const THUMB = {
  cantilever: `<rect class="t-fill t-ink" x="14" y="18" width="50" height="12.5" stroke-width="1.4"/><path class="t-ink" d="M13 9v30" stroke-width="2"/><path class="t-hatch" d="M13 12l-5 5M13 19l-5 5M13 26l-5 5M13 33l-5 5" stroke-width="1.2"/>`,
  l_bracket: `<path class="t-fill t-ink" d="M16 24H36V4H56V44H16Z" stroke-width="1.4" stroke-linejoin="round"/><path class="t-dash" d="M36 4V24M36 24V44M36 24H56" stroke-width="1" stroke-dasharray="2 2"/>`,
  plate_hole: `<path class="t-fill t-ink" d="M23 44A7 7 0 0 0 16 37V4H56V44Z" stroke-width="1.4" stroke-linejoin="round"/>`,
  steel_insert: `<path class="t-fill t-ink" d="M16 4H56V44H16Z" stroke-width="1.4" stroke-linejoin="round"/><path class="t-solid t-ink" d="M16 34A10 10 0 0 1 26 44H16Z" stroke-width="1.4" stroke-linejoin="round"/>`,
  bimaterial: `<path class="t-fill t-ink" d="M16 4H56V44H16Z" stroke-width="1.4" stroke-linejoin="round"/><path class="t-page t-ink" d="M16 34A10 10 0 0 1 26 44H16Z" stroke-width="1.4" stroke-linejoin="round"/>`,
  filled_slot: `<path class="t-fill t-ink" d="M16 4H56V44H16Z" stroke-width="1.4" stroke-linejoin="round"/><path class="t-dashfill t-ink" d="M16 26A6 18 0 0 1 22 44H16Z" stroke-width="1.4" stroke-linejoin="round"/>`,
};
const thumb = (key, w, h) => `<svg width="${w}" height="${h}" viewBox="0 0 72 48" fill="none" aria-hidden="true">${THUMB[key] || ""}</svg>`;

const readJSON = (k) => { try { return JSON.parse(store.get(k) || "null"); } catch { return null; } };
const reduced = () => document.documentElement.getAttribute("data-motion") === "reduced"
  || (window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches);
const state = {
  results: null, meta: null, bench: store.get("atlas-bench"), fig: {},
  liveBench: store.get("atlas-live-bench"), job: null, jobs: [], poll: null, health: null,
  workspace: null, filesTab: "runs", filesFilter: "all", femTest: null, filesMsg: "",
  params: readJSON("atlas-params") || {}, heroShown: false, tabsShown: false, geo: null,
  advOpen: store.get("atlas-adv-open") === "1", settingsOpen: false, settingsMsg: "",
  budget: (() => { const b = { ...FULL, ...(readJSON("atlas-budget") || {}) }; return { epochs: b.epochs, lbfgs: snapLbfgs(b.lbfgs) }; })(),
  solver: (() => {
    const s = readJSON("atlas-solver") || {};
    return {
      ...SOLVER_DEFAULT, ...s,
      inv_metric: { ...(s.inv_metric || {}) },
      inv_ratio: { ...(s.inv_ratio || {}) },
    };
  })(),
};

function countUp(el, delay = 0) {
  const m = /^([−+-]?)(\d[\d,]*\.?\d*)(.*)$/s.exec(el.textContent.trim());
  if (!m || reduced()) return;
  const [, sign, num, rest] = m;
  const target = parseFloat(num.replace(/,/g, ""));
  const decimals = (num.split(".")[1] || "").length;
  const grouped = num.includes(",");
  const fmt = (v) => sign + (grouped ? v.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) : v.toFixed(decimals)) + rest;
  const t0 = performance.now() + delay;
  el.textContent = fmt(0);
  const step = (now) => {
    const k = Math.min(1, Math.max(0, (now - t0) / 1100));
    el.textContent = fmt(target * (1 - Math.pow(1 - k, 3)));
    if (k < 1) requestAnimationFrame(step); else el.textContent = fmt(target);
  };
  requestAnimationFrame(step);
}

function setRevealOrigin(x, y) {
  const r = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
  const s = document.documentElement.style;
  s.setProperty("--vt-x", `${x}px`);
  s.setProperty("--vt-y", `${y}px`);
  s.setProperty("--vt-r", `${r}px`);
}

function withTransition(fn) {
  if (document.startViewTransition && !reduced()) document.startViewTransition(fn);
  else fn();
}

function switchTheme(t) {
  if (t === document.documentElement.getAttribute("data-theme")) return;
  withTransition(() => {
    applyTheme(t);
    flow.recolor();
    if (currentView() === "settings") renderSettingsDrawer();
  });
}

function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  store.set("atlas-theme", t);
  $$("[data-theme-pick]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themePick === t)));
}

function applyScale(sc) {
  if (sc === "default" || !sc) {
    document.documentElement.removeAttribute("data-scale");
    store.set("atlas-scale", "");
  } else {
    document.documentElement.setAttribute("data-scale", sc);
    store.set("atlas-scale", sc);
  }
  if (currentView() === "settings") renderSettingsDrawer();
}

function applyMotion(enableAnimations) {
  if (enableAnimations) {
    document.documentElement.removeAttribute("data-motion");
    store.set("atlas-motion", "full");
  } else {
    document.documentElement.setAttribute("data-motion", "reduced");
    store.set("atlas-motion", "reduced");
  }
  flow.recolor();
  if (currentView() === "settings") renderSettingsDrawer();
}

function toggleSidebar() {
  const collapsed = document.documentElement.getAttribute("data-sidebar") === "collapsed";
  if (collapsed) {
    document.documentElement.removeAttribute("data-sidebar");
    store.set("atlas-sidebar", "open");
  } else {
    document.documentElement.setAttribute("data-sidebar", "collapsed");
    store.set("atlas-sidebar", "collapsed");
  }
}

function closeMobileSidebar() {
  const sb = $("#sidebar"), bd = $("#sideBackdrop");
  if (sb) sb.classList.remove("mobile-open");
  if (bd) bd.hidden = true;
}

function currentView() {
  const h = (location.hash || "").replace(/^#/, "");
  if (h === "live" || h === "files" || h === "settings") return h;
  return "results";
}

function setStaticText() {
  $$("[data-level-pick]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.levelPick === (simple() ? "simple" : "expert"))));
  const liveNav = $('.side-nav a[data-view="live"] .nav-text');
  if (liveNav) liveNav.textContent = S("Try it live", "Live solver");
  const filesNav = $('.side-nav a[data-view="files"] .nav-text');
  if (filesNav) filesNav.textContent = S("Folder & runs", "Folder & files");
  $("#heroEyebrow").textContent = S("What is ATLAS?", "Stored study · validated against FEniCS");
  $("#heroLead").innerHTML = S(
    `Engineers simulate parts on a computer to see how they bend under load and where they might break. The standard method, the finite element method ${tip("fem")}, is accurate but slow to repeat for every design change. ATLAS trains a neural network on the laws of physics instead. Browse all six benchmark parts below, or pick Live Solver on the left to simulate your own.`,
    "Relative L² error of displacement and von Mises stress against FEniCS reference solutions. Every number on this page is read from the stored study in results/ — nothing is retrained in the browser.");
  $("#runTitle").textContent = S("Try it yourself", "New run");
  const intro = $("#liveIntro");
  intro.hidden = !simple();
  intro.textContent = "Pick a part, choose its materials and load, and ATLAS works out how it bends and where it is most stressed. A finite-element solver (FEniCS or Built-in P2 FEM) then verifies the answer.";
  renderWorkspaceHeader();
}

function applyLevel(level) {
  closeTip();
  document.documentElement.setAttribute("data-level", level);
  store.set("atlas-level", level);
  setStaticText();
  if (state.results) { renderHero(); renderResults(); }
  if (state.meta) { renderBenchPick(); renderProps(); renderCompute(); renderBudget(); renderSolverCfg(); renderJob(); }
  if (currentView() === "files") renderFilesView();
  if (currentView() === "settings") renderSettingsDrawer();
  pollHealth();
}

const shortGpu = (name) => (name ? name.replace(/^NVIDIA\s+GeForce\s+/i, "").replace(/\s+Laptop\s+GPU$/i, " Laptop") : "CUDA GPU");
const humanBytes = (b) => {
  if (!b) return "0 KB";
  return b >= 1048576 ? `${(b / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`;
};
const selectedDevice = () => state.health?.selected_device || store.get("atlas-device") || "auto";
const effectiveDevice = () => {
  const pref = selectedDevice();
  if (pref === "cpu") return "cpu";
  if (pref === "cuda") return state.health?.cuda_available ? "cuda" : "cpu";
  return state.health?.device || "cpu";
};
const selectedFem = () => state.health?.selected_fem_backend || store.get("atlas-fem") || "auto";
const effectiveFemName = () => {
  const h = state.health;
  if (!h) return "Built-in P2 FEM";
  if (h.fem_backend === "fenics") return `FEniCS ${h.fenics_version || "2019.1"}`;
  return "Built-in P2 FEM";
};
const saveSolver = () => store.set("atlas-solver", JSON.stringify(state.solver));

async function updateAppSettings(patch) {
  try {
    const h = await api("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    state.health = h;
    state.settingsMsg = "";
    if (patch.device) store.set("atlas-device", patch.device);
    if (patch.fem_backend) store.set("atlas-fem", patch.fem_backend);
    applyHealthUI(h);
    if (state.meta) { renderCompute(); renderSpec(); renderBudget(); }
    if (currentView() === "settings") renderSettingsDrawer();
  } catch (e) {
    state.settingsMsg = e.message;
    if (currentView() === "settings") renderSettingsDrawer();
  }
}

function updateNavBadges() {
  const liveBadge = $("#navLiveBadge");
  const filesBadge = $("#navFilesBadge");
  const resBadge = $("#navResultsBadge");
  const busy = isBusy() || (state.health?.active_jobs > 0);
  if (liveBadge) {
    liveBadge.hidden = !busy;
    liveBadge.textContent = S("Running", "Active");
  }
  if (filesBadge) {
    const n = (state.health?.plots_count ?? 0) + state.jobs.length;
    filesBadge.textContent = String(n);
  }
  if (resBadge && state.results?.order) {
    resBadge.textContent = String(state.results.order.length);
  }
}

function applyHealthUI(h) {
  const pill = $("#health");
  const sub = $("#brandSub");
  if (sub) sub.textContent = h.app_mode === "desktop" ? S("Desktop Studio", "Desktop Studio · Local") : S("Design Studio", "Structural Studio");
  const busy = h.active_jobs > 0;
  const runDev = (h.running_device || h.device) === "cuda" ? "GPU" : "CPU";
  const idleDev = h.device === "cuda"
    ? S(`GPU · ${shortGpu(h.gpu_name)}`, `GPU · ${shortGpu(h.gpu_name)}${h.selected_device !== "auto" ? " (pinned)" : ""}`)
    : S(`CPU · ${h.cpu_threads} threads`, `CPU · ${h.cpu_threads}/${h.cpu_count} threads${h.selected_device === "cpu" && h.cuda_available ? " (GPU idle)" : ""}`);
  const mainLine = busy
    ? S(`${runDev} busy · ${plural(h.active_jobs, "run", "runs")}`, `${runDev} · ${plural(h.active_jobs, "job", "jobs")} running`)
    : idleDev;
  const femShort = h.fem_backend === "fenics" ? `FEM: FEniCS ${h.fenics_version || ""}`.trim() : "FEM: Built-in P2";
  pill.innerHTML = `<span class="dot ${busy ? "busy" : "ok"}"></span><span class="runtime-lines"><span class="runtime-main">${esc(mainLine)}</span><span class="runtime-sub">${esc(femShort)}</span></span>`;
  updateNavBadges();
}

function renderWorkspaceHeader() {
  const v = currentView();
  const eye = $("#wsEyebrow"), title = $("#wsTitle"), act = $("#wsActions");
  if (!eye || !title || !act) return;
  const activeRunPill = isBusy() && v !== "live"
    ? `<a href="#live" class="mini-btn"><span class="dot busy"></span><span>View active run</span></a>`
    : "";

  if (v === "results") {
    const b = state.results?.benchmarks?.[state.bench];
    eye.textContent = S("Benchmark Study", "Stored FEniCS Study");
    title.textContent = b ? b.title : "Overview";
    act.innerHTML = `${activeRunPill}
      <button type="button" class="mini-btn" data-try-bench="${esc(state.bench || "cantilever")}">${S("Open in Live Solver →", "Simulate this benchmark →")}</button>
      <a href="#files" class="mini-btn">${S("Browse Folder", "Study Files")}</a>`;
  } else if (v === "live") {
    const m = liveMeta(state.liveBench);
    eye.textContent = S("Interactive Simulation", "Live Solver & Inverse Design");
    title.textContent = m ? m.title : "Workbench";
    act.innerHTML = `<a href="#files" class="mini-btn">${S("Saved Runs & Plots", "Output Folder")}</a>
      <a href="#settings" class="mini-btn">${S("Compute & FEM Settings", "Engine Settings")}</a>`;
  } else if (v === "files") {
    eye.textContent = S("Workspace Explorer", "Output & Reference Storage");
    title.textContent = S("Folder & Saved Runs", "Runs, Plots & Reference Data");
    act.innerHTML = `${activeRunPill}
      <button type="button" class="mini-btn" id="refreshWorkspaceBtn">Refresh</button>
      <button type="button" class="mini-btn" data-open-folder="plots">Open Plots Folder</button>`;
  } else {
    eye.textContent = S("Preferences", "Application Configuration");
    title.textContent = S("Appearance, Compute & FEM Engine", "Studio & Runtime Settings");
    act.innerHTML = `${activeRunPill}
      <button type="button" class="mini-btn" id="quickTestFem">Test Active FEM Engine</button>
      <button type="button" class="mini-btn" id="resetAllBtn">Reset Defaults</button>`;
  }
}

async function loadWorkspace() {
  try {
    state.workspace = await api("/api/workspace");
    if (currentView() === "files") renderFilesView();
  } catch (e) {
    state.filesMsg = `Could not read workspace files: ${e.message}`;
    if (currentView() === "files") renderFilesView();
  }
}

function renderFilesView() {
  const box = $("#filesContent");
  if (!box) return;
  const ws = state.workspace;
  const plots = ws?.plots || { count: 0, bytes: 0, path: "webapp/static/plots", items: [] };
  const resFiles = ws?.results || { count: 0, bytes: 0, path: "results", items: [] };
  const refs = ws?.references || { count: 0, bytes: 0, path: "fem/reference", items: [] };
  const tab = state.filesTab || "runs";

  let tabBody = "";
  if (tab === "runs") {
    const jobCards = state.jobs.map((j) => {
      const m = liveMeta(j.benchmark);
      const mx = j.metrics || {};
      const img = j.plots?.fem_comparison || j.plots?.design || j.plots?.deformed || j.plots?.convergence;
      const devLabel = (mx.device || j.device) === "cuda" ? "GPU" : "CPU";
      const femLabel = mx.fem_engine || j.fem_engine || "FEM";
      const statLine = j.status === "done"
        ? (j.kind === "inverse"
          ? `Design: ${mx.final?.toFixed(3) ?? "—"} m · ${dur(mx.train_seconds)}`
          : `Disp ${pct(mx.relative_L2_displacement)} · VM ${pct(mx.von_mises_l2_excl ?? mx.von_mises_l2)} · ${dur(mx.train_seconds)}`)
        : `Status: ${j.status}`;
      return `<article class="file-card">
        <div>
          ${img ? `<div class="file-thumb" data-open-img="${esc(img)}"><img src="${esc(img)}" alt="${esc(m?.title || j.benchmark)}"></div>`
            : `<div class="file-thumb">${thumb(j.benchmark, 96, 64)}</div>`}
          <div class="file-info" style="margin-top:12px">
            <span class="file-name">${j.kind === "inverse" ? "Inverse Design" : "Forward Solve"} · ${esc(m?.title || j.benchmark)}</span>
            <span class="file-meta">${esc(statLine)}</span>
            <span class="file-meta">${esc(`${j.formulation === "energy" ? "ATLAS single" : "ATLAS-XPINN"} · ${devLabel} · ${femLabel}`)}</span>
          </div>
        </div>
        <div class="file-actions">
          <button type="button" class="mini-btn" data-inspect-job="${esc(j.id)}">Open in Live Solver</button>
          ${j.status === "done" ? `<a class="mini-btn" href="/api/jobs/${esc(j.id)}/export" download>Export JSON</a>` : ""}
        </div>
      </article>`;
    }).join("");

    const loosePlots = plots.items.map((f) => `<article class="file-card">
      <div>
        ${f.is_image ? `<div class="file-thumb" data-open-img="${esc(f.url)}"><img src="${esc(f.url)}" alt="${esc(f.name)}"></div>` : ""}
        <div class="file-info" style="margin-top:10px">
          <span class="file-name">${esc(f.name)}</span>
          <span class="file-meta">${esc(humanBytes(f.bytes))} · ${esc(f.modified.replace("T", " "))}</span>
        </div>
      </div>
      <div class="file-actions">
        <a class="mini-btn" href="${esc(f.url)}" target="_blank" rel="noopener">Open Full Size</a>
        <a class="mini-btn" href="${esc(f.url)}" download="${esc(f.name)}">Download</a>
      </div>
    </article>`).join("");

    tabBody = (jobCards || loosePlots)
      ? `<div class="files-grid">${jobCards}${loosePlots}</div>`
      : `<div class="card"><p class="note" style="margin:0">No live runs or saved plot files yet in this session. Start a simulation in <a href="#live" style="text-decoration:underline;font-weight:600">Live Solver</a> to populate this folder automatically.</p></div>`;
  } else if (tab === "results") {
    const flt = state.filesFilter || "all";
    const items = resFiles.items.filter((i) => flt === "all" || (flt === "png" ? i.is_image : !i.is_image));
    tabBody = `
      <div class="toolbar-row">
        <div class="seg" role="group" aria-label="File type filter">
          <button type="button" data-files-filter="all" aria-pressed="${flt === "all"}">All (${resFiles.items.length})</button>
          <button type="button" data-files-filter="png" aria-pressed="${flt === "png"}">Figures (.png)</button>
          <button type="button" data-files-filter="data" aria-pressed="${flt === "data"}">JSON &amp; Reports</button>
        </div>
        <span class="meta">Directory: ${esc(resFiles.path)}</span>
      </div>
      <div class="files-grid">
        ${items.map((item) => `<article class="file-card">
          <div>
            ${item.is_image ? `<div class="file-thumb" data-open-img="${esc(item.url)}"><img src="${esc(item.url)}" alt="${esc(item.path)}" loading="lazy"></div>` : ""}
            <div class="file-info" style="margin-top:${item.is_image ? "10px" : "0"}">
              <span class="file-name">${esc(item.path)}</span>
              <span class="file-meta">${esc(item.group)} · ${esc(humanBytes(item.bytes))}</span>
            </div>
          </div>
          <div class="file-actions">
            <a class="mini-btn" href="${esc(item.url)}" target="_blank" rel="noopener">View</a>
            <a class="mini-btn" href="${esc(item.url)}" download>Download</a>
          </div>
        </article>`).join("")}
      </div>`;
  } else {
    tabBody = `
      <section class="card">
        <div class="card-head">
          <h2>Precomputed FEniCS 2019.1.0 Reference Datasets</h2>
          <span class="meta">${esc(refs.path)}</span>
        </div>
        <p class="note">Genuine legacy FEniCS (dolfin + mshr) reference solutions bundled with ATLAS so all six benchmarks validate immediately on Windows, macOS, and Linux even before WSL2 is configured.</p>
        <div class="data-table-wrap" style="margin-top:16px">
          <table class="data-table">
            <thead><tr><th>Dataset</th><th>Mesh / Grid Points</th><th>Reference Compliance</th><th>Files</th><th>Size</th></tr></thead>
            <tbody>
              ${refs.items.map((r) => `<tr>
                <td><strong>${esc(r.key)}</strong></td>
                <td>${r.points != null ? r.points.toLocaleString("en-US") : "—"}</td>
                <td>${r.compliance != null ? sci(r.compliance) : "—"}</td>
                <td>${r.files}</td>
                <td>${esc(humanBytes(r.bytes))}</td>
              </tr>`).join("")}
            </tbody>
          </table>
        </div>
      </section>`;
  }

  box.innerHTML = `
    <div class="page-banner">
      <div>
        <div class="eyebrow">Workspace &amp; Output Directory</div>
        <h1>${S("Folder, Runs & Reference Data", "Workspace Explorer")}</h1>
        <p>${S("Browse every simulation figure and JSON report created during your session, open your local folders in File Explorer, or inspect the bundled study figures and FEniCS references.",
          "Inspect live session outputs in webapp/static/plots/, browse study artifacts in results/, and verify FEniCS reference datasets in fem/reference/.")}</p>
      </div>
      <div class="set-actions">
        <button type="button" class="mini-btn" data-open-folder="plots">Open Live Plots</button>
        <button type="button" class="mini-btn" data-open-folder="results">Open Study Results</button>
        <button type="button" class="mini-btn" data-open-folder="references">Open FEniCS Refs</button>
        <button type="button" class="mini-btn danger" id="clearStorageBtnFiles">Clear Finished Runs &amp; Plots</button>
      </div>
    </div>

    ${state.filesMsg ? `<div class="status-banner">${esc(state.filesMsg)}</div>` : ""}

    <div class="tiles">
      ${tile("Session runs", String(state.jobs.length), `${state.jobs.filter((j) => j.status === "done").length} completed this session`)}
      ${tile("Live plot files", String(plots.count), `${humanBytes(plots.bytes)} in webapp/static/plots`)}
      ${tile("Study artifacts", String(resFiles.count), `${humanBytes(resFiles.bytes)} in results/`)}
      ${tile("FEniCS references", String(refs.count), `${humanBytes(refs.bytes)} in fem/reference/`)}
    </div>

    <div class="toolbar-row">
      <div class="seg" role="group" aria-label="Workspace sections">
        <button type="button" data-files-tab="runs" aria-pressed="${tab === "runs"}">Session Runs &amp; Live Plots (${state.jobs.length + plots.count})</button>
        <button type="button" data-files-tab="results" aria-pressed="${tab === "results"}">Study Figures &amp; JSON (${resFiles.count})</button>
        <button type="button" data-files-tab="refs" aria-pressed="${tab === "refs"}">FEniCS References (${refs.count})</button>
      </div>
    </div>

    ${tabBody}`;
}

function openSettingsDrawer() {
  state.settingsMsg = "";
  if (location.hash !== "#settings") location.hash = "#settings";
  else renderSettingsDrawer();
}

function closeSettingsDrawer() {
  if (location.hash === "#settings") location.hash = "#results";
}

function renderSettingsDrawer() {
  const body = $("#settingsBody");
  if (!body) return;
  const h = state.health;
  const curTheme = document.documentElement.getAttribute("data-theme") || "light";
  const curScale = document.documentElement.getAttribute("data-scale") || "default";
  const animOn = !reduced();
  const prefDev = selectedDevice();
  const prefFem = selectedFem();
  const cudaOk = !!h?.cuda_available;
  const fenicsOk = !!h?.fenics_available;
  const cfg = h?.settings || { plot_dpi: 140, poll_ms: 1500, auto_open_log: true };
  const mem = h?.gpu_memory;
  const vram = mem ? `${(mem.free_mb / 1024).toFixed(1)} GB free / ${(mem.total_mb / 1024).toFixed(1)} GB total` : null;
  const threads = h?.cpu_threads || 4;
  const maxThreads = h?.cpu_count || 8;
  const dpis = [[120, "120 DPI · Fast"], [180, "180 DPI · High"], [300, "300 DPI · Print"]];
  const polls = [[750, "0.75 s"], [1500, "1.5 s"], [3000, "3.0 s"]];

  const themeCards = [
    ["light", "Light Studio", "Clean daylight paper surface & high-contrast blue accents", "#FCFCFA", "#F3F2EE", "#15171A", "#1E5BC6"],
    ["dark", "Dark Studio", "Low-glare charcoal slate tuned for long GPU training sessions", "#17191B", "#131416", "#EDEDEA", "#7FA6EE"],
    ["atlas", "Atlas Parchment", "Editorial serif typography on warm drafting parchment", "#F2E9D4", "#EDE3CC", "#2A2016", "#2F5A7A"],
  ];

  body.innerHTML = `
    <div class="page-banner">
      <div>
        <div class="eyebrow">Application Preferences</div>
        <h1>${S("Appearance, Compute & FEM Engine", "Studio & Runtime Configuration")}</h1>
        <p>${S("Customize your theme and layout scale, switch between GPU and CPU execution, and choose whether custom simulations verify against WSL2 FEniCS or the built-in cross-platform FEM solver.",
          "Configure theme, detail level, UI scale, PyTorch GPU/CPU execution, and FEM validation backend (genuine WSL2/Conda FEniCS 2019.1.0 vs Built-in SciPy P2 FEM).")}</p>
      </div>
    </div>

    ${state.settingsMsg ? `<div class="status-banner">${esc(state.settingsMsg)}</div>` : ""}

    <div class="settings-grid">
      <section class="card settings-card full">
        <div class="settings-card-head">
          <div>
            <h2>${S("Theme & Interface Layout", "Appearance & Display")}</h2>
            <p>${S("Switch color palette, explanation style, text scaling, and background animations.", "Persisted automatically across desktop sessions.")}</p>
          </div>
        </div>
        <div class="theme-cards" role="group" aria-label="Color theme">
          ${themeCards.map(([id, name, desc, railBg, mainBg, inkCol, accCol]) => `
            <button type="button" class="theme-card" data-theme-pick="${id}" aria-pressed="${curTheme === id}">
              <div class="theme-swatch" style="background:${mainBg}">
                <div class="sw-rail" style="background:${railBg};border-right:1px solid rgba(128,128,128,.2)">
                  <span class="sw-bar" style="background:${accCol}"></span>
                  <span class="sw-bar" style="background:${inkCol};opacity:.25"></span>
                  <span class="sw-bar" style="background:${inkCol};opacity:.25"></span>
                </div>
                <div class="sw-main">
                  <span class="sw-bar" style="background:${inkCol};opacity:.8;width:58%"></span>
                  <span class="sw-accent" style="background:${accCol}"></span>
                </div>
              </div>
              <div class="theme-card-title"><span>${esc(name)}</span>${curTheme === id ? '<span class="badge custom">Active</span>' : ""}</div>
              <span class="theme-card-sub">${esc(desc)}</span>
            </button>`).join("")}
        </div>

        <div class="grid-2" style="margin-top:4px">
          <div class="field compact">
            <span>Explanation &amp; metric detail level</span>
            <div class="seg seg-block two" role="group" aria-label="Detail mode">
              <button type="button" data-level-pick="simple" aria-pressed="${simple()}">Simple (Guided)</button>
              <button type="button" data-level-pick="expert" aria-pressed="${!simple()}">Expert (Technical)</button>
            </div>
          </div>
          <div class="field compact">
            <span>Interface zoom / density</span>
            <div class="seg seg-block" role="group" aria-label="UI scale">
              <button type="button" data-scale-pick="compact" aria-pressed="${curScale === "compact"}">Compact (93%)</button>
              <button type="button" data-scale-pick="default" aria-pressed="${curScale === "default"}">Default (100%)</button>
              <button type="button" data-scale-pick="large" aria-pressed="${curScale === "large"}">Large (107%)</button>
            </div>
          </div>
        </div>

        <label class="toggle-row">
          <span>Enable stress-flow background animation &amp; smooth theme transitions</span>
          <input type="checkbox" id="setMotion" ${animOn ? "checked" : ""}>
        </label>
      </section>

      <section class="card settings-card">
        <div class="settings-card-head">
          <div>
            <h2>${S("FEM Validation Engine", "Finite-Element Reference Backend")}</h2>
            <p>${S("Choose how custom geometries and inverse shape designs are verified.", "Select between untouched WSL2/Conda FEniCS 2019.1.0 and Built-in SciPy P2 FEM.")}</p>
          </div>
          <button type="button" class="mini-btn" id="testFemBtn">Run Self-Test</button>
        </div>

        <div class="engine-list" role="group" aria-label="FEM backend">
          <button type="button" class="engine-opt" data-fem-pick="auto" aria-pressed="${prefFem === "auto"}">
            <span class="engine-radio"></span>
            <span class="engine-body">
              <span class="engine-title"><span>Auto (Recommended)</span><span class="badge">Active: ${esc(effectiveFemName())}</span></span>
              <span class="engine-desc">Uses genuine WSL2 / Conda FEniCS 2019.1.0 automatically when present, and seamlessly falls back to Built-in P2 FEM on machines without WSL2.</span>
            </span>
          </button>
          <button type="button" class="engine-opt" data-fem-pick="fenics" aria-pressed="${prefFem === "fenics"}" ${fenicsOk ? "" : 'disabled title="WSL2 / Conda FEniCS (dolfin + mshr) not detected on this host"'}>
            <span class="engine-radio"></span>
            <span class="engine-body">
              <span class="engine-title"><span>WSL2 / Conda FEniCS 2019.1.0</span><span class="badge ${fenicsOk ? "custom" : ""}">${fenicsOk ? `Ready (${esc(h.fenics_version)})` : "Not installed"}</span></span>
              <span class="engine-desc">Runs untouched fem/fem_live.py &amp; fem/fem_parametric.py (dolfin + mshr) natively or through WSL2 for thesis-identical validation.</span>
            </span>
          </button>
          <button type="button" class="engine-opt" data-fem-pick="builtin" aria-pressed="${prefFem === "builtin"}">
            <span class="engine-radio"></span>
            <span class="engine-body">
              <span class="engine-title"><span>Built-in Quadratic P2 FEM (No WSL Needed)</span><span class="badge custom">Ready</span></span>
              <span class="engine-desc">Cross-platform body-fitted 6-node P2 triangular FEM solver (${esc(h?.builtin_fem_version || "SciPy sparse")}). Works natively on Windows, macOS &amp; Linux with zero WSL setup.</span>
            </span>
          </button>
        </div>

        ${state.femTest ? `<div class="status-banner">${esc(state.femTest)}</div>` : ""}
      </section>

      <section class="card settings-card">
        <div class="settings-card-head">
          <div>
            <h2>${S("Compute Hardware (GPU / CPU)", "PyTorch Compute Device & Threads")}</h2>
            <p>${S("Switch neural-network training between CUDA GPU and multi-core CPU.", "Applies immediately to upcoming forward and inverse runs.")}</p>
          </div>
        </div>

        <div class="seg seg-block" role="group" aria-label="Compute device">
          <button type="button" data-dev-pick="auto" aria-pressed="${prefDev === "auto"}">Auto</button>
          <button type="button" data-dev-pick="cuda" aria-pressed="${prefDev === "cuda"}" ${cudaOk ? "" : 'disabled title="No CUDA GPU detected in this Python environment"'}>GPU (CUDA)</button>
          <button type="button" data-dev-pick="cpu" aria-pressed="${prefDev === "cpu"}">CPU</button>
        </div>

        <div class="prow">
          <div class="prow-top"><span>CPU worker threads</span><span class="pnum"><input type="number" id="setThreadsNum" min="1" max="${maxThreads}" step="1" value="${threads}"><span>/ ${maxThreads}</span></span></div>
          <input type="range" class="slider" id="setThreadsRange" min="1" max="${maxThreads}" step="1" value="${threads}" style="--fill:${maxThreads > 1 ? ((threads - 1) / (maxThreads - 1)) * 100 : 100}%">
        </div>

        <dl class="hw-list">
          <div><dt>Runtime mode</dt><dd>${h?.app_mode === "desktop" ? "Standalone Desktop App" : "Local Application Console"}</dd></div>
          <div><dt>PyTorch</dt><dd>${esc(h?.torch || "—")}</dd></div>
          ${h?.system_gpu_name ? `<div><dt>Physical GPU</dt><dd>${esc(h.system_gpu_name)}${h.nvidia_driver ? ` · Driver ${esc(h.nvidia_driver)}` : ""}${h.driver_cuda_version ? ` (CUDA ${esc(h.driver_cuda_version)})` : ""}</dd></div>` : ""}
          <div><dt>CUDA Runtime</dt><dd>${cudaOk ? esc(h.gpu_name) : (h?.system_gpu_name ? "CPU PyTorch installed — click below to activate CUDA" : "Not detected (CPU active)")}</dd></div>
          ${vram ? `<div><dt>GPU VRAM</dt><dd>${esc(vram)}</dd></div>` : ""}
        </dl>
        ${cudaOk
          ? `<div class="set-actions"><button type="button" class="mini-btn" id="clearCudaBtn">Free GPU memory cache</button></div>`
          : (h?.system_gpu_name || h?.platform === "Windows")
            ? `<div class="set-actions"><button type="button" class="mini-btn" id="installCudaBtn" data-install-cuda>Install NVIDIA CUDA Runtime</button></div>`
            : ""}
      </section>

      <section class="card settings-card full">
        <div class="settings-card-head">
          <div>
            <h2>${S("Figures, Polling & Workspace Storage", "Output Resolution & Session Storage")}</h2>
            <p>${S("Control figure DPI quality, live chart update speed, and saved run files.", "Saved settings persist in webapp/.app_settings.json.")}</p>
          </div>
        </div>
        <div class="grid-2">
          <div class="field compact">
            <span>Figure export resolution</span>
            <div class="seg seg-block" role="group" aria-label="Plot DPI">
              ${dpis.map(([v, l]) => `<button type="button" data-dpi="${v}" aria-pressed="${cfg.plot_dpi === v || (v === 120 && cfg.plot_dpi < 160)}">${esc(l)}</button>`).join("")}
            </div>
          </div>
          <div class="field compact">
            <span>Live progress polling interval</span>
            <div class="seg seg-block" role="group" aria-label="Refresh interval">
              ${polls.map(([v, l]) => `<button type="button" data-poll="${v}" aria-pressed="${cfg.poll_ms === v}">${esc(l)}</button>`).join("")}
            </div>
          </div>
        </div>
        <label class="toggle-row">
          <span>Auto-open solver log when a simulation starts</span>
          <input type="checkbox" id="setAutoLog" ${cfg.auto_open_log !== false ? "checked" : ""}>
        </label>
        <div class="toolbar-row" style="padding-top:4px;border-top:1px solid var(--line-soft)">
          <span class="meta">Saved live plots: <strong>${h ? `${h.plots_count} files (${humanBytes(h.plots_bytes)})` : "—"}</strong> · Session jobs: <strong>${h ? h.total_jobs : 0}</strong></span>
          <div class="set-actions">
            <a href="#files" class="mini-btn">Browse in Folder View</a>
            <button type="button" class="mini-btn" id="openPlotsBtn">Open Plots Folder</button>
            <button type="button" class="mini-btn danger" id="clearStorageBtn">Clear Finished Runs &amp; Plots</button>
          </div>
        </div>
      </section>

      <section class="card settings-card full">
        <div class="settings-card-head">
          <div>
            <h2>${S("Desktop Installer & Releases", "GitHub Releases & Native Win32 Setup Wizard")}</h2>
            <p>${S("Download the official native installer (Published by ATLAS) for Windows, macOS, or Linux.", "Includes native Windows x64 GUI Installer (ATLAS-Setup.exe), ATLAS.exe launcher, and complete release bundle.")}</p>
          </div>
          <a href="https://github.com/SultraZ/atlas/releases/latest" target="_blank" rel="noopener" class="mini-btn">All Releases ↗</a>
        </div>
        <div class="set-actions">
          <a href="https://github.com/SultraZ/atlas/archive/refs/tags/v1.1.4.zip" target="_blank" rel="noopener" class="mini-btn">Full Complete Package (ATLAS-v1.1.4.zip)</a>
          <a href="https://github.com/SultraZ/atlas/raw/v1.1.4/ATLAS-Setup.exe" target="_blank" rel="noopener" class="mini-btn">Windows Installer (ATLAS-Setup.exe)</a>
          <a href="https://github.com/SultraZ/atlas/raw/v1.1.4/setup_atlas.sh" target="_blank" rel="noopener" class="mini-btn">macOS / Linux Installer (setup_atlas.sh)</a>
        </div>
      </section>
    </div>`;
}

const tipState = { btn: null };
function closeTip() {
  const pop = $("#tipPop");
  if (pop) pop.hidden = true;
  if (tipState.btn) tipState.btn.setAttribute("aria-expanded", "false");
  tipState.btn = null;
}

function openTip(btn) {
  const [title, body] = GLOSSARY[btn.dataset.tip];
  const pop = $("#tipPop");
  pop.innerHTML = `<b>${esc(title)}</b>${esc(body)}`;
  pop.hidden = false;
  const r = btn.getBoundingClientRect();
  const w = Math.min(320, innerWidth - 24);
  pop.style.width = `${w}px`;
  const left = Math.max(12, Math.min(innerWidth - w - 12, r.left + r.width / 2 - w / 2));
  const below = r.bottom + 8 + pop.offsetHeight < innerHeight;
  pop.style.left = `${left}px`;
  pop.style.top = `${below ? r.bottom + 8 : r.top - 8 - pop.offsetHeight}px`;
  btn.setAttribute("aria-expanded", "true");
  tipState.btn = btn;
}

function route() {
  const view = currentView();
  closeMobileSidebar();
  ["results", "live", "files", "settings"].forEach((k) => {
    const p = $(`#view-${k}`);
    if (p) p.hidden = k !== view;
  });
  $$(".side-nav a[data-view]").forEach((a) => (a.dataset.view === view ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  renderWorkspaceHeader();
  if (view === "files") {
    renderFilesView();
    loadWorkspace();
  } else if (view === "settings") {
    renderSettingsDrawer();
    pollHealth();
  }
  const el = $(`#view-${view}`);
  if (el) {
    el.classList.remove("view");
    void el.offsetWidth;
    el.classList.add("view");
  }
}

async function pollHealth() {
  const pill = $("#health");
  try {
    const prevCuda = state.health?.cuda_available;
    const prevSel = state.health?.selected_device;
    const prevFem = state.health?.fem_backend;
    const h = await api("/api/health");
    state.health = h;
    applyHealthUI(h);
    if (state.meta && (prevCuda !== h.cuda_available || prevSel !== h.selected_device || prevFem !== h.fem_backend)) {
      renderCompute();
      renderSpec();
      renderBudget();
    }
    if (currentView() === "settings" && document.activeElement?.tagName !== "INPUT") renderSettingsDrawer();
  } catch {
    pill.innerHTML = `<span class="dot bad"></span><span class="runtime-lines"><span class="runtime-main">${S("App offline", "Backend unreachable")}</span><span class="runtime-sub">Disconnected</span></span>`;
  }
}

const SOLVER_NAMES = {
  paper: {
    atlas_xpinn: ["ATLAS-XPINN", "Domain-decomposed", "a"], atlas: ["ATLAS", "Single network", "a"],
    deepxde: ["DeepXDE PINN", "Default settings", "b"], deepxde_configured: ["DeepXDE PINN", "Configured", "b"],
  },
  multi: {
    atlas_xpinn: ["ATLAS-XPINN", "Split at the material interface", "a"], atlas: ["ATLAS", "Single network", "b"],
    atlas_wide: ["ATLAS", "Single network, parameter-matched", "b"],
  },
};
const SOLVER_NAMES_SIMPLE = {
  paper: {
    atlas_xpinn: ["ATLAS-XPINN", "split into regions", "a"], atlas: ["ATLAS", "one network", "a"],
    deepxde: ["Standard PINN", "default settings", "b"], deepxde_configured: ["Standard PINN", "tuned settings", "b"],
  },
  multi: {
    atlas_xpinn: ["ATLAS-XPINN", "one network per material", "a"], atlas: ["ATLAS", "one network", "b"],
    atlas_wide: ["ATLAS", "one network, same size", "b"],
  },
};
const solverName = (group, key) => (simple() ? SOLVER_NAMES_SIMPLE : SOLVER_NAMES)[group][key];
const ceilNice = (x) => [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1].find((v) => x <= v) ?? x;

function atlasRows(b) {
  const keys = b.group === "paper" ? ["atlas_xpinn", "atlas"] : ["atlas_xpinn", "atlas", "atlas_wide"];
  return keys.map((k) => ({ key: k, bench: b, ...b.solvers[k] })).filter((r) => r.disp != null);
}

function renderHero() {
  const { results } = state;
  const benches = results.order.map((k) => results.benchmarks[k]);
  const paper = benches.filter((b) => b.group === "paper");
  const multi = benches.filter((b) => b.group === "multi");
  const t = results.targets;
  const pass = (r) => r.disp < t.displacement && r.vm < t.von_mises;
  const atlas = benches.flatMap(atlasRows);
  const pinnBenches = paper.filter((b) => b.solvers.deepxde || b.solvers.deepxde_configured);
  const pinnRows = pinnBenches.flatMap((b) => ["deepxde", "deepxde_configured"].map((k) => b.solvers[k] && { ...b.solvers[k], bench: b, key: k }).filter(Boolean));
  const pinnMiss = pinnBenches.filter((b) => ["deepxde", "deepxde_configured"].every((k) => !b.solvers[k] || (b.solvers[k].disp >= t.displacement && b.solvers[k].vm >= t.von_mises))).length;

  const worstD = atlas.reduce((a, r) => (r.disp > a.disp ? r : a), atlas[0]);
  const worstV = atlas.reduce((a, r) => (r.vm > a.vm ? r : a), atlas[0]);
  let title;
  if (simple() && worstD && atlas.every(pass)) {
    title = `On all ${word(benches.length)} test parts, ATLAS's bending is off by under ${+(ceilNice(worstD.disp) * 100).toFixed(1)} % and its stress by under ${+(ceilNice(worstV.vm) * 100).toFixed(1)} %, compared with the trusted simulation.`;
    if (pinnRows.length) {
      const errs = pinnRows.flatMap((r) => [r.disp, r.vm]);
      title += ` A standard physics AI is off by ${Math.round(Math.min(...errs) * 100)} to ${Math.round(Math.max(...errs) * 100)} %.`;
    }
  } else {
    title = atlas.every(pass)
      ? `ATLAS stays inside both accuracy targets on all ${word(benches.length)} benchmarks.`
      : `ATLAS meets both accuracy targets on ${word(new Set(atlas.filter(pass).map((r) => r.bench.key)).size)} of ${word(benches.length)} benchmarks.`;
    if (pinnBenches.length) {
      title += pinnMiss === pinnBenches.length
        ? ` A standard PINN misses both on all ${word(pinnBenches.length)} it was run on.`
        : ` A standard PINN misses both on ${word(pinnMiss)} of the ${word(pinnBenches.length)} it was run on.`;
    }
  }
  const h = $("#heroTitle");
  if (!state.heroShown && !reduced()) {
    h.innerHTML = title.split(" ").map((w, i) => `<span class="w" style="animation-delay:${60 + i * 36}ms">${esc(w)}</span>`).join(" ");
  } else h.textContent = title;

  const kpis = [];
  const nameOf = (r) => solverName(r.bench.group, r.key)[0];
  if (worstD) kpis.push([pct(worstD.disp), S(`Largest bending error ${tip("displacement")}`, "Worst ATLAS displacement error"), S(`ATLAS vs the trusted answer · goal under ${t.displacement * 100} % · ${worstD.bench.title}`, `Target < ${t.displacement * 100} % · ${worstD.bench.title}, ${nameOf(worstD)}`), ""]);
  if (worstV) kpis.push([pct(worstV.vm), S(`Largest stress error ${tip("vonmises")}`, "Worst ATLAS von Mises error"), S(`goal under ${t.von_mises * 100} % · ${worstV.bench.title}`, `Target < ${t.von_mises * 100} % · ${worstV.bench.title}, ${nameOf(worstV)}`), ""]);
  if (pinnRows.length) {
    const best = pinnRows.reduce((a, r) => (r.disp < a.disp ? r : a), pinnRows[0]);
    kpis.push([pct(best.disp), S(`Best bending error of a standard physics AI ${tip("pinn")}`, "Best standard-PINN displacement error"), S(`${best.bench.title} · ${Math.round(best.disp / t.displacement)}× over the goal`, `DeepXDE · ${best.bench.title} · ${Math.round(best.disp / t.displacement)}× the target`), "soft"]);
  }
  const gains = multi.map((b) => ({ b, x: b.seeds.atlas_xpinn, w: b.seeds.atlas_wide })).filter((g) => g.x && g.w);
  if (gains.length) {
    const g = gains.reduce((a, c) => (c.w.vm_mean / c.x.vm_mean > a.w.vm_mean / a.x.vm_mean ? c : a), gains[0]);
    kpis.push([`${(g.w.vm_mean / g.x.vm_mean).toFixed(1)}×`, S(`Smaller stress error from splitting the network ${tip("xpinn")}`, "Lower von Mises error from decomposition"),
      S(`${g.b.title} · vs one network of the same size, ${g.x.n} runs`, `${g.b.title} · ATLAS-XPINN vs a parameter-matched single network, ${g.x.n} seeds`), ""]);
  }
  $("#heroKpis").innerHTML = kpis.map(([v, l, s, cls]) => `<div class="kpi"><span class="kpi-value ${cls}">${esc(v)}</span><div class="kpi-text"><span class="kpi-label">${l}</span><span class="kpi-sub">${esc(s)}</span></div></div>`).join("");
  if (!state.heroShown) $$("#heroKpis .kpi-value").forEach((el, i) => countUp(el, 250 + i * 110));
  state.heroShown = true;
}

function benchSubtitle(b) { return S(SIMPLE_SUB[b.key] || b.geometry, `${b.geometry} · ${plural(b.subdomains, "subdomain", "subdomains")}`); }

function renderBenchTabs() {
  const { results } = state;
  const groups = [["paper", S("Test parts made of one material", "Paper benchmarks")], ["multi", S("Test parts made of two materials", "Multi-material · where decomposition pays off")]];
  const enter = !state.tabsShown;
  let n = 0;
  $("#benches").innerHTML = groups.map(([g, label]) => {
    const keys = results.order.filter((k) => results.benchmarks[k].group === g);
    if (!keys.length) return "";
    return `<div><div class="eyebrow bench-group-label">${esc(label)}</div><div class="bench-tabs">${keys.map((k) => {
      const b = results.benchmarks[k];
      return `<button type="button" class="bench-tab ${enter ? "enter" : ""}" style="animation-delay:${300 + 55 * n++}ms" data-bench="${k}" aria-pressed="${k === state.bench}"><span class="thumb">${thumb(k, 72, 48)}</span><span class="bench-tab-text"><span class="bench-tab-title">${esc(b.title)}</span><span class="bench-tab-meta">${esc(benchSubtitle(b))}</span></span></button>`;
    }).join("")}</div></div>`;
  }).join("");
  state.tabsShown = true;
}

function axisScale(values) {
  const lo = Math.min(0.001, ...values.filter((v) => v > 0).map((v) => 10 ** Math.floor(Math.log10(v))));
  const dec = Math.max(1, Math.round(-Math.log10(Math.max(lo, 1e-5))));
  const l0 = -dec;
  return {
    pos: (x) => `${Math.max(0, Math.min(100, ((Math.log10(Math.max(x, 1e-9)) - l0) / -l0) * 100)).toFixed(2)}%`,
    ticks: Array.from({ length: dec + 1 }, (_, i) => ({ v: 10 ** (l0 + i), p: (i / dec) * 100 })),
  };
}

function metricCell(v, target, scale, dot, label) {
  const grid = scale.ticks.map((t) => `<span class="gl" style="left:${t.p}%"></span>`).join("");
  return `<div class="metric"><span class="metric-val">${passIcon(v < target)}<span>${pct(v)}</span></span><div class="plot" title="${esc(label)}">${grid}<span class="bl"></span><span class="tl" style="left:${scale.pos(target)}"></span><span class="pt ${dot}" data-left="${scale.pos(v)}"></span></div></div>`;
}

function renderErrors(b) {
  const t = state.results.targets;
  const isPaper = b.group === "paper";
  const lib = b.solvers.deepxde?.library || b.solvers.deepxde_configured?.library;
  const groups = isPaper
    ? [[S(`ATLAS · energy method ${tip("energy")}`, "ATLAS · deep-energy solvers"), ["atlas_xpinn", "atlas"]], [S(`Standard physics AI (PINN) ${tip("pinn")}`, `Standard PINN · ${esc(lib ? lib.replace(/^deepxde/i, "DeepXDE") : "DeepXDE")}`), ["deepxde", "deepxde_configured"]]]
    : [[S(`Split at the material boundary ${tip("xpinn")}`, "ATLAS-XPINN · decomposed at the interface"), ["atlas_xpinn"]], [S(`One network for both materials ${tip("matched")}`, "ATLAS · one network over both materials"), ["atlas", "atlas_wide"]]];
  const all = groups.flatMap(([, ks]) => ks.map((k) => b.solvers[k]).filter(Boolean));
  const scale = axisScale(all.flatMap((r) => [r.disp, r.vm]));
  const vmLabel = S(`Stress error ${tip("vonmises")}`, isPaper ? "von Mises L², excl." : "von Mises L²");
  const dLabel = S(`Bending error ${tip("displacement")}`, "Displacement L²");
  const rows = groups.map(([label, ks]) => {
    const body = ks.filter((k) => b.solvers[k]).map((k, i) => {
      const r = b.solvers[k];
      const [name, sub, dot] = solverName(b.group, k);
      const size = r.params && !simple() ? ` · ${r.params.toLocaleString("en-US")} params` : "";
      return `<div class="erow body" style="animation-delay:${i * 60}ms"><div class="who"><span class="who-name">${esc(name)}</span><span class="who-sub">${esc(sub)}${size}</span></div>${metricCell(r.disp, t.displacement, scale, dot, `${name}: displacement ${pct(r.disp)}`)}${metricCell(r.vm, t.von_mises, scale, dot, `${name}: von Mises ${pct(r.vm)}`)}<span class="etime">${mins(r.seconds)}</span></div>`;
    }).join("");
    return body ? `<div class="egroup">${label}</div>${body}` : "";
  }).join("");
  const last = scale.ticks.length - 1;
  const every = last > 3 ? 2 : 1;
  const tickLabel = (v, i) => `${+(v * 100).toPrecision(2)}${i === last ? " %" : ""}`;
  const axis = `<div class="axis"><div class="axis-bar">${scale.ticks.slice(1).map(() => "<span></span>").join("")}</div>${scale.ticks.map((tk, i) => (i % every && i !== last ? "" : `<span class="tick ${i === 0 ? "first" : i === last ? "last" : ""}" style="left:${tk.p}%">${tickLabel(tk.v, i)}</span>`)).join("")}</div>`;
  const note = S(isPaper
    ? `Lower is better; the orange line is each goal. At sharp corners and where supports meet edges the true stress is infinite, so a tiny area there is left out of the stress error ${tip("excl")}. One training run per method.`
    : `Lower is better; the orange line is each goal. "Same size" is one network with as many adjustable numbers as ATLAS-XPINN, so the difference is not just network size. One training run per method.`,
  esc(isPaper
    ? `The von Mises error leaves out a small disk around each stress singularity (radius ${b.exclusion?.factor ?? 2.5} × the local point spacing). One training run per solver, seed 42.`
    : "There are no stress singularities here, so no points are left out. One training run per solver, seed 42. The parameter-matched single network has as many parameters as ATLAS-XPINN, which rules out network size as the reason for the difference."));
  $("#errors").innerHTML = `
    <div class="card-head"><h2 class="l">${S("How close is each method to the trusted answer?", "Error against FEniCS")}</h2><span class="meta">${esc(b.title)} · ${S("one run each", "seed 42")}</span></div>
    <div class="erow head"><span class="col">${S("Method", "Solver")}</span><span class="col"><span>${dLabel}</span><small>${S("goal", "target")} ${t.displacement * 100} %</small></span><span class="col"><span>${vmLabel}</span><small>${S("goal", "target")} ${t.von_mises * 100} %</small></span><span class="col" style="text-align:right">${S("Time", "Train time")}</span></div>
    ${rows}
    <div class="erow eaxis"><span></span><div class="metric"><span class="metric-val"></span>${axis}</div><div class="metric"><span class="metric-val"></span>${axis}</div><span></span></div>
    <div class="legend"><span><span class="sw a"></span>${isPaper ? "ATLAS" : "ATLAS-XPINN"}</span><span><span class="sw b"></span>${isPaper ? S("Standard physics AI", "Standard PINN") : S("One network", "Single network")}</span><span><span class="sw t"></span>${S("Goal", "Accuracy target")}${S(` ${tip("target")}`, "")}</span><span style="color:var(--muted)">${S("Log scale: each grid line is 10× smaller", "Log scale")}</span></div>
    <p class="note">${note}</p>`;
  requestAnimationFrame(() => requestAnimationFrame(() => $$("#errors .pt").forEach((p) => (p.style.left = p.dataset.left))));
}

function renderSeeds(b) {
  const entries = b.group === "paper"
    ? [["ATLAS-XPINN", b.seeds.atlas_xpinn], [S("ATLAS, one network", "ATLAS"), b.seeds.atlas]]
    : [["ATLAS-XPINN", b.seeds.atlas_xpinn], [S("ATLAS, one network of the same size", "ATLAS, parameter-matched"), b.seeds.atlas_wide]];
  const have = entries.filter(([, s]) => s);
  const title = S(`Repeat runs ${tip("seeds")}`, "Repeatability");
  if (!have.length) { $("#seeds").innerHTML = `<div class="card-head"><h2>${title}</h2></div><p class="note">No multi-seed study stored for this benchmark.</p>`; return; }
  const seeds = have[0][1].seeds;
  const total = have.reduce((a, [, s]) => a + s.n, 0);
  const ok = have.reduce((a, [, s]) => a + s.n_pass_both, 0);
  const vmLabel = S("Stress error", b.group === "paper" ? "von Mises L², excl." : "von Mises L²");
  let html = `<div class="card-head"><h2>${title}</h2><span class="meta">${S(`${seeds.length} runs, random starts`, `${plural(seeds.length, "seed", "seeds")} · ${seeds[0]}–${seeds[seeds.length - 1]}`)}</span></div>`;
  html += have.map(([name, s]) => `<div class="stat"><span class="stat-name">${esc(name)}</span><div class="kv"><span>${S("Bending error", "Displacement L²")}</span><span>${meanStd(s.disp_mean, s.disp_std)}</span></div><div class="kv"><span>${vmLabel}</span><span>${meanStd(s.vm_mean, s.vm_std)}</span></div></div>`).join("");
  html += `<div class="verdict">${passIcon(ok === total, 16)}<span>${ok === total ? `All ${total} runs meet both ${S("goals", "targets")}` : `${ok} of ${total} runs meet both ${S("goals", "targets")}`}</span></div>`;
  if (b.paired && b.paired.n) {
    const p = b.paired;
    const unit = S("runs", "seeds");
    const count = (k) => (k === p.n ? `all ${p.n} ${unit}` : `${k} of ${p.n} ${unit}`);
    const txt = p.xpinn_lower_both === p.n
      ? S(`ATLAS-XPINN beats the single network on both errors in all ${p.n} runs`, `ATLAS-XPINN is lower on both errors in all ${p.n} seeds`)
      : S(`ATLAS-XPINN beats the single network on bending in ${count(p.xpinn_lower_disp)} and on stress in ${count(p.xpinn_lower_vm)}`,
        `ATLAS-XPINN is lower on displacement in ${count(p.xpinn_lower_disp)} and on von Mises in ${count(p.xpinn_lower_vm)}`);
    const icon = p.xpinn_lower_both === p.n ? ICON.pass(16)
      : p.xpinn_lower_disp + p.xpinn_lower_vm === 0 ? ICON.fail(16) : ICON.mixed(16);
    html += `<div class="verdict">${icon}<span>${esc(txt)}</span></div>`;
  }
  $("#seeds").innerHTML = html;
}

function renderSide(b) {
  if (b.group === "paper") {
    const inv = b.inverse;
    if (!inv) { $("#sideCard").innerHTML = `<div class="card-head"><h2>Inverse design</h2></div><p class="note">No inverse design stored.</p>`; return; }
    if (simple()) {
      const was = inv.baseline != null ? `<span class="was">was ${(+inv.baseline).toFixed(2)} m</span>` : "";
      $("#sideCard").innerHTML = `
        <div class="card-head"><h2>Design it backwards ${tip("inverse")}</h2><span class="meta">ATLAS-XPINN</span></div>
        <p class="note" style="margin-top:8px">Instead of asking how a shape behaves, ATLAS finds the shape that behaves the way you ask.</p>
        <div class="big"><span>${esc(cap(inv.param || "design value"))} found</span><span class="num">${inv.value != null ? `${inv.value.toFixed(3)} m` : "—"}${was}</span></div>
        <div class="stat" style="gap:8px">
          <div class="kv"><span>Goal</span><span>${esc(goalText(inv))}</span></div>
          <div class="kv"><span>Checked with FEM ${tip("fem")}</span><span>${passIcon(inv.within_10pct)}${spct(inv.error_vs_target)} from the goal</span></div>
        </div>
        <p class="note" style="margin-top:14px">The trusted simulation re-checks the shape ATLAS found. ATLAS itself never sees the trusted answers.</p>`;
      return;
    }
    $("#sideCard").innerHTML = `
      <div class="card-head"><h2>Inverse design</h2><span class="meta">ATLAS-XPINN</span></div>
      <div class="big"><span>${esc(cap(inv.param || "Design parameter"))} found</span><span class="num">${inv.value?.toFixed(4) ?? "—"}</span></div>
      <div class="stat" style="gap:8px">
        <div class="kv"><span>Objective</span><span>${esc(inv.objective || inv.metric || "")}</span></div>
        <div class="kv"><span>FEM check vs target</span><span>${passIcon(inv.within_10pct)}${spct(inv.error_vs_target)}</span></div>
      </div>
      <p class="note" style="margin-top:14px">FEniCS re-solves the design ATLAS found. The solver itself never sees FEM data.</p>`;
    return;
  }
  const rows = [["atlas_xpinn", "ATLAS-XPINN", ""], ["atlas", "ATLAS", S("one network", "single network")], ["atlas_wide", "ATLAS", S("one network, same size", "parameter-matched")]]
    .filter(([k]) => b.solvers[k]);
  $("#sideCard").innerHTML = `
    <div class="card-head"><h2>${S(`Where the materials meet ${tip("interface")}`, "At the material interface")}</h2><span class="meta">${S("one run each", "seed 42")}</span></div>
    <table class="band-table"><thead><tr><th>${S("Stress error within 5 cm of the boundary", "von Mises L², within 0.05 m")}</th><th>${esc(b.inside)}</th><th>${esc(b.outside)}</th></tr></thead><tbody>
    ${rows.map(([k, n, s]) => `<tr><td>${esc(n)}${s ? `<small>${esc(s)}</small>` : ""}</td><td>${pct(b.solvers[k].band_inside)}</td><td>${pct(b.solvers[k].band_outside)}</td></tr>`).join("")}
    </tbody></table>
    <p class="note" style="margin-top:14px">${S("A single network is smooth, so it blurs the sudden change in stretching where two materials meet. ATLAS-XPINN gives each material its own network and joins them at the boundary.", "A single smooth network has to bend around the jump in strain at the interface. ATLAS-XPINN puts a subdomain boundary there instead.")}</p>`;
}

function goalText(inv) {
  const r = inv.target_ratio;
  const what = inv.param || "a dimension";
  if (inv.metric === "compliance" && r) return `a ${what} that makes it ${/hole/.test(what) ? "stretch" : "bend"} ${Math.round(Math.abs(1 - r) * 100)} % ${r < 1 ? "less" : "more"}`;
  if (inv.metric === "peak_vm" && r) return `a ${what} that cuts the peak stress by ${Math.round((1 - r) * 100)} %`;
  return inv.objective || inv.metric || "";
}

const FIGS = {
  paper: [
    ["fem_comparison", "ATLAS vs FEM", "Top row |displacement|, bottom row von Mises stress. Left to right: ATLAS-XPINN, the FEniCS reference, and the pointwise error between them."],
    ["displacement", "Displacement", "Displacement field predicted by ATLAS-XPINN."],
    ["stress", "von Mises", "von Mises stress predicted by ATLAS-XPINN, in Pa."],
    ["convergence", "Convergence", "Left: relative energy gap to the final value. Right: strain energy U and external work W, which settle at U = W / 2. Dashed line: switch from Adam to L-BFGS; the jump at iteration 1000 is where the shared warm-start network is split into subnetworks."],
    ["all_solvers", "All solvers", "Top to bottom: |displacement|, von Mises stress and |displacement error| (saturated at 20 % of the FEM maximum). Left to right: FEM, DeepXDE with library defaults, DeepXDE configured, ATLAS-XPINN."],
  ],
  multi: [
    ["fields", "Fields", "Top: von Mises stress from FEM and each solver. Bottom: the von Mises error, saturated at 10 % of the FEM 99.8th percentile. Dashed line: the material interface."],
    ["profiles", "Stress profiles", "von Mises stress along the two symmetry lines, through the interface (dotted line): FEM against each solver."],
    ["convergence", "Error vs time", "Displacement and von Mises error against training wall-clock, seed 42."],
  ],
};

const FIGS_SIMPLE = {
  fem_comparison: ["AI vs trusted answer", "Top row: how far each point moves. Bottom row: stress. Left to right: ATLAS-XPINN, the trusted simulation (FEM), and the difference between them."],
  displacement: ["How it moves", "How far each point of the part moves under the load, predicted by ATLAS-XPINN."],
  stress: ["Stress", "Stress (von Mises) predicted by ATLAS-XPINN, in pascals. The highest values mark where the part would start to give way first."],
  convergence: ["Training", "How training settled. Left: how far the stored energy still is from its final value; it falls as the network learns. Right: the energy stored in the part and the work done by the load, which settle into a fixed balance (stored energy = half the work)."],
  all_solvers: ["All methods", "Rows: movement, stress and the movement error. Columns: the trusted simulation (FEM), the standard PINN with default and tuned settings, and ATLAS-XPINN. The PINNs' maps differ visibly from FEM; ATLAS-XPINN's match it."],
  fields: ["Stress maps", "Top: stress from the trusted simulation and from each method. Bottom: each method's error. The dashed line is the boundary between the two materials."],
  profiles: ["Stress along a line", "Stress along the two edges of the plate, through the material boundary (dotted line). Where the single networks wobble near the boundary, ATLAS-XPINN follows the trusted answer more closely."],
  multi_convergence: ["Error over time", "How the bending and stress errors fall during training, for one run of each method."],
};

function renderFigures(b) {
  const figs = FIGS[b.group].filter(([k]) => b.plots[k]).map((f) => {
    const plain = simple() && FIGS_SIMPLE[b.group === "multi" && f[0] === "convergence" ? "multi_convergence" : f[0]];
    return plain ? [f[0], ...plain] : f;
  });
  if (!figs.length) { $("#figCard").innerHTML = `<div class="fig-head"><h2 class="l">Fields and validation</h2></div><div class="fig-empty">No figures stored for this benchmark.</div>`; return; }
  const cur = figs.find(([k]) => k === state.fig[b.key]) || figs[0];
  const src = b.plots[cur[0]];
  $("#figCard").innerHTML = `
    <div class="fig-head"><div><h2 class="l">${S("See the results", "Fields and validation")}</h2><span class="meta">${b.group === "paper" ? "ATLAS-XPINN" : "All solvers"} · ${esc(b.title)}</span></div>
    <div class="seg" role="group" aria-label="Figure">${figs.map(([k, l]) => `<button type="button" data-fig="${k}" aria-pressed="${k === cur[0]}">${esc(l)}</button>`).join("")}</div></div>
    <div class="fig-frame"><img src="${esc(src)}" alt="${esc(`${cur[1]}, ${b.title}`)}" data-full="${esc(src)}"></div>
    <div class="fig-foot"><p>${esc(cur[2])}</p><a class="open-link" href="${esc(src)}" target="_blank" rel="noopener">Open full size${ICON.open}</a></div>`;
}

function renderResults() {
  const r = state.results;
  if (!r.benchmarks[state.bench]) state.bench = r.order[0];
  const b = r.benchmarks[state.bench];
  renderBenchTabs();
  renderErrors(b);
  renderSeeds(b);
  renderSide(b);
  renderFigures(b);
}

async function loadResults() {
  try {
    state.results = await api("/api/results");
    renderHero();
    renderResults();
    if (state.meta) { renderSpec(); renderBudget(); }
  } catch (e) {
    $("#heroTitle").textContent = "Could not load the stored study.";
    $("#errors").innerHTML = `<p class="note">${esc(e.message)}</p>`;
  }
}

const STEPS = {
  forward: [["setup", "Setup"], ["train", "Training"], ["predict", "Prediction"], ["validate", "FEM validation"], ["plots", "Plots"]],
  inverse: [["setup", "Setup"], ["train", "Design loop"], ["validate", "FEM check"], ["plots", "Plots"]],
};
const STEPS_SIMPLE = {
  forward: [["setup", "Getting ready"], ["train", "Training"], ["predict", "Predicting"], ["validate", "Checking with FEM"], ["plots", "Drawing"]],
  inverse: [["setup", "Getting ready"], ["train", "Trying shapes"], ["validate", "Checking with FEM"], ["plots", "Drawing"]],
};

function liveMeta(key) { return (state.meta || []).find((m) => m.key === key); }

function renderBenchPick() {
  const groups = [["paper", S("One material", "Paper benchmarks")], ["multi", S("Two materials", "Multi-material")]];
  const html = groups.map(([g, label]) => {
    const ms = state.meta.filter((m) => m.group === g);
    return `<div class="opt-group-label">${esc(label)}</div>${ms.map((m) => `<label class="opt ${m.key === state.liveBench ? "on" : ""}"><input type="radio" name="bench" value="${m.key}" ${m.key === state.liveBench ? "checked" : ""}><span class="thumb">${thumb(m.key, 48, 32)}</span><span class="opt-text"><span class="opt-title">${esc(m.title)}</span><span class="opt-meta">${esc(benchSubtitle(m))}</span></span></label>`).join("")}`;
  }).join("");
  $("#benchPick").innerHTML = `<legend class="eyebrow">${S("Part", "Benchmark")}</legend>${html}`;
}

function invDefaultMetric(key) {
  return liveMeta(key)?.inverse?.metric || inverseOf(key)?.metric || "compliance";
}
function invDefaultRatio(key) {
  return liveMeta(key)?.inverse?.default_ratio ?? inverseOf(key)?.target_ratio ?? 0.75;
}
function invSelectedMetric(key) {
  return state.solver.inv_metric[key] || invDefaultMetric(key);
}
function invSelectedRatio(key) {
  const saved = state.solver.inv_ratio[key];
  return saved != null ? saved : invDefaultRatio(key);
}

function isSolverModified(key) {
  const s = state.solver;
  const d = SOLVER_DEFAULT;
  if (s.formulation !== d.formulation || s.hidden_layers !== d.hidden_layers || s.neurons !== d.neurons
    || Math.abs(s.lr - d.lr) > 1e-8 || s.precision !== d.precision || s.seed !== d.seed
    || s.warm_start !== d.warm_start || s.al_stages !== d.al_stages
    || Math.abs(s.gamma_u - d.gamma_u) > 1e-6 || Math.abs(s.gamma_t - d.gamma_t) > 1e-6
    || s.forward_epochs !== d.forward_epochs || s.forward_lbfgs !== d.forward_lbfgs || s.n_outer !== d.n_outer) {
    return true;
  }
  if (key && liveMeta(key)?.inverse) {
    if (invSelectedMetric(key) !== invDefaultMetric(key)) return true;
    if (Math.abs(invSelectedRatio(key) - invDefaultRatio(key)) > 1e-6) return true;
  }
  return false;
}

function renderCompute() {
  const box = $("#computeBox");
  if (!box) return;
  const h = state.health;
  const pref = selectedDevice();
  const prefFem = selectedFem();
  const cudaOk = !!h?.cuda_available;
  const fenicsOk = !!h?.fenics_available;
  const eff = effectiveDevice();
  const form = state.solver.formulation;
  const devDesc = eff === "cuda"
    ? `${shortGpu(h?.gpu_name)}${pref === "auto" ? " (auto)" : " (pinned)"}`
    : `CPU · ${h?.cpu_threads || 4} threads${pref === "cpu" && cudaOk ? " (GPU idle)" : ""}`;
  box.innerHTML = `
    <div class="props-head"><span class="eyebrow">${S(`Compute, Solver & FEM ${tip("device")}`, "Compute, Solver & FEM")}</span><a href="#settings" class="link-btn">${S("All settings", "Settings")}</a></div>
    <div class="seg seg-block" role="group" aria-label="Compute device">
      <button type="button" data-dev-pick="auto" aria-pressed="${pref === "auto"}">Auto</button>
      <button type="button" data-dev-pick="cuda" aria-pressed="${pref === "cuda"}" ${cudaOk ? "" : 'disabled title="No CUDA GPU detected in this Python environment"'}>GPU</button>
      <button type="button" data-dev-pick="cpu" aria-pressed="${pref === "cpu"}">CPU</button>
    </div>
    <div class="compute-meta"><span>Active: <b>${esc(devDesc)}</b></span><span>FEM: <b>${esc(effectiveFemName())}</b></span></div>
    ${!cudaOk && h?.system_gpu_name ? `<div class="compute-meta"><span>GPU detected: <b>${esc(shortGpu(h.system_gpu_name))}</b></span><button type="button" class="link-btn" data-install-cuda>Enable CUDA →</button></div>` : ""}
    <div class="seg seg-block two" role="group" aria-label="Solver architecture">
      <button type="button" data-form-pick="energy_xpinn" aria-pressed="${form === "energy_xpinn"}">ATLAS-XPINN</button>
      <button type="button" data-form-pick="energy" aria-pressed="${form === "energy"}">${S("ATLAS (1 network)", "ATLAS single")}</button>
    </div>
    <div class="field compact">
      <span>FEM validator for custom sliders &amp; design</span>
      <div class="seg seg-block" role="group" aria-label="FEM validator">
        <button type="button" data-fem-pick="auto" aria-pressed="${prefFem === "auto"}">Auto</button>
        <button type="button" data-fem-pick="fenics" aria-pressed="${prefFem === "fenics"}" ${fenicsOk ? "" : 'disabled title="WSL2 / Conda FEniCS not detected — Auto uses Built-in P2 FEM"'}>FEniCS</button>
        <button type="button" data-fem-pick="builtin" aria-pressed="${prefFem === "builtin"}">Built-in P2</button>
      </div>
    </div>`;
}

function renderSpec() {
  const m = liveMeta(state.liveBench);
  if (!m) return;
  const s = state.solver;
  const isXpinn = s.formulation === "energy_xpinn";
  const nSub = isXpinn ? m.subdomains : 1;
  const rows = [
    ["Solver", isXpinn ? "ATLAS-XPINN" : "ATLAS (single network)"],
    ["Compute", effectiveDevice() === "cuda" ? `GPU (${shortGpu(state.health?.gpu_name)})` : `CPU (${state.health?.cpu_threads || 4} threads)`],
    ["Network", `${nSub} × (${s.hidden_layers} × ${s.neurons}), tanh · ${s.precision}`],
  ];
  $("#spec").hidden = simple();
  $("#spec").innerHTML = rows.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");
  const inv = m.inverse;
  const custom = isCustom(m.key);
  const devWord = effectiveDevice() === "cuda" ? "GPU" : "CPU";
  $("#runInverse").disabled = !inv || isBusy();
  $("#runForward").textContent = isBusy() ? "Running…" : S("Run simulation", "Run forward + validate");
  $("#runInverse").textContent = S("Find a shape (inverse design)", "Run inverse design");
  const goalObj = inv ? { ...inv, target_ratio: invSelectedRatio(m.key), metric: invSelectedMetric(m.key) } : null;
  $("#runNote").innerHTML = simple()
    ? (!inv
      ? `Runs locally on ${devWord}, one run at a time, then validates against FEniCS. Finding a shape is available for the one-material parts.`
      : custom
        ? `Find a shape ${tip("inverse")} searches for ${esc(goalText(goalObj))}, starting from your settings, then FEniCS checks the result on ${devWord}.`
        : `Find a shape ${tip("inverse")} searches for ${esc(goalText(goalObj))}, then FEniCS checks the result. Runs on ${devWord}.`)
    : esc(!inv
      ? `Inverse design is available on the three paper benchmarks. Runs on ${devWord} and validates against FEniCS.`
      : custom
        ? `Inverse design optimizes ${inv.param} (${invSelectedMetric(m.key)} = ${invSelectedRatio(m.key).toFixed(2)}× baseline) from custom settings, then validates with FEniCS.`
        : `Inverse design optimizes ${inv.param} (${invSelectedMetric(m.key)} = ${invSelectedRatio(m.key).toFixed(2)}× baseline), then validates with FEniCS.`);
}

const inverseOf = (key) => state.results?.benchmarks?.[key]?.inverse;
const inverseRatio = (key) => inverseOf(key)?.target_ratio;
const inverseMetric = (key) => inverseOf(key)?.metric;

function runEstimate(key, budget) {
  const solverKey = state.solver.formulation === "energy_xpinn" ? "atlas_xpinn" : "atlas";
  const full = state.results?.benchmarks?.[key]?.solvers?.[solverKey]?.seconds
    || state.results?.benchmarks?.[key]?.solvers?.atlas_xpinn?.seconds;
  if (!full) return "";
  const share = ADAM_SHARE * budget.epochs / FULL.epochs + (1 - ADAM_SHARE) * budget.lbfgs / FULL.lbfgs;
  const cpuMult = effectiveDevice() === "cpu" ? 2.4 : 1.0;
  const widthMult = (state.solver.neurons / 64) ** 1.35 * (state.solver.hidden_layers / 4);
  const sec = full * share * cpuMult * widthMult + (isCustom(key) ? 30 : 0);
  return sec < 90 ? `about ${Math.max(5, Math.round(sec / 5) * 5)} s` : `about ${Math.max(1, Math.round(sec / 60))} min`;
}

function runText() {
  const i = stepOf(state.budget);
  const est = cap(runEstimate(state.liveBench, state.budget));
  const end = est ? ` ${est}.` : "";
  const r = RUN_STEPS[i];
  if (i === FULL_STEP) return ["Full", `The same training as the study.${end}`];
  if (i === 0) return [r.label, `${r.label} of the study's training: the fastest, but it may miss the accuracy goals.${end}`];
  if (i > 0) return [r.label, `${r.label} of the study's training, so less accurate.${end}`];
  return ["Custom", `${state.budget.epochs} Adam + ${state.budget.lbfgs} L-BFGS steps.${end}`];
}

function renderBudget() {
  const box = $("#budget");
  if (!box) return;
  if (simple()) {
    const n = RUN_STEPS.length;
    const found = stepOf(state.budget);
    const idx = found >= 0 ? found : FULL_STEP;
    const [val, hint] = runText();
    box.innerHTML = `<div class="field">
        <div class="run-top"><span class="field-label">Run length ${tip("training")}</span><b id="runVal">${esc(val)}</b></div>
        <input type="range" class="slider" data-run min="0" max="${n - 1}" step="1" value="${idx}" style="--fill:${(idx / (n - 1)) * 100}%" aria-label="Run length" aria-valuetext="${esc(val)}">
        <div class="run-ticks"><span>Quicker</span><span>Full training</span></div>
        <span class="hint" id="runHint">${esc(hint)}</span></div>`;
    return;
  }
  box.innerHTML = `<div class="grid-2">
      <label class="field">Adam epochs<input id="epochs" type="number" min="50" max="40000" step="50" value="${state.budget.epochs}"></label>
      <label class="field">L-BFGS iterations<input id="lbfgs" type="number" min="0" max="${state.maxLbfgs ?? MAX_LBFGS}" step="100" value="${state.budget.lbfgs}"></label>
    </div>
    <span class="hint">Study budget: ${FULL.epochs} Adam + ${state.lbfgs ?? FULL.lbfgs} L-BFGS, seed ${state.solver.seed}; up to ${(state.maxLbfgs ?? MAX_LBFGS).toLocaleString("en-US")} L-BFGS (${runEstimate(state.liveBench, state.budget) || "custom"}).</span>`;
}

function renderSolverCfg() {
  const box = $("#solverCfg");
  if (!box || !state.meta) return;
  const m = liveMeta(state.liveBench);
  if (!m) return;
  const s = state.solver;
  const isXpinn = s.formulation === "energy_xpinn";
  const modified = isSolverModified(m.key);
  const presets = [[3, 32, "3×32 Fast"], [4, 64, "4×64 Study"], [4, 91, "4×91 Wide"], [5, 128, "5×128 Deep"]];
  const seeds = [42, 43, 44, 45, 46];
  const inv = m.inverse;
  const invMet = inv ? invSelectedMetric(m.key) : null;
  const invRat = inv ? invSelectedRatio(m.key) : null;
  const metricLabels = { compliance: S("Stiffness / compliance", "Compliance"), peak_vm: S("Peak stress (von Mises)", "Peak von Mises"), volume: "Volume" };

  const simpleSteps = simple() ? `
    <div class="adv-sec">
      <div class="pgroup-label">Exact training steps</div>
      <div class="grid-2">
        <label class="field compact">Adam epochs<input id="epochs" type="number" min="50" max="40000" step="50" value="${state.budget.epochs}"></label>
        <label class="field compact">L-BFGS steps<input id="lbfgs" type="number" min="0" max="${state.maxLbfgs ?? MAX_LBFGS}" step="100" value="${state.budget.lbfgs}"></label>
      </div>
    </div>` : "";

  const inner = `
    ${simpleSteps}
    <div class="adv-sec">
      <div class="adv-head"><span class="pgroup-label">${S("Network & optimizer", "Architecture & optimizer")}</span></div>
      <div class="preset-row" role="group" aria-label="Network size presets">
        ${presets.map(([hl, nn, label]) => `<button type="button" class="preset-pill ${s.hidden_layers === hl && s.neurons === nn ? "on" : ""}" data-net-preset="${hl}x${nn}">${esc(label)}</button>`).join("")}
      </div>
      <div class="grid-2">
        <label class="field compact">Hidden layers<input id="cfgLayers" type="number" min="2" max="8" step="1" value="${s.hidden_layers}"></label>
        <label class="field compact">Neurons / layer<input id="cfgNeurons" type="number" min="16" max="256" step="8" value="${s.neurons}"></label>
      </div>
      <div class="grid-2">
        <label class="field compact">Learning rate<input id="cfgLr" type="number" min="0.00001" max="0.05" step="0.0005" value="${s.lr}"></label>
        <label class="field compact">Seed<input id="cfgSeed" type="number" min="0" max="999999" step="1" value="${s.seed}"></label>
      </div>
      <div class="preset-row" role="group" aria-label="Study seeds">
        ${seeds.map((sd) => `<button type="button" class="preset-pill ${s.seed === sd ? "on" : ""}" data-seed-pick="${sd}">Seed ${sd}</button>`).join("")}
      </div>
      <div class="field compact">
        <span>Floating-point precision</span>
        <div class="seg seg-block two" role="group" aria-label="Precision">
          <button type="button" data-prec-pick="float64" aria-pressed="${s.precision === "float64"}">float64 (Study)</button>
          <button type="button" data-prec-pick="float32" aria-pressed="${s.precision === "float32"}">float32 (Faster)</button>
        </div>
      </div>
    </div>

    ${isXpinn ? `
    <div class="adv-sec">
      <div class="pgroup-label">${S("Subdomain coupling (XPINN)", "XPINN warm-start & interface penalties")}</div>
      <div class="grid-2">
        <label class="field compact">Warm-start steps<input id="cfgWarm" type="number" min="0" max="20000" step="100" value="${s.warm_start}"></label>
        <label class="field compact">AL stages<input id="cfgAl" type="number" min="0" max="8" step="1" value="${s.al_stages}"></label>
      </div>
      <div class="grid-2">
        <label class="field compact">Disp. penalty γ_u<input id="cfgGu" type="number" min="0.1" max="1000" step="5" value="${s.gamma_u}"></label>
        <label class="field compact">Traction penalty γ_t<input id="cfgGt" type="number" min="0" max="100" step="0.05" value="${s.gamma_t}"></label>
      </div>
    </div>` : ""}

    ${inv ? `
    <div class="adv-sec">
      <div class="pgroup-label">${S(`Shape search settings (${inv.param})`, `Inverse design (${inv.param})`)}</div>
      <div class="grid-2">
        <label class="field compact">Objective
          <select id="cfgInvMetric">
            ${(inv.metrics || ["compliance", "peak_vm", "volume"]).map((mk) => `<option value="${mk}" ${invMet === mk ? "selected" : ""}>${esc(metricLabels[mk] || mk)}</option>`).join("")}
          </select>
        </label>
        <label class="field compact">Target × baseline<input id="cfgInvRatio" type="number" min="0.1" max="5" step="0.05" value="${(+invRat).toFixed(2)}"></label>
      </div>
      <input type="range" class="slider" id="cfgInvRatioRange" min="0.3" max="1.6" step="0.05" value="${Math.min(1.6, Math.max(0.3, invRat))}" style="--fill:${((Math.min(1.6, Math.max(0.3, invRat)) - 0.3) / 1.3) * 100}%" aria-label="Target ratio">
      <div class="grid-2">
        <label class="field compact">Max outer steps<input id="cfgInvOuter" type="number" min="1" max="100" step="1" value="${s.n_outer}"></label>
        <label class="field compact">Forward L-BFGS<input id="cfgInvLbfgs" type="number" min="0" max="8000" step="100" value="${s.forward_lbfgs}"></label>
      </div>
    </div>` : ""}`;

  const headRight = `<span style="display:flex;align-items:center;gap:8px"><span class="badge ${modified ? "custom" : ""}">${modified ? S("Modified", "Custom") : S("Defaults", "Study")}</span><button type="button" class="link-btn" id="resetSolver" ${modified ? "" : "disabled"}>Reset</button></span>`;

  if (simple()) {
    box.innerHTML = `<details class="solver-adv" id="solverAdv" ${state.advOpen ? "open" : ""}>
      <summary><span>Adjustable solver &amp; design settings</span>${headRight}</summary>
      <div class="adv-body">${inner}</div>
    </details>`;
  } else {
    box.innerHTML = `<div class="solver-adv">
      <div class="props-head"><span class="eyebrow">Solver &amp; design settings</span>${headRight}</div>
      <div class="adv-body" style="margin-top:2px">${inner}</div>
    </div>`;
  }
}

function setBudget(epochs, lbfgs) {
  state.budget = { epochs, lbfgs };
  store.set("atlas-budget", JSON.stringify(state.budget));
}

function matOptions(E, nu, lo, hi) {
  const match = MATERIALS.find(([, , e, n]) => Math.abs(e - E) < 0.05 && Math.abs(n - nu) < 0.005);
  const opts = MATERIALS.filter(([, , e]) => e >= lo && e <= hi)
    .map(([id, name, e, n]) => `<option value="${id}" ${match && match[0] === id ? "selected" : ""}>${esc(name)} · ${+e} GPa</option>`);
  if (!match) opts.unshift(`<option value="" selected>Other · ${+(+E).toFixed(1)} GPa, ν ${(+nu).toFixed(2)}</option>`);
  return opts.join("");
}

function paramValues(key) {
  const m = liveMeta(key);
  const saved = state.params[key] || {};
  const out = {};
  for (const s of m?.params || []) out[s.key] = saved[s.key] ?? s.value;
  return out;
}

function isCustom(key) {
  const m = liveMeta(key);
  const v = paramValues(key);
  return (m?.params || []).some((s) => Math.abs(v[s.key] - s.value) > 1e-9 * Math.max(1, Math.abs(s.value)));
}

const fmtParam = (v, step) => { const d = (String(step).split(".")[1] || "").length; return (+v).toFixed(d); };
const fillPct = (s, v) => `${((v - s.min) / (s.max - s.min)) * 100}%`;

function renderProps() {
  const m = liveMeta(state.liveBench);
  if (!m) return;
  const vals = paramValues(m.key);
  const groups = [];
  for (const s of m.params) {
    let g = groups.find((x) => x.name === s.group);
    if (!g) groups.push((g = { name: s.group, items: [] }));
    g.items.push(s);
  }
  const label = (s) => {
    const base = s.key.replace(/\d$/, "");
    const text = simple() && SIMPLE_LABEL[base] ? SIMPLE_LABEL[base] : s.label;
    return `${esc(text)}${base === "E" || base === "nu" ? ` ${tip(base)}` : ""}`;
  };
  const row = (s) => `
      <div class="prow" data-key="${s.key}">
        <div class="prow-top"><span>${label(s)}</span><span class="pnum"><input type="number" data-num="${s.key}" min="${s.min}" max="${s.max}" step="${s.step}" value="${fmtParam(vals[s.key], s.step)}" aria-label="${esc(s.label)}"><span>${esc(s.unit)}</span></span></div>
        <input type="range" class="slider" data-range="${s.key}" min="${s.min}" max="${s.max}" step="${s.step}" value="${vals[s.key]}" style="--fill:${fillPct(s, vals[s.key])}" aria-label="${esc(s.label)}">
      </div>`;
  const groupHtml = (g) => {
    const eSpec = g.items.find((x) => /^E\d?$/.test(x.key));
    const nuSpec = eSpec && g.items.find((x) => x.key === `nu${eSpec.key.slice(1)}`);
    if (!eSpec || !nuSpec) return g.items.map(row).join("");
    const rest = g.items.filter((x) => x !== eSpec && x !== nuSpec);
    const select = `<label class="mat"><span class="sr-only">${esc(g.name)} material</span><select class="mat-select" data-mat="${eSpec.key}" data-mat-nu="${nuSpec.key}">${matOptions(vals[eSpec.key], vals[nuSpec.key], eSpec.min, eSpec.max)}</select></label>`;
    const exact = simple()
      ? `<details class="fine"><summary>Exact values</summary>${row(eSpec)}${row(nuSpec)}</details>`
      : row(eSpec) + row(nuSpec);
    return select + exact + rest.map(row).join("");
  };
  $("#props").innerHTML = `
    <div class="props-head"><span class="eyebrow">${S("Materials and load", "Properties")}</span><span style="display:flex;align-items:center;gap:10px"><span class="badge" id="propsBadge"></span><button type="button" class="link-btn" id="resetProps">Reset</button></span></div>
    <div class="geo" id="geo" role="img" aria-label="Geometry preview"></div>
    ${groups.map((g) => `<div class="pgroup"><div class="pgroup-label">${esc(g.name)}</div>${groupHtml(g)}</div>`).join("")}
    <p class="props-note" id="propsNote"></p>`;
  updatePropsChrome();
  geoTarget(m.key, vals, true);
}

function updatePropsChrome() {
  const m = liveMeta(state.liveBench);
  const vals = paramValues(m.key);
  const custom = isCustom(m.key);
  const badge = $("#propsBadge");
  badge.className = `badge ${custom ? "custom" : ""}`;
  badge.textContent = custom ? S("Your settings", "Custom settings") : S("Original settings", "Study settings");
  $("#resetProps").disabled = !custom;
  $("#propsNote").textContent = custom
    ? S("The server first runs the trusted simulation (FEM) for exactly these settings, which takes a few seconds, then compares ATLAS with it.",
      "The server solves a fresh FEniCS reference for exactly these inputs (a few seconds), then scores ATLAS-XPINN against it.")
    : S("These are the settings used in the study, so ATLAS is compared with the stored trusted answer.",
      "These are the study's settings: the run is scored against the stored FEniCS reference.");
  for (const s of m.params) {
    const row = $(`.prow[data-key="${s.key}"]`);
    if (row) row.classList.toggle("changed", Math.abs(vals[s.key] - s.value) > 1e-9 * Math.max(1, Math.abs(s.value)));
  }
  $$(".mat-select").forEach((sel) => {
    const e = m.params.find((x) => x.key === sel.dataset.mat);
    sel.innerHTML = matOptions(vals[sel.dataset.mat], vals[sel.dataset.matNu], e.min, e.max);
  });
  renderSpec();
  renderBudget();
}

let saveTimer = null;
function setParam(key, raw, source) {
  const m = liveMeta(state.liveBench);
  const s = m.params.find((x) => x.key === key);
  let v = parseFloat(raw);
  if (!Number.isFinite(v)) return;
  v = Math.min(s.max, Math.max(s.min, v));
  state.params[m.key] = { ...(state.params[m.key] || {}), [key]: v };
  const range = $(`[data-range="${key}"]`), num = $(`[data-num="${key}"]`);
  if (source !== "range") range.value = v;
  if (source !== "num") num.value = fmtParam(v, s.step);
  range.style.setProperty("--fill", fillPct(s, v));
  updatePropsChrome();
  geoTarget(m.key, paramValues(m.key));
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => store.set("atlas-params", JSON.stringify(state.params)), 300);
}

function geoTarget(bench, vals, snap = false) {
  const m = liveMeta(bench);
  const target = { ...vals };
  for (const s of m.params) target[`_d_${s.key}`] = s.value;
  if (!state.geo || state.geo.bench !== bench || snap || reduced()) state.geo = { bench, cur: { ...target }, target, raf: null };
  else state.geo.target = target;
  const g = state.geo;
  const tick = () => {
    let moving = false;
    for (const k of Object.keys(g.target)) {
      const d = g.target[k] - g.cur[k];
      if (Math.abs(d) > 1e-4 * Math.max(1, Math.abs(g.target[k]))) { g.cur[k] += d * 0.2; moving = true; } else g.cur[k] = g.target[k];
    }
    drawGeo(g.bench, g.cur);
    g.raf = moving ? requestAnimationFrame(tick) : null;
  };
  if (!g.raf) tick();
}

function stiffShade(ratio) {
  const k = Math.min(92, (Math.abs(Math.log10(ratio)) / 2.2) * 100);
  return ratio >= 1 ? `color-mix(in srgb, var(--ink) ${k.toFixed(0)}%, var(--thumb-fill))` : `color-mix(in srgb, var(--surface) ${k.toFixed(0)}%, var(--thumb-fill))`;
}

function arrow(x1, y1, x2, y2, px, py) {
  const a = Math.atan2(y2 - y1, x2 - x1), h = 6;
  const p1 = [x2 - h * Math.cos(a - 0.45), y2 - h * Math.sin(a - 0.45)], p2 = [x2 - h * Math.cos(a + 0.45), y2 - h * Math.sin(a + 0.45)];
  const st = `style="--push-x:${px}px;--push-y:${py}px"`;
  return `<line class="g-load" ${st} x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}"/><path class="g-head" ${st} d="M${x2.toFixed(1)} ${y2.toFixed(1)}L${p1[0].toFixed(1)} ${p1[1].toFixed(1)}L${p2[0].toFixed(1)} ${p2[1].toFixed(1)}Z"/>`;
}

function drawGeo(bench, p) {
  const el = $("#geo");
  if (!el) return;
  const W = 320, Hh = 176;
  const loadScale = (key) => Math.max(0.5, Math.min(2.2, Math.sqrt(p[key] / p[`_d_${key}`])));
  const txt = (x, y, s, anchor = "start") => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${anchor}" style="font:10px var(--font-mono);fill:var(--muted)">${esc(s)}</text>`;
  let svg = "";
  if (bench === "cantilever") {
    const k = Math.min((W - 84) / 8, (Hh - 50) / 2), x0 = 30, yc = 84;
    const L = p.L * k, H = p.H * k;
    svg += `<rect class="g-body" x="${x0}" y="${(yc - H / 2).toFixed(1)}" width="${L.toFixed(1)}" height="${H.toFixed(1)}"/>`;
    svg += `<line class="g-dash" x1="${(x0 + L / 2).toFixed(1)}" x2="${(x0 + L / 2).toFixed(1)}" y1="${(yc - H / 2).toFixed(1)}" y2="${(yc + H / 2).toFixed(1)}"/>`;
    svg += `<line class="g-hatch" x1="${x0}" x2="${x0}" y1="${(yc - H / 2 - 8).toFixed(1)}" y2="${(yc + H / 2 + 8).toFixed(1)}" style="stroke-width:2"/>`;
    for (let y = yc - H / 2 - 6; y < yc + H / 2 + 6; y += 7) svg += `<line class="g-hatch" x1="${x0}" y1="${y.toFixed(1)}" x2="${x0 - 7}" y2="${(y + 7).toFixed(1)}"/>`;
    const len = 26 * loadScale("load"), xa = x0 + L + 12;
    svg += arrow(xa, yc - len / 2, xa, yc + len / 2, 0, 3);
    svg += txt(x0, Hh - 10, `L ${(+p.L).toFixed(1)} m · H ${(+p.H).toFixed(1)} m · ${gpa(p.E * 1e9)}`);
  } else if (bench === "l_bracket") {
    const k = (Hh - 44) / 2, x0 = 70, yb = Hh - 24;
    const X = (x) => x0 + x * k, Y = (y) => yb - y * k;
    svg += `<path class="g-body" d="M${X(0)} ${Y(0)}L${X(2)} ${Y(0)}L${X(2)} ${Y(2)}L${X(1)} ${Y(2)}L${X(1)} ${Y(1)}L${X(0)} ${Y(1)}Z"/>`;
    svg += `<path class="g-dash" d="M${X(1)} ${Y(0)}L${X(1)} ${Y(1)}L${X(2)} ${Y(1)}"/>`;
    svg += `<line class="g-hatch" x1="${X(0)}" x2="${X(0)}" y1="${Y(1) - 6}" y2="${Y(0) + 6}" style="stroke-width:2"/>`;
    for (let y = Y(1) - 4; y < Y(0) + 4; y += 7) svg += `<line class="g-hatch" x1="${X(0)}" y1="${y.toFixed(1)}" x2="${X(0) - 7}" y2="${(y + 7).toFixed(1)}"/>`;
    const sc = loadScale("load");
    for (let i = 1; i <= 5; i++) { const u = i / 6, len = 34 * sc * 4 * u * (1 - u); svg += arrow(X(2) + 4, Y(1 + u), X(2) + 4 + len, Y(1 + u), 3, 0); }
    svg += txt(12, 20, gpa(p.E * 1e9)) + txt(12, 34, `ν ${(+p.nu).toFixed(2)}`);
  } else {
    const k = (Hh - 48) / 2, x0 = 64, yb = Hh - 16;
    const X = (x) => x0 + x * k, Y = (y) => yb - y * k;
    const sc = loadScale("sigma");
    const groupOf = (key) => liveMeta(bench).params.find((s) => s.key === key)?.group || "";
    if (bench === "plate_hole") {
      const r = p.r * k;
      svg += `<path class="g-body" d="M${(x0 + r).toFixed(1)} ${yb}L${X(2)} ${yb}L${X(2)} ${Y(2)}L${x0} ${Y(2)}L${x0} ${(yb - r).toFixed(1)}A${r.toFixed(1)} ${r.toFixed(1)} 0 0 1 ${(x0 + r).toFixed(1)} ${yb}Z"/>`;
      for (const R of [p.r + 0.3, p.r + 0.6]) svg += `<path class="g-dash" d="M${x0} ${(yb - R * k).toFixed(1)}A${(R * k).toFixed(1)} ${(R * k).toFixed(1)} 0 0 1 ${(x0 + R * k).toFixed(1)} ${yb}"/>`;
      svg += txt(x0 - 8, 14, `${gpa(p.E * 1e9)} · ν ${(+p.nu).toFixed(2)} · r ${(+p.r).toFixed(2)} m`);
    } else {
      const a = (bench === "filled_slot" ? p.a : p.r) * k, b = (bench === "filled_slot" ? p.b : p.r) * k;
      svg += `<rect class="g-body" x="${x0}" y="${Y(2)}" width="${2 * k}" height="${2 * k}"/>`;
      svg += `<path class="g-inc" style="fill:${stiffShade(p.E2 / p.E1)}" d="M${x0} ${yb}L${(x0 + a).toFixed(1)} ${yb}A${a.toFixed(1)} ${b.toFixed(1)} 0 0 0 ${x0} ${(yb - b).toFixed(1)}Z"/>`;
      svg += txt(x0 - 8, 14, `${groupOf("E1")} ${gpa(p.E1 * 1e9)} · ${groupOf("E2")} ${gpa(p.E2 * 1e9)}`);
    }
    for (let i = 0; i < 6; i++) svg += `<circle class="g-roll" cx="${(X(0.2 + i * 0.33)).toFixed(1)}" cy="${yb + 4}" r="3"/><circle class="g-roll" cx="${x0 - 4}" cy="${(Y(0.2 + i * 0.33)).toFixed(1)}" r="3"/>`;
    for (let i = 0; i < 5; i++) { const y = Y(0.2 + i * 0.4); svg += arrow(X(2) + 3, y, X(2) + 3 + 22 * sc, y, 3, 0); }
  }
  el.innerHTML = `<svg viewBox="0 0 ${W} ${Hh}" preserveAspectRatio="xMidYMid meet">${svg}</svg>`;
}

function renderEnergy(j) {
  const box = $("#energy");
  const pts = j?.progress?.points || [];
  if (j?.kind === "inverse" || pts.length < 3) { box.hidden = true; return; }
  box.hidden = false;
  if (!$("#eLine")) {
    box.innerHTML = `<div class="energy-head"><span id="eHead"></span><span><b id="eVal"></b> · <span id="eN"></span></span></div>
      <div class="energy-plot"><svg viewBox="0 0 600 120" preserveAspectRatio="none" aria-hidden="true"><defs><linearGradient id="energyFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="var(--step-active)" stop-opacity=".3"/><stop offset="1" stop-color="var(--step-active)" stop-opacity="0"/></linearGradient></defs><line class="energy-grid" x1="0" x2="600" y1="119.5" y2="119.5"/><path class="energy-area" id="eArea"/><path class="energy-line" id="eLine"/></svg><span class="energy-dot" id="eDot"></span></div>
      <div class="energy-note" id="eNote"></div>`;
  }
  const n = pts[pts.length - 1][0];
  const best = Math.min(...pts.map((q) => q[1]));
  const positive = pts.map((q) => q[1] - best).filter((g) => g > 0);
  const floor = positive.length ? Math.min(...positive) : 1e-9;
  const gap = pts.map(([x, y]) => [x, Math.log10(Math.max(y - best, floor))]);
  const top = Math.max(...gap.map((q) => q[1]));
  const bottom = Math.log10(floor) - 0.08 * (top - Math.log10(floor) || 1);
  const span = top - bottom || 1;
  const N = 120, sample = [];
  let j0 = 0;
  for (let i = 0; i < N; i++) {
    const x = (i / (N - 1)) * n;
    while (j0 < gap.length - 2 && gap[j0 + 1][0] < x) j0++;
    const [xa, ya] = gap[j0], [xb, yb] = gap[Math.min(j0 + 1, gap.length - 1)];
    const w = xb > xa ? Math.min(1, Math.max(0, (x - xa) / (xb - xa))) : 0;
    sample.push([(x / n) * 600, 6 + (1 - (ya + w * (yb - ya) - bottom) / span) * 108]);
  }
  const d = sample.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join("");
  const area = `${d}L600 120L0 120Z`;
  const line = $("#eLine"), fill = $("#eArea");
  line.setAttribute("d", d); fill.setAttribute("d", area);
  line.style.d = `path("${d}")`; fill.style.d = `path("${area}")`;
  const last = sample[N - 1];
  const dot = $("#eDot");
  dot.style.left = `${(last[0] / 600) * 100}%`;
  dot.style.top = `${(last[1] / 120) * 100}%`;
  dot.classList.toggle("live", j.status === "running");
  $("#eHead").textContent = S("Training progress: distance from the lowest energy reached (log scale)", "Energy gap to the lowest Π reached, log scale");
  $("#eVal").textContent = S(`energy ${pts[pts.length - 1][1].toFixed(4)}`, `Π = ${pts[pts.length - 1][1].toFixed(4)}`);
  $("#eN").textContent = S(`${n.toLocaleString("en-US")} steps`, `${n.toLocaleString("en-US")} energy evaluations`);
  $("#eNote").textContent = j.status === "running"
    ? S("The network keeps adjusting itself to lower the energy stored in the part; when the line flattens, it has found the answer. The small upward jumps are planned: they are where the separate networks start training on their own and where their borders are tightened.",
      "Updates as the solver runs. Upward jumps are the switch from the shared warm start to separate subnetworks and each augmented-Lagrangian update.")
    : S("The whole training run.", "The complete descent of this run.");
}

function isBusy() { return state.jobs.some((j) => j.status === "queued" || j.status === "running"); }

function renderJobs() {
  const ul = $("#jobs");
  const clearBtn = $("#clearJobsBtn");
  if (clearBtn) clearBtn.hidden = !state.jobs.some((j) => j.status !== "queued" && j.status !== "running");
  if (!state.jobs.length) { ul.innerHTML = `<li><p class="empty">No jobs yet this session.</p></li>`; return; }
  ul.innerHTML = state.jobs.slice(0, 10).map((j) => {
    const m = liveMeta(j.benchmark);
    const cls = j.status === "done" ? "ok" : j.status === "error" ? "bad" : j.status === "cancelled" ? "" : "busy";
    const right = j.status === "done" ? dur(secondsBetween(j.started, j.finished)) : j.status === "error" ? "failed" : j.status === "cancelled" ? "stopped" : j.status;
    const devTag = j.device === "cuda" ? "GPU" : j.device === "cpu" ? "CPU" : "";
    const formTag = j.formulation === "energy" ? "1-net" : "XPINN";
    return `<li><button type="button" data-job="${j.id}" aria-current="${state.job?.id === j.id}"><span class="dot ${cls}"></span><span class="job-label">${j.kind === "inverse" ? S("Shape search", "Inverse") : S("Simulation", "Forward")} · ${esc(m?.title || j.benchmark)} <span class="job-meta-tag">${esc(devTag ? `${formTag} · ${devTag}` : formTag)}</span></span><span class="job-right">${esc(right)}</span></button></li>`;
  }).join("");
}

function renderJob() {
  const j = state.job;
  const busy = isBusy();
  $("#runForward").disabled = busy;
  $("#runForward").textContent = busy ? "Running…" : S("Run simulation", "Run forward + validate");
  const m = liveMeta(state.liveBench);
  $("#runInverse").disabled = busy || !m?.inverse;
  const cancelBtn = $("#cancelJob");
  const exportBtn = $("#exportJob");
  const activeRun = j && (j.status === "queued" || j.status === "running");
  if (cancelBtn) {
    cancelBtn.hidden = !activeRun;
    cancelBtn.disabled = !!j?.cancel_requested;
    cancelBtn.textContent = j?.cancel_requested ? "Stopping…" : S("Stop", "Stop run");
  }
  if (exportBtn) {
    exportBtn.hidden = !(j && j.status === "done");
    if (j && j.status === "done") exportBtn.href = `/api/jobs/${j.id}/export`;
  }
  const steps = (simple() ? STEPS_SIMPLE : STEPS)[j?.kind === "inverse" ? "inverse" : "forward"];
  if (!j) {
    $("#jobTitle").textContent = S("Nothing run yet", "No run yet");
    $("#jobSub").textContent = S("Pick a part on the left, choose GPU or CPU if needed, then press Run simulation.", "Pick a benchmark, choose compute/solver settings on the left, then run.");
    $("#jobChip").className = "chip";
    $("#jobChip").textContent = "Idle";
    $("#progress").hidden = true;
    $("#jobError").hidden = true;
    $("#log").textContent = "—";
  } else {
    const jm = liveMeta(j.benchmark);
    const secs = j.metrics?.train_seconds ?? secondsBetween(j.started, j.finished);
    const bud = j.budget;
    const step = bud ? stepOf(bud) : FULL_STEP;
    const fullRun = step === FULL_STEP;
    const solverLabel = j.formulation === "energy" ? "ATLAS single" : "ATLAS-XPINN";
    const devLabel = j.device === "cuda" ? `GPU (${shortGpu(j.device_name)})` : j.device ? (j.device_name || "CPU") : "";
    $("#jobTitle").textContent = j.kind === "inverse" ? S("Finding a shape", "Inverse design") : S("Simulation, checked with FEniCS", "Forward solve + FEniCS validation");
    $("#jobSub").textContent = simple()
      ? `${jm?.title || j.benchmark} · ${devLabel ? `${devLabel} · ` : ""}${j.custom ? "your settings" : "original settings"}${j.kind === "inverse" ? "" : fullRun ? " · full training" : step >= 0 ? ` · ${RUN_STEPS[step].label} training` : " · custom steps"}${j.status === "done" ? ` · finished in ${dur(secs)}` : ""}`
      : `${jm?.title || j.benchmark} · ${solverLabel}${devLabel ? ` · ${devLabel}` : ""} · ${j.custom ? "custom settings" : "study settings"}${bud && !fullRun && j.kind !== "inverse" ? ` · ${bud.epochs} Adam + ${bud.lbfgs} L-BFGS` : ""} · seed ${j.metrics?.seed ?? j.solver_config?.seed ?? state.seed ?? 42}${j.status === "done" ? ` · finished in ${dur(secs)}` : ""}`;
    const chip = {
      queued: ["run", S("Waiting in line", "Queued")],
      running: ["run", j.cancel_requested ? "Stopping…" : "Running"],
      done: ["done", "Done"],
      cancelled: ["cancel", "Stopped"],
      error: ["fail", "Failed"],
    }[j.status] || ["", j.status];
    $("#jobChip").className = `chip ${chip[0]}`;
    $("#jobChip").textContent = chip[1];
    $("#progress").hidden = !(j.status === "running" || j.status === "queued");
    $("#jobError").hidden = !j.error;
    $("#jobError").textContent = j.error || "";
    const pre = $("#log");
    const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 24;
    pre.textContent = j.log || (j.status === "queued" ? S("Waiting for the run before yours to finish…", "Waiting in queue…") : "Solver output appears here as it is written.");
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  }
  const idx = !j ? -1 : j.status === "done" ? steps.length : Math.max(0, steps.findIndex(([k]) => k === j.stage));
  $("#steps").innerHTML = steps.map(([, label], i) => {
    const cls = i < idx ? "done" : i === idx ? (j?.status === "error" ? "failed" : j?.status === "queued" || j?.status === "cancelled" ? "" : "active") : "";
    return `<li class="${cls}"><span class="step-dot">${i < idx ? ICON.check : ""}</span><span class="step-label">${esc(label)}</span>${i < steps.length - 1 ? '<span class="step-line"></span>' : ""}</li>`;
  }).join("");
  renderEnergy(j);
  renderTiles(j);
  renderLiveFigure(j);
  renderDetails(j);
}

function tile(label, value, sub, ok) {
  const dash = value == null;
  return `<article class="card tile"><span class="tile-label">${esc(label)}</span><span class="tile-value ${dash ? "dash" : ""}"><span>${dash ? "—" : esc(value)}</span>${ok != null && !dash ? passIcon(ok, 18) : ""}</span><span class="tile-sub">${esc(sub)}</span></article>`;
}

function renderTiles(j) {
  const t = state.results?.targets || { displacement: 0.05, von_mises: 0.1 };
  const done = j?.status === "done";
  const mx = done ? j.metrics : {};
  const runDevText = (mx.device || j?.device) === "cuda" ? `GPU (${shortGpu(mx.device_name || j?.device_name)})` : (mx.device_name || j?.device_name || state.health?.device || "CPU");
  const waiting = j && !done ? (j.status === "error" ? "failed" : j.status === "cancelled" ? "stopped" : "running") : "no run yet";
  if (j?.kind === "inverse") {
    const fc = mx.fem_check || {};
    const ratio = fc.target_ratio;
    const change = ratio != null ? `${ratio < 1 ? "−" : "+"}${Math.round(Math.abs(1 - ratio) * 100)} %` : null;
    const what = mx.metric === "peak_vm" ? "peak stress" : mx.metric === "compliance" ? (/hole/.test(mx.param || "") ? "stretching" : "bending") : mx.metric;
    const near = (x) => mx.bounds && Math.abs(mx.final - x) < 1e-6;
    const edge = done && !mx.converged ? (near(mx.bounds?.[1]) ? "largest" : near(mx.bounds?.[0]) ? "smallest" : "") : null;
    const stop = edge == null ? "" : edge ? S(` · stopped at the ${edge} size allowed`, ` · at the ${edge === "largest" ? "upper" : "lower"} bound`) : S(" · did not fully settle", ` · ${mx.status}`);
    $("#tiles").innerHTML = [
      tile(done ? `${cap(mx.param)} found` : S("Shape found", "Design found"), done ? `${mx.final?.toFixed(3)} m` : null, done ? `started from ${mx.initial?.toFixed(3)} m${stop}` : waiting),
      tile(S("Goal", "Objective"), done ? change : null, done ? S(`change in ${what} from the start`, `${mx.objective}: ${sci(mx.target)}`) : waiting),
      tile(S("Checked with FEM", "FEM check"), done && fc.error_vs_target != null ? spct(fc.error_vs_target) : null,
        done ? S("from the goal, verified by FEniCS", `FEniCS ${what} at the design vs its target`) : waiting, done && fc.within_10pct != null ? fc.within_10pct : null),
      tile("Run time", done ? dur(mx.train_seconds) : null, done ? `design loop on ${runDevText}` : waiting),
    ].join("");
    return;
  }
  const vmKey = mx.von_mises_l2_excl != null ? "von_mises_l2_excl" : "von_mises_l2";
  $("#tiles").innerHTML = [
    tile(S("Bending error", "Displacement L²"), done ? pct(mx.relative_L2_displacement) : null, S(`vs FEniCS · goal under ${t.displacement * 100} %`, `relative to FEniCS · target < ${t.displacement * 100} %`), done ? mx.relative_L2_displacement < t.displacement : null),
    tile(S("Stress error", vmKey === "von_mises_l2_excl" || !done ? "von Mises L², excl." : "von Mises L²"), done ? pct(mx[vmKey]) : null, S(`vs FEniCS · goal under ${t.von_mises * 100} %`, `relative to FEniCS · target < ${t.von_mises * 100} %`), done ? mx[vmKey] < t.von_mises : null),
    tile(S("Training time", "Run time"), done ? dur(mx.train_seconds) : null, done ? `wall-clock on ${runDevText}` : waiting),
    tile(S("Network size", "Parameters"), done ? mx.params?.toLocaleString("en-US") : null, done ? S(`adjustable numbers · ${plural(mx.subdomains, "network", "networks")}`, `${plural(mx.subdomains, "subnetwork", "subnetworks")} · ${mx.hidden_layers || 4}×${mx.neurons || 64}`) : waiting),
  ].join("");
}

function renderLiveFigure(j) {
  const card = $("#liveFig");
  const done = j?.status === "done";
  const figs = j?.kind === "inverse" ? [["design", S("Shape", "Design")], ["inverse_history", S("Steps", "History")]]
    : [["fem_comparison", S("Comparison", "ATLAS vs FEM")], ["deformed", S("Deformed shape", "Deformed")], ["convergence", S("Training", "Convergence")]];
  const avail = done ? figs.filter(([k]) => j.plots?.[k]) : [];
  const cur = avail.find(([k]) => k === state.fig.live) || avail[0];
  const m = j ? liveMeta(j.benchmark) : null;
  const solverTag = j?.formulation === "energy" ? "ATLAS single" : "ATLAS-XPINN";
  const head = `<div class="fig-head"><div><h2>${j?.kind === "inverse" ? S("The shape ATLAS found", "Design") : S("ATLAS vs the trusted answer", "ATLAS vs FEniCS")}</h2>${m ? `<span class="meta">${esc(m.title)} · ${esc(solverTag)}</span>` : ""}</div>${avail.length > 1 ? `<div class="seg" role="group" aria-label="Figure">${avail.map(([k, l]) => `<button type="button" data-livefig="${k}" aria-pressed="${k === cur[0]}">${esc(l)}</button>`).join("")}</div>` : ""}</div>`;
  if (!cur) {
    const msg = !j ? "The FEniCS comparison appears here when a run finishes."
      : j.status === "error" ? "The run failed before producing figures."
        : j.status === "cancelled" ? "Run stopped before figures were generated."
          : "The figures appear here when the run finishes.";
    card.innerHTML = `${head}<div class="fig-empty">${esc(msg)}</div>`;
    return;
  }
  const src = j.plots[cur[0]];
  card.innerHTML = `${head}<div class="fig-frame sm"><img src="${esc(src)}" alt="${esc(`${cur[1]}, ${m?.title || ""}`)}" data-full="${esc(src)}"></div><div class="fig-foot"><p>${esc(liveCaption(j, cur[0]))}</p><a class="open-link" href="${esc(src)}" target="_blank" rel="noopener">Open full size${ICON.open}</a></div>`;
}

function liveCaption(j, fig) {
  const studyPlot = !j.custom && liveMeta(j.benchmark)?.group === "paper";
  if (fig === "fem_comparison") {
    return studyPlot
      ? S("Top row: how far each point moves. Bottom row: stress. Left to right: ATLAS-XPINN, the trusted simulation (FEM), and the difference between them.",
        "Top row |displacement|, bottom row von Mises stress. Left to right: ATLAS-XPINN, the FEniCS reference, and the pointwise error.")
      : S("How far each point moves (|u|) and the stress, each shown three ways: the trusted simulation (FEM), ATLAS-XPINN, and the difference between them. The more alike the first two look, and the darker the difference, the closer the match.",
        "|u| and von Mises stress: FEM, ATLAS-XPINN and the pointwise error (saturated at 10 % of the FEM 99.8th percentile of each field).");
  }
  if (fig === "deformed") {
    return studyPlot
      ? S("The shape ATLAS-XPINN predicts under the load (red) over the unloaded part (blue), with the movement exaggerated so it is visible (the factor is in the title).",
        "Deformed (red) and undeformed (blue) grids from ATLAS-XPINN, displacement scaled by the factor in the title.")
      : S("The shape ATLAS-XPINN predicts under the load, with the movement exaggerated so it is visible (the factor is in the title). Colour shows the stress; grey is the unloaded part.",
        "Deformed configuration predicted by ATLAS-XPINN, displacement scaled by the factor in the title, coloured by von Mises stress.");
  }
  if (fig === "convergence") {
    return S("How training settled: the energy gap falls as the network learns, and the stored energy and the work done by the load settle into balance.",
      "Relative energy gap to the final value, and strain energy U against external work W (U = W / 2 at the solution).");
  }
  if (fig === "design") {
    return S("Left: the starting shape (dashed) and the shape ATLAS found (solid). Middle and right: how far each point of the new shape moves, and its stress.",
      "Baseline (dashed) and designed (solid) geometry, with |u| and von Mises stress of the design from ATLAS-XPINN.");
  }
  if (fig === "inverse_history") {
    return S("Each step of the search: the size tried, how close it came to the goal (red dashed line), and the gap left, which shrinks as the search settles.",
      "Design parameter, objective against its target (dashed) and residual |metric / target − 1| per outer iteration.");
  }
  return "";
}

function renderDetails(j) {
  const card = $("#details");
  card.hidden = simple();
  const done = j?.status === "done";
  if (!done) {
    card.innerHTML = `<h2>${j?.kind === "inverse" ? "Design detail" : "Validation detail"}</h2><p class="note" style="margin-top:12px">${j?.kind === "inverse" ? "The design history appears when the run finishes." : "Per-component errors appear after FEM validation."}</p>`;
    return;
  }
  const mx = j.metrics;
  const m = liveMeta(j.benchmark);
  let rows;
  if (j.kind === "inverse") {
    const fc = mx.fem_check || {};
    const unit = mx.metric === "peak_vm" ? "Pa" : mx.metric === "compliance" ? "N·m" : "";
    const val = (x) => (x == null ? "—" : `${sci(x)} ${unit}`.trim());
    rows = [
      [`${cap(mx.param)}, start`, "baseline geometry", `${mx.initial?.toFixed(4)} m`],
      [`${cap(mx.param)}, found`, "ATLAS-XPINN design", `${mx.final?.toFixed(4)} m`],
      ["Objective", mx.objective, val(mx.target)],
      ["Status", `outer design loop, ${plural(mx.n_outer, "step", "steps")}`, mx.converged ? "converged" : mx.status],
      ["FEM at the start", `${mx.metric}, starting shape`, val(fc.fem_baseline)],
      ["FEM target", `${fc.target_ratio != null ? (+fc.target_ratio).toFixed(2) : "—"} × the FEM start`, val(fc.fem_target)],
      ["FEM at the design", "the shape ATLAS-XPINN found", val(fc.fem_at_design)],
      ["ATLAS-XPINN vs FEM", `${mx.metric} at the design`, spct(fc.atlas_vs_fem)],
    ];
  } else if (m?.group === "multi") {
    rows = [
      ["Displacement L², u", "horizontal component", pct(mx.relative_L2_u)],
      ["Displacement L², v", "vertical component", pct(mx.relative_L2_v)],
      ["von Mises L²", "whole part, no exclusion", pct(mx.von_mises_l2 ?? mx.von_mises_l2_global)],
      ["Compliance vs FEM", "work of the applied load", spct(mx.compliance_error)],
      [`von Mises L², ${m.inside.toLowerCase()}`, "within 0.05 m of the interface", pct(mx.band_vm_l2_inside)],
      [`von Mises L², ${m.outside.toLowerCase()}`, "within 0.05 m of the interface", pct(mx.band_vm_l2_outside)],
    ];
  } else {
    const loc = mx.max_pointwise_von_mises_error_location;
    rows = [
      ["Displacement L², u", "horizontal component", pct(mx.relative_L2_u)],
      ["Displacement L², v", "vertical component", pct(mx.relative_L2_v)],
      ["von Mises L², global", "no exclusion", pct(mx.von_mises_l2_global)],
      mx.custom
        ? ["Compliance vs FEM", "work of the applied load", spct(mx.compliance_error)]
        : ["von Mises L², ROI", `${mx.von_mises_roi_desc || "region of interest"} left out`, pct(mx.von_mises_l2_roi)],
      ["Max pointwise von Mises error", Array.isArray(loc) ? `at (x=${(+loc[0]).toFixed(4)}, y=${(+loc[1]).toFixed(4)})` : "location not reported", mx.max_pointwise_von_mises_error != null ? `${sci(mx.max_pointwise_von_mises_error)} Pa` : "—"],
      ["Points left out near singularities", mx.von_mises_exclusion_radius != null ? `r = ${(+mx.von_mises_exclusion_radius).toPrecision(4)}` : "", mx.von_mises_total_points != null ? `${mx.von_mises_excluded_points} of ${mx.von_mises_total_points.toLocaleString("en-US")}` : "—"],
    ];
  }
  if (mx.device_name || mx.precision) {
    rows.push(["Compute runtime", `${mx.precision || "float64"} · seed ${mx.seed ?? 42}`, mx.device_name || mx.device || "—"]);
  }
  if (mx.fem_engine) {
    rows.push(["FEM reference engine", mx.custom ? "solved on demand" : "reference source", mx.fem_engine]);
  }
  if (mx.custom && mx.settings && m) {
    const units = Object.fromEntries(m.params.map((s) => [s.key, [s.label, s.unit, s.step, s.group]]));
    const changed = m.params.filter((s) => Math.abs(mx.settings[s.key] - s.value) > 1e-9 * Math.max(1, Math.abs(s.value)));
    if (mx.fem_cells) rows.push(["FEM mesh cells", mx.fem_engine || "solved for these settings", `${mx.fem_cells.toLocaleString("en-US")} cells`]);
    for (const s of changed) rows.push([`${s.group}: ${s.label}`, `study value ${fmtParam(s.value, s.step)} ${s.unit}`.trim(), `${fmtParam(mx.settings[s.key], units[s.key][2])} ${s.unit}`.trim()]);
  }
  card.innerHTML = `<h2>${j.kind === "inverse" ? "Design detail" : "Validation detail"}</h2><dl class="detail-grid">${rows.map(([l, s, v]) => `<div><dt><span>${esc(l)}</span><span>${esc(s)}</span></dt><dd>${esc(v ?? "—")}</dd></div>`).join("")}</dl>`;
}

async function refreshJobs() {
  try {
    const { jobs } = await api("/api/jobs");
    state.jobs = jobs;
    renderJobs();
    updateNavBadges();
  } catch {}
}

async function watchJob(id) {
  clearTimeout(state.poll);
  try {
    const j = await api(`/api/jobs/${id}`);
    const wasOpen = state.job?.id === id;
    const prevStatus = state.job?.status;
    state.job = j;
    const justDone = wasOpen && prevStatus && prevStatus !== "done" && j.status === "done";
    const autoLog = state.health?.settings?.auto_open_log !== false;
    if ((!wasOpen || (prevStatus !== j.status && j.status === "running")) && autoLog) {
      $("#logBox").open = !simple() && (j.status === "running" || j.status === "queued");
    }
    const i = state.jobs.findIndex((x) => x.id === j.id);
    if (i >= 0) state.jobs[i] = { ...state.jobs[i], ...j }; else state.jobs.unshift(j);
    renderJobs();
    renderJob();
    if (justDone) $$("#tiles .tile-value > span:first-child").forEach((el, i) => countUp(el, i * 90));
    const waitMs = j.cancel_requested ? 350 : (state.health?.settings?.poll_ms || 1500);
    if (j.status === "queued" || j.status === "running") state.poll = setTimeout(() => watchJob(id), waitMs);
    else { refreshJobs(); pollHealth(); }
  } catch (e) {
    $("#jobError").hidden = false;
    $("#jobError").textContent = `Lost contact with the job: ${e.message}`;
    state.poll = setTimeout(() => watchJob(id), 4000);
  }
}

async function submit(kind) {
  const bench = state.liveBench;
  const s = state.solver;
  const common = {
    benchmark: bench,
    device: selectedDevice(),
    fem_backend: selectedFem(),
    formulation: s.formulation,
    hidden_layers: s.hidden_layers,
    neurons: s.neurons,
    precision: s.precision,
    seed: s.seed,
    warm_start: s.warm_start,
    al_stages: s.al_stages,
    gamma_u: s.gamma_u,
    gamma_t: s.gamma_t,
    params: paramValues(bench),
  };
  const body = kind === "inverse"
    ? {
      ...common,
      metric: invSelectedMetric(bench),
      target_ratio: invSelectedRatio(bench),
      forward_epochs: state.budget.epochs,
      forward_lbfgs: s.forward_lbfgs,
      n_outer: s.n_outer,
    }
    : {
      ...common,
      epochs: state.budget.epochs,
      lbfgs: state.budget.lbfgs,
      lr: s.lr,
    };
  $("#runForward").disabled = true;
  $("#runInverse").disabled = true;
  $("#jobChip").className = "chip run";
  $("#jobChip").textContent = S("Starting…", "Submitting…");
  try {
    const { job_id } = await api(`/api/${kind}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    state.jobs.unshift({
      id: job_id, kind, benchmark: bench, formulation: s.formulation,
      device: effectiveDevice(), status: "queued", stage: "queued",
    });
    state.fig.live = null;
    await watchJob(job_id);
    pollHealth();
  } catch (e) {
    $("#jobError").hidden = false;
    $("#jobError").textContent = `Could not start the run: ${e.message}`;
    renderJob();
  }
}

async function loadLive() {
  try {
    const data = await api("/api/benchmarks");
    state.meta = data.benchmarks;
    state.lbfgs = data.lbfgs_iters;
    state.maxLbfgs = data.max_lbfgs ?? MAX_LBFGS;
    state.seed = data.seed;
  } catch (e) {
    $("#benchPick").innerHTML = `<legend class="eyebrow">Benchmark</legend><p class="empty">Could not load the benchmarks: ${esc(e.message)}</p>`;
    return;
  }
  if (!liveMeta(state.liveBench)) state.liveBench = state.meta[0].key;
  renderBenchPick();
  renderCompute();
  renderSpec();
  renderProps();
  renderBudget();
  renderSolverCfg();
  await refreshJobs();
  const active = state.jobs.find((j) => j.status === "running" || j.status === "queued") || state.jobs[0];
  if (active) await watchJob(active.id); else renderJob();
}

document.addEventListener("click", async (e) => {
  const tipBtn = e.target.closest(".tip");
  if (tipBtn) {
    e.preventDefault();
    const same = tipState.btn === tipBtn;
    closeTip();
    if (!same) openTip(tipBtn);
    return;
  }
  if (!e.target.closest("#tipPop")) closeTip();

  if (e.target.closest("#collapseSidebar")) {
    toggleSidebar();
    return;
  }
  if (e.target.closest("#mobileMenuBtn")) {
    const sb = $("#sidebar"), bd = $("#sideBackdrop");
    if (sb) sb.classList.toggle("mobile-open");
    if (bd) bd.hidden = !sb?.classList.contains("mobile-open");
    return;
  }
  if (e.target === $("#sideBackdrop")) {
    closeMobileSidebar();
    return;
  }

  if (e.target.closest("#health") || e.target.closest("[data-open-settings]")) {
    openSettingsDrawer();
    return;
  }

  const tryBench = e.target.closest("[data-try-bench]");
  if (tryBench) {
    const k = tryBench.dataset.tryBench;
    if (k && liveMeta(k)) {
      state.liveBench = k;
      store.set("atlas-live-bench", k);
      renderBenchPick();
      renderCompute();
      renderSpec();
      renderProps();
      renderSolverCfg();
      renderJob();
    }
    location.hash = "#live";
    return;
  }

  const inspectJob = e.target.closest("[data-inspect-job]");
  if (inspectJob) {
    const id = inspectJob.dataset.inspectJob;
    state.fig.live = null;
    location.hash = "#live";
    await watchJob(id);
    return;
  }

  const openImg = e.target.closest("[data-open-img]");
  if (openImg) {
    window.open(openImg.dataset.openImg, "_blank", "noopener");
    return;
  }

  const filesTabBtn = e.target.closest("[data-files-tab]");
  if (filesTabBtn) {
    state.filesTab = filesTabBtn.dataset.filesTab;
    renderFilesView();
    return;
  }
  const filesFltBtn = e.target.closest("[data-files-filter]");
  if (filesFltBtn) {
    state.filesFilter = filesFltBtn.dataset.filesFilter;
    renderFilesView();
    return;
  }
  if (e.target.closest("#refreshWorkspaceBtn")) {
    await refreshJobs();
    await loadWorkspace();
    state.filesMsg = "Refreshed workspace directory listing.";
    renderFilesView();
    return;
  }

  const openFolderBtn = e.target.closest("[data-open-folder]");
  if (openFolderBtn) {
    const target = openFolderBtn.dataset.openFolder;
    const r = await api("/api/open-folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target }),
    });
    const msg = r.opened ? `Opened folder: ${r.path}` : `Folder path: ${r.path}`;
    state.filesMsg = msg;
    state.settingsMsg = msg;
    if (currentView() === "files") renderFilesView();
    if (currentView() === "settings") renderSettingsDrawer();
    return;
  }

  const femBtn = e.target.closest("[data-fem-pick]");
  if (femBtn && !femBtn.disabled) {
    state.femTest = null;
    await updateAppSettings({ fem_backend: femBtn.dataset.femPick });
    return;
  }
  const scaleBtn = e.target.closest("[data-scale-pick]");
  if (scaleBtn) {
    applyScale(scaleBtn.dataset.scalePick);
    return;
  }
  if (e.target.closest("#testFemBtn") || e.target.closest("#quickTestFem")) {
    state.femTest = "Running cantilever verification check…";
    if (currentView() === "settings") renderSettingsDrawer();
    try {
      const r = await api("/api/fem/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fem_backend: selectedFem() }),
      });
      state.femTest = `Verified ${r.engine_name}: cantilever compliance = ${r.compliance.toFixed(2)} N·m in ${r.elapsed_ms} ms.`;
      state.settingsMsg = state.femTest;
    } catch (err) {
      state.femTest = `FEM self-test failed: ${err.message}`;
      state.settingsMsg = state.femTest;
    }
    if (currentView() === "settings") renderSettingsDrawer();
    return;
  }

  const devBtn = e.target.closest("[data-dev-pick]");
  if (devBtn && !devBtn.disabled) {
    await updateAppSettings({ device: devBtn.dataset.devPick });
    return;
  }
  const formBtn = e.target.closest("[data-form-pick]");
  if (formBtn) {
    state.solver.formulation = formBtn.dataset.formPick;
    saveSolver();
    renderCompute();
    renderSpec();
    renderBudget();
    renderSolverCfg();
    return;
  }
  const precBtn = e.target.closest("[data-prec-pick]");
  if (precBtn) {
    state.solver.precision = precBtn.dataset.precPick;
    saveSolver();
    renderSpec();
    renderSolverCfg();
    return;
  }
  const netPreset = e.target.closest("[data-net-preset]");
  if (netPreset) {
    const [hl, nn] = netPreset.dataset.netPreset.split("x").map(Number);
    state.solver.hidden_layers = hl;
    state.solver.neurons = nn;
    saveSolver();
    renderSpec();
    renderBudget();
    renderSolverCfg();
    return;
  }
  const seedPick = e.target.closest("[data-seed-pick]");
  if (seedPick) {
    state.solver.seed = parseInt(seedPick.dataset.seedPick, 10);
    saveSolver();
    renderBudget();
    renderSolverCfg();
    return;
  }
  if (e.target.closest("#resetSolver") || e.target.closest("#resetAllBtn")) {
    state.solver = { ...SOLVER_DEFAULT, inv_metric: {}, inv_ratio: {} };
    saveSolver();
    setBudget(FULL.epochs, FULL.lbfgs);
    applyScale("default");
    applyMotion(true);
    await updateAppSettings({ device: "auto", fem_backend: "auto", plot_dpi: 140, poll_ms: 1500, auto_open_log: true });
    state.settingsMsg = "Restored default appearance, solver, compute, and FEM settings.";
    renderCompute();
    renderSpec();
    renderBudget();
    renderSolverCfg();
    if (currentView() === "settings") renderSettingsDrawer();
    return;
  }

  const dpiBtn = e.target.closest("[data-dpi]");
  if (dpiBtn) {
    await updateAppSettings({ plot_dpi: parseInt(dpiBtn.dataset.dpi, 10) });
    return;
  }
  const pollBtn = e.target.closest("[data-poll]");
  if (pollBtn) {
    await updateAppSettings({ poll_ms: parseInt(pollBtn.dataset.poll, 10) });
    return;
  }
  if (e.target.closest("#clearCudaBtn")) {
    const h = await api("/api/cuda/clear", { method: "POST" });
    state.health = h;
    state.settingsMsg = "Freed unused CUDA memory cache.";
    if (currentView() === "settings") renderSettingsDrawer();
    return;
  }
  if (e.target.closest("[data-install-cuda]")) {
    state.settingsMsg = "Starting NVIDIA CUDA runtime setup…";
    if (currentView() === "settings") renderSettingsDrawer();
    const res = await api("/api/cuda/setup", { method: "POST" });
    state.settingsMsg = res.message || `CUDA setup status: ${res.status}`;
    if (currentView() === "settings") renderSettingsDrawer();
    else openSettingsDrawer();
    return;
  }
  if (e.target.closest("#openPlotsBtn")) {
    const r = await api("/api/open-plots", { method: "POST" });
    state.settingsMsg = r.opened ? `Opened ${r.path}` : `Plots directory: ${r.path}`;
    if (currentView() === "settings") renderSettingsDrawer();
    return;
  }
  if (e.target.closest("#clearStorageBtn") || e.target.closest("#clearStorageBtnFiles") || e.target.closest("#clearJobsBtn")) {
    const h = await api("/api/jobs/clear", { method: "POST" });
    state.health = h;
    if (state.job && state.job.status !== "queued" && state.job.status !== "running") state.job = null;
    await refreshJobs();
    await loadWorkspace();
    renderJob();
    const msg = `Cleared ${h.removed_jobs || 0} finished runs and ${h.deleted_files || 0} plot files.`;
    state.settingsMsg = msg;
    state.filesMsg = msg;
    if (currentView() === "settings") renderSettingsDrawer();
    if (currentView() === "files") renderFilesView();
    applyHealthUI(h);
    return;
  }
  if (e.target.closest("#cancelJob") && state.job) {
    const j = await api(`/api/jobs/${state.job.id}/cancel`, { method: "POST" });
    state.job = j;
    renderJob();
    watchJob(j.id);
    return;
  }

  const themeBtn = e.target.closest("[data-theme-pick]");
  if (themeBtn) return switchTheme(themeBtn.dataset.themePick);
  const levelBtn = e.target.closest("[data-level-pick]");
  if (levelBtn) {
    const level = levelBtn.dataset.levelPick;
    if (level !== (simple() ? "simple" : "expert")) withTransition(() => applyLevel(level));
    return;
  }

  if (e.target.closest("#resetProps")) {
    delete state.params[state.liveBench];
    store.set("atlas-params", JSON.stringify(state.params));
    renderProps();
    return;
  }
  const bt = e.target.closest("[data-bench]");
  if (bt) {
    state.bench = bt.dataset.bench;
    store.set("atlas-bench", state.bench);
    renderResults();
    renderWorkspaceHeader();
    return;
  }
  const fb = e.target.closest("[data-fig]");
  if (fb) {
    state.fig[state.bench] = fb.dataset.fig;
    renderFigures(state.results.benchmarks[state.bench]);
    return;
  }
  const lf = e.target.closest("[data-livefig]");
  if (lf) { state.fig.live = lf.dataset.livefig; renderLiveFigure(state.job); return; }
  const jb = e.target.closest("[data-job]");
  if (jb) { state.fig.live = null; watchJob(jb.dataset.job); return; }
  const img = e.target.closest(".fig-frame img");
  if (img) window.open(img.dataset.full, "_blank", "noopener");
});

document.addEventListener("toggle", (e) => {
  if (e.target.id === "solverAdv") {
    state.advOpen = e.target.open;
    store.set("atlas-adv-open", e.target.open ? "1" : "0");
  }
}, true);

document.addEventListener("change", async (e) => {
  if (e.target.name === "bench") {
    state.liveBench = e.target.value;
    store.set("atlas-live-bench", state.liveBench);
    $$(".opt").forEach((o) => o.classList.toggle("on", $("input", o).checked));
    renderCompute();
    renderSpec();
    renderProps();
    renderSolverCfg();
    renderJob();
    renderWorkspaceHeader();
    return;
  }
  if (e.target.dataset.num) setParam(e.target.dataset.num, e.target.value, "num");
  if (e.target.dataset.mat) {
    const mat = MATERIALS.find(([id]) => id === e.target.value);
    if (mat) { setParam(e.target.dataset.mat, mat[2], "mat"); setParam(e.target.dataset.matNu, mat[3], "mat"); }
  }
  if (e.target.id === "epochs" || e.target.id === "lbfgs") {
    const epEl = $("#epochs"), lbEl = $("#lbfgs");
    const ep = Math.min(40000, Math.max(50, parseInt(epEl ? epEl.value : state.budget.epochs, 10) || FULL.epochs));
    const rawLb = lbEl ? parseInt(lbEl.value, 10) : state.budget.lbfgs;
    const lb = snapLbfgs(Number.isFinite(rawLb) ? rawLb : FULL.lbfgs);
    setBudget(ep, lb);
    renderBudget();
    renderSolverCfg();
    return;
  }
  const s = state.solver;
  if (e.target.id === "cfgLayers") { s.hidden_layers = Math.min(8, Math.max(2, parseInt(e.target.value, 10) || 4)); saveSolver(); renderSpec(); renderBudget(); renderSolverCfg(); }
  if (e.target.id === "cfgNeurons") { s.neurons = Math.min(256, Math.max(16, parseInt(e.target.value, 10) || 64)); saveSolver(); renderSpec(); renderBudget(); renderSolverCfg(); }
  if (e.target.id === "cfgLr") { s.lr = Math.min(0.05, Math.max(0.00001, parseFloat(e.target.value) || 0.002)); saveSolver(); renderSolverCfg(); }
  if (e.target.id === "cfgSeed") { s.seed = Math.min(999999, Math.max(0, parseInt(e.target.value, 10) || 42)); saveSolver(); renderBudget(); renderSolverCfg(); }
  if (e.target.id === "cfgWarm") { s.warm_start = Math.min(20000, Math.max(0, parseInt(e.target.value, 10) || 0)); saveSolver(); renderSolverCfg(); }
  if (e.target.id === "cfgAl") { s.al_stages = Math.min(8, Math.max(0, parseInt(e.target.value, 10) || 0)); saveSolver(); renderSolverCfg(); }
  if (e.target.id === "cfgGu") { s.gamma_u = Math.min(1000, Math.max(0.1, parseFloat(e.target.value) || 30)); saveSolver(); renderSolverCfg(); }
  if (e.target.id === "cfgGt") { s.gamma_t = Math.min(100, Math.max(0, parseFloat(e.target.value) || 0)); saveSolver(); renderSolverCfg(); }
  if (e.target.id === "cfgInvMetric") { s.inv_metric[state.liveBench] = e.target.value; saveSolver(); renderSpec(); renderSolverCfg(); }
  if (e.target.id === "cfgInvRatio") {
    s.inv_ratio[state.liveBench] = Math.min(5, Math.max(0.1, parseFloat(e.target.value) || invDefaultRatio(state.liveBench)));
    saveSolver();
    renderSpec();
    renderSolverCfg();
  }
  if (e.target.id === "cfgInvOuter") { s.n_outer = Math.min(100, Math.max(1, parseInt(e.target.value, 10) || 12)); saveSolver(); renderSolverCfg(); }
  if (e.target.id === "cfgInvLbfgs") { s.forward_lbfgs = Math.min(8000, Math.max(0, parseInt(e.target.value, 10) || 0)); saveSolver(); renderSolverCfg(); }
  if (e.target.id === "setThreadsNum" || e.target.id === "setThreadsRange") {
    const n = Math.max(1, parseInt(e.target.value, 10) || 1);
    await updateAppSettings({ cpu_threads: n });
  }
  if (e.target.id === "setAutoLog") {
    await updateAppSettings({ auto_open_log: e.target.checked });
  }
  if (e.target.id === "setMotion") {
    applyMotion(e.target.checked);
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeTip();
    closeMobileSidebar();
  }
});
window.addEventListener("scroll", closeTip, { passive: true });
window.addEventListener("resize", closeTip);

document.addEventListener("input", (e) => {
  if (e.target.dataset.range) setParam(e.target.dataset.range, e.target.value, "range");
  if (e.target.dataset.run !== undefined) {
    const idx = +e.target.value;
    const r = RUN_STEPS[idx];
    setBudget(r.epochs, r.lbfgs);
    const [val, hint] = runText();
    $("#runVal").textContent = val;
    $("#runHint").textContent = hint;
    e.target.setAttribute("aria-valuetext", val);
    e.target.style.setProperty("--fill", `${(idx / (RUN_STEPS.length - 1)) * 100}%`);
    if ($("#epochs")) $("#epochs").value = r.epochs;
    if ($("#lbfgs")) $("#lbfgs").value = r.lbfgs;
  }
  if (e.target.id === "setThreadsRange") {
    const v = +e.target.value;
    const maxT = +e.target.max || 8;
    e.target.style.setProperty("--fill", `${((v - 1) / Math.max(1, maxT - 1)) * 100}%`);
    if ($("#setThreadsNum")) $("#setThreadsNum").value = v;
  }
  if (e.target.id === "cfgInvRatioRange") {
    const v = +e.target.value;
    e.target.style.setProperty("--fill", `${((v - 0.3) / 1.3) * 100}%`);
    state.solver.inv_ratio[state.liveBench] = v;
    saveSolver();
    if ($("#cfgInvRatio")) $("#cfgInvRatio").value = v.toFixed(2);
    renderSpec();
  }
});

document.addEventListener("pointerdown", (e) => setRevealOrigin(e.clientX, e.clientY), { passive: true });

let spotRaf = null, spotEv = null;
document.addEventListener("pointermove", (e) => {
  spotEv = e;
  if (spotRaf) return;
  spotRaf = requestAnimationFrame(() => {
    spotRaf = null;
    const card = spotEv.target.closest && spotEv.target.closest(".card");
    if (!card) return;
    const r = card.getBoundingClientRect();
    card.style.setProperty("--mx", `${spotEv.clientX - r.left}px`);
    card.style.setProperty("--my", `${spotEv.clientY - r.top}px`);
  });
}, { passive: true });

const flow = (() => {
  const cv = $("#flow");
  const ctx = cv && cv.getContext("2d");
  if (!ctx) return { recolor() {}, update() {} };
  let w = 0, h = 0, hole = null, parts = [], raf = null, colors = { a: "#2A78D6", b: "#C9561A" }, visible = true;
  const running = () => visible && !document.hidden && !$("#view-results").hidden && !reduced()
    && document.documentElement.getAttribute("data-theme") !== "atlas";

  function field(x, y) {
    const dx = (x - hole.x) / hole.a, dy = (hole.y - y) / hole.a;
    const r2 = dx * dx + dy * dy;
    if (r2 < 1.02) return null;
    const th = Math.atan2(dy, dx), c2 = Math.cos(2 * th), s2 = Math.sin(2 * th);
    const a2 = 1 / r2, a4 = a2 * a2;
    const srr = 0.5 * (1 - a2) + 0.5 * (1 - 4 * a2 + 3 * a4) * c2;
    const stt = 0.5 * (1 + a2) - 0.5 * (1 + 3 * a4) * c2;
    const srt = -0.5 * (1 + 2 * a2 - 3 * a4) * s2;
    const c = Math.cos(th), s = Math.sin(th);
    const sxx = srr * c * c + stt * s * s - 2 * srt * s * c;
    const syy = srr * s * s + stt * c * c + 2 * srt * s * c;
    const sxy = (srr - stt) * s * c + srt * (c * c - s * s);
    const phi = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    const s1 = 0.5 * (sxx + syy) + Math.sqrt(0.25 * (sxx - syy) ** 2 + sxy * sxy);
    let ux = Math.cos(phi), uy = -Math.sin(phi);
    if (ux < 0) { ux = -ux; uy = -uy; }
    return [ux, uy, s1];
  }

  function spawn(p, edge) {
    p.x = edge ? Math.random() * w * 0.05 : Math.random() * w;
    p.y = Math.random() * h;
    p.age = 0;
    p.life = 160 + Math.random() * 340;
  }

  function drawStatic() {
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 1;
    ctx.strokeStyle = colors.a;
    ctx.globalAlpha = 0.18;
    for (let i = 0; i < 44; i++) {
      let x = 0, y = ((i + 0.5) / 44) * h;
      ctx.beginPath();
      ctx.moveTo(x, y);
      for (let k = 0; k < 2000 && x < w; k++) {
        const f = field(x, y);
        if (!f) break;
        x += f[0] * 2; y += f[1] * 2;
        ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const r = cv.getBoundingClientRect();
    w = r.width; h = r.height;
    if (!w || !h) return;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    hole = w > 900 ? { x: w * 0.47, y: h * 0.2, a: Math.max(30, h * 0.1) } : { x: w * 0.78, y: h * 0.1, a: Math.max(22, w * 0.07) };
    parts = Array.from({ length: Math.min(520, Math.round((w * h) / 1700)) }, () => { const p = {}; spawn(p, false); return p; });
    if (reduced()) drawStatic();
  }

  function frame() {
    raf = null;
    if (!running()) return;
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "destination-out";
    ctx.fillStyle = "rgba(0,0,0,0.032)";
    ctx.fillRect(0, 0, w, h);
    ctx.globalCompositeOperation = "source-over";
    ctx.lineWidth = 1.15;
    ctx.lineCap = "round";
    for (const p of parts) {
      const f = field(p.x, p.y);
      if (!f || p.x > w + 2 || p.y < -2 || p.y > h + 2 || p.age++ > p.life) { spawn(p, Math.random() < 0.7); continue; }
      const sp = 0.45 + 0.5 * Math.min(f[2], 3);
      const nx = p.x + f[0] * sp, ny = p.y + f[1] * sp;
      const hot = Math.min(1, Math.max(0, (f[2] - 1.15) / 1.5));
      ctx.strokeStyle = hot > 0.4 ? colors.b : colors.a;
      ctx.globalAlpha = 0.24 + 0.46 * hot;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(nx, ny);
      ctx.stroke();
      p.x = nx; p.y = ny;
    }
    ctx.globalAlpha = 0.5;
    ctx.strokeStyle = colors.a;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.arc(hole.x, hole.y, hole.a, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    raf = requestAnimationFrame(frame);
  }

  function update() {
    if (!hole) resize();
    if (running() && !raf) raf = requestAnimationFrame(frame);
  }

  function recolor() {
    const cs = getComputedStyle(document.documentElement);
    colors = { a: cs.getPropertyValue("--dot-a").trim() || colors.a, b: cs.getPropertyValue("--target").trim() || colors.b };
    if (w && h) ctx.clearRect(0, 0, w, h);
    if (reduced() && hole) drawStatic();
    update();
  }

  new IntersectionObserver(([e]) => { visible = e.isIntersecting; update(); }).observe(cv);
  document.addEventListener("visibilitychange", update);
  let rt = null, lastSize = "";
  new ResizeObserver(() => {
    const r = cv.getBoundingClientRect();
    const size = `${Math.round(r.width)}x${Math.round(r.height)}`;
    if (size === lastSize) return;
    lastSize = size;
    clearTimeout(rt);
    rt = setTimeout(() => { resize(); update(); }, 120);
  }).observe(cv);
  resize();
  recolor();
  return { recolor, update };
})();

$("#runForward").addEventListener("click", () => submit("forward"));
$("#runInverse").addEventListener("click", () => submit("inverse"));
window.addEventListener("hashchange", () => withTransition(() => { route(); flow.update(); }));

applyTheme(document.documentElement.getAttribute("data-theme") || "light");
setStaticText();
route();
pollHealth();
setInterval(pollHealth, 5000);
setInterval(() => { if (!isBusy()) refreshJobs(); }, 10000);
loadLive().then(loadResults);
