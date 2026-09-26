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
  check: `<svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true"><path d="M2.5 6.2l2.2 2.2 4.8-5" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  open: `<svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M6 3h7v7M13 3L4 12" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
};
const passIcon = (ok, s) => (ok ? ICON.pass(s) : ICON.fail(s));

const THUMB = {
  cantilever: `<rect class="t-fill t-ink" x="14" y="18" width="50" height="12.5" stroke-width="1.4"/><path class="t-ink" d="M13 9v30" stroke-width="2"/><path class="t-hatch" d="M13 12l-5 5M13 19l-5 5M13 26l-5 5M13 33l-5 5" stroke-width="1.2"/>`,
  l_bracket: `<path class="t-fill t-ink" d="M16 24H36V4H56V44H16Z" stroke-width="1.4" stroke-linejoin="round"/><path class="t-dash" d="M36 4V24M36 24V44M36 24H56" stroke-width="1" stroke-dasharray="2 2"/>`,
  plate_hole: `<path class="t-fill t-ink" d="M23 44A7 7 0 0 0 16 37V4H56V44Z" stroke-width="1.4" stroke-linejoin="round"/>`,
  steel_insert: `<path class="t-fill t-ink" d="M16 4H56V44H16Z" stroke-width="1.4" stroke-linejoin="round"/><path class="t-solid t-ink" d="M16 34A10 10 0 0 1 26 44H16Z" stroke-width="1.4" stroke-linejoin="round"/>`,
  bimaterial: `<path class="t-fill t-ink" d="M16 4H56V44H16Z" stroke-width="1.4" stroke-linejoin="round"/><path class="t-page t-ink" d="M16 34A10 10 0 0 1 26 44H16Z" stroke-width="1.4" stroke-linejoin="round"/>`,
  filled_slot: `<path class="t-fill t-ink" d="M16 4H56V44H16Z" stroke-width="1.4" stroke-linejoin="round"/><path class="t-dashfill t-ink" d="M16 26A6 18 0 0 1 22 44H16Z" stroke-width="1.4" stroke-linejoin="round"/>`,
};
const thumb = (key, w, h) => `<svg width="${w}" height="${h}" viewBox="0 0 72 48" fill="none" aria-hidden="true">${THUMB[key] || ""}</svg>`;

const state = {
  results: null, meta: null, bench: store.get("atlas-bench"), fig: {},
  liveBench: store.get("atlas-live-bench"), job: null, jobs: [], poll: null, health: null,
};

function applyTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  store.set("atlas-theme", t);
  $$("[data-theme-pick]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.themePick === t)));
}

function route() {
  const view = location.hash === "#live" ? "live" : "results";
  $("#view-results").hidden = view !== "results";
  $("#view-live").hidden = view !== "live";
  $$("nav.seg a").forEach((a) => (a.dataset.view === view ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  const el = $(`#view-${view}`);
  el.classList.remove("view");
  void el.offsetWidth;
  el.classList.add("view");
}

async function pollHealth() {
  const pill = $("#health");
  try {
    const h = await api("/api/health");
    state.health = h;
    const busy = h.active_jobs > 0;
    pill.innerHTML = `<span class="dot ${busy ? "busy" : "ok"}"></span><span>${busy ? `${esc(h.device)} · ${plural(h.active_jobs, "job", "jobs")} running` : `${h.device === "cuda" ? "GPU" : "CPU"} ready · ${esc(h.device)}`}</span>`;
  } catch {
    pill.innerHTML = `<span class="dot bad"></span><span>Backend unreachable</span>`;
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

  let title = atlas.every(pass)
    ? `ATLAS stays inside both accuracy targets on all ${word(benches.length)} benchmarks.`
    : `ATLAS meets both accuracy targets on ${word(new Set(atlas.filter(pass).map((r) => r.bench.key)).size)} of ${word(benches.length)} benchmarks.`;
  if (pinnBenches.length) {
    title += pinnMiss === pinnBenches.length
      ? ` A standard PINN misses both on all ${word(pinnBenches.length)} it was run on.`
      : ` A standard PINN misses both on ${word(pinnMiss)} of the ${word(pinnBenches.length)} it was run on.`;
  }
  $("#heroTitle").textContent = title;

  const kpis = [];
  const worstD = atlas.reduce((a, r) => (r.disp > a.disp ? r : a), atlas[0]);
  const worstV = atlas.reduce((a, r) => (r.vm > a.vm ? r : a), atlas[0]);
  const nameOf = (r) => SOLVER_NAMES[r.bench.group][r.key][0];
  if (worstD) kpis.push([pct(worstD.disp), "Worst ATLAS displacement error", `Target < ${t.displacement * 100} % · ${worstD.bench.title}, ${nameOf(worstD)}`, ""]);
  if (worstV) kpis.push([pct(worstV.vm), "Worst ATLAS von Mises error", `Target < ${t.von_mises * 100} % · ${worstV.bench.title}, ${nameOf(worstV)}`, ""]);
  if (pinnRows.length) {
    const best = pinnRows.reduce((a, r) => (r.disp < a.disp ? r : a), pinnRows[0]);
    kpis.push([pct(best.disp), "Best standard-PINN displacement error", `DeepXDE · ${best.bench.title} · ${Math.round(best.disp / t.displacement)}× the target`, "soft"]);
  }
  const gains = multi.map((b) => ({ b, x: b.seeds.atlas_xpinn, w: b.seeds.atlas_wide })).filter((g) => g.x && g.w);
  if (gains.length) {
    const g = gains.reduce((a, c) => (c.w.vm_mean / c.x.vm_mean > a.w.vm_mean / a.x.vm_mean ? c : a), gains[0]);
    kpis.push([`${(g.w.vm_mean / g.x.vm_mean).toFixed(1)}×`, "Lower von Mises error from decomposition",
      `${g.b.title} · ATLAS-XPINN vs a parameter-matched single network, ${g.x.n} seeds`, ""]);
  }
  $("#heroKpis").innerHTML = kpis.map(([v, l, s, cls]) => `<div class="kpi"><span class="kpi-value ${cls}">${esc(v)}</span><div class="kpi-text"><span class="kpi-label">${esc(l)}</span><span class="kpi-sub">${esc(s)}</span></div></div>`).join("");
}

function benchSubtitle(b) { return `${b.geometry} · ${plural(b.subdomains, "subdomain", "subdomains")}`; }

function renderBenchTabs() {
  const { results } = state;
  const groups = [["paper", "Paper benchmarks"], ["multi", "Multi-material · where decomposition pays off"]];
  $("#benches").innerHTML = groups.map(([g, label]) => {
    const keys = results.order.filter((k) => results.benchmarks[k].group === g);
    if (!keys.length) return "";
    return `<div><div class="eyebrow bench-group-label">${esc(label)}</div><div class="bench-tabs">${keys.map((k) => {
      const b = results.benchmarks[k];
      return `<button type="button" class="bench-tab" data-bench="${k}" aria-pressed="${k === state.bench}"><span class="thumb">${thumb(k, 72, 48)}</span><span class="bench-tab-text"><span class="bench-tab-title">${esc(b.title)}</span><span class="bench-tab-meta">${esc(benchSubtitle(b))}</span></span></button>`;
    }).join("")}</div></div>`;
  }).join("");
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
  const names = SOLVER_NAMES[b.group];
  const isPaper = b.group === "paper";
  const lib = b.solvers.deepxde?.library || b.solvers.deepxde_configured?.library;
  const groups = isPaper
    ? [["ATLAS · deep-energy solvers", ["atlas_xpinn", "atlas"]], [`Standard PINN · ${lib ? lib.replace(/^deepxde/i, "DeepXDE") : "DeepXDE"}`, ["deepxde", "deepxde_configured"]]]
    : [["ATLAS-XPINN · decomposed at the interface", ["atlas_xpinn"]], ["ATLAS · one network over both materials", ["atlas", "atlas_wide"]]];
  const all = groups.flatMap(([, ks]) => ks.map((k) => b.solvers[k]).filter(Boolean));
  const scale = axisScale(all.flatMap((r) => [r.disp, r.vm]));
  const vmLabel = isPaper ? "von Mises L², excl." : "von Mises L²";
  const rows = groups.map(([label, ks]) => {
    const body = ks.filter((k) => b.solvers[k]).map((k, i) => {
      const r = b.solvers[k];
      const [name, sub, dot] = names[k];
      return `<div class="erow body" style="animation-delay:${i * 60}ms"><div class="who"><span class="who-name">${esc(name)}</span><span class="who-sub">${esc(sub)}${r.params ? ` · ${r.params.toLocaleString("en-US")} params` : ""}</span></div>${metricCell(r.disp, t.displacement, scale, dot, `${name}: displacement ${pct(r.disp)}`)}${metricCell(r.vm, t.von_mises, scale, dot, `${name}: von Mises ${pct(r.vm)}`)}<span class="etime">${mins(r.seconds)}</span></div>`;
    }).join("");
    return body ? `<div class="egroup">${esc(label)}</div>${body}` : "";
  }).join("");
  const last = scale.ticks.length - 1;
  const every = last > 3 ? 2 : 1;
  const tickLabel = (v, i) => `${+(v * 100).toPrecision(2)}${i === last ? " %" : ""}`;
  const axis = `<div class="axis"><div class="axis-bar">${scale.ticks.slice(1).map(() => "<span></span>").join("")}</div>${scale.ticks.map((tk, i) => (i % every && i !== last ? "" : `<span class="tick ${i === 0 ? "first" : i === last ? "last" : ""}" style="left:${tk.p}%">${tickLabel(tk.v, i)}</span>`)).join("")}</div>`;
  const note = isPaper
    ? `The von Mises error leaves out a small disk around each stress singularity (radius ${b.exclusion?.factor ?? 2.5} × the local point spacing). One training run per solver, seed 42.`
    : "There are no stress singularities here, so no points are left out. One training run per solver, seed 42. The parameter-matched single network has as many parameters as ATLAS-XPINN, which rules out network size as the reason for the difference.";
  $("#errors").innerHTML = `
    <div class="card-head"><h2 class="l">Error against FEniCS</h2><span class="meta">${esc(b.title)} · seed 42</span></div>
    <div class="erow head"><span class="col">Solver</span><span class="col">Displacement L²<small>target ${t.displacement * 100} %</small></span><span class="col">${vmLabel}<small>target ${t.von_mises * 100} %</small></span><span class="col" style="text-align:right">Train time</span></div>
    ${rows}
    <div class="erow eaxis"><span></span><div class="metric"><span class="metric-val"></span>${axis}</div><div class="metric"><span class="metric-val"></span>${axis}</div><span></span></div>
    <div class="legend"><span><span class="sw a"></span>${isPaper ? "ATLAS" : "ATLAS-XPINN"}</span><span><span class="sw b"></span>${isPaper ? "Standard PINN" : "Single network"}</span><span><span class="sw t"></span>Accuracy target</span><span style="color:var(--muted)">Log scale</span></div>
    <p class="note">${esc(note)}</p>`;
  requestAnimationFrame(() => requestAnimationFrame(() => $$("#errors .pt").forEach((p) => (p.style.left = p.dataset.left))));
}

function renderSeeds(b) {
  const entries = b.group === "paper"
    ? [["ATLAS-XPINN", b.seeds.atlas_xpinn], ["ATLAS", b.seeds.atlas]]
    : [["ATLAS-XPINN", b.seeds.atlas_xpinn], ["ATLAS, parameter-matched", b.seeds.atlas_wide]];
  const have = entries.filter(([, s]) => s);
  if (!have.length) { $("#seeds").innerHTML = `<div class="card-head"><h2>Repeatability</h2></div><p class="note">No multi-seed study stored for this benchmark.</p>`; return; }
  const seeds = have[0][1].seeds;
  const total = have.reduce((a, [, s]) => a + s.n, 0);
  const ok = have.reduce((a, [, s]) => a + s.n_pass_both, 0);
  const vmLabel = b.group === "paper" ? "von Mises L², excl." : "von Mises L²";
  let html = `<div class="card-head"><h2>Repeatability</h2><span class="meta">${plural(seeds.length, "seed", "seeds")} · ${seeds[0]}–${seeds[seeds.length - 1]}</span></div>`;
  html += have.map(([name, s]) => `<div class="stat"><span class="stat-name">${esc(name)}</span><div class="kv"><span>Displacement L²</span><span>${meanStd(s.disp_mean, s.disp_std)}</span></div><div class="kv"><span>${vmLabel}</span><span>${meanStd(s.vm_mean, s.vm_std)}</span></div></div>`).join("");
  html += `<div class="verdict">${passIcon(ok === total, 16)}<span>${ok === total ? `All ${total} runs meet both targets` : `${ok} of ${total} runs meet both targets`}</span></div>`;
  if (b.paired && b.paired.n) {
    const p = b.paired;
    const txt = p.xpinn_lower_both === p.n
      ? `ATLAS-XPINN is lower on both errors in all ${p.n} seeds`
      : `ATLAS-XPINN is lower on displacement in ${p.xpinn_lower_disp} of ${p.n} seeds and on von Mises in ${p.xpinn_lower_vm} of ${p.n}`;
    html += `<div class="verdict">${passIcon(p.xpinn_lower_both === p.n, 16)}<span>${esc(txt)}</span></div>`;
  }
  $("#seeds").innerHTML = html;
}

function renderSide(b) {
  if (b.group === "paper") {
    const inv = b.inverse;
    if (!inv) { $("#sideCard").innerHTML = `<div class="card-head"><h2>Inverse design</h2></div><p class="note">No inverse design stored.</p>`; return; }
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
  const rows = [["atlas_xpinn", "ATLAS-XPINN", ""], ["atlas", "ATLAS", "single network"], ["atlas_wide", "ATLAS", "parameter-matched"]]
    .filter(([k]) => b.solvers[k]);
  $("#sideCard").innerHTML = `
    <div class="card-head"><h2>At the material interface</h2><span class="meta">seed 42</span></div>
    <table class="band-table"><thead><tr><th>von Mises L², within 0.05 m</th><th>${esc(b.inside)}</th><th>${esc(b.outside)}</th></tr></thead><tbody>
    ${rows.map(([k, n, s]) => `<tr><td>${esc(n)}${s ? `<small>${esc(s)}</small>` : ""}</td><td>${pct(b.solvers[k].band_inside)}</td><td>${pct(b.solvers[k].band_outside)}</td></tr>`).join("")}
    </tbody></table>
    <p class="note" style="margin-top:14px">A single smooth network has to bend around the jump in strain at the interface. ATLAS-XPINN puts a subdomain boundary there instead.</p>`;
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

function renderFigures(b) {
  const figs = FIGS[b.group].filter(([k]) => b.plots[k]);
  if (!figs.length) { $("#figCard").innerHTML = `<div class="fig-head"><h2 class="l">Fields and validation</h2></div><div class="fig-empty">No figures stored for this benchmark.</div>`; return; }
  const cur = figs.find(([k]) => k === state.fig[b.key]) || figs[0];
  const src = b.plots[cur[0]];
  $("#figCard").innerHTML = `
    <div class="fig-head"><div><h2 class="l">Fields and validation</h2><span class="meta">${b.group === "paper" ? "ATLAS-XPINN" : "All solvers"} · ${esc(b.title)}</span></div>
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
  } catch (e) {
    $("#heroTitle").textContent = "Could not load the stored study.";
    $("#errors").innerHTML = `<p class="note">${esc(e.message)}</p>`;
  }
}

const STEPS = {
  forward: [["setup", "Setup"], ["train", "Training"], ["predict", "Prediction"], ["validate", "FEM validation"], ["plots", "Plots"]],
  inverse: [["setup", "Setup"], ["train", "Design loop"], ["plots", "Plots"]],
};

function liveMeta(key) { return (state.meta || []).find((m) => m.key === key); }

function renderBenchPick() {
  const groups = [["paper", "Paper benchmarks"], ["multi", "Multi-material"]];
  const html = groups.map(([g, label]) => {
    const ms = state.meta.filter((m) => m.group === g);
    return `<div class="opt-group-label">${esc(label)}</div>${ms.map((m) => `<label class="opt ${m.key === state.liveBench ? "on" : ""}"><input type="radio" name="bench" value="${m.key}" ${m.key === state.liveBench ? "checked" : ""}><span class="thumb">${thumb(m.key, 48, 32)}</span><span class="opt-text"><span class="opt-title">${esc(m.title)}</span><span class="opt-meta">${esc(benchSubtitle(m))}</span></span></label>`).join("")}`;
  }).join("");
  $("#benchPick").innerHTML = `<legend class="eyebrow">Benchmark</legend>${html}`;
}

function renderSpec() {
  const m = liveMeta(state.liveBench);
  if (!m) return;
  const mats = m.materials.length === 1
    ? [["Material", m.materials[0].name], ["Young's modulus", gpa(m.materials[0].E)], ["Poisson's ratio", m.materials[0].nu.toFixed(2)]]
    : m.materials.map((x) => [x.name, `${gpa(x.E)} · ν ${x.nu.toFixed(2)}`]);
  const rows = [...mats, ["Solver", "ATLAS-XPINN"], ["Subdomains", String(m.subdomains)],
    ["Network", `${m.subdomains} × (${m.network.hidden_layers} × ${m.network.neurons}), ${m.network.activation}`]];
  $("#spec").innerHTML = rows.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("");
  $("#epochsHint").textContent = `Then ${state.lbfgs ?? 2000} L-BFGS iterations, the same budget as the stored study. Seed ${state.seed ?? 42}.`;
  const inv = m.inverse;
  $("#runInverse").disabled = !inv || isBusy();
  $("#runNote").textContent = inv
    ? `Inverse design finds the ${inv.param} that meets a ${inv.objective.toLowerCase()}. One GPU job runs at a time; new runs wait in a queue.`
    : "Inverse design is set up for the three paper benchmarks. One GPU job runs at a time; new runs wait in a queue.";
}

function isBusy() { return state.jobs.some((j) => j.status === "queued" || j.status === "running"); }

function renderJobs() {
  const ul = $("#jobs");
  if (!state.jobs.length) { ul.innerHTML = `<li><p class="empty">No jobs yet this session.</p></li>`; return; }
  ul.innerHTML = state.jobs.slice(0, 8).map((j) => {
    const m = liveMeta(j.benchmark);
    const cls = j.status === "done" ? "ok" : j.status === "error" ? "bad" : "busy";
    const right = j.status === "done" ? dur(secondsBetween(j.started, j.finished)) : j.status === "error" ? "failed" : j.status;
    return `<li><button type="button" data-job="${j.id}" aria-current="${state.job?.id === j.id}"><span class="dot ${cls}"></span><span class="job-label">${j.kind === "inverse" ? "Inverse" : "Forward"} · ${esc(m?.title || j.benchmark)}</span><span class="job-right">${esc(right)}</span></button></li>`;
  }).join("");
}

function renderJob() {
  const j = state.job;
  const busy = isBusy();
  $("#runForward").disabled = busy;
  $("#runForward").textContent = busy ? "Running…" : "Run forward + validate";
  const m = liveMeta(state.liveBench);
  $("#runInverse").disabled = busy || !m?.inverse;
  const steps = STEPS[j?.kind === "inverse" ? "inverse" : "forward"];
  if (!j) {
    $("#jobTitle").textContent = "No run yet";
    $("#jobSub").textContent = "Pick a benchmark, then run a forward solve.";
    $("#jobChip").className = "chip";
    $("#jobChip").textContent = "Idle";
    $("#progress").hidden = true;
    $("#jobError").hidden = true;
    $("#log").textContent = "—";
  } else {
    const jm = liveMeta(j.benchmark);
    const secs = j.metrics?.train_seconds ?? secondsBetween(j.started, j.finished);
    $("#jobTitle").textContent = j.kind === "inverse" ? "Inverse design" : "Forward solve + FEM validation";
    $("#jobSub").textContent = `${jm?.title || j.benchmark} · ATLAS-XPINN · seed ${j.metrics?.seed ?? state.seed ?? 42}${j.status === "done" ? ` · finished in ${dur(secs)}` : ""}`;
    const chip = { queued: ["run", "Queued"], running: ["run", "Running"], done: ["done", "Done"], error: ["fail", "Failed"] }[j.status] || ["", j.status];
    $("#jobChip").className = `chip ${chip[0]}`;
    $("#jobChip").textContent = chip[1];
    $("#progress").hidden = !(j.status === "running" || j.status === "queued");
    $("#jobError").hidden = !j.error;
    $("#jobError").textContent = j.error || "";
    const pre = $("#log");
    const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 24;
    pre.textContent = j.log || (j.status === "queued" ? "Waiting for the GPU…" : "Solver output appears here as it is written.");
    if (atBottom) pre.scrollTop = pre.scrollHeight;
  }
  const idx = !j ? -1 : j.status === "done" ? steps.length : Math.max(0, steps.findIndex(([k]) => k === j.stage));
  $("#steps").innerHTML = steps.map(([, label], i) => {
    const cls = i < idx ? "done" : i === idx ? (j?.status === "error" ? "failed" : j?.status === "queued" ? "" : "active") : "";
    return `<li class="${cls}"><span class="step-dot">${i < idx ? ICON.check : ""}</span><span class="step-label">${esc(label)}</span>${i < steps.length - 1 ? '<span class="step-line"></span>' : ""}</li>`;
  }).join("");
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
  const waiting = j && !done ? (j.status === "error" ? "failed" : "running") : "no run yet";
  if (j?.kind === "inverse") {
    $("#tiles").innerHTML = [
      tile(done ? `${cap(mx.param)} found` : "Design found", done ? mx.final?.toFixed(4) : null, done ? `started from ${mx.initial?.toFixed(3)}` : waiting),
      tile("Objective", done ? mx.objective : null, done ? `target ${sci(mx.target)}` : waiting),
      tile("Outer iterations", done ? String(mx.n_outer) : null, done ? (mx.converged ? "converged" : mx.status) : waiting),
      tile("Run time", done ? dur(mx.train_seconds) : null, done ? `wall-clock on ${state.health?.device || "the server"}` : waiting),
    ].join("");
    return;
  }
  const vmKey = mx.von_mises_l2_excl != null ? "von_mises_l2_excl" : "von_mises_l2";
  $("#tiles").innerHTML = [
    tile("Displacement L²", done ? pct(mx.relative_L2_displacement) : null, `relative to FEM · target < ${t.displacement * 100} %`, done ? mx.relative_L2_displacement < t.displacement : null),
    tile(vmKey === "von_mises_l2_excl" || !done ? "von Mises L², excl." : "von Mises L²", done ? pct(mx[vmKey]) : null, `relative to FEM · target < ${t.von_mises * 100} %`, done ? mx[vmKey] < t.von_mises : null),
    tile("Run time", done ? dur(mx.train_seconds) : null, done ? `training wall-clock on ${state.health?.device || "the server"}` : waiting),
    tile("Parameters", done ? mx.params?.toLocaleString("en-US") : null, done ? plural(mx.subdomains, "subnetwork", "subnetworks") : waiting),
  ].join("");
}

function renderLiveFigure(j) {
  const card = $("#liveFig");
  const done = j?.status === "done";
  const figs = j?.kind === "inverse" ? [["design", "Design"], ["inverse_history", "History"]] : [["fem_comparison", "ATLAS vs FEM"], ["convergence", "Convergence"]];
  const avail = done ? figs.filter(([k]) => j.plots?.[k]) : [];
  const cur = avail.find(([k]) => k === state.fig.live) || avail[0];
  const m = j ? liveMeta(j.benchmark) : null;
  const head = `<div class="fig-head"><div><h2>${j?.kind === "inverse" ? "Design" : "ATLAS vs FEM"}</h2>${m ? `<span class="meta">${esc(m.title)} · ATLAS-XPINN</span>` : ""}</div>${avail.length > 1 ? `<div class="seg" role="group" aria-label="Figure">${avail.map(([k, l]) => `<button type="button" data-livefig="${k}" aria-pressed="${k === cur[0]}">${esc(l)}</button>`).join("")}</div>` : ""}</div>`;
  if (!cur) {
    const msg = !j ? "The FEM comparison appears here when a run finishes." : j.status === "error" ? "The run failed before producing figures." : "The figures appear here when the run finishes.";
    card.innerHTML = `${head}<div class="fig-empty">${esc(msg)}</div>`;
    return;
  }
  const src = j.plots[cur[0]];
  card.innerHTML = `${head}<div class="fig-frame sm"><img src="${esc(src)}" alt="${esc(`${cur[1]}, ${m?.title || ""}`)}" data-full="${esc(src)}"></div><div class="fig-foot"><p></p><a class="open-link" href="${esc(src)}" target="_blank" rel="noopener">Open full size${ICON.open}</a></div>`;
}

function renderDetails(j) {
  const card = $("#details");
  const done = j?.status === "done";
  if (!done) {
    card.innerHTML = `<h2>${j?.kind === "inverse" ? "Design detail" : "Validation detail"}</h2><p class="note" style="margin-top:12px">${j?.kind === "inverse" ? "The design history appears when the run finishes." : "Per-component errors appear after FEM validation."}</p>`;
    return;
  }
  const mx = j.metrics;
  const m = liveMeta(j.benchmark);
  let rows;
  if (j.kind === "inverse") {
    rows = [
      [`${cap(mx.param)}, start`, "baseline geometry", mx.initial?.toFixed(4)],
      [`${cap(mx.param)}, found`, "ATLAS-XPINN design", mx.final?.toFixed(4)],
      ["Objective", mx.objective, sci(mx.target)],
      ["Status", "outer design loop", mx.converged ? "converged" : mx.status],
    ];
  } else if (m?.group === "multi") {
    rows = [
      ["Displacement L², u", "horizontal component", pct(mx.relative_L2_u)],
      ["Displacement L², v", "vertical component", pct(mx.relative_L2_v)],
      ["von Mises L²", "whole part, no exclusion", pct(mx.von_mises_l2)],
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
      ["von Mises L², ROI", `${mx.von_mises_roi_desc || "region of interest"} left out`, pct(mx.von_mises_l2_roi)],
      ["Max pointwise von Mises error", Array.isArray(loc) ? `at (x=${(+loc[0]).toFixed(4)}, y=${(+loc[1]).toFixed(4)})` : "location not reported", mx.max_pointwise_von_mises_error != null ? `${sci(mx.max_pointwise_von_mises_error)} Pa` : "—"],
      ["Points left out near singularities", mx.von_mises_exclusion_radius != null ? `r = ${(+mx.von_mises_exclusion_radius).toPrecision(4)}` : "", mx.von_mises_total_points != null ? `${mx.von_mises_excluded_points} of ${mx.von_mises_total_points.toLocaleString("en-US")}` : "—"],
    ];
  }
  card.innerHTML = `<h2>${j.kind === "inverse" ? "Design detail" : "Validation detail"}</h2><dl class="detail-grid">${rows.map(([l, s, v]) => `<div><dt><span>${esc(l)}</span><span>${esc(s)}</span></dt><dd>${esc(v ?? "—")}</dd></div>`).join("")}</dl>`;
}

async function refreshJobs() {
  try {
    const { jobs } = await api("/api/jobs");
    state.jobs = jobs;
    renderJobs();
  } catch {}
}

async function watchJob(id) {
  clearTimeout(state.poll);
  try {
    const j = await api(`/api/jobs/${id}`);
    const wasOpen = state.job?.id === id;
    const prevStatus = state.job?.status;
    state.job = j;
    if (!wasOpen || (prevStatus !== j.status && j.status === "running")) $("#logBox").open = j.status === "running" || j.status === "queued";
    const i = state.jobs.findIndex((x) => x.id === j.id);
    if (i >= 0) state.jobs[i] = { ...state.jobs[i], ...j }; else state.jobs.unshift(j);
    renderJobs();
    renderJob();
    if (j.status === "queued" || j.status === "running") state.poll = setTimeout(() => watchJob(id), 1500);
    else { refreshJobs(); pollHealth(); }
  } catch (e) {
    $("#jobError").hidden = false;
    $("#jobError").textContent = `Lost contact with the job: ${e.message}`;
    state.poll = setTimeout(() => watchJob(id), 4000);
  }
}

async function submit(kind) {
  const bench = state.liveBench;
  const body = kind === "inverse"
    ? { benchmark: bench }
    : { benchmark: bench, formulation: "energy_xpinn", epochs: Math.max(50, parseInt($("#epochs").value, 10) || 2000),
        n_interior: +$("#nInterior").value, prefit_epochs: +$("#prefit").value, warm_start_eb: $("#ebPrefit").checked };
  $("#runForward").disabled = true;
  $("#runInverse").disabled = true;
  try {
    const { job_id } = await api(`/api/${kind}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    state.jobs.unshift({ id: job_id, kind, benchmark: bench, status: "queued", stage: "queued" });
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
    state.seed = data.seed;
  } catch (e) {
    $("#benchPick").innerHTML = `<legend class="eyebrow">Benchmark</legend><p class="empty">Could not load the benchmarks: ${esc(e.message)}</p>`;
    return;
  }
  if (!liveMeta(state.liveBench)) state.liveBench = state.meta[0].key;
  renderBenchPick();
  renderSpec();
  await refreshJobs();
  const active = state.jobs.find((j) => j.status === "running" || j.status === "queued") || state.jobs[0];
  if (active) await watchJob(active.id); else renderJob();
}

document.addEventListener("click", (e) => {
  const themeBtn = e.target.closest("[data-theme-pick]");
  if (themeBtn) return applyTheme(themeBtn.dataset.themePick);
  const bt = e.target.closest("[data-bench]");
  if (bt) {
    state.bench = bt.dataset.bench;
    store.set("atlas-bench", state.bench);
    renderResults();
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

document.addEventListener("change", (e) => {
  if (e.target.name === "bench") {
    state.liveBench = e.target.value;
    store.set("atlas-live-bench", state.liveBench);
    $$(".opt").forEach((o) => o.classList.toggle("on", $("input", o).checked));
    renderSpec();
    renderJob();
  }
});

$("#runForward").addEventListener("click", () => submit("forward"));
$("#runInverse").addEventListener("click", () => submit("inverse"));
window.addEventListener("hashchange", route);

applyTheme(document.documentElement.getAttribute("data-theme") || "light");
route();
pollHealth();
setInterval(pollHealth, 5000);
setInterval(() => { if (!isBusy()) refreshJobs(); }, 10000);
loadLive().then(loadResults);
