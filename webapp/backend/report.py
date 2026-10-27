import base64
import html
from datetime import datetime
from pathlib import Path

from atlas import live

TITLES = {
    "cantilever": "Cantilever beam", "l_bracket": "L-shaped bracket", "plate_hole": "Plate with circular hole",
    "steel_insert": "Steel insert", "bimaterial": "Soft inclusion", "filled_slot": "Epoxy-filled slot",
}
TARGETS = {"displacement": 0.05, "von_mises": 0.10}


def _e(x) -> str:
    return html.escape(str(x))


def _pct(x) -> str:
    return "—" if x is None else f"{100 * float(x):.2f} %"


def _spct(x) -> str:
    return "—" if x is None else f"{100 * float(x):+.2f} %"


def _sci(x) -> str:
    return "—" if x is None else f"{float(x):.4g}"


def _img(path: Path, alt: str) -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{_e(alt)}"><figcaption>{_e(alt)}</figcaption></figure>'


def _table(rows, head=None) -> str:
    th = f"<thead><tr>{''.join(f'<th>{_e(h)}</th>' for h in head)}</tr></thead>" if head else ""
    body = "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>{th}<tbody>{body}</tbody></table>"


def _check(value, target) -> str:
    return "—" if value is None else ("pass" if value < target else "above target")


def _forward_rows(m: dict) -> list:
    vm_key = "von_mises_l2_excl" if m.get("von_mises_l2_excl") is not None else "von_mises_l2"
    vm = m.get(vm_key, m.get("von_mises_l2_global"))
    rows = [
        ["Displacement L² (relative)", _pct(m.get("relative_L2_displacement")), f"< {TARGETS['displacement']:.0%}",
         _check(m.get("relative_L2_displacement"), TARGETS["displacement"])],
        ["Displacement L², u", _pct(m.get("relative_L2_u")), "", ""],
        ["Displacement L², v", _pct(m.get("relative_L2_v")), "", ""],
        ["von Mises L²" + (" (singular points excluded)" if vm_key == "von_mises_l2_excl" else ""), _pct(vm),
         f"< {TARGETS['von_mises']:.0%}", _check(vm, TARGETS["von_mises"])],
    ]
    if m.get("von_mises_l2_global") is not None and vm_key == "von_mises_l2_excl":
        rows.append(["von Mises L², whole domain", _pct(m["von_mises_l2_global"]), "", ""])
    if m.get("compliance_error") is not None:
        rows.append(["Compliance vs FEM", _spct(m["compliance_error"]), "", ""])
    if m.get("band_vm_l2_inside") is not None:
        rows.append(["von Mises L², interface band (inside)", _pct(m["band_vm_l2_inside"]), "", ""])
        rows.append(["von Mises L², interface band (outside)", _pct(m["band_vm_l2_outside"]), "", ""])
    if m.get("max_pointwise_von_mises_error") is not None:
        loc = m.get("max_pointwise_von_mises_error_location")
        where = f" at ({loc[0]:.4f}, {loc[1]:.4f}) m" if loc else ""
        rows.append(["Max pointwise von Mises error", f"{m['max_pointwise_von_mises_error'] / 1e6:.4g} MPa{where}", "", ""])
    return rows


def build_report(job: dict, run_dir: Path, version: str, boundary_conditions: dict) -> str:
    bench = job["benchmark"]
    m = job.get("metrics") or {}
    cfg = job.get("solver_config") or {}
    budget = job.get("budget") or {}
    settings = job.get("settings") or live.values(bench)
    inverse = job["kind"] == "inverse"
    title = TITLES.get(bench, bench)
    when = job.get("finished") or job.get("created")
    stamp = datetime.fromisoformat(when).strftime("%d %B %Y, %H:%M") if when else "—"

    groups: dict = {}
    for s in live.PARAMS[bench]:
        val = settings.get(s["key"], s["value"])
        groups.setdefault(s["group"], []).append([s["label"], f"{val:g} {s['unit']}".strip(),
                                                  "" if abs(val - s["value"]) < 1e-9 * max(1, abs(s["value"]))
                                                  else f"study value {s['value']:g}"])
    props = "".join(f"<h3>{_e(g)}</h3>" + _table(r, ["Property", "Value", "Note"]) for g, r in groups.items())
    bcs = "".join(f"<li>{_e(b)}</li>" for b in boundary_conditions.get(bench, []))

    solver = "ATLAS-XPINN (domain-decomposed deep energy)" if job.get("formulation") == "energy_xpinn" \
        else "ATLAS single network (deep energy)"
    solver_rows = [
        ["Formulation", solver],
        ["Network", f"{m.get('hidden_layers', cfg.get('hidden_layers'))} hidden layers × {m.get('neurons', cfg.get('neurons'))} neurons"
                    + (f", {m['subdomains']} subnetworks" if (m.get("subdomains") or 1) > 1 else "")],
        ["Trainable parameters", f"{m['params']:,}" if m.get("params") else "—"],
        ["Optimiser budget", f"{budget.get('epochs')} Adam epochs + {budget.get('lbfgs')} L-BFGS iterations"
                             + (f", {budget['n_outer']} outer design steps" if budget.get("n_outer") else "")],
    ]
    if job.get("formulation") == "energy_xpinn":
        solver_rows.append(["Interface coupling", f"augmented Lagrangian, γ_u = {cfg.get('gamma_u')}, γ_t = {cfg.get('gamma_t')}, "
                                                  f"{cfg.get('al_stages')} stages"])
    solver_rows += [
        ["Precision / seed", f"{m.get('precision', cfg.get('precision'))} · seed {m.get('seed', cfg.get('seed'))}"],
        ["Compute device", m.get("device_name") or job.get("device_name") or "—"],
        ["Training time", f"{m['train_seconds']:.1f} s" if m.get("train_seconds") is not None else "—"],
        ["FEM reference", m.get("fem_engine") or job.get("fem_engine") or "—"],
    ]
    if m.get("fem_cells"):
        solver_rows.append(["FEM mesh", f"{m['fem_cells']:,} cells"])

    if inverse:
        fc = m.get("fem_check") or {}
        unit = "Pa" if m.get("metric") == "peak_vm" else "N·m" if m.get("metric") == "compliance" else ""
        results = _table([
            [f"{m.get('param', 'parameter').capitalize()}, start", f"{m.get('initial', 0):.4f} m"],
            [f"{m.get('param', 'parameter').capitalize()}, found", f"{m.get('final', 0):.4f} m"],
            ["Objective", f"{m.get('objective', '')}: {_sci(m.get('target'))} {unit}"],
            ["Outer design steps", f"{m.get('n_outer', '—')} ({'converged' if m.get('converged') else m.get('status', '')})"],
            ["FEM at the start", f"{_sci(fc.get('fem_baseline'))} {unit}"],
            ["FEM target", f"{_sci(fc.get('fem_target'))} {unit}"],
            ["FEM at the design", f"{_sci(fc.get('fem_at_design'))} {unit}"],
            ["FEM at the design vs target", _spct(fc.get("error_vs_target"))],
            ["ATLAS vs FEM at the design", _spct(fc.get("atlas_vs_fem"))],
        ], ["Quantity", "Value"])
        figs = _img(run_dir / f"{bench}_history.png", "Design history: parameter, objective relative to target, residual") + \
            _img(run_dir / f"{bench}_design.png", "Baseline and optimised geometry with predicted fields")
        conv = ""
    else:
        results = _table(_forward_rows(m), ["Metric", "Value", "Target", "Check"])
        if m.get("von_mises_exclusion_radius") is not None:
            results += (f"<p class='note'>Stress is singular at re-entrant corners and load or support edges. "
                        f"{m.get('von_mises_excluded_points', '')} of {m.get('von_mises_total_points', '')} points within "
                        f"r = {m['von_mises_exclusion_radius']:.4g} m of those points are excluded from the von Mises "
                        f"error; the whole-domain value is listed separately.</p>")
        conv = _img(run_dir / "convergence.png", "Convergence: relative energy gap and strain energy against iterations")
        figs = _img(run_dir / "fem_comparison.png", "ATLAS and FEM: |u| and von Mises stress with pointwise error") + \
            _img(run_dir / "deformed.png", "Deformed shape (displacement magnified)")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ATLAS Calculation Report · {_e(title)}</title>
<style>
:root {{ color-scheme: light; --ink:#111827; --muted:#4b5563; --line:#d1d5db; --soft:#f3f4f6; --accent:#1d4ed8; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: #e5e7eb; color: var(--ink); font: 13px/1.5 "Segoe UI", system-ui, -apple-system, sans-serif; }}
.page {{ max-width: 210mm; margin: 24px auto; background: #fff; padding: 18mm 16mm; box-shadow: 0 2px 12px rgba(0,0,0,.12); }}
header {{ border-bottom: 2px solid var(--ink); padding-bottom: 10px; margin-bottom: 18px; display: flex; justify-content: space-between; gap: 16px; align-items: flex-end; }}
header h1 {{ margin: 0; font-size: 22px; letter-spacing: -.01em; }}
header .eyebrow {{ font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }}
header .meta {{ text-align: right; color: var(--muted); font-size: 12px; }}
h2 {{ font-size: 15px; margin: 22px 0 8px; padding-bottom: 4px; border-bottom: 1px solid var(--line); }}
h3 {{ font-size: 12.5px; margin: 12px 0 4px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
table {{ width: 100%; border-collapse: collapse; margin: 4px 0 8px; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ background: var(--soft); font-weight: 600; font-size: 12px; }}
ul {{ margin: 4px 0; padding-left: 18px; }}
figure {{ margin: 12px 0; break-inside: avoid; }}
figure img {{ width: 100%; height: auto; border: 1px solid var(--line); }}
figcaption {{ color: var(--muted); font-size: 11.5px; margin-top: 4px; }}
.note {{ color: var(--muted); font-size: 12px; }}
footer {{ margin-top: 24px; padding-top: 8px; border-top: 1px solid var(--line); color: var(--muted); font-size: 11px; display: flex; justify-content: space-between; }}
.toolbar {{ max-width: 210mm; margin: 16px auto 0; display: flex; justify-content: flex-end; gap: 8px; padding: 0 4px; }}
.toolbar button {{ font: inherit; padding: 8px 16px; border-radius: 8px; border: 1px solid var(--accent); background: var(--accent); color: #fff; cursor: pointer; }}
@media (max-width: 600px) {{ .page {{ padding: 16px; margin: 0; }} header {{ flex-direction: column; align-items: flex-start; }} header .meta {{ text-align: left; }} }}
@media print {{ body {{ background: #fff; }} .toolbar {{ display: none; }} .page {{ box-shadow: none; margin: 0; padding: 0; max-width: none; }}
  @page {{ size: A4; margin: 15mm; }} h2 {{ break-after: avoid; }} table {{ break-inside: avoid; }} }}
</style></head>
<body>
<div class="toolbar"><button type="button" onclick="window.print()">Print / Save as PDF</button></div>
<main class="page">
<header>
  <div><div class="eyebrow">ATLAS engineering calculation report</div><h1>{_e(title)} · {"Inverse shape design" if inverse else "Forward solve and FEM validation"}</h1></div>
  <div class="meta">{_e(stamp)}<br>Run {_e(job['id'])}{" · custom settings" if job.get("custom") else " · study settings"}</div>
</header>
<h2>1. Model: plane stress, linear elastic</h2>
{props}
<h3>Boundary conditions</h3>
<ul>{bcs}</ul>
<h2>2. Solver</h2>
{_table(solver_rows)}
<h2>3. {"Design result and FEM check" if inverse else "Validation against FEM"}</h2>
{results}
{f"<h2>4. Convergence</h2>{conv}" if conv else ""}
<h2>{"4" if not conv else "5"}. Figures</h2>
{figs}
<footer><span>Generated by ATLAS Desktop Studio v{_e(version)}</span><span>{_e(job.get('folder', ''))}</span></footer>
</main>
</body></html>
"""
