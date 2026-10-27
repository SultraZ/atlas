"""Run from the project root with the Atlas env (FEniCS on PATH):
    PATH=$CONDA_PREFIX/bin:$PATH python scripts/run_inverse_seed_study.py [--seeds 42 43 44 45 46] [--benchmarks ...]

Example:
    PATH=$CONDA_PREFIX/bin:$PATH python scripts/run_inverse_seed_study.py --seeds 42 43 44 45 46
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fem"))

from atlas.atlas_main import design_result_to_json, setup_device
from atlas.energy_design import run_inverse_design
from atlas.paths import RESULTS_DIR
from fem_parametric import fem_metrics

BENCHMARKS = ["cantilever", "plate_hole", "l_bracket", "steel_insert", "bimaterial", "filled_slot"]
TITLES = {"cantilever": "Cantilever", "plate_hole": "Plate with a hole", "l_bracket": "L-bracket",
          "steel_insert": "Steel insert", "bimaterial": "Soft inclusion", "filled_slot": "Epoxy-filled slot"}
REF = RESULTS_DIR / "atlas_xpinn_energy" / "inverse_design"
OUT = REF / "seeds"


def run_one(bench, seed, device):
    path = OUT / f"{bench}_seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    ref = json.loads((REF / f"{bench}_fem_check.json").read_text())
    t0 = time.perf_counter()
    res = run_inverse_design(bench, device=device, decomposed=True, seed=seed, verbose=True)
    seconds = time.perf_counter() - t0
    data = design_result_to_json(res)
    metric, p = data["metric"], float(data["p_star"])
    fem = fem_metrics(bench, p)[metric]
    row = dict(benchmark=bench, seed=seed, param_name=data["param_name"], metric=metric, p_atlas=p,
               status=data["status"], converged=bool(data["converged"]),
               n_steps=len([h for h in data["history"] if h["tag"] not in ("probe", "return")]),
               atlas_metric=float(data["metrics"][metric]), atlas_target=float(data["target"]),
               fem_target=ref["fem_target"], fem_at_p_atlas=fem, p_fem_reference=ref["p_fem_reference"],
               fem_error_vs_target=fem / ref["fem_target"] - 1.0,
               atlas_vs_fem=float(data["metrics"][metric]) / fem - 1.0,
               param_difference=abs(p - ref["p_fem_reference"]) / ref["p_fem_reference"],
               seconds=seconds, design=data)
    path.write_text(json.dumps(row, indent=2))
    print(f"{bench} seed {seed}: {data['param_name']} = {p:.4f}, FEM vs target {row['fem_error_vs_target']:+.2%}, "
          f"ATLAS vs FEM {row['atlas_vs_fem']:+.2%}, {seconds / 60:.1f} min", flush=True)
    return row


def table_png(header, body, path, title, group_rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    widths = [max([1.2 * len(col[0])] + [len(str(c)) for c in col[1:]]) + 2 for col in zip(header, *body)]
    fig_w = 0.105 * sum(widths) + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, 0.32 * (len(body) + 1) + 0.6), dpi=200)
    ax.axis("off")
    tab = ax.table(cellText=body, colLabels=header, loc="center", cellLoc="center",
                   colWidths=[w / sum(widths) for w in widths])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.45)
    shade = group_rows or [i % 2 for i in range(len(body))]
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#c8ccd2")
        if r == 0:
            cell.set_facecolor("#1f3b5c")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#eef2f7" if shade[r - 1] else "white")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)


def summarize(rows, seeds):
    benches = [b for b in BENCHMARKS if any(r["benchmark"] == b for r in rows)]
    summary, lines = {}, []
    lines.append(f"# ATLAS-XPINN inverse design, seeds {', '.join(map(str, seeds))}\n")
    lines.append("FEM = FEniCS 2019.1.0. The FEM target is the target ratio times the FEM value at the starting shape.\n")
    lines.append("## Per seed\n")
    lines.append("| Benchmark | Seed | Parameter found | FEM optimum | Parameter vs FEM optimum | FEM at design vs target | ATLAS vs FEM | Steps | Status | Time |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for b in benches:
        rs = sorted((r for r in rows if r["benchmark"] == b), key=lambda r: r["seed"])
        for r in rs:
            lines.append(f"| {TITLES[b]} | {r['seed']} | {r['p_atlas']:.4f} | {r['p_fem_reference']:.4f} | "
                         f"{100 * r['param_difference']:.2f} % | {100 * r['fem_error_vs_target']:+.2f} % | "
                         f"{100 * r['atlas_vs_fem']:+.2f} % | {r['n_steps']} | {'converged' if r['converged'] else r['status']} | "
                         f"{r['seconds'] / 60:.1f} min |")
        p = np.array([r["p_atlas"] for r in rs])
        e = np.array([r["fem_error_vs_target"] for r in rs])
        a = np.array([r["atlas_vs_fem"] for r in rs])
        summary[b] = dict(title=TITLES[b], param_name=rs[0]["param_name"], metric=rs[0]["metric"],
                          seeds=[r["seed"] for r in rs], n=len(rs), p_mean=float(p.mean()), p_std=float(p.std(ddof=1)) if len(p) > 1 else 0.0,
                          p_min=float(p.min()), p_max=float(p.max()), p_fem_reference=rs[0]["p_fem_reference"],
                          fem_error_mean=float(e.mean()), fem_error_max_abs=float(np.abs(e).max()),
                          atlas_vs_fem_mean=float(a.mean()), atlas_vs_fem_max_abs=float(np.abs(a).max()),
                          n_within_10pct=int(np.sum(np.abs(e) < 0.10)), n_converged=int(sum(r["converged"] for r in rs)))
    per_seed = []
    groups = []
    for gi, b in enumerate(benches):
        for r in sorted((r for r in rows if r["benchmark"] == b), key=lambda r: r["seed"]):
            per_seed.append([TITLES[b], r["seed"], f"{r['p_atlas']:.4f}", f"{r['p_fem_reference']:.4f}",
                             f"{100 * r['param_difference']:.2f} %", f"{100 * r['fem_error_vs_target']:+.2f} %",
                             f"{100 * r['atlas_vs_fem']:+.2f} %", r["n_steps"], f"{r['seconds'] / 60:.1f} min"])
            groups.append(gi % 2)
    table_png(["Benchmark", "Seed", "Parameter found", "FEM optimum", "Parameter vs FEM optimum",
               "FEM at design vs goal", "ATLAS vs FEM", "Steps", "Time"], per_seed, OUT / "per_seed_table.png",
              f"ATLAS-XPINN inverse design, per seed ({', '.join(map(str, seeds))})", groups)
    lines.append("\n## Summary across seeds\n")
    lines.append("| Benchmark | Parameter | Goal metric | Seeds | Parameter, mean ± sd | Range | FEM optimum | FEM vs target, mean | FEM vs target, worst | ATLAS vs FEM, worst | Within 10 % |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for b, s in summary.items():
        lines.append(f"| {s['title']} | {s['param_name']} | {s['metric']} | {s['n']} | {s['p_mean']:.4f} ± {s['p_std']:.4f} | "
                     f"{s['p_min']:.4f}–{s['p_max']:.4f} | {s['p_fem_reference']:.4f} | {100 * s['fem_error_mean']:+.2f} % | "
                     f"{100 * s['fem_error_max_abs']:.2f} % | {100 * s['atlas_vs_fem_max_abs']:.2f} % | {s['n_within_10pct']}/{s['n']} |")
    goals = {"compliance": "compliance", "peak_vm": "peak stress"}
    table_png(["Benchmark", "Goal", "Parameter", "Mean ± sd", "Range", "FEM optimum", "FEM vs goal, mean",
               "FEM vs goal, worst", "ATLAS vs FEM, worst", "Within 10 %"],
              [[s["title"], goals.get(s["metric"], s["metric"]), s["param_name"], f"{s['p_mean']:.4f} ± {s['p_std']:.4f}",
                f"{s['p_min']:.4f}–{s['p_max']:.4f}", f"{s['p_fem_reference']:.4f}", f"{100 * s['fem_error_mean']:+.2f} %",
                f"{100 * s['fem_error_max_abs']:.2f} %", f"{100 * s['atlas_vs_fem_max_abs']:.2f} %", f"{s['n_within_10pct']}/{s['n']}"]
               for s in summary.values()], OUT / "summary_table.png",
              f"ATLAS-XPINN inverse design, summary across seeds ({', '.join(map(str, seeds))})")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, choices=BENCHMARKS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = setup_device()
    rows = [run_one(b, s, device) for b in args.benchmarks for s in args.seeds]
    summarize(rows, args.seeds)
