"""Figures and table for the ATLAS energy vs ATLAS-XPINN convergence /
processing-time study (input: results/comparison/convergence_timing/*.json written
by scripts/run_convergence_timing.py).

Outputs (same folder):
  convergence_vs_iteration.png   error vs iteration, both solvers, 3 benchmarks
  convergence_vs_time.png        error vs training wall-clock
  processing_time.png            total training time and time-to-target bars
  CONVERGENCE_TIMING.md          the numbers
"""
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np

from atlas.paths import RESULTS_DIR

D = RESULTS_DIR / "comparison" / "convergence_timing"
BENCH = [("cantilever", "Cantilever"), ("plate_hole", "Plate with hole"), ("l_bracket", "L-bracket")]
SOLVERS = [("deepxde", "PINN (DeepXDE, library defaults)", "#c0504d"),
           ("deepxde_configured", "PINN (DeepXDE, configured)", "#8c6d1f"),
           ("xpinn", "ATLAS-XPINN", "#d9822b")]


def load(b, s):
    p = D / f"{b}_{s}.json"
    return json.load(open(p)) if p.exists() else None


def first_below(rows, key, thr, x):
    """First probe where the metric is below thr and stays below afterwards."""
    vals = [r[key] for r in rows]
    for i in range(len(vals)):
        if all(v is not None and v < thr for v in vals[i:]):
            return rows[i][x]
    return None


def curves(xkey, xlabel, fname, metric, ylabel, thr):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, (b, title) in zip(axs, BENCH):
        for s, name, col in SOLVERS:
            d = load(b, s)
            if d is None:
                continue
            x = [r[xkey] for r in d["probes"]]
            y = [max(r[metric], 1e-5) for r in d["probes"]]
            ax.semilogy(x, y, "-o", ms=2.5, lw=1.4, color=col, label=name)
        ax.axhline(thr, color="r", ls="--", lw=1, label=f"target {thr:.0%}")
        ax.set_xscale("log")
        ax.set(title=title, xlabel=xlabel, ylim=(1e-3, 3))
        ax.grid(True, which="both", alpha=0.3)
    axs[0].set_ylabel(ylabel)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(D / fname, dpi=140)
    plt.close(fig)


def main():
    if not any(load(b, s) for b, _ in BENCH for s, *_ in SOLVERS):
        sys.exit("no study data found")
    curves("iter", "iteration (Adam, then L-BFGS; log axis)", "convergence_vs_iteration.png",
           "disp_vec_l2", "displacement rel-L2 vs FEM", 0.05)
    curves("seconds", "training wall-clock [s] (log axis)", "convergence_vs_time.png",
           "disp_vec_l2", "displacement rel-L2 vs FEM", 0.05)
    curves("seconds", "training wall-clock [s] (log axis)", "convergence_vs_time_vonmises.png",
           "vm_l2_excl", "von Mises rel-L2 vs FEM (excl.)", 0.10)

    lines = ["# Convergence and processing time: DeepXDE PINN (defaults / configured) vs ATLAS-XPINN", "",
             "Same seed (42), GPU, one job at a time. ATLAS-XPINN: 2000 Adam + 2000 L-BFGS "
             "iterations. PINN (DeepXDE): 10000 Adam + 2000 L-BFGS iterations, library defaults "
             "(float32) or the configured run (exact essential BCs via an output transform, float64, "
             "extra collocation points at the stress concentration, loss weights balanced from the "
             "initial losses; never chosen by the FEM error). Errors are scored against the FEM "
             "reference during training (monitoring only, never in the loss); probing time is excluded from the clock. Displacement "
             "error is the vector L2 (stricter than the |u| metric in validation.py). "
             "\"Time to X\" = first probe after which the error stays below X.", "",
             "| benchmark | solver | params | train time [s] | s / iteration | final disp L2 | "
             "final vM L2 (excl.) | time to 5% disp [s] | time to 1% disp [s] | iters to 5% | "
             "iters to 1% |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, f: "-" if v is None else format(v, f)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    times = {}
    for b, title in BENCH:
        for s, name, col in SOLVERS:
            d = load(b, s)
            if d is None:
                continue
            rows = d["probes"]
            n_it = rows[-1]["iter"]
            t5 = first_below(rows, "disp_vec_l2", 0.05, "seconds")
            t1 = first_below(rows, "disp_vec_l2", 0.01, "seconds")
            i5 = first_below(rows, "disp_vec_l2", 0.05, "iter")
            i1 = first_below(rows, "disp_vec_l2", 0.01, "iter")
            times[(b, s)] = (d["train_seconds"], t5, t1)
            lines.append(f"| {title} | {name} | {d['params']} | {d['train_seconds']:.0f} | "
                         f"{d['train_seconds'] / n_it:.3f} | {rows[-1]['disp_vec_l2']:.2%} | "
                         f"{rows[-1]['vm_l2_excl']:.2%} | {fmt(t5, '.0f')} | {fmt(t1, '.0f')} | "
                         f"{fmt(i5, 'd')} | {fmt(i1, 'd')} |")
    for ax, (b, title) in zip(axs, BENCH):
        w = 0.27
        for k, (s, name, col) in enumerate(SOLVERS):
            if (b, s) not in times:
                continue
            raw = times[(b, s)]
            vals = [v if v is not None else 0 for v in raw]
            xs = np.arange(3) + (k - 1.0) * w
            ax.bar(xs, vals, w, color=col, label=name)
            for x, v, r in zip(xs, vals, raw):
                ax.text(x, v, "not reached" if r is None else f"{v:.0f}", ha="center",
                        va="bottom", fontsize=8, rotation=90 if r is None else 0)
        ax.set_xticks(range(3), ["total training", "to 5% disp", "to 1% disp"])
        ax.set(title=title, ylabel="wall-clock [s]")
        ax.grid(True, axis="y", alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(D / "processing_time.png", dpi=140)
    plt.close(fig)
    (D / "CONVERGENCE_TIMING.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
