"""Independent FEM check of an inverse-design result (docs/ATLAS_spec.md: compliance
of the ATLAS-optimised design vs the FEM parametric reference, target within 10%).

  1. solves the design ATLAS chose (p*) with FEM and compares the metric with the
     target and with ATLAS's own value;
  2. sweeps the parameter with FEM, finds where the FEM metric equals the target
     (the "FEM parametric reference" design) and compares it with p*.

Run in the Atlas env with its bin first on PATH (pkg-config for dolfin):
    python scripts/verify_inverse_design.py --benchmark plate_hole
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "fem"))

import numpy as np

from fem_parametric import fem_metrics

DEFAULT_RATIO = {'cantilever': 0.6, 'plate_hole': 1.10, 'l_bracket': 0.75}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(DEFAULT_RATIO))
    ap.add_argument("--json", default=None)
    ap.add_argument("--solver", default="atlas_energy", choices=["atlas_energy", "atlas_xpinn_energy"],
                    help="which solver's results/<solver>/inverse_design/ folder to read and write")
    ap.add_argument("--n", type=int, default=6, help="FEM sweep points")
    ap.add_argument("--window", type=float, default=0.35, help="sweep +/- this fraction of p*")
    args = ap.parse_args()

    folder = os.path.join(ROOT, "results", args.solver, "inverse_design")
    path = args.json or os.path.join(folder, f"{args.benchmark}_inverse.json")
    res = json.load(open(path))
    b, metric = args.benchmark, res["metric"]
    p_star, p0 = res["p_star"], res["param_baseline"]
    lo, hi = res["bounds"]
    ratio = res["target"] / res["baseline"][metric]

    fem_base = fem_metrics(b, p0)
    target = ratio * fem_base[metric]
    at_star = fem_metrics(b, p_star)
    print(f"FEM baseline {metric} = {fem_base[metric]:.6e}; target = {ratio:.4f} x = {target:.6e}")
    print(f"FEM at ATLAS design p*={p_star:.4f}: {metric} = {at_star[metric]:.6e} "
          f"({at_star[metric] / target - 1:+.2%} vs target)")

    ps = np.unique(np.clip(np.concatenate([
        np.linspace(p_star * (1 - args.window), p_star * (1 + args.window), args.n), [p_star]]), lo, hi))
    sweep = [(float(p), fem_metrics(b, float(p))[metric]) for p in ps]
    for p, v in sweep:
        print(f"  FEM p={p:.4f}: {metric}={v:.6e} ({v / target - 1:+.2%})")
    arr = np.array(sweep)
    order = np.argsort(arr[:, 1])
    p_fem = float(np.interp(target, arr[order, 1], arr[order, 0])) \
        if arr[:, 1].min() <= target <= arr[:, 1].max() else None

    out = dict(benchmark=b, metric=metric, target_ratio=ratio, fem_target=target,
               p_atlas=p_star, metric_fem_at_p_atlas=at_star[metric],
               metric_atlas_at_p_atlas=res["metrics"][metric],
               metric_error_vs_target=at_star[metric] / target - 1.0,
               atlas_vs_fem_metric=res["metrics"][metric] / at_star[metric] - 1.0,
               p_fem_reference=p_fem,
               param_difference=None if p_fem is None else (p_star - p_fem) / p_fem,
               within_10pct=abs(at_star[metric] / target - 1.0) < 0.10,
               fem_baseline=fem_base, fem_at_p_atlas=at_star, sweep=sweep)
    op = os.path.join(folder, f"{b}_fem_check.json")
    json.dump(out, open(op, "w"), indent=2)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(arr[:, 0], arr[:, 1], "o-", color="#4f81bd", label="FEM sweep")
    ax.axhline(target, color="r", ls="--", label=f"FEM target ({ratio:.2f} x baseline)")
    ax.axvline(p_star, color="#2e8b57", ls=":", label=f"ATLAS design p*={p_star:.3f}")
    if p_fem is not None:
        ax.axvline(p_fem, color="k", ls=":", label=f"FEM reference p={p_fem:.3f}")
    ax.plot([p_star], [res["metrics"][metric]], "s", color="#2e8b57", label="ATLAS metric at p*")
    ax.set(xlabel=res["param_name"], ylabel=metric, title=f"{b}: ATLAS design vs FEM parametric reference")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(op.replace(".json", ".png"), dpi=140)
    print(f"\nRESULT {b}: FEM {metric} at ATLAS p* is {out['metric_error_vs_target']:+.2%} from target "
          f"({'PASS' if out['within_10pct'] else 'FAIL'} <10%); "
          f"p* vs FEM reference p: {out['param_difference'] if p_fem is None else format(out['param_difference'], '+.2%')}")


if __name__ == "__main__":
    main()
