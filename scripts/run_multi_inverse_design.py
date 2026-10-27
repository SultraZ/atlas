"""Run from the project root with the Atlas env (FEniCS on PATH):
    PATH=$CONDA_PREFIX/bin:$PATH python scripts/run_multi_inverse_design.py [BENCHMARK ...]

Example:
    PATH=$CONDA_PREFIX/bin:$PATH python scripts/run_multi_inverse_design.py steel_insert bimaterial filled_slot
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fem"))

from atlas.atlas_main import run_inverse_design_all, setup_device
from atlas.paths import RESULTS_DIR
from fem_parametric import fem_metrics

OUT = RESULTS_DIR / "atlas_xpinn_energy" / "inverse_design"


def fem_reference(bench, metric, target, p_atlas, lo, hi):
    pts = np.clip(np.linspace(p_atlas - 0.12, p_atlas + 0.12, 7), lo, hi)
    sweep = [(float(p), fem_metrics(bench, float(p))[metric]) for p in np.unique(pts)]
    print(f"  FEM sweep: {[(round(p, 4), round(v, 6)) for p, v in sweep]}", flush=True)
    g = [(p, v / target - 1.0) for p, v in sweep]
    pairs = [(a, b) for a, b in zip(g, g[1:]) if a[1] * b[1] <= 0]
    if not pairs:
        return None, sweep
    (p0, g0), (p1, g1) = pairs[0]
    for _ in range(3):
        p = p0 - g0 * (p1 - p0) / (g1 - g0)
        v = fem_metrics(bench, p)[metric]
        sweep.append((p, v))
        gp = v / target - 1.0
        if abs(gp) < 1e-4:
            break
        (p0, g0), (p1, g1) = ((p0, g0), (p, gp)) if g0 * gp <= 0 else ((p, gp), (p1, g1))
    return float(p), sorted(sweep)


def run(bench, device):
    res = run_inverse_design_all(bench, device, decomposed=True, out_dir=OUT)
    (OUT / f"{bench}_design_net.pt").unlink(missing_ok=True)
    metric, p_atlas = res["metric"], float(res["p_star"])
    ratio = res["target"] / res["baseline"][metric]
    base = fem_metrics(bench, float(res["case"].p0))
    found = fem_metrics(bench, p_atlas)
    fem_target = ratio * base[metric]
    p_ref, sweep = fem_reference(bench, metric, fem_target, p_atlas, *res["bounds"])
    check = dict(benchmark=bench, metric=metric, target_ratio=ratio, fem_target=fem_target, p_atlas=p_atlas,
                 metric_fem_at_p_atlas=found[metric], metric_atlas_at_p_atlas=float(res["metrics"][metric]),
                 metric_error_vs_target=found[metric] / fem_target - 1.0,
                 atlas_vs_fem_metric=float(res["metrics"][metric]) / found[metric] - 1.0,
                 p_fem_reference=p_ref, param_difference=None if p_ref is None else abs(p_atlas - p_ref) / p_ref,
                 within_10pct=bool(abs(found[metric] / fem_target - 1.0) < 0.10),
                 fem_baseline=base, fem_at_p_atlas=found, sweep=[list(s) for s in sweep])
    with open(OUT / f"{bench}_fem_check.json", "w") as f:
        json.dump(check, f, indent=2)
    print(f"{bench}: ATLAS {res['param_name']} = {p_atlas:.4f}, FEM optimum {p_ref}, "
          f"FEM metric vs target {check['metric_error_vs_target']:+.2%}, ATLAS vs FEM {check['atlas_vs_fem_metric']:+.2%}",
          flush=True)


if __name__ == "__main__":
    device = setup_device()
    for b in sys.argv[1:] or ["steel_insert", "bimaterial", "filled_slot"]:
        run(b, device)
