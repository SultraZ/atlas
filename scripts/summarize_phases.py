"""Build the phase-by-phase ATLAS forward-solver summary table.

Gate definitions match validation.py:
  - displacement pass : relative_L2_displacement < 5%
  - stress pass       : von_mises_l2_roi < 10% (fixed singular-ROI reading;
                      von_mises_l2_excl is kept as context alongside it)
Global von Mises is reported as context only and never gates.

Usage:
    python scripts/summarize_phases.py
    python scripts/summarize_phases.py --results-dir results --out results/PHASE_SUMMARY.md
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.paths import RESULTS_DIR

BENCH_ORDER = ["cantilever", "l_bracket", "plate_hole"]


def collect(results_dir):
    rows = []
    for root, _dirs, files in os.walk(results_dir):
        if "baseline_pre_phase0" in root.replace(os.sep, "/"):
            continue
        for name in files:
            if not name.endswith("_validation.json"):
                continue
            bench = name[: -len("_validation.json")]
            if bench not in BENCH_ORDER:
                continue
            path = os.path.join(root, name)
            with open(path) as f:
                res = json.load(f)
            rel = os.path.relpath(root, results_dir).replace(os.sep, "/")
            cfg = res.get("train_config") or {}
            rows.append(
                {
                    "snapshot": rel,
                    "benchmark": bench,
                    "epochs": cfg.get("epochs"),
                    "warm_start_eb": cfg.get("warm_start_eb", False),
                    "lbfgs": cfg.get("use_lbfgs", False),
                    "lbfgs_steps": cfg.get("lbfgs_steps"),
                    "disp": res.get("relative_L2_displacement"),
                    "disp_ok": res.get("displacement_within_5pct_target"),
                    "vm_excl": res.get("von_mises_l2_excl"),
                    "vm_excl_ok": res.get("von_mises_l2_excl_within_10pct"),
                    "vm_roi": res.get("von_mises_l2_roi"),
                    "vm_roi_ok": res.get("von_mises_l2_roi_within_10pct"),
                    "vm_global": res.get("von_mises_l2_global"),
                }
            )
    rows.sort(key=lambda r: (r["snapshot"], BENCH_ORDER.index(r["benchmark"])))
    return rows


def fmt_pct(v):
    return "n/a" if v is None else f"{v:.2%}"


def fmt_flag(v):
    return "n/a" if v is None else ("PASS" if v else "FAIL")


def build_markdown(rows):
    lines = [
        "# ATLAS forward-solver phase summary",
        "",
        "Metrics from the unchanged validation path. Displacement gate: <5% "
        "(relative L2). Stress gate: <10% von Mises L2 on the fixed singular "
        "ROI (von_mises_l2_roi); von_mises_l2_excl is kept as context. Global "
        "von Mises is context only.",
        "",
        "| snapshot | benchmark | epochs | warm-EB | L-BFGS | disp L2% | disp | "
        "vm excl% | vm excl | vm ROI% | vm ROI | vm global% |",
        "|---|---|---|:---:|:---:|---|:---:|---|:---:|---|:---:|---|",
    ]
    for r in rows:
        lines.append(
            "| {snapshot} | {benchmark} | {epochs} | {warm} | {lbfgs} | {disp} | "
            "{dispf} | {vme} | {vmef} | {vmroi} | {vmroif} | {vmg} |".format(
                snapshot=r["snapshot"],
                benchmark=r["benchmark"],
                epochs=r["epochs"],
                warm="yes" if r["warm_start_eb"] else "-",
                lbfgs=(
                    f"yes({r['lbfgs_steps']})" if r["lbfgs"] else "-"
                ),
                disp=fmt_pct(r["disp"]),
                dispf=fmt_flag(r["disp_ok"]),
                vme=fmt_pct(r["vm_excl"]),
                vmef=fmt_flag(r["vm_excl_ok"]),
                vmroi=fmt_pct(r["vm_roi"]),
                vmroif=fmt_flag(r["vm_roi_ok"]),
                vmg=fmt_pct(r["vm_global"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out", default=str(RESULTS_DIR / "PHASE_SUMMARY.md"))
    args = parser.parse_args()

    rows = collect(args.results_dir)
    md = build_markdown(rows)
    with open(args.out, "w") as f:
        f.write(md)
    print(md)
    print(f"[summary] {len(rows)} runs written to {args.out}")


if __name__ == "__main__":
    main()
