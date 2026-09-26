"""Aggregate the per-solver *.{npz,json} written by scripts/compare_solvers.py
(results/comparison, results/atlas_energy/forward, results/atlas_xpinn_energy/forward) into a
folder results/comparison/<out-sub>/ containing

    COMPARISON.md                 table of errors vs FEM (+ compliance, pointwise maxima)
    <benchmark>_comparison.png    FEM | each solver: fields + displacement error
    comparison_summary.png        grouped bars: displacement, von Mises, compliance errors

    python scripts/make_comparison_report.py                                  # all_solvers/
    python scripts/make_comparison_report.py --only deepxde xpinn_energy \
        --out-sub standard_pinn_vs_atlas_xpinn
    python scripts/make_comparison_report.py --only atlas xpinn_energy --out-sub atlas_vs_atlas_xpinn
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

from atlas.paths import RESULTS_DIR
from atlas.sampling import CantileverBeamSampler, LBracketSampler, PlateWithHoleSampler
from atlas.visualization import _inside_material

CMP = RESULTS_DIR / "comparison"
DATA = CMP / "data"                 # PINN / DeepXDE npz+json written by the baseline scripts
DATA_DIR = {"atlas": RESULTS_DIR / "atlas_energy" / "forward",
            "xpinn_energy": RESULTS_DIR / "atlas_xpinn_energy" / "forward"}
BENCHES = ["cantilever", "plate_hole", "l_bracket"]
TITLES = {"cantilever": "Cantilever beam", "plate_hole": "Plate with circular hole",
          "l_bracket": "L-shaped bracket"}
VARIANTS = [("pinn", "Standard PINN (in-house)"), ("deepxde", "PINN (DeepXDE, library defaults)"),
            ("deepxde_configured", "PINN (DeepXDE, configured)"),
            ("atlas", "ATLAS (energy, single net)"), ("xpinn_energy", "ATLAS-XPINN")]
COLORS = {"pinn": "#c0504d", "deepxde": "#e69f00", "deepxde_configured": "#8c6d1f", "atlas": "#2e8b57",
          "xpinn_energy": "#7b3fa0"}
OUT = CMP / "all_solvers"          # set in __main__ (--out-sub); data files stay in CMP
SAMPLERS = {"cantilever": CantileverBeamSampler, "plate_hole": PlateWithHoleSampler,
            "l_bracket": LBracketSampler}


def compliance(bench, coords, u, v):
    """Compliance C = int t.u ds over the loaded edge [N m per unit thickness], from the
    displacement at the reference nodes on that edge (P2 nodes, trapezoid rule). The
    tractions are those of the benchmarks: parabolic tip shear (cantilever, P=-1e6, H=1),
    uniform sigma_inf=1e6 (plate), parabolic t_x on the upper right face (L-bracket).
    """
    x, y = coords[:, 0], coords[:, 1]
    if bench == "cantilever":
        m = np.isclose(x, 4.0)
        yy, vv = y[m], v[m]
        o = np.argsort(yy)
        s = yy[o] / 1.0
        return float(np.trapz(6.0 * (-1e6 / 1.0) * s * (1.0 - s) * vv[o], yy[o]))
    if bench == "plate_hole":
        m = np.isclose(x, 2.0)
        o = np.argsort(y[m])
        return float(np.trapz(1e6 * u[m][o], y[m][o]))
    m = np.isclose(x, 2.0) & (y >= 1.0 - 1e-9)
    o = np.argsort(y[m])
    s = y[m][o] - 1.0
    return float(np.trapz(6.0 * 1e6 * s * (1.0 - s) * u[m][o], y[m][o]))


def vector_l2(z):
    """Relative L2 error of the displacement VECTOR field, sqrt(sum |d - d_fem|^2 / sum |d_fem|^2).
    Unlike the magnitude-based metric of atlas.validation (which compares sqrt(u^2+v^2)),
    this cannot hide a sign flip: a mirror-image solution scores ~200 %, not ~10 %.
    """
    du, dv = z["u"] - z["u_fem"], z["v"] - z["v_fem"]
    return float(np.sqrt((du ** 2 + dv ** 2).sum() / (z["u_fem"] ** 2 + z["v_fem"] ** 2).sum()))


def load(bench, var):
    d = DATA_DIR.get(var, DATA)
    j, z = d / f"{bench}_{var}.json", d / f"{bench}_{var}.npz"
    if not (j.exists() and z.exists()):
        return None
    return json.load(open(j)), np.load(z)


def table():
    lines = ["# " + " vs ".join(label for _, label in VARIANTS) + " vs FEM", "",
             "Errors are relative L2 against the FEniCS reference in `fem/reference/` "
             "(displacement magnitude, and von Mises with the singular region excluded, "
             "the same metrics `atlas.validation` reports). Targets: displacement < 5 %, "
             "von Mises < 10 %. No solver sees FEM data in training.", "",
             "Compliance C = int t.u ds over the loaded edge, computed from each solver's "
             "displacement at the reference nodes; its error is relative to the FEM value "
             "(spec target: within 10 %).", "",
             "Two displacement errors are shown. *Magnitude* is the metric `atlas.validation` "
             "reports (compares sqrt(u^2+v^2)); *vector* compares the (u, v) field itself and "
             "therefore catches a sign-flipped solution that the magnitude metric hides.", "",
             "| benchmark | solver | displacement L2 (magnitude) | displacement L2 (vector) "
             "| von Mises L2 (excl.) | compliance [N m] | compliance vs FEM | parameters | train time |",
             "|---|---|---|---|---|---|---|---|---|"]
    for b in BENCHES:
        for var, label in VARIANTS:
            r = load(b, var)
            if r is None:
                continue
            m = r[0]
            d, v = m["relative_L2_displacement"], m["von_mises_l2_excl"]
            ok = lambda x, t: "PASS" if x < t else "fail"
            t = m.get("train_seconds")
            z = r[1]
            c_pred = compliance(b, z["coords"], z["u"], z["v"])
            c_fem = compliance(b, z["coords"], z["u_fem"], z["v_fem"])
            c_err = c_pred / c_fem - 1.0
            vec = vector_l2(z)
            lines.append(f"| {TITLES[b]} | {label} | {d:.2%} ({ok(d, .05)}) | {vec:.2%} ({ok(vec, .05)}) | "
                         f"{v:.2%} ({ok(v, .10)}) | "
                         f"{c_pred:.4g} (FEM {c_fem:.4g}) | {c_err:+.2%} ({ok(abs(c_err), .10)}) | "
                         f"{m['params']:,} | {'-' if t is None else f'{t/60:.1f} min'} |")
    lines += ["", "Compliance difference and pointwise maxima", "",
              "Compliance difference is C - C_FEM. Maximum pointwise errors are the single worst "
              "node over the whole reference mesh, as a percentage of the FEM maximum of that "
              "field. The von Mises maximum usually sits at a stress concentration (hole edge, "
              "re-entrant corner, clamp corner), where the FEM reference itself is least "
              "reliable, so treat it as a worst-case indicator, not a pass/fail metric.", "",
              "| benchmark | solver | compliance difference [N m] | max pointwise displacement error "
              "| max pointwise von Mises error | at (x, y) |",
              "|---|---|---|---|---|---|"]
    for b in BENCHES:
        for var, label in VARIANTS:
            r = load(b, var)
            if r is None:
                continue
            z = r[1]
            c_diff = compliance(b, z["coords"], z["u"], z["v"]) - compliance(b, z["coords"], z["u_fem"], z["v_fem"])
            dd = np.hypot(z["u"] - z["u_fem"], z["v"] - z["v_fem"])
            dmax = dd.max() / np.hypot(z["u_fem"], z["v_fem"]).max()
            dv = np.abs(z["vm"] - z["vm_fem"])
            k = int(np.argmax(dv))
            lines.append(f"| {TITLES[b]} | {label} | {c_diff:+.4g} | {dmax:.2%} | "
                         f"{dv[k] / z['vm_fem'].max():.2%} ({dv[k]:.3g} Pa) | "
                         f"({z['coords'][k, 0]:.3f}, {z['coords'][k, 1]:.3f}) |")
    defs = {
        "pinn": "- **Standard PINN (in-house)**: one network over the whole domain, strong-form residual loss "
                "(same terms, scaling and load-amplitude constraints as ATLAS), no domain "
                "decomposition and no accuracy aids; parameter count matched to ATLAS-XPINN.",
        "deepxde": "- **PINN (DeepXDE library)**: an independent baseline built with the open-source "
                   "DeepXDE library (Lu et al., SIAM Review 2021): one tanh network, soft boundary "
                   "conditions, library-default loss weights, Adam then L-BFGS; nothing from the ATLAS "
                   "solver. Five cantilever settings (loss weights 1-100, float32/float64) were tried; "
                   "none got below 88 % displacement error, so the library defaults are reported.",
        "deepxde_configured": "- **PINN (DeepXDE, configured)**: the same library with settings fixed in "
                              "advance from standard DeepXDE practice and never chosen by the FEM error: the "
                              "essential BCs imposed exactly through an output transform, float64, extra "
                              "collocation points near each benchmark's stress concentration, and loss weights "
                              "balanced so every term starts at 1 (from the initial training losses only).",
        "atlas": "- **ATLAS (energy, single net)**: a deep-energy (Ritz) formulation with exact hard "
                 "boundary conditions and ONE network (no domain decomposition).",
        "xpinn_energy": "- **ATLAS-XPINN**: the same energy formulation and hard boundary "
                        "conditions, domain-decomposed: one subnetwork per subdomain with interface "
                        "displacement and traction continuity (augmented-Lagrangian coupling, "
                        "single-network warm start; docs/BUGFIXES.md section 16)."}
    lines += ["", "Solver definitions", ""] + [defs[v] for v, _ in VARIANTS] + [""]
    (OUT / "COMPARISON.md").write_text("\n".join(lines))
    print("\n".join(lines[:11 + 12]))


def field_figure(bench, only=None, suffix="comparison"):
    runs = [(var, label, load(bench, var)) for var, label in VARIANTS if only is None or var in only]
    runs = [(v, l, r) for v, l, r in runs if r is not None]
    if not runs:
        return
    z0 = runs[0][2][1]
    coords, uf, vf, mf = z0["coords"], z0["u_fem"], z0["v_fem"], z0["vm_fem"]
    x, y = coords[:, 0], coords[:, 1]
    tri = Triangulation(x, y)
    cen = tri.triangles
    tri.set_mask(~_inside_material(SAMPLERS[bench](), x[cen].mean(1), y[cen].mean(1)))
    dfem = np.hypot(uf, vf)
    d_max = dfem.max()
    m_max = np.percentile(mf, 99.5)
    wide = bench == "cantilever"
    ncol = 1 + len(runs)
    fig, axes = plt.subplots(3, ncol, figsize=(3.6 * ncol, 7.2 if wide else 10.5))
    lv_d, lv_m = np.linspace(0, d_max, 41), np.linspace(0, m_max, 41)
    lv_e = np.linspace(0, 0.2 * d_max, 41)

    def panel(ax, val, levels, cmap, title):
        c = ax.tricontourf(tri, np.clip(val, levels[0], levels[-1]), levels=levels, cmap=cmap, extend="max")
        ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=6)
        return c

    c0 = panel(axes[0, 0], dfem, lv_d, "viridis", "FEM (reference)\n|displacement| [m]")
    panel(axes[1, 0], mf, lv_m, "hot", "FEM (reference)\nvon Mises [Pa]")
    axes[2, 0].axis("off")
    axes[2, 0].text(0.02, 0.5, "Row 3: |displacement error|\nvs FEM, saturated at\n20 % of the FEM maximum.",
                    fontsize=9, va="center")
    for j, (var, label, (meta, z)) in enumerate(runs, start=1):
        dp = np.hypot(z["u"], z["v"])
        panel(axes[0, j], dp, lv_d, "viridis",
              f"{label}\n|disp| L2 err {meta['relative_L2_displacement']:.2%}")
        panel(axes[1, j], z["vm"], lv_m, "hot",
              f"{label}\nvon Mises L2 err (excl.) {meta['von_mises_l2_excl']:.2%}")
        panel(axes[2, j], np.abs(dp - dfem), lv_e, "magma", f"{label}\n|disp error|")
    names = " vs ".join(l for _, l, _ in runs)
    fig.suptitle(f"{TITLES[bench]}: FEM vs {names}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / f"{bench}_{suffix}.png", dpi=130)
    plt.close(fig)


def summary_figure():
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    for ax, key, tgt, ttl in [(axes[0], "vector_error", 0.05, "Displacement relative L2 error (vector field)"),
                              (axes[1], "von_mises_l2_excl", 0.10, "von Mises relative L2 error (excl.)"),
                              (axes[2], "compliance_error", 0.10, "Compliance error vs FEM (|C/C_FEM - 1|)")]:
        n = len(VARIANTS)
        w = 0.8 / n
        for k, (var, label) in enumerate(VARIANTS):
            vals = []
            for b in BENCHES:
                r = load(b, var)
                if r is None:
                    vals.append(np.nan)
                elif key == "vector_error":
                    vals.append(vector_l2(r[1]))
                elif key == "compliance_error":
                    z = r[1]
                    vals.append(abs(compliance(b, z["coords"], z["u"], z["v"])
                                    / compliance(b, z["coords"], z["u_fem"], z["v_fem"]) - 1.0))
                else:
                    vals.append(r[0][key])
            ax.bar(np.arange(len(BENCHES)) + (k - (n - 1) / 2) * w, vals, w, label=label, color=COLORS[var])
        ax.axhline(tgt, color="k", ls="--", lw=1, label="target (5 % displacement, 10 % von Mises / compliance)")
        ax.set_yscale("log"); ax.set_xticks(range(len(BENCHES)))
        ax.set_xticklabels([TITLES[b] for b in BENCHES], fontsize=8)
        ax.set_title(ttl); ax.grid(True, axis="y", alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=len(l))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "comparison_summary.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=[v for v, _ in VARIANTS], default=None,
                    help="solvers to compare (default: all four)")
    ap.add_argument("--out-sub", default=None,
                    help="output folder under results/comparison/ (default: all_solvers, or the "
                         "--only solvers joined by _vs_)")
    ap.add_argument("--suffix", default="comparison", help="field-figure file suffix")
    args = ap.parse_args()
    if args.only is not None:
        VARIANTS = [(v, l) for v, l in VARIANTS if v in args.only]
    OUT = CMP / (args.out_sub or ("all_solvers" if args.only is None else "_vs_".join(args.only)))
    OUT.mkdir(parents=True, exist_ok=True)
    table()
    for b in BENCHES:
        field_figure(b, None, args.suffix)
    summary_figure()
    print(f"\nwritten to {OUT}")
