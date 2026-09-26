"""Table and figures for the two-material benchmark (results/multi_material/bimaterial/).

For each contrast found under results/multi_material/bimaterial/c<contrast>/ (or, with --bench steel_insert /
filled_slot, for results/multi_material/steel_insert/ or results/multi_material/filled_slot/) writes
  * table.md          seed-42 metrics of every solver (and 5-seed means where they exist)
  * convergence.png   displacement / von Mises error against training wall-clock
  * fields.png        FEM vs each solver: von Mises and displacement error
  * profiles.png      von Mises along y = 0 and x = 0, across the interface

    python scripts/multi_material/make_report.py
    python scripts/multi_material/make_report.py --bench steel_insert
    python scripts/multi_material/make_report.py --bench filled_slot
"""
import argparse
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

from atlas.benchmarks import bimaterial as bm
from atlas.paths import RESULTS_DIR

D = RESULTS_DIR / "multi_material" / "bimaterial"
SEMI_AXES = (bm.R_INC, bm.R_INC)       # interface (x/a)^2 + (y/b)^2 = 1; the filled slot is an ellipse
SOLVERS = [("atlas", "ATLAS energy (single 4x64)", "#2e75b6"),
           ("atlas_wide", "ATLAS energy (single 4x91, parameter-matched)", "#6fa8dc"),
           ("xpinn", "ATLAS-XPINN (inclusion + matrix)", "#d9822b")]


def load(cdir, s, seed=42):
    j, z = cdir / s / f"seed{seed}.json", cdir / s / f"seed{seed}.npz"
    return (json.load(open(j)), np.load(z)) if j.exists() else (None, None)


def seeds(cdir, s):
    rows = [json.load(open(p))["final"] for p in sorted((cdir / s).glob("seed*.json"))] if (cdir / s).exists() else []
    return rows


def table(cdir, data):
    pct = lambda x: f"{100 * x:.2f}%"
    lines = ["| solver | params | train time | displacement L2 | von Mises L2 | band vM, inclusion side | "
             "band vM, matrix side | compliance vs FEM |", "|---|---|---|---|---|---|---|---|"]
    for s, name, _ in SOLVERS:
        d, _ = data[s]
        if d is None:
            continue
        f = d["final"]
        lines.append(f"| {name} | {d['params']:,} | {d['train_seconds']:.0f} s | {pct(f['disp_vec_l2'])} | "
                     f"{pct(f['vm_l2'])} | {pct(f['band_vm_l2_inclusion'])} | {pct(f['band_vm_l2_matrix'])} | "
                     f"{100 * f['compliance_err']:+.3f}% |")
    multi = [(name, seeds(cdir, s)) for s, name, _ in SOLVERS]
    multi = [(n, r) for n, r in multi if len(r) >= 2]
    if multi:
        lines += ["", "Seeds (mean +- std over the seeds found):", "",
                  "| solver | seeds | displacement L2 | von Mises L2 |", "|---|---|---|---|"]
        for n, r in multi:
            a = np.array([x["disp_vec_l2"] for x in r]) * 100
            b = np.array([x["vm_l2"] for x in r]) * 100
            lines.append(f"| {n} | {len(r)} | {a.mean():.2f} +- {a.std():.2f}% | {b.mean():.2f} +- {b.std():.2f}% |")
    (cdir / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def convergence(cdir, data):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, ttl in zip(axs, ["disp_vec_l2", "vm_l2"], ["displacement rel-L2 vs FEM", "von Mises rel-L2 vs FEM"]):
        for s, name, col in SOLVERS:
            d, _ = data[s]
            if d is not None:
                ax.semilogy([r["seconds"] for r in d["probes"]], [max(r[key], 1e-5) for r in d["probes"]],
                            "-", lw=1.5, color=col, label=name)
        ax.set(xlabel="training wall-clock [s]", title=ttl)
        ax.grid(True, which="both", alpha=0.3)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(cdir / "convergence.png", dpi=140)
    plt.close(fig)


def fields(cdir, data):
    have = [(s, n) for s, n, _ in SOLVERS if data[s][1] is not None]
    ref = data[have[0][0]][1]
    c = ref["coords"]
    tri = Triangulation(c[:, 0], c[:, 1])
    cap = np.percentile(ref["vm_fem"], 99.8)
    umax = np.hypot(ref["u_fem"], ref["v_fem"]).max()
    fig, axes = plt.subplots(2, len(have) + 1, figsize=(3.3 * (len(have) + 1), 6.6))
    axes[0, 0].tricontourf(tri, np.clip(ref["vm_fem"], 0, cap), levels=np.linspace(0, cap, 30), cmap="inferno")
    axes[0, 0].set_title("FEM von Mises", fontsize=8)
    axes[1, 0].axis("off")
    for j, (s, name) in enumerate(have, start=1):
        z = data[s][1]
        axes[0, j].tricontourf(tri, np.clip(z["vm"], 0, cap), levels=np.linspace(0, cap, 30), cmap="inferno")
        err = np.abs(z["vm"] - z["vm_fem"]) / cap
        axes[1, j].tricontourf(tri, np.clip(err, 0, 0.1), levels=np.linspace(0, 0.1, 30), cmap="magma")
        axes[0, j].set_title(name, fontsize=8)
        axes[1, j].set_title("|von Mises error| / FEM 99.8th pct (sat. 10%)", fontsize=7)
    th = np.linspace(0, np.pi / 2, 100)
    for ax in axes.ravel():
        if ax.axison:
            ax.plot(SEMI_AXES[0] * np.cos(th), SEMI_AXES[1] * np.sin(th), "w--", lw=0.6)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(cdir / "fields.png", dpi=130)
    plt.close(fig)


def profiles(cdir, data):
    have = [(s, n, col) for s, n, col in SOLVERS if data[s][1] is not None]
    ref = data[have[0][0]][1]
    c = ref["coords"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (axis, other, lbl) in zip(axs, [(0, 1, "along y = 0 (x varies)"), (1, 0, "along x = 0 (y varies)")]):
        m = (np.abs(c[:, other]) < 1e-9) & (c[:, axis] < 0.8)
        o = np.argsort(c[m, axis])
        ax.plot(c[m, axis][o], ref["vm_fem"][m][o] / 1e6, "k-", lw=2, label="FEM")
        for s, n, col in have:
            ax.plot(c[m, axis][o], data[s][1]["vm"][m][o] / 1e6, "-", lw=1.3, color=col, label=n)
        ax.axvline(SEMI_AXES[axis], color="gray", ls=":", lw=1)
        ax.set(xlabel="distance from the inclusion centre [m]", ylabel="von Mises [MPa]", title=lbl)
        ax.grid(True, alpha=0.3)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(cdir / "profiles.png", dpi=140)
    plt.close(fig)


def main():
    global D, SOLVERS, SEMI_AXES
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="bimaterial", choices=["bimaterial", "steel_insert", "filled_slot"])
    args = ap.parse_args()
    if args.bench != "bimaterial":
        D = RESULTS_DIR / "multi_material" / args.bench
        part = {"steel_insert": "insert + plate", "filled_slot": "filler + plate"}[args.bench]
        SOLVERS = [(s, n.replace("inclusion + matrix", part), c) for s, n, c in SOLVERS]
        if args.bench == "filled_slot":
            from atlas.benchmarks import filled_slot as fs
            SEMI_AXES = (fs.AX, fs.BY)
        dirs = [D]
    else:
        dirs = sorted(p for p in D.glob("c*") if p.is_dir())
    for cdir in dirs:
        data = {s: load(cdir, s) for s, _, _ in SOLVERS}
        if all(d is None for d, _ in data.values()):
            continue
        print(f"\n## {cdir.name}")
        table(cdir, data)
        convergence(cdir, data)
        fields(cdir, data)
        profiles(cdir, data)
    print(f"written to {D}")


if __name__ == "__main__":
    main()
