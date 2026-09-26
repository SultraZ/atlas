"""Figures for the multi-cutout bracket (results/hard_single_material/multicutout/): convergence.png and fields.png.

Reads the seed-42 runs listed in RUNS and draws
  * error against training wall-clock (displacement, von Mises, cutout zones),
  * FEM vs each solver: |displacement|, von Mises, displacement error map.
"""
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

from atlas.benchmarks import multicutout as mc
from atlas.paths import RESULTS_DIR

D = RESULTS_DIR / "hard_single_material" / "multicutout"
RUNS = [("deepxde", "PINN (DeepXDE, defaults)", "#c0504d"),
        ("atlas_long", "ATLAS energy (single network)", "#2e75b6"),
        ("xpinn_long", "ATLAS-XPINN (default recipe)", "#d9822b"),
        ("xpinn_long_keepmem_coarse", "ATLAS-XPINN + coarse-to-fine start", "#2e8b57"),
        ("xpinn_long_coarse_localnorm", "ATLAS-XPINN + local normalisation", "#7b3fa0")]


def load(tag):
    j, z = D / tag / "seed42.json", D / tag / "seed42.npz"
    return (json.load(open(j)), np.load(z)) if j.exists() and z.exists() else (None, None)


def main():
    data = {t: load(t) for t, _, _ in RUNS}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, ttl, thr in zip(axs, ["disp_vec_l2", "vm_l2_excl", "cutout_vm_l2_mean"],
                                 ["displacement rel-L2 vs FEM", "von Mises rel-L2 (excl.) vs FEM",
                                  "von Mises rel-L2 near the cutouts"], [0.05, 0.10, None]):
        for t, name, col in RUNS:
            d, _ = data[t]
            if d is None:
                continue
            pts = [(r["seconds"], max(r[key], 1e-4)) for r in d["probes"] if r.get(key) is not None]
            if not pts:          # e.g. the DeepXDE runs only record the cutout metric at the end
                continue
            ax.semilogy(*zip(*pts), "-", lw=1.5, color=col, label=name)
        if thr:
            ax.axhline(thr, color="k", ls="--", lw=0.9, label=f"target {thr:.0%}")
        ax.set(xlabel="training wall-clock [s]", title=ttl, ylim=(2e-3, 3))
        ax.grid(True, which="both", alpha=0.3)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(D / "convergence.png", dpi=140)
    plt.close(fig)

    ref = next(z for t, *_ in RUNS if (z := data[t][1]) is not None)
    c = ref["coords"]
    tri = Triangulation(c[:, 0], c[:, 1])
    cx, cy = c[tri.triangles].mean(axis=1).T                 # blank the cutouts and the outside
    tri.set_mask(~mc.material(cx, cy))
    cols = [("FEM (reference)", None)] + [(n, t) for t, n, _ in RUNS if data[t][1] is not None]
    fig, axes = plt.subplots(3, len(cols), figsize=(3.1 * len(cols), 9.6))
    mag_f = np.hypot(ref["u_fem"], ref["v_fem"])
    vm_cap = np.percentile(ref["vm_fem"], 99.5)
    for j, (name, t) in enumerate(cols):
        z = ref if t is None else data[t][1]
        mag = mag_f if t is None else np.hypot(z["u"], z["v"])
        vm = ref["vm_fem"] if t is None else z["vm"]
        axes[0, j].tricontourf(tri, np.clip(mag, 0, mag_f.max()), levels=np.linspace(0, mag_f.max(), 25),
                               cmap="viridis")
        axes[1, j].tricontourf(tri, np.clip(vm, 0, vm_cap), levels=np.linspace(0, vm_cap, 25), cmap="inferno")
        if t is None:
            axes[2, j].axis("off")
        else:
            err = np.hypot(z["u"] - z["u_fem"], z["v"] - z["v_fem"]) / mag_f.max()
            axes[2, j].tricontourf(tri, np.clip(err, 0, 0.2), levels=np.linspace(0, 0.2, 25), cmap="magma")
        axes[0, j].set_title(name, fontsize=8)
        for i in range(3):
            axes[i, j].set_aspect("equal")
            axes[i, j].tick_params(labelsize=6)
    axes[0, 0].set_ylabel("|displacement|")
    axes[1, 0].set_ylabel("von Mises (capped at the FEM 99.5th pct)")
    fig.text(0.01, 0.28, "row 3: |displacement error| / max FEM |displacement|,\nsaturated at 20 %", fontsize=8)
    fig.tight_layout()
    fig.savefig(D / "fields.png", dpi=130)
    plt.close(fig)
    print(f"written to {D}")


if __name__ == "__main__":
    main()
