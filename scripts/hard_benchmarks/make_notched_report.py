"""Table and figures for the notched-plate benchmark (results/hard_single_material/notched_plate/)."""
import json
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np

from atlas.paths import RESULTS_DIR

D = RESULTS_DIR / "hard_single_material" / "notched_plate"
RUNS = [("deepxde", "PINN (DeepXDE library)", D / "deepxde" / "seed42", "#c0504d"),
        ("atlas", "ATLAS energy (single network)", D / "atlas" / "seed42", "#2e75b6"),
        ("xpinn", "ATLAS-XPINN, patch layout (2000+2000)", D / "xpinn" / "seed42", "#d9822b"),
        ("xpinn_long", "ATLAS-XPINN, patch layout (2000+8000)", D / "long_budget" / "xpinn" / "seed42", "#7b3fa0"),
        ("split", "ATLAS-XPINN, split layout, penalty", D / "fix_study" / "split_penalty" / "seed42", "#8c8c8c"),
        ("split_al", "ATLAS-XPINN, split layout + AL", D / "fix_study" / "split_al" / "seed42", "#2e8b57"),
        ("split_al_warm", "ATLAS-XPINN, split + AL + warm start", D / "xpinn_split" / "seed42", "#b8860b")]


def load(stem):
    j = stem.with_suffix(".json")
    return (json.load(open(j)), np.load(stem.with_suffix(".npz"))) if j.exists() else (None, None)


def first_below(rows, key, thr):
    for i in range(len(rows)):
        if all(r[key] is not None and r[key] < thr for r in rows[i:]):
            return rows[i]["seconds"]
    return None


def main():
    data = {k: load(p) for k, _, p, _ in RUNS}
    lines = ["# Notched plate (h/r = 10): ATLAS energy vs ATLAS-XPINN vs DeepXDE PINN", "",
             "Plate 1 x 2 m, U-notch depth 0.25 m / root radius 0.025 m, clamped at y = 0, uniform tension at "
             "y = 2. Seed 42, one GPU job at a time. Errors vs the FEniCS reference (never used in training). "
             "Displacement error is the vector L2; von Mises excludes the notch apex and clamp corners; "
             "'notch zone' is the von Mises L2 on a dense grid around the root (excluding 0.005 m around the apex); "
             "'root peak' is the predicted / FEM peak von Mises there (1 = exact). The FEM peak itself is converged "
             "to about 4 % (change between the last two mesh levels).", "",
             "| solver | params | train time [s] | displacement L2 | von Mises L2 (excl.) | notch-zone von Mises L2 "
             "| root peak vs FEM | time to 5 % disp [s] |", "|---|---|---|---|---|---|---|---|"]
    for k, name, _, _ in RUNS:
        d, _ = data[k]
        if d is None:
            continue
        f = d["final"]
        rows = d["probes"]
        t5 = first_below(rows, "disp_vec_l2", 0.05)
        lines.append(f"| {name} | {d['params']} | {d['train_seconds']:.0f} | {f['disp_vec_l2']:.2%} | "
                     f"{f['vm_l2_excl']:.2%} | {f['notch_vm_l2']:.2%} | {f['peak_vm_ratio']:.3f} | "
                     f"{'-' if t5 is None else format(t5, '.0f')} |")
    (D / "NOTCHED_PLATE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, ttl, thr in zip(axs, ["disp_vec_l2", "vm_l2_excl", "notch_vm_l2"],
                                 ["displacement rel-L2 vs FEM", "von Mises rel-L2 (excl.) vs FEM",
                                  "notch-zone von Mises rel-L2 vs FEM"], [0.05, 0.10, 0.10]):
        for k, name, _, col in RUNS:
            d, _ = data[k]
            if d is None:
                continue
            ax.semilogy([r["seconds"] for r in d["probes"]], [max(r[key], 1e-4) for r in d["probes"]],
                        "-", lw=1.5, color=col, label=name)
        ax.axhline(thr, color="k", ls="--", lw=0.9, label=f"target {thr:.0%}")
        ax.set(xlabel="training wall-clock [s]", title=ttl, xscale="log", ylim=(5e-4, 4))
        ax.grid(True, which="both", alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(D / "convergence.png", dpi=140)
    plt.close(fig)

    # field comparison on the uniform grid
    ref = None
    for k, *_ in RUNS:
        if data[k][1] is not None:
            ref = data[k][1]
            break
    c = ref["coords"]
    ix, iy = np.round(c[:, 0] / 0.01).astype(int), np.round(c[:, 1] / 0.01).astype(int)
    shape = (ix.max() + 1, iy.max() + 1)

    def img(v):
        a = np.full(shape, np.nan)
        a[ix, iy] = v
        return a.T

    cols = [("FEM (reference)", None)] + [(name, k) for k, name, _, _ in RUNS if data[k][1] is not None]
    fig, axes = plt.subplots(3, len(cols), figsize=(3.0 * len(cols), 9.5))
    mag_f = np.hypot(ref["u_fem"], ref["v_fem"])
    vm_cap = np.percentile(ref["vm_fem"], 99.5)
    for j, (name, k) in enumerate(cols):
        z = ref if k is None else data[k][1]
        mag = np.hypot(z["u_fem"], z["v_fem"]) if k is None else np.hypot(z["u"], z["v"])
        vm = z["vm_fem"] if k is None else z["vm"]
        axes[0, j].imshow(img(mag), origin="lower", extent=(0, 1, 0, 2), vmin=0, vmax=mag_f.max(), cmap="viridis")
        axes[1, j].imshow(img(vm), origin="lower", extent=(0, 1, 0, 2), vmin=0, vmax=vm_cap, cmap="inferno")
        if k is None:
            axes[2, j].axis("off")
        else:
            err = np.hypot(z["u"] - z["u_fem"], z["v"] - z["v_fem"]) / mag_f.max()
            axes[2, j].imshow(img(np.minimum(err, 0.2)), origin="lower", extent=(0, 1, 0, 2), vmin=0, vmax=0.2,
                              cmap="magma")
        axes[0, j].set_title(name, fontsize=8)
        for i in range(3):
            axes[i, j].tick_params(labelsize=6)
    axes[0, 0].set_ylabel("|displacement|")
    axes[1, 0].set_ylabel("von Mises (capped at FEM 99.5th pct)")
    fig.text(0.01, 0.30, "row 3: |disp error| / max FEM |disp|,\nsaturated at 20 %", fontsize=8)
    fig.tight_layout()
    fig.savefig(D / "fields.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
