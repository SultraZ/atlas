"""Run from the project root (all options: --help):
    python scripts/multi_material/run_sandwich.py --solver {atlas,xpinn,atlas_wide} [--seed N]

Example:
    python scripts/multi_material/run_sandwich.py --solver xpinn --seed 42
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from atlas.benchmarks import sandwich as sw
from atlas.paths import FEM_REFERENCE_DIR, RESULTS_DIR
from atlas.validation import default_exclusion_radius

WIDE_NEURONS = 112


def rel(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True, choices=["atlas", "xpinn", "atlas_wide"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lbfgs", type=int, default=2000)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default=None, help="result subfolder (default: the solver name)")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.solver == "xpinn":
        model, prob = sw.build_decomposed(device=device)
    else:
        model, prob = sw.build_single(device=device, neurons=WIDE_NEURONS if args.solver == "atlas_wide" else 64)
    n_params = sum(p.numel() for p in model.parameters())

    R = FEM_REFERENCE_DIR
    coords = np.load(R / "sandwich_coords.npy")
    u_f, v_f = np.load(R / "sandwich_u.npy").ravel(), np.load(R / "sandwich_v.npy").ravel()
    vm_f = np.load(R / "sandwich_vonmises.npy").ravel()
    meta = json.load(open(R / "sandwich_meta.json"))
    r_excl = default_exclusion_radius(coords, sw.SINGULAR)
    keep = np.min([np.hypot(coords[:, 0] - a, coords[:, 1] - b) for a, b in sw.SINGULAR], axis=0) > r_excl
    core = sw.layer_of(coords[:, 1]) == 1
    tip = np.array([[sw.L, 0.5 * sw.H]])

    def score():
        up, vp, vm = sw.predict(model, coords, device)
        _, v_tip, _ = sw.predict(model, tip, device)
        vec = float(np.sqrt(np.sum((up - u_f) ** 2 + (vp - v_f) ** 2) / np.sum(u_f ** 2 + v_f ** 2)))
        C = sw.compliance(model, device)
        return dict(disp_vec_l2=vec, vm_l2_excl=rel(vm[keep], vm_f[keep]),
                    face_vm_l2=rel(vm[keep & ~core], vm_f[keep & ~core]),
                    core_vm_l2=rel(vm[keep & core], vm_f[keep & core]),
                    tip_err=float(v_tip[0] / meta["tip_deflection"] - 1.0),
                    compliance_err=(C - meta["compliance"]) / meta["compliance"]), (up, vp, vm)

    rows, st = [], {"probe_s": 0.0}
    sync = lambda: torch.cuda.synchronize() if device.startswith("cuda") else None

    def probe(it):
        sync()
        t = time.perf_counter()
        el = t - t0 - st["probe_s"]
        m, _ = score()
        rows.append(dict(iter=int(it), seconds=el, **m))
        sync()
        st["probe_s"] += time.perf_counter() - t
        print(f"  it {it:5d}  t={el:7.1f}s  disp={m['disp_vec_l2']:.3%}  vm={m['vm_l2_excl']:.3%}  "
              f"face/core vm={m['face_vm_l2']:.3%}/{m['core_vm_l2']:.3%}  tip={m['tip_err']:+.3%}", flush=True)

    print(f"sandwich / {args.solver} seed {args.seed}: {n_params} params, {prob.n_points} quadrature points, "
          f"device {device}, von Mises exclusion radius {r_excl:.4f}", flush=True)
    sync()
    t0 = time.perf_counter()
    if args.solver == "xpinn":
        from atlas.energy_xpinn import train_decomposed
        hist = train_decomposed(model, prob, epochs=args.epochs, lbfgs_per_stage=args.lbfgs,
                                gammas=(30.0,), verbose=False, probe=probe)
        jump = prob.last.get("jump_u")
    else:
        from atlas.energy_solver import train_energy
        hist = train_energy(model, prob, epochs=args.epochs, lbfgs_iters=args.lbfgs, lbfgs_chunk=100,
                            verbose=False, probe=probe, probe_every=args.probe_every)
        jump = None
    sync()
    total = time.perf_counter() - t0 - st["probe_s"]
    final, (up, vp, vm) = score()
    out = RESULTS_DIR / "multi_material" / "failed_candidates" / "sandwich" / (args.tag or args.solver)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}"
    np.savez_compressed(out / f"{tag}.npz", coords=coords, u=up, v=vp, vm=vm, u_fem=u_f, v_fem=v_f, vm_fem=vm_f,
                        keep=keep)
    torch.save(model.state_dict(), out / f"{tag}_model.pt")
    json.dump(dict(benchmark="sandwich", solver=args.solver, seed=args.seed, params=int(n_params),
                   train_seconds=total, probe_seconds=st["probe_s"], epochs=args.epochs, lbfgs=args.lbfgs,
                   r_excl=float(r_excl), final=final, final_energy=float(hist['total'][-1]),
                   interface_jump_sq=jump, probes=rows),
              open(out / f"{tag}.json", "w"))
    print(f"DONE sandwich/{args.solver} seed {args.seed}: train {total:.1f}s  disp {final['disp_vec_l2']:.3%}  "
          f"vM {final['vm_l2_excl']:.3%}  face/core vM {final['face_vm_l2']:.3%}/{final['core_vm_l2']:.3%}  "
          f"tip {final['tip_err']:+.3%}  C err {final['compliance_err']:+.3%}", flush=True)


if __name__ == "__main__":
    main()
