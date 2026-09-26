"""Run from the project root (all options: --help):
    python scripts/multi_material/run_transition_joint.py --solver {atlas,xpinn,atlas_wide} [--seed N]

Example:
    python scripts/multi_material/run_transition_joint.py --solver xpinn --seed 42
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

from atlas.benchmarks import transition_joint as tj
from atlas.validation import default_exclusion_radius
from atlas.paths import FEM_REFERENCE_DIR, RESULTS_DIR

WIDE_NEURONS = 91


def load_ref(tag):
    R = FEM_REFERENCE_DIR
    return (np.load(R / f"{tag}_coords.npy"), np.load(R / f"{tag}_u.npy").ravel(),
            np.load(R / f"{tag}_v.npy").ravel(), np.load(R / f"{tag}_vonmises.npy").ravel())


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
    ref = "transition_joint"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.solver == "xpinn":
        model, prob = tj.build_decomposed(device=device)
    else:
        model, prob = tj.build_single(device=device, neurons=WIDE_NEURONS if args.solver == "atlas_wide" else 64)
    n_params = sum(p.numel() for p in model.parameters())

    coords, u_f, v_f, vm_f = load_ref(ref)
    roi_c, _, _, vmr_f = load_ref(ref + "_roi")
    meta = json.load(open(FEM_REFERENCE_DIR / f"{ref}_meta.json"))
    st_r = tj.in_steel(roi_c[:, 0], roi_c[:, 1])
    r_excl = default_exclusion_radius(coords, tj.SINGULAR)
    sx, sy = tj.SINGULAR[0]
    keep = np.hypot(coords[:, 0] - sx, coords[:, 1] - sy) > r_excl
    keep_r = np.hypot(roi_c[:, 0] - sx, roi_c[:, 1] - sy) > r_excl

    def score():
        up, vp, vm = tj.predict(model, coords, device)
        _, _, vmr = tj.predict(model, roi_c, device)
        vec = float(np.sqrt(np.sum((up - u_f) ** 2 + (vp - v_f) ** 2) / np.sum(u_f ** 2 + v_f ** 2)))
        C = tj.compliance(model, device)
        return dict(disp_vec_l2=vec, vm_l2=rel(vm[keep], vm_f[keep]), band_vm_l2=rel(vmr[keep_r], vmr_f[keep_r]),
                    band_vm_l2_steel=rel(vmr[keep_r & st_r], vmr_f[keep_r & st_r]),
                    band_vm_l2_alu=rel(vmr[keep_r & ~st_r], vmr_f[keep_r & ~st_r]),
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
        print(f"  it {it:5d}  t={el:7.1f}s  disp={m['disp_vec_l2']:.3%}  vm={m['vm_l2']:.3%}  "
              f"band vm st/al={m['band_vm_l2_steel']:.3%}/{m['band_vm_l2_alu']:.3%}  "
              f"C err={m['compliance_err']:+.3%}", flush=True)

    print(f"{ref} / {args.solver} seed {args.seed}: {n_params} params, {prob.n_points} quadrature points, "
          f"device {device}", flush=True)
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
    out = RESULTS_DIR / "multi_material" / "failed_candidates" / "transition_joint" / (args.tag or args.solver)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}"
    np.savez_compressed(out / f"{tag}.npz", coords=coords, u=up, v=vp, vm=vm, u_fem=u_f, v_fem=v_f, vm_fem=vm_f)
    torch.save(model.state_dict(), out / f"{tag}_model.pt")
    json.dump(dict(benchmark="transition_joint", solver=args.solver, seed=args.seed, r_excl=float(r_excl),
                   params=int(n_params), train_seconds=total, probe_seconds=st["probe_s"],
                   epochs=args.epochs, lbfgs=args.lbfgs, final=final, final_energy=float(hist['total'][-1]),
                   interface_jump_sq=jump, probes=rows),
              open(out / f"{tag}.json", "w"))
    print(f"DONE {ref}/{args.solver} seed {args.seed}: train {total:.1f}s  disp {final['disp_vec_l2']:.3%}  "
          f"vM {final['vm_l2']:.3%}  band vM st/al {final['band_vm_l2_steel']:.3%}/"
          f"{final['band_vm_l2_alu']:.3%}  C err {final['compliance_err']:+.3%}", flush=True)


if __name__ == "__main__":
    main()
