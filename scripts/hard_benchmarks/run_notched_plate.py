"""Run from the project root (all options: --help):
    python scripts/hard_benchmarks/run_notched_plate.py --solver {atlas,xpinn,atlas_wide} [--seed N]

Example:
    python scripts/hard_benchmarks/run_notched_plate.py --solver xpinn --seed 42
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

from atlas.benchmarks.notched_plate import (E_MOD, NU, STRESS_SCALE, NotchGeometry, build_decomposed,
                                 build_single, predict)
from atlas.paths import FEM_REFERENCE_DIR, RESULTS_DIR
from atlas.physics import PlaneStressElasticity
from atlas.validation import compute_von_mises_metrics, default_exclusion_radius, relative_l2

WIDE_NEURONS = 91


def load_ref(tag):
    R = FEM_REFERENCE_DIR
    return (np.load(R / f"{tag}_coords.npy"), np.load(R / f"{tag}_u.npy").ravel(),
            np.load(R / f"{tag}_v.npy").ravel(), np.load(R / f"{tag}_vonmises.npy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True, choices=["atlas", "xpinn", "atlas_wide"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lbfgs", type=int, default=2000)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="hard_single_material/notched_plate")
    ap.add_argument("--layout", default="patch", choices=["patch", "split"],
                    help="xpinn only: root patch + rest, or top/bottom halves cut along the notch line")
    ap.add_argument("--al-stages", type=int, default=3,
                    help="xpinn only: augmented-Lagrangian multiplier updates (0 = plain penalty)")
    ap.add_argument("--tag", default=None, help="result subfolder (default: the solver name)")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    g = NotchGeometry()
    phys = PlaneStressElasticity(E=E_MOD / STRESS_SCALE, nu=NU)
    if args.solver == "xpinn":
        model, prob = build_decomposed(phys, g, device=device, layout=args.layout)
    else:
        model, prob = build_single(phys, g, neurons=WIDE_NEURONS if args.solver == "atlas_wide" else 64,
                                   device=device)
    n_params = sum(p.numel() for p in model.parameters())

    coords, u_f, v_f, vm_f = load_ref("notched_plate")
    roi_c, ur_f, vr_f, vmr_f = load_ref("notched_plate_roi")
    centers = [g.apex, (0.0, 0.0), (g.W, 0.0)]
    r_excl = default_exclusion_radius(coords, centers)
    far = np.hypot(roi_c[:, 0] - g.apex[0], roi_c[:, 1] - g.apex[1]) > 0.005
    mag_f = np.hypot(u_f, v_f)

    def score():
        up, vp, vm = predict(model, coords, device)
        vec = float(np.sqrt(np.sum((up - u_f) ** 2 + (vp - v_f) ** 2) / np.sum(u_f ** 2 + v_f ** 2)))
        mag = float(relative_l2(np.hypot(up, vp), mag_f))
        vml = compute_von_mises_metrics(vm.reshape(-1, 1), vm_f, coords, centers, r_excl)['von_mises_l2_excl']
        ur, vr, vmr = predict(model, roi_c, device)
        roi_vm = float(np.sqrt(np.sum((vmr[far] - vmr_f.ravel()[far]) ** 2) / np.sum(vmr_f.ravel()[far] ** 2)))
        roi_disp = float(np.sqrt(np.sum((ur - ur_f) ** 2 + (vr - vr_f) ** 2) / np.sum(ur_f ** 2 + vr_f ** 2)))
        return dict(disp_vec_l2=vec, disp_mag_l2=mag, vm_l2_excl=None if vml is None else float(vml),
                    notch_vm_l2=roi_vm, notch_disp_l2=roi_disp,
                    peak_vm_ratio=float(vmr[far].max() / vmr_f.ravel()[far].max())), (up, vp, vm)

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
        print(f"  it {it:5d}  t={el:7.1f}s  disp(vec)={m['disp_vec_l2']:.3%}  vm={m['vm_l2_excl']:.3%}  "
              f"notch vm={m['notch_vm_l2']:.3%}  peak ratio={m['peak_vm_ratio']:.3f}", flush=True)

    print(f"notched_plate / {args.solver} seed {args.seed}: {n_params} params, "
          f"{prob.n_points} quadrature points, device {device}", flush=True)
    sync()
    t0 = time.perf_counter()
    if args.solver == "xpinn":
        from atlas.energy_xpinn import train_decomposed
        hist = train_decomposed(model, prob, epochs=args.epochs, lbfgs_per_stage=args.lbfgs,
                                gammas=(30.0,), verbose=False, probe=probe,
                                al_stages=args.al_stages)
        jump = prob.last.get("jump_u")
    else:
        from atlas.energy_solver import train_energy
        hist = train_energy(model, prob, epochs=args.epochs, lbfgs_iters=args.lbfgs, lbfgs_chunk=100,
                            verbose=False, probe=probe, probe_every=args.probe_every)
        jump = None
    sync()
    total = time.perf_counter() - t0 - st["probe_s"]
    final, (up, vp, vm) = score()
    out = RESULTS_DIR / args.out / (args.tag or args.solver)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}"
    np.savez_compressed(out / f"{tag}.npz", coords=coords, u=up, v=vp, vm=vm, u_fem=u_f, v_fem=v_f,
                        vm_fem=vm_f.ravel())
    torch.save(model.state_dict(), out / f"{tag}_model.pt")
    json.dump(dict(benchmark="notched_plate", solver=args.solver, layout=args.layout,
                   al_stages=args.al_stages, seed=args.seed, params=int(n_params),
                   train_seconds=total, probe_seconds=st["probe_s"], epochs=args.epochs, lbfgs=args.lbfgs,
                   final=final, final_energy=float(hist['total'][-1]), interface_jump_sq=jump, probes=rows),
              open(out / f"{tag}.json", "w"))
    print(f"DONE notched_plate/{args.solver} seed {args.seed}: train {total:.1f}s  "
          f"disp(vec) {final['disp_vec_l2']:.3%}  vM(excl) {final['vm_l2_excl']:.3%}  "
          f"notch vM {final['notch_vm_l2']:.3%}  peak ratio {final['peak_vm_ratio']:.3f}", flush=True)


if __name__ == "__main__":
    main()
