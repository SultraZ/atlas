"""Convergence and processing-time study: ATLAS energy (single network) vs
ATLAS-XPINN (domain-decomposed), same iteration budget, same seed.

Writes results/comparison/convergence_timing/<benchmark>_<solver>.json.

    python scripts/run_convergence_timing.py --benchmark plate_hole --solver atlas
"""
import argparse
import copy
import json
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from atlas.paths import FEM_REFERENCE_DIR, RESULTS_DIR
from atlas.physics import PlaneStressElasticity
from atlas.validation import (compute_von_mises_metrics, default_exclusion_radius,
                              relative_l2, sampler_geometry_dict, singular_locations)
from atlas.visualization import _evaluate_model_on_grid, evaluate_von_mises_on_points

E_MOD, NU, STRESS_SCALE = 210e9, 0.3, 1e11

# --local-norm: bounding box of each subdomain, so every subnetwork maps ITS OWN region to
# [-1, 1]^2 (default: every subnetwork normalises over the whole domain).
LOCAL_BOXES = {'cantilever': [((0.0, 0.0), (2.0, 1.0)), ((2.0, 0.0), (4.0, 1.0))],
               'plate_hole': [((0.0, 0.0), (0.6, 0.6)), ((0.0, 0.0), (0.9, 0.9)),
                              ((0.0, 0.0), (2.0, 2.0))],
               'l_bracket': [((0.0, 0.0), (1.0, 1.0)), ((1.0, 0.0), (2.0, 1.0)),
                             ((1.0, 1.0), (2.0, 2.0))]}


def build(benchmark, solver, device, layout='paper', gamma_t=0.1):
    from atlas.sampling import CantileverBeamSampler, LBracketSampler, PlateWithHoleSampler
    sampler = {'cantilever': CantileverBeamSampler, 'plate_hole': PlateWithHoleSampler,
               'l_bracket': LBracketSampler}[benchmark](device=device)
    phys_train = PlaneStressElasticity(E=E_MOD / STRESS_SCALE, nu=NU)
    if solver == 'xpinn':
        from atlas.energy_xpinn import build_xpinn_energy
        model, problem, _ = build_xpinn_energy(benchmark, phys_train, device=device, layout=layout,
                                               gamma_t=gamma_t)
        return model, problem, sampler
    from atlas import energy_solver as es
    from atlas.networks import create_xpinn_for_benchmark
    if benchmark == 'cantilever':
        L, H = 4.0, 1.0
        u_s = 1e6 * L ** 3 / (3.0 * E_MOD * (H ** 3 / 12.0))
        net, prob = es.build_cantilever(phys_train, -1e6 / STRESS_SCALE, u_s, L, H,
                                        hidden_layers=4, neurons=64, device=device,
                                        dtype=torch.float64)
    elif benchmark == 'l_bracket':
        u_s = 1e6 / (3.0 * E_MOD / 12.0)
        net, prob = es.build_l_bracket(phys_train, 1e6 / STRESS_SCALE, u_s, 1.0,
                                       hidden_layers=4, neurons=64, device=device,
                                       dtype=torch.float64)
    else:
        u_s = 1e6 * 2.0 / E_MOD
        net, prob = es.build_plate_hole(phys_train, 1e6 / STRESS_SCALE, u_s, 0.3, 2.0, 2.0,
                                        hidden_layers=4, neurons=64, device=device,
                                        dtype=torch.float64)
    model = create_xpinn_for_benchmark(benchmark, device=device, monolith=True)
    model.subnets[0] = net
    return model, prob, sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["cantilever", "plate_hole", "l_bracket"])
    ap.add_argument("--solver", required=True, choices=["atlas", "xpinn", "pinn"])
    ap.add_argument("--epochs", type=int, default=None,
                    help="Adam steps (default 2000; 8000 for the residual PINN)")
    ap.add_argument("--lbfgs", type=int, default=None,
                    help="L-BFGS iterations (default 2000; 1200 = 60 x 20 for the PINN)")
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="comparison/convergence_timing")
    ap.add_argument("--al-stages", type=int, default=3,
                    help="xpinn only: augmented-Lagrangian multiplier updates (0 = plain penalty)")
    ap.add_argument("--local-norm", action="store_true",
                    help="xpinn only: normalise each subnetwork's inputs over its own subdomain")
    ap.add_argument("--layout", default="paper", choices=["paper", "concentration"],
                    help="xpinn only: decomposition (see build_xpinn_energy)")
    ap.add_argument("--gamma-t", type=float, default=0.1, help="xpinn only: traction-continuity weight")
    ap.add_argument("--warm-start", type=int, default=1000,
                    help="xpinn only: Adam steps of a shared single network copied into all subnetworks")
    ap.add_argument("--tag", default=None, help="output file name tag (default: the solver name)")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.epochs is None:
        args.epochs = 8000 if args.solver == "pinn" else 2000
    if args.lbfgs is None:
        args.lbfgs = 1200 if args.solver == "pinn" else 2000

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    b = args.benchmark
    if args.solver == "pinn":
        from atlas.sampling import CantileverBeamSampler, LBracketSampler, PlateWithHoleSampler
        sampler = {'cantilever': CantileverBeamSampler, 'plate_hole': PlateWithHoleSampler,
                   'l_bracket': LBracketSampler}[b](device=device)
        model = problem = None
    else:
        model, problem, sampler = build(b, args.solver, device, args.layout, args.gamma_t)
        if args.solver == 'xpinn' and args.local_norm:
            for net, (lo, hi) in zip(model.subnets, LOCAL_BOXES[b]):
                net._lo.copy_(torch.tensor(lo))
                net._hi.copy_(torch.tensor(hi))
    physics = PlaneStressElasticity(E=E_MOD, nu=NU)
    R = FEM_REFERENCE_DIR
    coords = np.load(R / f"{b}_coords.npy")
    u_f = np.load(R / f"{b}_u.npy").ravel()
    v_f = np.load(R / f"{b}_v.npy").ravel()
    vm_f = np.load(R / f"{b}_vonmises.npy")
    centers = [loc for _, loc in singular_locations(b, sampler_geometry_dict(sampler, b))]
    r_excl = default_exclusion_radius(coords, centers)
    mag_f = np.sqrt(u_f ** 2 + v_f ** 2)
    n_params = None if model is None else sum(p.numel() for p in model.parameters())

    rows = []
    state = {"probe_s": 0.0}

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    def probe(it, net=None, samp=None):
        net = model if net is None else net
        samp = sampler if samp is None else samp
        sync()
        t = time.perf_counter()
        elapsed = t - t_start - state["probe_s"]
        snap = copy.deepcopy(net).float()   # scored in float32, as validation.py does
        u, v = _evaluate_model_on_grid(snap, samp, coords[:, 0:1], coords[:, 1:2], device)
        vec = float(np.sqrt(np.sum((u - u_f) ** 2 + (v - v_f) ** 2) / np.sum(u_f ** 2 + v_f ** 2)))
        mag = float(relative_l2(np.sqrt(u ** 2 + v ** 2), mag_f))
        vm = evaluate_von_mises_on_points(snap, physics, samp, coords, device=device)
        vm_l2 = compute_von_mises_metrics(vm, vm_f, coords, centers, r_excl)['von_mises_l2_excl']
        rows.append(dict(iter=int(it), seconds=elapsed, disp_vec_l2=vec, disp_mag_l2=mag,
                         vm_l2_excl=None if vm_l2 is None else float(vm_l2)))
        sync()
        state["probe_s"] += time.perf_counter() - t
        print(f"  it {it:5d}  t={elapsed:7.1f}s  disp(vec)={vec:.3%}  disp(|u|)={mag:.3%}  "
              f"vm={'n/a' if vm_l2 is None else format(vm_l2, '.3%')}", flush=True)

    print(f"{b} / {args.solver}: device {device}", flush=True)
    sync()
    t_start = time.perf_counter()
    if args.solver == 'pinn':
        from atlas.atlas_main import train_forward_solver
        model, trainer, _, _ = train_forward_solver(
            benchmark=b, device=device, epochs=args.epochs, print_every=1000, use_lbfgs=True,
            lbfgs_steps=args.lbfgs // 20, lbfgs_max_iter_per_step=20, formulation="residual",
            seed=args.seed, standard_pinn=True, probe=probe, probe_every=args.probe_every)
        n_params = sum(p.numel() for p in model.parameters())
        hist = {'total': trainer.history['total'], 'iter': list(range(1, len(trainer.history['total']) + 1))}
    elif args.solver == 'xpinn':
        from atlas.energy_xpinn import train_decomposed
        hist = train_decomposed(model, problem, epochs=args.epochs, lbfgs_per_stage=args.lbfgs,
                                gammas=(30.0,), verbose=False, probe=probe,
                                al_stages=args.al_stages, warm_start=args.warm_start)
    else:
        from atlas.energy_solver import train_energy
        hist = train_energy(model.subnets[0], problem, epochs=args.epochs,
                            lbfgs_iters=args.lbfgs, lbfgs_chunk=100, verbose=False,
                            probe=probe, probe_every=args.probe_every)
    sync()
    total_s = time.perf_counter() - t_start - state["probe_s"]
    pi = np.asarray(hist['total'], dtype=np.float64)
    out = RESULTS_DIR / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.solver
    json.dump(dict(benchmark=b, solver=args.solver, tag=tag, al_stages=args.al_stages,
                   local_norm=args.local_norm, layout=args.layout, warm_start=args.warm_start, gamma_t=args.gamma_t,
                   seed=args.seed, device=device,
                   params=n_params, epochs=args.epochs, lbfgs=args.lbfgs,
                   train_seconds=total_s, probe_seconds=state["probe_s"],
                   final_energy=float(pi[-1]), probes=rows,
                   energy_iter=hist['iter'], energy=pi.tolist()),
              open(out / f"{b}_{tag}.json", "w"))
    last = rows[-1]
    print(f"DONE {b}/{args.solver}: train {total_s:.1f}s (probing {state['probe_s']:.1f}s excluded), "
          f"final disp(vec) {last['disp_vec_l2']:.3%}, vm {last['vm_l2_excl']:.3%}", flush=True)


if __name__ == "__main__":
    main()
