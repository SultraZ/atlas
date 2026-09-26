"""Train and score one solver on the multi-cutout bracket (atlas/benchmarks/multicutout.py).

Solvers
    atlas        ATLAS energy: one network
    xpinn        ATLAS-XPINN: three subnetworks (upper arm / corner region / horizontal
                 arm), augmented-Lagrangian interfaces + warm start (the default recipe)
    atlas_wide   one network with about the parameter count of `xpinn` (control)

Scored against the FEM reference (fem/reference/multicutout_*.npy, never used in training)
every --probe-every iterations (probing time excluded from the clock) and at the end:
  displacement vector L2 / |u| L2, von Mises L2 excluding the re-entrant corner and the two
  clamp corners, von Mises L2 in a 0.03 m band around each cutout, and the predicted / FEM
  peak von Mises at each cutout.

Writes results/hard_single_material/multicutout/<tag>/seed<seed>.{json,npz,_model.pt}.

    python scripts/hard_benchmarks/run_multicutout.py --solver xpinn --seed 42
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

from atlas.benchmarks import multicutout as mc
from atlas.paths import FEM_REFERENCE_DIR, RESULTS_DIR
from atlas.physics import PlaneStressElasticity
from atlas.validation import compute_von_mises_metrics, default_exclusion_radius, relative_l2

GOOD_TOL, STOP_TOL = 0.0015, 0.003   # energy-check guard (no FEM data): keep / stop thresholds on |ratio - 1|
WIDE_NEURONS = 112        # one 4 x 112 network ~ 38.4k parameters = three 4 x 64 subnetworks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True, choices=["atlas", "xpinn", "atlas_wide"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lbfgs", type=int, default=2000)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--al-stages", type=int, default=3)
    ap.add_argument("--warm-start", type=int, default=1000)
    ap.add_argument("--al-keep-memory", action="store_true",
                    help="xpinn: one L-BFGS through all augmented-Lagrangian stages")
    ap.add_argument("--warm-start-lbfgs", type=int, default=0,
                    help="xpinn: L-BFGS iterations of the shared warm-start network (from the same budget)")
    ap.add_argument("--local-norm", action="store_true",
                    help="xpinn: normalise each subnetwork's inputs over its own subdomain")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="hard_single_material/multicutout")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    phys = PlaneStressElasticity(E=mc.E_MOD / mc.STRESS_SCALE, nu=mc.NU)
    if args.solver == "xpinn":
        model, prob = mc.build_decomposed(phys, device=device, local_norm=args.local_norm)
    else:
        model, prob = mc.build_single(phys, neurons=WIDE_NEURONS if args.solver == "atlas_wide" else 64,
                                      device=device)
    n_params = sum(p.numel() for p in model.parameters())

    R = FEM_REFERENCE_DIR
    coords = np.load(R / "multicutout_coords.npy")
    u_f = np.load(R / "multicutout_u.npy").ravel()
    v_f = np.load(R / "multicutout_v.npy").ravel()
    vm_f = np.load(R / "multicutout_vonmises.npy")
    centers = mc.singular_points()
    r_excl = default_exclusion_radius(coords, centers)
    Z = mc.cutout_zones(coords)
    mag_f = np.hypot(u_f, v_f)

    from atlas.energy_solver import EnergyProblem
    chk = EnergyProblem(phys, mc.load_check_quadrature(), [], energy_scale=1.0, device=device)
    train_all = EnergyProblem(phys, [(p, w) for _, p, w in mc.load_quadrature()], [], energy_scale=1.0,
                              device=device)

    def energy_check():
        """Strain energy on an independent quadrature / on the training quadrature (no FEM data):
        ~1 while the network is resolved by the training points; drifts if it exploits the gaps.
        """
        return float(chk.strain_energy(model).detach() / train_all.strain_energy(model).detach())

    def score():
        up, vp, vm = mc.predict(model, coords, device)
        vec = float(np.sqrt(np.sum((up - u_f) ** 2 + (vp - v_f) ** 2) / np.sum(u_f ** 2 + v_f ** 2)))
        mag = float(relative_l2(np.hypot(up, vp), mag_f))
        vml = compute_von_mises_metrics(vm.reshape(-1, 1), vm_f, coords, centers, r_excl)['von_mises_l2_excl']
        f = vm_f.ravel()
        cut = {k: float(np.sqrt(np.sum((vm[m] - f[m]) ** 2) / np.sum(f[m] ** 2))) for k, m in Z.items()}
        peak = {k: float(vm[m].max() / f[m].max()) for k, m in Z.items()}
        return dict(disp_vec_l2=vec, disp_mag_l2=mag, vm_l2_excl=None if vml is None else float(vml),
                    cutout_vm_l2=cut, cutout_peak_ratio=peak,
                    cutout_vm_l2_mean=float(np.mean(list(cut.values()))),
                    energy_check_ratio=energy_check()), (up, vp, vm)

    class StopTraining(Exception):
        pass

    rows, st = [], {"probe_s": 0.0, "good_state": None, "good_it": None, "stopped_at": None}
    sync = lambda: torch.cuda.synchronize() if device.startswith("cuda") else None

    def probe(it):
        sync()
        t = time.perf_counter()
        el = t - t0 - st["probe_s"]
        m, _ = score()
        rows.append(dict(iter=int(it), seconds=el, **m))
        dev = abs(m["energy_check_ratio"] - 1.0)
        if dev < GOOD_TOL:
            st["good_state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
            st["good_it"] = int(it)
        sync()
        st["probe_s"] += time.perf_counter() - t
        print(f"  it {it:5d}  t={el:7.1f}s  disp(vec)={m['disp_vec_l2']:.3%}  vm={m['vm_l2_excl']:.3%}  "
              f"cutouts vm={m['cutout_vm_l2_mean']:.3%}  energy check={m['energy_check_ratio']:.4f}", flush=True)
        if dev > STOP_TOL:
            st["stopped_at"] = int(it)
            raise StopTraining()

    print(f"multicutout / {args.solver} seed {args.seed}: {n_params} params, {prob.n_points} "
          f"quadrature points, device {device}", flush=True)
    sync()
    t0 = time.perf_counter()
    hist = {'total': [float('nan')]}
    try:
        if args.solver == "xpinn":
            from atlas.energy_xpinn import train_decomposed
            hist = train_decomposed(model, prob, epochs=args.epochs, lbfgs_per_stage=args.lbfgs, gammas=(30.0,),
                                    verbose=False, probe=probe, al_stages=args.al_stages,
                                    warm_start=args.warm_start, al_keep_memory=args.al_keep_memory,
                                    warm_start_lbfgs=args.warm_start_lbfgs)
        else:
            from atlas.energy_solver import train_energy
            hist = train_energy(model, prob, epochs=args.epochs, lbfgs_iters=args.lbfgs, lbfgs_chunk=100,
                                verbose=False, probe=probe, probe_every=args.probe_every)
    except StopTraining:
        print(f"  energy-check guard: stopped at iteration {st['stopped_at']}, restoring the weights of "
              f"iteration {st['good_it']}", flush=True)
        model.load_state_dict(st["good_state"])
    jump = prob.last.get("jump_u") if args.solver == "xpinn" else None
    sync()
    total = time.perf_counter() - t0 - st["probe_s"]
    final, (up, vp, vm) = score()
    # compliance C = int t.u over the loaded edge, from the network, vs the FEM value
    lp, lw, tr = mc.loads(mc.SIGMA)
    ul, vl, _ = mc.predict(model, lp, device)
    C = float(np.sum(lw * (tr[:, 0] * ul + tr[:, 1] * vl)))
    C_fem = json.load(open(R / "multicutout_meta.json"))["compliance"]
    final.update(compliance=C, compliance_fem=C_fem, compliance_error=C / C_fem - 1.0)
    tag = args.tag or args.solver
    out = RESULTS_DIR / args.out / tag
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"seed{args.seed}.npz", coords=coords, u=up, v=vp, vm=vm, u_fem=u_f, v_fem=v_f,
                        vm_fem=vm_f.ravel())
    torch.save(model.state_dict(), out / f"seed{args.seed}_model.pt")
    json.dump(dict(benchmark="multicutout", solver=args.solver, tag=tag, seed=args.seed, params=int(n_params),
                   epochs=args.epochs, lbfgs=args.lbfgs, al_stages=args.al_stages, warm_start=args.warm_start,
                   al_keep_memory=args.al_keep_memory, warm_start_lbfgs=args.warm_start_lbfgs,
                   local_norm=args.local_norm,
                   train_seconds=total, probe_seconds=st["probe_s"], final=final,
                   guard=dict(good_tol=GOOD_TOL, stop_tol=STOP_TOL, stopped_at=st["stopped_at"],
                              restored_iteration=st["good_it"] if st["stopped_at"] else None),
                   final_energy=float(hist['total'][-1]), interface_jump_sq=jump, probes=rows),
              open(out / f"seed{args.seed}.json", "w"), indent=1)
    print(f"DONE multicutout/{tag} seed {args.seed}: train {total:.1f}s  disp(vec) {final['disp_vec_l2']:.3%}  "
          f"vM(excl) {final['vm_l2_excl']:.3%}  cutouts vM {final['cutout_vm_l2_mean']:.3%}  "
          f"compliance {final['compliance_error']:+.2%}", flush=True)


if __name__ == "__main__":
    main()
