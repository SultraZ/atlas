"""Run from the project root (all options: --help):
    python scripts/compare_solvers.py --benchmark {cantilever,l_bracket,plate_hole} --variant {pinn,atlas,xpinn_energy}

Example:
    python scripts/compare_solvers.py --benchmark plate_hole --variant xpinn_energy
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from atlas.atlas_main import train_forward_solver
from atlas.paths import FEM_REFERENCE_DIR, RESULTS_DIR
from atlas.validation import run_validation
from atlas.visualization import _evaluate_model_on_grid, evaluate_von_mises_on_points

VARIANTS = ("pinn", "atlas", "xpinn_energy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["cantilever", "l_bracket", "plate_hole"])
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--epochs", type=int, default=None,
                    help="Adam steps (default 8000 for residual variants, 2000 for xpinn_energy)")
    ap.add_argument("--lbfgs-steps", type=int, default=60)
    ap.add_argument("--lbfgs-max-iter", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--plots-only", action="store_true",
                    help="xpinn_energy only: skip training, load the saved model and "
                         "(re)write the plots")
    ap.add_argument("--out-dir", default=None,
                    help="override the per-variant default output folder under results/")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    default_dir = {"atlas": "atlas_energy/forward",
                   "xpinn_energy": "atlas_xpinn_energy/forward"}.get(args.variant, "comparison/data")
    out = RESULTS_DIR / (args.out_dir or default_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.benchmark}_{args.variant}"
    if args.epochs is None:
        args.epochs = 2000 if args.variant == "xpinn_energy" else 8000

    t0 = time.time()
    if args.variant == "xpinn_energy":
        from atlas.energy_xpinn import build_xpinn_energy, train_decomposed
        from atlas.physics import PlaneStressElasticity
        from atlas.sampling import CantileverBeamSampler, LBracketSampler, PlateWithHoleSampler
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        physics = PlaneStressElasticity(E=210e9, nu=0.3)
        model, problem, _ = build_xpinn_energy(
            args.benchmark, PlaneStressElasticity(E=210e9 / 1e11, nu=0.3), device=device)
        if args.plots_only:
            model.load_state_dict(torch.load(out / f"{tag}_model.pt", map_location=device))
            hist = None
        else:
            hist = train_decomposed(model, problem, epochs=args.epochs)
        model = model.float()
        if not args.plots_only:
            torch.save(model.state_dict(), out / f"{tag}_model.pt")
            from atlas.energy_solver import history_for_plot
            with open(out / f"{tag}_history.json", "w") as f:
                json.dump({"history": history_for_plot(hist), "adam_epoch_count": args.epochs}, f)
        sampler = {"cantilever": CantileverBeamSampler, "plate_hole": PlateWithHoleSampler,
                   "l_bracket": LBracketSampler}[args.benchmark](device=device)
        train_s = None if args.plots_only else time.time() - t0
        cfg = {"formulation": "energy_xpinn", "epochs": args.epochs, "lbfgs_iters": 2000,
               "gammas": [30.0], "gamma_t": 0.1, "al_stages": 3, "warm_start": 1000,
               "final_interface_jump_sq": problem.last.get("jump_u")}
    elif args.variant == "atlas":
        from atlas.energy_solver import load_energy_model
        model, sampler, physics = load_energy_model(
            args.benchmark, str(RESULTS_DIR / "atlas_energy" / "forward" / f"{args.benchmark}_model.pt"),
            device=device)
        train_s = None
        cfg = {"formulation": "energy", "note": "loaded from results/atlas_energy/forward"}
    else:
        common = dict(benchmark=args.benchmark, device=device, epochs=args.epochs,
                      print_every=1000, use_lbfgs=True, lbfgs_steps=args.lbfgs_steps,
                      lbfgs_max_iter_per_step=args.lbfgs_max_iter, formulation="residual",
                      seed=args.seed)
        if args.variant == "pinn":
            common["standard_pinn"] = True
        model, trainer, sampler, physics = train_forward_solver(**common)
        train_s = time.time() - t0
        cfg = {k: v for k, v in common.items() if k not in ("benchmark", "device")}

    res = run_validation(model, sampler, physics, args.benchmark, device=device, use_mock=False)

    if args.variant == "xpinn_energy":
        from atlas.visualization import (plot_deformed_shape, plot_displacement_field,
                                         plot_fem_comparison, plot_training_convergence,
                                         plot_von_mises_stress)
        plot_fem_comparison(model, sampler, physics, benchmark=args.benchmark, use_mock=False,
                            save_path=str(out / f"{args.benchmark}_fem_comparison.png"))
        plot_displacement_field(model, sampler, physics,
                                save_path=str(out / f"{args.benchmark}_displacement.png"))
        plot_von_mises_stress(model, sampler, physics,
                              save_path=str(out / f"{args.benchmark}_stress.png"))
        plot_deformed_shape(model, sampler, n_grid=50 if args.benchmark == "cantilever" else 70,
                            save_path=str(out / f"{args.benchmark}_deformed.png"))
        hp = out / f"{tag}_history.json"
        if hp.exists():
            h = json.load(open(hp))
            plot_training_convergence(h["history"], adam_epoch_count=h["adam_epoch_count"],
                                      save_path=str(out / f"{args.benchmark}_convergence.png"))
        if args.plots_only:
            print(f"plots written to {out}")
            return

    R = FEM_REFERENCE_DIR
    coords = np.load(R / f"{args.benchmark}_coords.npy")
    u_f = np.load(R / f"{args.benchmark}_u.npy").ravel()
    v_f = np.load(R / f"{args.benchmark}_v.npy").ravel()
    vm_f = np.load(R / f"{args.benchmark}_vonmises.npy").ravel()
    up, vp = _evaluate_model_on_grid(model, sampler, coords[:, 0:1], coords[:, 1:2], device)
    vm_p = evaluate_von_mises_on_points(model, physics, sampler, coords, device=device).ravel()
    np.savez_compressed(out / f"{tag}.npz", coords=coords, u=up, v=vp, vm=vm_p,
                        u_fem=u_f, v_fem=v_f, vm_fem=vm_f)

    n_params = int(sum(p.numel() for p in model.parameters()))
    keep = ("relative_L2_u", "relative_L2_v", "relative_L2_displacement", "von_mises_l2_excl",
            "von_mises_l2_global", "von_mises_l2_roi", "max_pointwise_von_mises_error")
    meta = {k: res.get(k) for k in keep}
    meta.update(benchmark=args.benchmark, variant=args.variant, seed=args.seed,
                params=n_params, train_seconds=None if train_s is None else round(train_s, 1),
                pred_max_abs_disp=float(np.hypot(up, vp).max()),
                fem_max_abs_disp=float(np.hypot(u_f, v_f).max()), config=cfg)
    with open(out / f"{tag}.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nRESULT {tag}: disp L2 {meta['relative_L2_displacement']:.2%}  "
          f"vM excl L2 {meta['von_mises_l2_excl']:.2%}  params {n_params}")


if __name__ == "__main__":
    main()
