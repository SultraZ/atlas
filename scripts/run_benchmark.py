"""Train one ATLAS benchmark, validate against the FEM reference, save artifacts.

    python scripts/run_benchmark.py --benchmark cantilever
    python scripts/run_benchmark.py --benchmark l_bracket --seed 1 --tag seed1

Writes into results/<out-dir>/ (default results/atlas_energy/forward/): <benchmark>_model.pt (state dict),
<benchmark>_validation.json, <benchmark>_fem_comparison.png,
<benchmark>_convergence.png, <benchmark>_history.json, <benchmark>_displacement.png,
<benchmark>_stress.png, <benchmark>_deformed.png. Exit code 0 only if both targets pass
(displacement L2 < 5%, von Mises L2 excluded-region < 10%).
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from atlas.atlas_main import train_forward_solver
from atlas.paths import RESULTS_DIR
from atlas.validation import run_validation
from atlas.visualization import (plot_deformed_shape, plot_displacement_field,
                                 plot_fem_comparison, plot_training_convergence,
                                 plot_von_mises_stress)


def save_field_plots(model, sampler, physics, benchmark, out, name):
    plot_displacement_field(model, sampler, physics, save_path=str(out / f"{name}_displacement.png"))
    plot_von_mises_stress(model, sampler, physics, save_path=str(out / f"{name}_stress.png"))
    plot_deformed_shape(model, sampler, n_grid=50 if benchmark == "cantilever" else 70,
                        save_path=str(out / f"{name}_deformed.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True,
                    choices=["cantilever", "l_bracket", "plate_hole"])
    ap.add_argument("--formulation", default="energy", choices=["energy", "residual"])
    ap.add_argument("--epochs", type=int, default=2000, help="Adam steps")
    ap.add_argument("--lbfgs-iters", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden-layers", type=int, default=4)
    ap.add_argument("--neurons", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default="atlas_energy/forward")
    ap.add_argument("--tag", default="")
    ap.add_argument("--plots-only", default=None, metavar="MODEL.pt",
                    help="skip training; load this saved model and only write the "
                         "comparison/displacement/stress(/deformed) plots")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = RESULTS_DIR / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    name = args.benchmark + (f"_{args.tag}" if args.tag else "")

    if args.plots_only:
        from atlas.energy_solver import load_energy_model
        model, sampler, physics = load_energy_model(
            args.benchmark, args.plots_only, device=device,
            hidden_layers=args.hidden_layers, neurons=args.neurons)
        plot_fem_comparison(model, sampler, physics, benchmark=args.benchmark,
                            use_mock=False, save_path=str(out / f"{name}_fem_comparison.png"))
        save_field_plots(model, sampler, physics, args.benchmark, out, name)
        hist_path = out / f"{name}_history.json"
        if hist_path.exists():
            h = json.load(open(hist_path))
            plot_training_convergence(h["history"], save_path=str(out / f"{name}_convergence.png"),
                                      adam_epoch_count=h["adam_epoch_count"])
        print(f"plots written to {out}")
        return

    t0 = time.time()
    chunk = 100
    model, trainer, sampler, physics = train_forward_solver(
        benchmark=args.benchmark, device=device, epochs=args.epochs,
        print_every=500, use_lbfgs=args.lbfgs_iters > 0,
        lbfgs_steps=max(1, args.lbfgs_iters // chunk),
        lbfgs_max_iter_per_step=chunk, hidden_layers=args.hidden_layers,
        neurons=args.neurons, formulation=args.formulation, seed=args.seed)
    train_s = time.time() - t0

    res = run_validation(model, sampler, physics, args.benchmark, device=device,
                         use_mock=False)
    res.update({"benchmark": args.benchmark, "formulation": args.formulation,
                "seed": args.seed, "epochs": args.epochs,
                "lbfgs_iters": args.lbfgs_iters, "train_seconds": round(train_s, 1)})
    with open(out / f"{name}_validation.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    torch.save(model.state_dict(), out / f"{name}_model.pt")
    plot_fem_comparison(model, sampler, physics, benchmark=args.benchmark,
                        use_mock=False, save_path=str(out / f"{name}_fem_comparison.png"))
    with open(out / f"{name}_history.json", "w") as f:
        json.dump({"history": trainer.history, "adam_epoch_count": trainer.adam_epoch_count}, f)
    plot_training_convergence(trainer.history, save_path=str(out / f"{name}_convergence.png"),
                              adam_epoch_count=trainer.adam_epoch_count)
    save_field_plots(model, sampler, physics, args.benchmark, out, name)

    disp = res["relative_L2_displacement"]
    vm = res["von_mises_l2_excl"]
    ok = disp < 0.05 and vm is not None and vm < 0.10
    print(f"\nRESULT {name}: disp L2 {disp:.2%} (<5%)  vM excl L2 "
          f"{vm:.2%} (<10%)  -> {'PASS' if ok else 'FAIL'}  [{train_s:.0f}s]")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
