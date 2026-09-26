"""Run from the project root (all options: --help):
    python tests/validate_all_benchmarks.py [--benchmark B] [--epochs N]

Example:
    python tests/validate_all_benchmarks.py --benchmark cantilever
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from atlas.atlas_main import train_forward_solver
from atlas.validation import run_validation
from atlas.visualization import (plot_fem_comparison, plot_displacement_field,
                                 plot_von_mises_stress, plot_deformed_shape,
                                 plot_training_convergence)
from atlas.paths import RESULTS_DIR, FEM_REFERENCE_DIR

BENCHMARKS = ['cantilever', 'l_bracket', 'plate_hole']

BENCH_DEFAULTS = {
    'cantilever': dict(epochs=11000, warm_start_eb=True, prefit_epochs=6000, use_lbfgs=False,
                       formulation='residual'),
    'l_bracket':   dict(epochs=5000, use_lbfgs=False, formulation='residual'),
    'plate_hole':  dict(epochs=5000, use_lbfgs=False, formulation='residual'),
}


def train_and_report(benchmark: str, device: str, epoch_override, overrides=None,
                     no_plots: bool = False):
    print("\n" + "=" * 72)
    print(f"BENCHMARK: {benchmark.upper()}  (device={device}, seed=42)")
    print("=" * 72)

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    cfg = dict(BENCH_DEFAULTS[benchmark])
    if overrides:
        cfg.update(overrides)
    if epoch_override is not None:
        cfg['epochs'] = epoch_override

    print(f"  training config: {cfg}")

    model, trainer, sampler, phys_real = train_forward_solver(
        benchmark=benchmark, device=device, **cfg)

    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    torch.save(model.state_dict(), str(RESULTS_DIR / f"{benchmark}_model.pt"))

    res = run_validation(
        model=model, sampler=sampler, physics=phys_real,
        benchmark=benchmark, device=device,
        use_mock=False, data_dir=str(FEM_REFERENCE_DIR),
    )
    res['benchmark'] = benchmark
    res['device'] = device
    res['seed'] = 42
    res['train_config'] = cfg
    with open(str(RESULTS_DIR / f"{benchmark}_validation.json"), "w") as f:
        json.dump(res, f, indent=2)

    if no_plots:
        return res

    plot_fem_comparison(model, sampler, phys_real, benchmark=benchmark,
                        use_mock=False, data_dir=str(FEM_REFERENCE_DIR),
                        save_path=str(RESULTS_DIR / f"{benchmark}_fem_comparison.png"))
    plot_displacement_field(model, sampler, phys_real, n_grid=90,
                            save_path=str(RESULTS_DIR / f"{benchmark}_displacement.png"))
    plot_von_mises_stress(model, sampler, phys_real, n_grid=90,
                          save_path=str(RESULTS_DIR / f"{benchmark}_stress.png"))
    plot_deformed_shape(model, sampler, n_grid=50,
                        save_path=str(RESULTS_DIR / f"{benchmark}_deformed.png"))
    plot_training_convergence(
        trainer.history,
        save_path=str(RESULTS_DIR / f"{benchmark}_convergence.png"),
        adam_epoch_count=trainer.adam_epoch_count,
    )

    print("\n--- {} validation summary ---".format(benchmark))
    for k in ('relative_L2_u', 'relative_L2_v', 'relative_L2_displacement',
              'displacement_within_5pct_target', 'relative_L2_von_mises',
              'von_mises_l2_excl', 'von_mises_exclusion_radius',
              'von_mises_l2_excl_within_10pct',
              'von_mises_l2_roi', 'von_mises_l2_roi_within_10pct'):
        print(f"  {k}: {res.get(k)}")
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', default=None,
                        choices=BENCHMARKS, help='Run one benchmark (default: all).')
    parser.add_argument('--device', default=None, help='cuda/cpu (default: cuda if available).')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override the epoch budget for the chosen run(s).')
    parser.add_argument('--lbfgs', action='store_true',
                        help='Enable the Adam -> L-BFGS two-stage recipe (paper '
                             'methodology) for every selected benchmark. Without '
                             'this flag the Adam-only BENCH_DEFAULTS are used.')
    parser.add_argument('--lbfgs-steps', type=int, default=15,
                        help='L-BFGS outer steps when --lbfgs is set (default 15).')
    parser.add_argument('--lbfgs-max-iter', type=int, default=50,
                        help='L-BFGS internal iterations per step when --lbfgs is '
                             'set (default 50).')
    parser.add_argument('--adaptive-weights', action='store_true',
                        help='Phase-2 gradient-norm balancing of the per-component '
                             'loss weights during the Adam stage (default: fixed '
                             'weights).')
    parser.add_argument('--adapt-every', type=int, default=100,
                        help='Epoch interval between weight adaptations when '
                             '--adaptive-weights is set (default 100).')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip the heavyweight matplotlib figures (writes '
                             'validation.json / model.pt only). Screen mode: '
                             'avoids the OOM-prone plot stage on low-RAM boxes.')
    parser.add_argument('--warm-kirsch', action='store_true',
                        help='Phase-3 warm start: prefit every plate_hole '
                             'subdomain network onto the analytic Kirsch field '
                             '(per-subdomain region) before physics training.')
    parser.add_argument('--mpe', action='store_true',
                        help='Phase-4 MPE recipe (plate_hole only): monolith + '
                             'hard symmetry rollers + Fourier features, driven '
                             'by the signed total-potential-energy objective '
                             'instead of the strong-form residual losses.')
    args = parser.parse_args()

    overrides = {}
    if args.lbfgs:
        overrides['use_lbfgs'] = True
        overrides['lbfgs_steps'] = args.lbfgs_steps
        overrides['lbfgs_max_iter_per_step'] = args.lbfgs_max_iter
    if args.adaptive_weights:
        overrides['adaptive_weights'] = True
        overrides['adapt_every'] = args.adapt_every
    if args.warm_kirsch:
        overrides['warm_start_kirsch'] = True
    if args.mpe:
        overrides['use_mpe'] = True

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    targets = [args.benchmark] if args.benchmark else BENCHMARKS
    all_results = {}
    for bench in targets:
        all_results[bench] = train_and_report(bench, device, args.epochs, overrides,
                                              no_plots=args.no_plots)

    print("\n" + "=" * 72)
    print("ALL-BENCHMARK SUMMARY")
    print("=" * 72)
    for bench, res in all_results.items():
        disp = res.get('relative_L2_displacement')
        disp_ok = res.get('displacement_within_5pct_target')
        vm_excl = res.get('von_mises_l2_excl')
        vm_ok = res.get('von_mises_l2_excl_within_10pct')
        vm_roi = res.get('von_mises_l2_roi')
        vm_roi_ok = res.get('von_mises_l2_roi_within_10pct')
        print(f"  {bench:12s} |disp| L2 {disp:.2%}  [5% target {'PASS' if disp_ok else 'FAIL'}]  | "
              f"von Mises excl {vm_excl if vm_excl is None else f'{vm_excl:.2%}'}  "
              f"[10% {'PASS' if vm_ok else 'FAIL'}]  | "
              f"von Mises ROI {vm_roi if vm_roi is None else f'{vm_roi:.2%}'}  "
              f"[10% {'PASS' if vm_roi_ok else 'FAIL'}]")
    print("\nArtifacts written under", RESULTS_DIR)


if __name__ == '__main__':
    main()