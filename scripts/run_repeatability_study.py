"""Run from the project root (all options: --help):
    python scripts/run_repeatability_study.py --benchmark {cantilever,l_bracket,plate_hole} [--seeds N ...] [--formulation F]

Example:
    python scripts/run_repeatability_study.py --benchmark plate_hole --formulation energy_xpinn --seeds 42 43 44 45 46
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from atlas.atlas_main import setup_device, train_forward_solver
from atlas.validation import run_validation
from atlas.paths import FEM_REFERENCE_DIR, RESULTS_DIR


def run_one_seed(benchmark: str, seed: int, epochs: int, use_mock_fem: bool,
                  data_dir: str, l_bracket_fillet_init, formulation: str = 'residual'):
    device = setup_device(seed=seed)
    kwargs = {'formulation': formulation}
    if formulation == 'energy':
        kwargs.update(use_lbfgs=True, lbfgs_steps=15, lbfgs_max_iter_per_step=100)
    if formulation == 'energy_xpinn':
        kwargs.update(use_lbfgs=True, lbfgs_steps=20, lbfgs_max_iter_per_step=100)
    if formulation == 'residual' and benchmark == 'cantilever':
        kwargs['warm_start_eb'] = True
        kwargs['prefit_epochs'] = 6000
        kwargs['use_lbfgs'] = False
    if benchmark == 'l_bracket' and l_bracket_fillet_init is not None:
        kwargs['l_bracket_fillet_init'] = l_bracket_fillet_init

    model, trainer, sampler, physics_real = train_forward_solver(
        benchmark=benchmark, device=device, epochs=epochs, print_every=max(epochs, 1),
        seed=seed, **kwargs
    )

    final_loss = trainer.history['total'][-1]
    initial_loss = trainer.history['total'][0]
    loss_ratio = final_loss / initial_loss if initial_loss != 0 else float('nan')

    val_results = run_validation(
        model=model, sampler=sampler, physics=physics_real,
        benchmark=benchmark, device=device, use_mock=use_mock_fem, data_dir=data_dir,
    )

    return {
        'seed': seed,
        'final_loss': final_loss,
        'initial_loss': initial_loss,
        'loss_ratio': loss_ratio,
        'relative_L2_displacement': val_results.get('relative_L2_displacement'),
        'displacement_within_5pct_target': val_results.get('displacement_within_5pct_target'),
        'relative_L2_von_mises': val_results.get('relative_L2_von_mises'),
        'von_mises_l2_excl': val_results.get('von_mises_l2_excl'),
        'von_mises_exclusion_radius': val_results.get('von_mises_exclusion_radius'),
        'von_mises_l2_excl_within_10pct': val_results.get('von_mises_l2_excl_within_10pct'),
        'used_mock_data': val_results.get('used_mock_data'),
    }


def summarize(per_seed_results):
    def mean_std(key):
        vals = [r[key] for r in per_seed_results if r.get(key) is not None]
        if not vals:
            return None, None
        if len(vals) == 1:
            return vals[0], 0.0
        return statistics.mean(vals), statistics.stdev(vals)

    summary = {}
    for key in ['final_loss', 'loss_ratio', 'relative_L2_displacement',
                'relative_L2_von_mises', 'von_mises_l2_excl', 'von_mises_exclusion_radius']:
        m, s = mean_std(key)
        summary[f'{key}_mean'] = m
        summary[f'{key}_std'] = s

    pass_flags = [r['displacement_within_5pct_target'] for r in per_seed_results
                  if r.get('displacement_within_5pct_target') is not None]
    stress_flags = [r['von_mises_l2_excl_within_10pct'] for r in per_seed_results
                    if r.get('von_mises_l2_excl_within_10pct') is not None]
    summary['n_runs'] = len(per_seed_results)
    summary['n_runs_with_meaningful_fem_comparison'] = len(pass_flags)
    summary['n_runs_passed_5pct_target'] = sum(1 for p in pass_flags if p)
    summary['n_runs_passed_10pct_excl_stress'] = sum(1 for p in stress_flags if p)
    return summary


def plot_summary(benchmark, per_seed_results, summary, save_path):
    seeds = [r['seed'] for r in per_seed_results]
    errs = [r['relative_L2_displacement'] for r in per_seed_results]
    meaningful = [not r.get('used_mock_data', True) for r in per_seed_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['tab:blue' if m else 'tab:gray' for m in meaningful]
    ax.bar([str(s) for s in seeds], errs, color=colors)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=1, label="paper's 5% target")
    ax.set_xlabel('Random seed')
    ax.set_ylabel('Relative L2 error (|displacement|)')
    ax.set_title(f'{benchmark} — repeatability across {len(seeds)} independent runs')
    ax.legend()

    mean = summary.get('relative_L2_displacement_mean')
    std = summary.get('relative_L2_displacement_std')
    if mean is not None:
        subtitle = f'mean = {mean:.2%}, std = {std:.2%}' if std is not None else f'mean = {mean:.2%}'
        if any(not m for m in meaningful):
            subtitle += '  (gray bars = mock data, not counted toward the paper target)'
        ax.text(0.5, 1.08, subtitle, transform=ax.transAxes, ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--benchmark', required=True, choices=['cantilever', 'l_bracket', 'plate_hole'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46],
                        help='Random seeds for each independent run (default: 5 seeds, matching the paper).')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--use-mock-fem', action='store_true',
                        help='Use mock FEM data (default: False, i.e. expects a real fem_reference/). '
                             'Pass this flag only if you have not generated real FEM references yet - '
                             'the resulting L2-error numbers will not be meaningful for the paper.')
    parser.add_argument('--data-dir', default=str(FEM_REFERENCE_DIR))
    parser.add_argument('--l-bracket-fillet-init', type=float, default=None,
                        help='Only relevant for benchmark=l_bracket; leave unset for the default sharp-corner geometry.')
    parser.add_argument('--formulation', choices=['residual', 'energy', 'energy_xpinn'], default='residual',
                        help="Forward solver to repeat. 'residual' (default) is the original "
                             "strong-form XPINN recipe this study was written for; 'energy' "
                             "is the deep-energy solver (train_forward_solver's own default).")
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    default_dir = {'energy': RESULTS_DIR / 'atlas_energy' / 'repeatability' / args.benchmark,
                   'energy_xpinn': RESULTS_DIR / 'atlas_xpinn_energy' / 'repeatability' / args.benchmark,
                   }.get(args.formulation, RESULTS_DIR / f'repeatability_{args.benchmark}')
    out_dir = args.out_dir or str(default_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print(f"REPEATABILITY STUDY — {args.benchmark.upper()}")
    print(f"Seeds: {args.seeds}  |  Epochs per run: {args.epochs}  |  "
          f"FEM: {'MOCK' if args.use_mock_fem else 'real (' + args.data_dir + ')'}")
    print("=" * 70)

    per_seed_results = []
    for i, seed in enumerate(args.seeds):
        print(f"\n--- Run {i+1}/{len(args.seeds)} (seed={seed}) ---")
        result = run_one_seed(
            benchmark=args.benchmark, seed=seed, epochs=args.epochs,
            use_mock_fem=args.use_mock_fem, data_dir=args.data_dir,
            l_bracket_fillet_init=args.l_bracket_fillet_init,
            formulation=args.formulation,
        )
        per_seed_results.append(result)
        print(f"    -> final loss={result['final_loss']:.4e}, "
              f"L2 disp={result['relative_L2_displacement']:.4%}, "
              f"5%-target={result['displacement_within_5pct_target']}")

    summary = summarize(per_seed_results)

    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL RUNS")
    print("=" * 70)
    for key, val in summary.items():
        print(f"  {key}: {val}")

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({'per_seed': per_seed_results, 'summary': summary,
                   'benchmark': args.benchmark, 'epochs': args.epochs}, f, indent=2)

    plot_summary(args.benchmark, per_seed_results, summary,
                 os.path.join(out_dir, 'summary.png'))

    print(f"\nSaved {out_dir}/summary.json and {out_dir}/summary.png")


if __name__ == '__main__':
    main()
