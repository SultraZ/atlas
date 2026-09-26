"""design_plots.py - figures for inverse-design results (atlas/energy_design.py)."""
import os
from typing import Dict, Optional

import matplotlib
if os.environ.get("MPLBACKEND") is None and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atlas.energy_design import design_fields

_LABEL = {'compliance': 'Compliance C [N m]', 'peak_vm': 'Peak von Mises [Pa]',
          'volume': 'Solid area [m^2]'}


def plot_design_history(result: Dict, save_path: Optional[str] = None):
    """Design parameter and metric versus iteration, with the target marked."""
    h = [r for r in result['history'] if r['tag'] != 'probe' and r['tag'] != 'return']
    name, metric = result['param_name'], result['metric']
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    it = np.arange(len(h))
    ax[0].plot(it, [r['param'] for r in h], 'o-', color='#2e8b57')
    ax[0].axhline(result['p_star'], color='k', ls=':', lw=1)
    ax[0].set(xlabel='Iteration', ylabel=name.capitalize(), title=f'{name.capitalize()} evolution')
    ax[1].plot(it, [r[metric] for r in h], 's-', color='#4f81bd')
    ax[1].axhline(result['target'], color='r', ls='--', lw=1, label='target')
    ax[1].set(xlabel='Iteration', ylabel=_LABEL.get(metric, metric), title='Metric vs target')
    ax[1].legend()
    ax[2].semilogy(it, np.maximum(np.abs([r['g'] for r in h]), 1e-8), 'd-', color='#c0504d')
    ax[2].set(xlabel='Iteration', ylabel='|metric/target - 1|', title='Residual')
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.suptitle(f"{result['benchmark']}: {name} = {result['p_star']:.4f}  ({result['status']})")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    plt.close(fig)


def plot_design_result(result: Dict, save_path: Optional[str] = None, n: int = 140):
    """Baseline vs optimised geometry, and the displacement / von Mises fields of
    the optimised design.
    """
    case, net, p0, ps = result['case'], result['net'], result['case'].p0, result['p_star']
    f = design_fields(case, net, ps, n)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for p_, col, lab, ls in ((p0, '#888888', f'baseline ({p0:.3f})', '--'),
                             (ps, '#2e8b57', f'optimised ({ps:.3f})', '-')):
        o = case.outline(p_)
        ax[0].plot(o[:, 0], o[:, 1], ls, color=col, lw=2, label=lab)
    ax[0].set_aspect('equal'); ax[0].legend(fontsize=8)
    ax[0].set_title(f"Shape: {case.param_name}"); ax[0].grid(True, alpha=0.3)
    c1 = ax[1].contourf(f['X'], f['Y'], f['disp'], levels=40, cmap='viridis')
    ax[1].set_aspect('equal'); ax[1].set_title('|displacement| [m]'); plt.colorbar(c1, ax=ax[1], shrink=0.8)
    vmax = np.nanpercentile(f['vm'], 99.5)
    c2 = ax[2].contourf(f['X'], f['Y'], np.clip(f['vm'], 0, vmax), levels=40, cmap='hot')
    ax[2].set_aspect('equal'); ax[2].set_title('von Mises [Pa]'); plt.colorbar(c2, ax=ax[2], shrink=0.8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=140)
    plt.close(fig)
