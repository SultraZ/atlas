"""visualization.py - Plotting utilities for XPINN results
Now correctly routes points to the right subdomain network
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from atlas.paths import FEM_REFERENCE_DIR


def _show_if_interactive():
    """plt.show() only when a GUI backend is active; on headless/Agg runs it
    is a no-op instead of emitting a "non-interactive backend" warning.
    """
    if plt.get_backend().lower() != 'agg':
        plt.show()


def _colorbar(fig, im, ax, label=None):
    """Colorbar exactly as tall as its (equal-aspect) axes, so wide/short
    domains such as the cantilever don't get a figure-height bar and dead
    white space around the plot.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes('right', size='3%', pad=0.08)
    return fig.colorbar(im, cax=cax, label=label)


def _inside_material(sampler, X, Y) -> np.ndarray:
    """Boolean mask of grid points that lie in the actual material, so plots
    leave the plate's hole / the L-bracket's notch blank instead of showing
    the network extrapolated (or a spurious zero) inside the void.
    """
    if hasattr(sampler, 'L'):
        return np.ones_like(X, dtype=bool)
    if hasattr(sampler, 'W'):
        r = float(sampler.r.detach().cpu().item()) if torch.is_tensor(sampler.r) else float(sampler.r)
        return np.hypot(X, Y) >= r
    s = sampler.size
    inside = ~((X < s) & (Y > s))
    if getattr(sampler, 'fillet_enabled', False):
        f = float(sampler.fillet.detach().cpu().item())
        inside |= np.hypot(X - (s - f), Y - (s + f)) <= f
    return inside


def _route_and_forward(model, sampler, x_t: torch.Tensor, y_t: torch.Tensor):
    """    Single shared implementation of "which subdomain network owns this
    point" routing, used by every caller that needs (u, v) at arbitrary
    points: _evaluate_model_on_grid (no-grad, for plain field plots),
    plot_von_mises_stress (grad-tracked, for stress), and
    validation.evaluate_xpinn_stress_on_points (grad-tracked, for FEM
    stress comparison).
    """
    coords = torch.cat([x_t, y_t], dim=1)
    if getattr(model, 'self_routing', False):
        uv = model(coords)
        return uv[:, 0:1], uv[:, 1:2]
    N = coords.shape[0]
    device = x_t.device
    u = torch.zeros(N, 1, device=device)
    v = torch.zeros(N, 1, device=device)

    if hasattr(sampler, 'L'):  # Cantilever
        mid = sampler.L / 2
        mask0 = x_t.flatten() <= mid
        mask1 = ~mask0
        for m, idx in [(mask0, 0), (mask1, 1)]:
            if m.any():
                uv = model.get_subdomain_network(idx)(coords[m])
                u[m] = uv[:, 0:1]
                v[m] = uv[:, 1:2]

    elif hasattr(sampler, 'W'):  # Plate with hole
        r_mid = getattr(sampler, 'r_mid', 0.6)
        r_far = getattr(sampler, 'r_far', 1.2)
        dist = torch.sqrt(x_t**2 + y_t**2).flatten()
        mask0 = dist <= r_mid
        mask1 = (dist > r_mid) & (dist <= r_far)
        mask2 = dist > r_far
        for m, idx in [(mask0, 0), (mask1, 1), (mask2, 2)]:
            if m.any():
                uv = model.get_subdomain_network(idx)(coords[m])
                u[m] = uv[:, 0:1]
                v[m] = uv[:, 1:2]

    else:  # L-bracket
        s = sampler.size
        mask0 = (x_t.flatten() <= s) & (y_t.flatten() <= s)
        mask1 = (x_t.flatten() > s) & (y_t.flatten() <= s)
        mask2 = (x_t.flatten() >= s) & (y_t.flatten() > s)
        masks = [(mask0, 0), (mask1, 1), (mask2, 2)]
        if getattr(sampler, 'fillet_enabled', False):
            f = sampler.fillet.detach()
            Cx, Cy = s - f, s + f
            dist = torch.sqrt((x_t.flatten() - Cx)**2 + (y_t.flatten() - Cy)**2)
            already_covered = mask0 | mask1 | mask2
            mask_fillet = (dist <= f) & (~already_covered)
            masks.append((mask_fillet, 0))  # fillet patch belongs to sub0
        for m, idx in masks:
            if m.any():
                uv = model.get_subdomain_network(idx)(coords[m])
                u[m] = uv[:, 0:1]
                v[m] = uv[:, 1:2]

    return u, v


def _evaluate_model_on_grid(model, sampler, x_flat, y_flat, device):
    """    Evaluate the multi-subdomain XPINN correctly.
    Each point is sent to the network that owns that region.
    """
    x_t = torch.tensor(x_flat, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_flat, dtype=torch.float32, device=device)

    with torch.no_grad():
        u, v = _route_and_forward(model, sampler, x_t, y_t)

    return u.cpu().numpy().flatten(), v.cpu().numpy().flatten()


def evaluate_von_mises_on_points(model, physics, sampler, coords: np.ndarray, device: str = 'cpu'):
    """    Evaluate von Mises stress at arbitrary scattered (x,y) points -
    grad-tracked through _route_and_forward so strains/stresses can
    actually be computed, unlike _evaluate_model_on_grid (which is
    deliberately no-grad, for plain displacement plots). Used by
    validation.py to compare against a FEM stress reference field.
    """
    x_t = torch.tensor(coords[:, 0:1], dtype=torch.float32, device=device, requires_grad=True)
    y_t = torch.tensor(coords[:, 1:2], dtype=torch.float32, device=device, requires_grad=True)
    u, v = _route_and_forward(model, sampler, x_t, y_t)
    eps_xx, eps_yy, eps_xy = physics.compute_strains(u, v, x_t, y_t)
    sig_xx, sig_yy, sig_xy = physics.compute_stresses(eps_xx, eps_yy, eps_xy)
    von_mises = physics.compute_von_mises(sig_xx, sig_yy, sig_xy)
    return von_mises.detach().cpu().numpy().reshape(-1, 1)


def plot_fem_comparison(model, sampler, physics, benchmark: str,
                        data_dir: str = str(FEM_REFERENCE_DIR), use_mock: bool = True,
                        save_path: Optional[str] = None, r_excl: Optional[float] = None):
    """    The actual side-by-side ATLAS-vs-FEM comparison plot - until now,
    validation.py only ever printed relative L2 error NUMBERS to the
    console; nothing visualized what those numbers were computed from. This
    is the plot the paper's Visualization Module section calls for:
    "comparison of ATLAS and FEM results side by side."
    """
    from atlas.validation import (FEMValidator, sampler_geometry_dict, check_geometry_match,
                                  singular_locations, default_exclusion_radius,
                                  compute_von_mises_metrics)

    validator = FEMValidator(physics)
    fem_data = validator.load_fem_reference(benchmark, data_dir)
    used_mock = fem_data is None
    geometry_warning = None

    if fem_data is None:
        if not use_mock:
            raise FileNotFoundError(
                f"No FEM reference for '{benchmark}' in {data_dir}/ and use_mock=False - "
                "generate one with the fenics_scripts/*.py scripts, or pass use_mock=True."
            )
        fem_data = validator.generate_mock_fem_reference(sampler)
    else:
        current_geom = sampler_geometry_dict(sampler, benchmark)
        geometry_warning = check_geometry_match(fem_data.get('meta'), current_geom)

    coords = fem_data['coords']
    u_fem, v_fem = fem_data['u'], fem_data['v']
    vm_fem = fem_data.get('vonmises')
    u_pred, v_pred = validator.evaluate_xpinn_on_points(model, sampler, coords)

    disp_fem = np.sqrt(u_fem**2 + v_fem**2).flatten()
    disp_pred = np.sqrt(u_pred**2 + v_pred**2).flatten()
    abs_err = np.hypot(u_pred.flatten() - u_fem.flatten(), v_pred.flatten() - v_fem.flatten())
    # Label follows the model: one network over the whole domain vs domain-decomposed.
    name = 'ATLAS (single network)' if getattr(model, 'monolith', False) else 'ATLAS (XPINN)'
    rel_l2 = validator.compute_l2_error(disp_pred.reshape(-1, 1), disp_fem.reshape(-1, 1))

    x, y = coords[:, 0], coords[:, 1]
    vmax = max(disp_fem.max(), disp_pred.max(), 1e-30)

    show_stress = vm_fem is not None
    rel_l2_vm = None
    max_err_vm = None
    rel_l2_vm_excl = None
    vm_r_used = None
    vm_centers = []

    from matplotlib.tri import Triangulation
    tri = Triangulation(x, y)
    cen = tri.triangles
    cx, cy = x[cen].mean(axis=1), y[cen].mean(axis=1)
    tri.set_mask(~_inside_material(sampler, cx, cy))
    wide = hasattr(sampler, 'L')  # 4:1 beam needs far less height than the square domains
    fig, axes = plt.subplots(2 if show_stress else 1, 3,
                             figsize=(16, (5.5 if wide else 9) if show_stress else (3.2 if wide else 4.5)))
    row0 = axes[0] if show_stress else axes

    tpc0 = row0[0].tricontourf(tri, disp_pred, levels=np.linspace(0, vmax, 51), cmap='viridis')
    row0[0].set_title(f'{name} |displacement|')
    row0[0].set_aspect('equal')
    _colorbar(fig, tpc0, row0[0])

    tpc1 = row0[1].tricontourf(tri, disp_fem, levels=np.linspace(0, vmax, 51), cmap='viridis')
    row0[1].set_title('FEM (FEniCS)' + (' [MOCK DATA]' if used_mock else '') + ' |displacement|')
    row0[1].set_aspect('equal')
    _colorbar(fig, tpc1, row0[1])

    tpc2 = row0[2].tricontourf(tri, abs_err, levels=50, cmap='hot')
    row0[2].set_title('|ATLAS - FEM| vector error')
    row0[2].set_aspect('equal')
    _colorbar(fig, tpc2, row0[2])

    if show_stress:
        vm_pred = validator.evaluate_xpinn_vonmises_on_points(model, sampler, coords).flatten()
        vm_fem_flat = vm_fem.flatten()
        vm_abs_err = np.abs(vm_pred - vm_fem_flat)
        rel_l2_vm = validator.compute_l2_error(vm_pred.reshape(-1, 1), vm_fem.reshape(-1, 1))
        max_err_vm, max_idx_vm = validator.compute_max_pointwise_error(vm_pred.reshape(-1, 1), vm_fem.reshape(-1, 1))
        vm_max = max(vm_fem_flat.max(), vm_pred.max(), 1e-30)

        geom = sampler_geometry_dict(sampler, benchmark)
        vm_centers = [loc for _, loc in singular_locations(benchmark, geom)]
        vm_r_used = r_excl if r_excl is not None else default_exclusion_radius(coords, vm_centers)
        vm_metrics = compute_von_mises_metrics(vm_pred.reshape(-1, 1), vm_fem.reshape(-1, 1),
                                               coords, vm_centers, vm_r_used)
        rel_l2_vm_excl = vm_metrics['von_mises_l2_excl']

        row1 = axes[1]
        tpc3 = row1[0].tricontourf(tri, vm_pred, levels=np.linspace(0, vm_max, 51), cmap='hot')
        row1[0].set_title(f'{name} von Mises σ_vm')
        row1[0].set_aspect('equal')
        _colorbar(fig, tpc3, row1[0])

        tpc4 = row1[1].tricontourf(tri, vm_fem_flat, levels=np.linspace(0, vm_max, 51), cmap='hot')
        row1[1].set_title('FEM (FEniCS) von Mises σ_vm')
        row1[1].set_aspect('equal')
        _colorbar(fig, tpc4, row1[1])

        cap = max(0.1 * float(np.percentile(vm_fem_flat, 99.5)), 1e-30)
        tpc5 = row1[2].tricontourf(tri, np.minimum(vm_abs_err, cap),
                                   levels=np.linspace(0.0, cap, 51), cmap='hot', extend='max')
        row1[2].scatter([x[max_idx_vm]], [y[max_idx_vm]], marker='x', color='cyan', s=80, linewidths=2,
                        label=f'max error: {max_err_vm:.2e} Pa')
        row1[2].legend(loc='upper right', fontsize=8)
        row1[2].set_title('|ATLAS - FEM| von Mises pointwise error\n(scale capped at 10% of the FEM 99.5th-percentile stress)', fontsize=9)
        row1[2].set_aspect('equal')
        _colorbar(fig, tpc5, row1[2])

        if vm_r_used is not None and vm_centers:
            for cx, cy in vm_centers:
                circ = plt.Circle((cx, cy), vm_r_used, fill=False, linestyle='--',
                                  color='lime', linewidth=1.3)
                row1[2].add_patch(circ)
                row1[2].scatter([cx], [cy], marker='o', facecolors='none',
                                edgecolors='lime', s=40, linewidths=1.3)
            row1[2].text(0.02, 0.02, f'excluded region r_excl={vm_r_used:.4g}',
                         transform=row1[2].transAxes, fontsize=7, color='lime')
        row1[2].set_xlim(x.min(), x.max())
        row1[2].set_ylim(y.min(), y.max())

    meaningful = (not used_mock) and (geometry_warning is None)
    passed_5pct = (rel_l2 < 0.05) if meaningful else None

    title = f'{benchmark} — relative L2 error (|displacement|): {rel_l2:.2%}'
    if passed_5pct is True:
        title += '  [PASS: <5% target]'
    elif passed_5pct is False:
        title += '  [FAIL: >=5% target]'
    stress_fail = False
    if show_stress:
        title += f'  |  (von Mises): global {rel_l2_vm:.2%}'
        if rel_l2_vm_excl is not None and vm_r_used is not None:
            title += f' | excl.(r={vm_r_used:.4g}): {rel_l2_vm_excl:.2%}'
            if meaningful:
                stress_pass = rel_l2_vm_excl < 0.10
                stress_fail = not stress_pass
                title += f'  [STRESS <10% excl: {"PASS" if stress_pass else "FAIL"}]'
    if used_mock:
        title += '  [WARNING: comparing against MOCK FEM data, not a real solve]'
    elif geometry_warning:
        title += '  [WARNING: FEM reference geometry mismatch - see log]'
    fig.suptitle(title, color=('red' if (used_mock or geometry_warning or passed_5pct is False or stress_fail) else 'black'))

    for ax_ in fig.axes:                       # keep titles clear of the colour-bar exponent text
        if ax_.get_title():
            ax_.set_title(ax_.get_title(), pad=16, fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    _show_if_interactive()
    plt.close(fig)
    return {
        'relative_L2_displacement': rel_l2,
        'displacement_within_5pct_target': passed_5pct,
        'relative_L2_von_mises': rel_l2_vm,
        'von_mises_l2_global': rel_l2_vm,
        'von_mises_l2_excl': rel_l2_vm_excl,
        'von_mises_exclusion_radius': vm_r_used,
        'max_pointwise_von_mises_error': max_err_vm,
        'used_mock': used_mock,
        'geometry_warning': geometry_warning,
    }


def plot_displacement_field(model, sampler, physics, n_grid: int = 100,
                            save_path: Optional[str] = None):
    if hasattr(sampler, 'L'):
        x = np.linspace(0, sampler.L, n_grid)
        y = np.linspace(0, sampler.H, n_grid)
    elif hasattr(sampler, 'W'):
        x = np.linspace(0, sampler.W, n_grid)
        y = np.linspace(0, sampler.H, n_grid)
    else:
        x = np.linspace(0, 2*sampler.size, n_grid)
        y = np.linspace(0, 2*sampler.size, n_grid)

    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten().reshape(-1, 1)
    y_flat = Y.flatten().reshape(-1, 1)

    device = next(model.parameters()).device
    u, v = _evaluate_model_on_grid(model, sampler, x_flat, y_flat, device)

    void = ~_inside_material(sampler, X, Y)
    disp_mag = np.ma.masked_where(void, np.sqrt(u**2 + v**2).reshape(X.shape))
    u = np.ma.masked_where(void, u.reshape(X.shape))
    v = np.ma.masked_where(void, v.reshape(X.shape))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].contourf(X, Y, disp_mag, levels=50, cmap='viridis')
    axes[0].set_title('Displacement Magnitude |u| [m]')
    axes[0].set_aspect('equal')
    _colorbar(fig, im0, axes[0])

    im1 = axes[1].contourf(X, Y, u, levels=50, cmap='RdBu_r')
    axes[1].set_title('u displacement [m]')
    axes[1].set_aspect('equal')
    _colorbar(fig, im1, axes[1])

    im2 = axes[2].contourf(X, Y, v, levels=50, cmap='RdBu_r')
    axes[2].set_title('v displacement [m]')
    axes[2].set_aspect('equal')
    _colorbar(fig, im2, axes[2])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    _show_if_interactive()
    plt.close(fig)  # avoid leaking figures across repeated/headless (server) calls


def plot_von_mises_stress(model, sampler, physics, n_grid: int = 100,
                          save_path: Optional[str] = None):
    if hasattr(sampler, 'L'):
        x = np.linspace(0.01, sampler.L, n_grid)
        y = np.linspace(0.01, sampler.H, n_grid)
    elif hasattr(sampler, 'W'):
        x = np.linspace(0.01, sampler.W, n_grid)
        y = np.linspace(0.01, sampler.H, n_grid)
    else:
        x = np.linspace(0.01, 2*sampler.size, n_grid)
        y = np.linspace(0.01, 2*sampler.size, n_grid)

    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten().reshape(-1, 1)
    y_flat = Y.flatten().reshape(-1, 1)

    device = next(model.parameters()).device
    x_t = torch.tensor(x_flat, dtype=torch.float32, device=device, requires_grad=True)
    y_t = torch.tensor(y_flat, dtype=torch.float32, device=device, requires_grad=True)

    u, v = _route_and_forward(model, sampler, x_t, y_t)

    eps_xx, eps_yy, eps_xy = physics.compute_strains(u, v, x_t, y_t)
    sig_xx, sig_yy, sig_xy = physics.compute_stresses(eps_xx, eps_yy, eps_xy)
    von_mises = physics.compute_von_mises(sig_xx, sig_yy, sig_xy)
    VM = np.ma.masked_where(~_inside_material(sampler, X, Y),
                            von_mises.detach().cpu().numpy().reshape(X.shape))

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.contourf(X, Y, VM, levels=50, cmap='hot')
    ax.set_title('Von Mises Stress σ_vm')
    ax.set_aspect('equal')
    _colorbar(fig, im, ax, label='Stress [Pa]')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    _show_if_interactive()
    plt.close(fig)  # avoid leaking figures across repeated/headless (server) calls


def plot_training_convergence(history: Dict[str, List[float]],
                              save_path: Optional[str] = None,
                              adam_epoch_count: Optional[int] = None):
    """    adam_epoch_count: index into history['total'] where the Adam stage ends
    and L-BFGS fine-tuning begins (XPINNTrainer.adam_epoch_count after a
    two-stage train() call). Draws a vertical marker there so the plot makes
    the Adam -> L-BFGS handoff visible instead of looking like one
    homogeneous run. Pass None (default) to skip the marker, e.g. for an
    Adam-only run (use_lbfgs=False).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    energy_hist = 'iter' in history and len(history.get('strain', [])) > 0
    x_axis = history['iter'] if energy_hist else np.arange(len(history['total']))
    axes[0].semilogy(x_axis, history['total'], 'b-', linewidth=0.8)
    axes[0].set_xlabel('Iteration' if energy_hist else 'Epoch')
    axes[0].set_ylabel('Relative energy gap to final value' if energy_hist else 'Total Loss')
    axes[0].set_title('Training Convergence')
    axes[0].grid(True, alpha=0.3)

    if energy_hist:
        axes[1].plot(x_axis, history['strain'], color='green', label='strain energy U', linewidth=0.8)
        axes[1].plot(x_axis, history['work'], color='red', label='external work W', linewidth=0.8)
        axes[1].set_ylabel('Normalized energy')
        axes[1].set_title('Energy Components (U = W/2 at equilibrium)')
        axes[1].set_xlabel('Iteration')
        # The untrained network's first iterations have a huge strain energy;
        # scale the axis to the converged values so U and W stay readable.
        top = 1.6 * max(abs(history['strain'][-1]), abs(history['work'][-1]), 1e-30)
        axes[1].set_ylim(0.0, top)
    else:
        colors = {'pde': 'blue', 'dirichlet': 'green', 'neumann': 'red',
                  'interface_u': 'orange', 'interface_t': 'purple'}
        for key, color in colors.items():
            if key in history and len(history[key]) > 0:
                axes[1].semilogy(history[key], color=color, label=key, linewidth=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Components')
    axes[1].grid(True, alpha=0.3)

    n_total = len(history['total'])
    if adam_epoch_count is not None and 0 < adam_epoch_count < (x_axis[-1] if energy_hist else n_total):
        for ax in axes:
            ax.axvline(adam_epoch_count, color='black', linestyle='--',
                       linewidth=1.0, alpha=0.6)
        axes[0].text(adam_epoch_count, axes[0].get_ylim()[1], ' L-BFGS \u2192',
                    va='top', ha='left', fontsize=8, alpha=0.75)

    axes[1].legend(loc='center right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    _show_if_interactive()
    plt.close(fig)  # avoid leaking figures across repeated/headless (server) calls


def plot_deformed_shape(model, sampler, scale: Optional[float] = None,
                        target_fraction: float = 0.2,
                        n_grid: int = 50, save_path: Optional[str] = None):
    """    scale: exaggeration factor for the deformed overlay. Pass None (default)
    to auto-compute one from the actual predicted displacement magnitude, so
    the deformed shape always renders at a legible size regardless of how
    large or small the true physical displacements happen to be for this
    benchmark/load - a fixed magic-number scale (the old default was a
    hardcoded 1e6) looks fine by accident for whatever displacement range it
    was tuned against, but silently produces a nonsensical, wildly
    exaggerated shape once training actually converges to correct,
    physically-small displacements. Auto mode targets a deformed extent of
    `target_fraction` (default 20%) of the domain's largest dimension -
    visibly exaggerated (a real structure wouldn't bend the exact same
    material) but still recognizably beam/plate/bracket-shaped. Pass an
    explicit float to override auto mode (e.g. for a fixed scale across a
    side-by-side comparison of several runs).
    """
    if hasattr(sampler, 'L'):
        x = np.linspace(0, sampler.L, n_grid)
        y = np.linspace(0, sampler.H, n_grid)
        domain_extent = max(sampler.L, sampler.H)
    elif hasattr(sampler, 'W'):
        x = np.linspace(0, sampler.W, n_grid)
        y = np.linspace(0, sampler.H, n_grid)
        domain_extent = max(sampler.W, sampler.H)
    else:
        x = np.linspace(0, 2*sampler.size, n_grid)
        y = np.linspace(0, 2*sampler.size, n_grid)
        domain_extent = 2 * sampler.size

    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten().reshape(-1, 1)
    y_flat = Y.flatten().reshape(-1, 1)

    device = next(model.parameters()).device
    u, v = _evaluate_model_on_grid(model, sampler, x_flat, y_flat, device)
    u = u.reshape(X.shape)
    v = v.reshape(X.shape)

    # Only material points count: the plate's hole / the L-bracket's notch have
    # no displacement, and grid lines must not be drawn across them.
    inside = _inside_material(sampler, X, Y)
    u = np.where(inside, u, 0.0)
    v = np.where(inside, v, 0.0)

    if scale is None:
        max_disp = max(np.abs(u).max(), np.abs(v).max())
        scale = (target_fraction * domain_extent) / max_disp if max_disp > 1e-30 else 1.0

    X_def = X + scale * u
    Y_def = Y + scale * v

    def _segments(PX, PY):
        # a grid segment is drawn only if BOTH of its end points are material
        segs = []
        for (a, b) in [((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))]:
            ok = (inside[a] & inside[b]).ravel()
            p0 = np.stack([PX[a].ravel(), PY[a].ravel()], axis=1)[ok]
            p1 = np.stack([PX[b].ravel(), PY[b].ravel()], axis=1)[ok]
            segs.append(np.stack([p0, p1], axis=1))
        return np.concatenate(segs, axis=0)

    from matplotlib.collections import LineCollection
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.add_collection(LineCollection(_segments(X, Y), colors='b', alpha=0.3, linewidths=0.5))
    ax.add_collection(LineCollection(_segments(X_def, Y_def), colors='r', alpha=0.6, linewidths=0.5))
    ax.autoscale()
    ax.margins(0.03)

    ax.set_title(f'Deformed Shape (scale = {scale:.3g})')
    ax.set_aspect('equal')
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0], [0], color='b', alpha=0.4), Line2D([0], [0], color='r', alpha=0.7)],
              ['Original', 'Deformed'], loc='upper right')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    _show_if_interactive()
    plt.close(fig)  # avoid leaking figures across repeated/headless (server) calls