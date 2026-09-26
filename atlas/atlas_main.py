"""Run from the project root:
    python -m atlas.atlas_main
then pick a benchmark and an action when prompted.
"""

import torch
import numpy as np
import os
from typing import Tuple, Optional, Dict, List

from atlas.physics import PlaneStressElasticity
from atlas.networks import create_xpinn_for_benchmark, default_fourier_freqs
from atlas.sampling import CantileverBeamSampler, LBracketSampler, PlateWithHoleSampler
from atlas.loss_functions import XPINNTrainer
from atlas.inverse_design import InverseDesignOptimizer, InverseDesignConfig
from atlas.validation import run_validation
from atlas.visualization import (
    plot_displacement_field,
    plot_von_mises_stress,
    plot_training_convergence,
    plot_deformed_shape,
    plot_fem_comparison
)
from atlas.paths import RESULTS_DIR


def setup_device(seed: int = 42) -> str:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed_all(seed)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    return device


def _eb_prefit_cantilever(model, sampler, E: float, P: float,
                          device: str, epochs: int = 8000,
                          n_points: int = 30000, lr: float = 1e-3,
                          seed: int = 0) -> Tuple[float, float]:
    L, H = sampler.L, sampler.H
    I_c = H ** 3 / 12.0

    def eb_uv(x, y):
        eta = y - H / 2.0
        return (-P * eta * x * (2 * L - x) / (2.0 * E * I_c),
                P * x ** 2 * (3 * L - x) / (6.0 * E * I_c))

    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, L, (n_points, 1))
    ys = rng.uniform(0, H, (n_points, 1))
    u_t, v_t = eb_uv(xs, ys)
    coords = torch.tensor(np.hstack([xs, ys]), dtype=torch.float32, device=device)
    target = torch.tensor(np.hstack([u_t, v_t]), dtype=torch.float32, device=device)

    net = model.get_subdomain_network(0)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    last = 0.0
    for i in range(epochs):
        optim.zero_grad()
        loss = torch.mean((net(coords) - target) ** 2)
        loss.backward()
        optim.step()
        last = float(loss.item())
        if (i + 1) % 2000 == 0:
            print(f"  [prefit] epoch {i+1}: loss {last:.3e}")

    ytip = torch.linspace(0.15, 0.85, 5, device=device).reshape(-1, 1)
    xtip = torch.full_like(ytip, L)
    with torch.no_grad():
        uv = net(torch.cat([xtip, ytip], dim=1))
        tip_mm = float(uv[:, 1].mean() * 1000.0)
    print(f"  [prefit] tip deflection: {tip_mm:+.4f} mm (EB target: "
          f"{(P*L**3/(3*E*I_c))*1000:+.4f} mm)")
    return last, tip_mm


def _kirsch_region_points(sampler, i: int, n: int, seed: int = 0):
    sampler._update_subdomain_radii()
    rng = np.random.default_rng(seed + i)
    a = float(sampler.r.detach().cpu().item())
    W, H = sampler.W, sampler.H
    if i == 0:
        rho = np.sqrt(a ** 2 + rng.uniform(0, 1, n) * (sampler.r_mid ** 2 - a ** 2))
        theta = rng.uniform(0, np.pi / 2, n)
    elif i == 1:
        rho = np.sqrt(rng.uniform(sampler.r_mid ** 2, sampler.r_far ** 2, n))
        theta = rng.uniform(0, np.pi / 2, n)
    elif i == 2:
        pts = []
        while len(pts) < n:
            xc = rng.uniform(0, W, n)
            yc = rng.uniform(0, H, n)
            valid = np.hypot(xc, yc) > sampler.r_far
            pts.extend(np.stack([xc[valid], yc[valid]], axis=1).tolist())
        pts = np.array(pts[:n])
        return {'x': pts[:, 0], 'y': pts[:, 1]}
    else:
        raise ValueError(f"plate_hole has 3 subdomains, got {i}")
    return {'x': rho * np.cos(theta), 'y': rho * np.sin(theta)}


def _kirsch_prefit_plate(model, sampler, E: float, nu: float, sigma_inf: float,
                         device: str, lr: float = 1e-3, n_points: int = 4000,
                         min_epochs: int = 2000, max_epochs: int = 20000,
                         tol: float = 1e-4, patience: int = 200) -> Tuple[list, int]:
    a = float(sampler.r.detach().cpu().item())

    def kirsch_uv(x, y):
        r = np.hypot(x, y)
        th = np.arctan2(y, x)
        c2 = np.cos(2 * th)
        s2 = np.sin(2 * th)
        favg = sigma_inf / (2.0 * E)
        ur = favg * ((1.0 - nu) * r + (1.0 + nu) * a ** 2 / r
                     + ((1.0 + nu) * r + 4.0 * a ** 2 / r
                        - (1.0 + nu) * a ** 4 / r ** 3) * c2)
        uth = (sigma_inf / E) * (-(1.0 + nu) * r + 2.0 * (nu - 1.0) * a ** 2 / r
                                 - (1.0 + nu) * a ** 4 / r ** 3) * s2
        u = ur * np.cos(th) - uth * np.sin(th)
        v = ur * np.sin(th) + uth * np.cos(th)
        return u, v

    print("=" * 70)
    print("WARM START - Kirsch analytic prefit (per-subdomain regions)")
    print("=" * 70)
    print(f"  hole radius a={a:.4f}, sigma_inf={sigma_inf:.3e} Pa (real E={E:.3e})")

    losses_done = []
    epochs_used = 0
    for i in range(model.num_subdomains):
        pts = _kirsch_region_points(sampler, i, n_points, seed=0)
        u_t, v_t = kirsch_uv(pts['x'], pts['y'])
        coords = torch.tensor(np.stack([pts['x'], pts['y']], axis=1),
                              dtype=torch.float32, device=device)
        target = torch.tensor(np.stack([u_t, v_t], axis=1),
                              dtype=torch.float32, device=device)

        net = model.get_subdomain_network(i)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        ema = None
        stall = 0
        for ep in range(max_epochs):
            optim.zero_grad()
            loss = torch.mean((net(coords) - target) ** 2)
            loss.backward()
            optim.step()
            val = float(loss.item())
            if not np.isfinite(val):
                raise FloatingPointError(
                    f"Kirsch prefit subdomain {i} diverged (loss={val:.3e})")
            ema_old = ema
            ema = val if ema_old is None else 0.99 * ema_old + 0.01 * val
            if ema_old is not None:
                rel = (ema_old - ema) / ema_old if ema_old != 0.0 else 0.0
                stall = stall + 1 if rel < tol else 0
            if (ep + 1) >= min_epochs and stall >= patience:
                break
        losses_done.append(val)
        epochs_used = max(epochs_used, ep + 1)
        print(f"  [prefit] subdomain {i}: epochs {ep + 1}, final loss {val:.3e}")

    print(f"  [prefit] done: {epochs_used} epochs, losses={losses_done}")
    return losses_done, epochs_used


def _loss_scales(u_s: float, l_char: float, e_train: float, tau_ref: float) -> Dict[str, float]:
    sigma_s = e_train * u_s / l_char
    r_s = sigma_s / l_char
    energy_s = 0.5 * sigma_s * (u_s / l_char)
    return {
        'pde': r_s ** 2,
        'dirichlet': u_s ** 2,
        'neumann': tau_ref ** 2,
        'interface_u': u_s ** 2,
        'interface_t': sigma_s ** 2,
        'energy': energy_s,
    }


def _plate_hole_essential_mask(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return x[:, 0:1], x[:, 1:2]


def _plate_hole_energy_context(sampler, sigma_inf: float, device: str,
                               spacing: float = 1.0 / 16.0,
                               face_frac: float = 0.5):
    a = float(sampler.r.detach().cpu().item())
    W, H = sampler.W, sampler.H
    h = float(spacing)

    xs = (np.arange(int(W / h)) + 0.5) * h
    ys = (np.arange(int(H / h)) + 0.5) * h
    gm = np.meshgrid(xs, ys, indexing='ij')
    gx, gy = gm[0].reshape(-1), gm[1].reshape(-1)
    keep = np.hypot(gx, gy) >= a
    gx, gy = gx[keep], gy[keep]
    area_g = np.full(gx.shape[0], h * h, dtype=np.float32)

    fy = ys.copy()
    fx = np.full_like(fy, W)
    strip_area = face_frac * h * H
    area_f = np.full(fy.shape[0], strip_area / fy.shape[0], dtype=np.float32)

    xg = sampler.to_tensor(gx.astype(np.float32)).reshape(-1, 1)
    yg = sampler.to_tensor(gy.astype(np.float32)).reshape(-1, 1)
    wg = torch.tensor(area_g, dtype=torch.float32, device=device).reshape(-1, 1)
    xf = sampler.to_tensor(fx.astype(np.float32)).reshape(-1, 1)
    yf = sampler.to_tensor(fy.astype(np.float32)).reshape(-1, 1)
    wf = torch.tensor(area_f, dtype=torch.float32, device=device).reshape(-1, 1)

    interior = [
        {'x': xg, 'y': yg, 'area': wg},
        {'x': xf, 'y': yf, 'area': wf},
    ]

    y_edge = np.linspace(0.0, H, fy.shape[0]).reshape(-1, 1)
    loaded_edges = [{
        'x': sampler.to_tensor(np.full_like(y_edge, W)),
        'y': sampler.to_tensor(y_edge),
        't_x': sampler.to_tensor(np.full_like(y_edge, sigma_inf)),
        't_y': sampler.to_tensor(np.zeros_like(y_edge)),
        'length': float(H),
    }]
    return interior, loaded_edges


def train_forward_solver(
    benchmark: str,
    device: str,
    epochs: int = 5000,
    print_every: int = 500,
    n_interior: int = 800,
    n_boundary: int = 150,
    n_interface: int = 80,
    stress_scale: float = 1e11,
    use_lbfgs: bool = True,
    lbfgs_steps: int = 25,
    lbfgs_max_iter_per_step: int = 20,
    l_bracket_fillet_init: Optional[float] = None,
    hidden_layers: int = 4,
    neurons: int = 64,
    warm_start_eb: bool = False,
    prefit_epochs: int = 8000,
    warm_start_kirsch: bool = False,
    adaptive_weights: bool = False,
    adapt_every: int = 100,
    use_mpe: bool = False,
    use_singular_features: bool = False,
    cantilever_anchor_weight: float = 3000.0,
    formulation: str = 'energy',
    seed: Optional[int] = 42,
    standard_pinn: bool = False,
    probe=None,
    probe_every: int = 100,
) -> Tuple:
    if formulation == 'energy' and l_bracket_fillet_init is not None:
        print("[Info] fillet geometry is only supported by the residual "
              "formulation; falling back to formulation='residual'.")
        formulation = 'residual'
    if formulation == 'energy':
        from atlas.energy_solver import train_energy_forward_solver
        return train_energy_forward_solver(
            benchmark, device=device, epochs=epochs,
            lbfgs_iters=(lbfgs_steps * lbfgs_max_iter_per_step) if use_lbfgs else 0,
            lbfgs_chunk=lbfgs_max_iter_per_step, print_every=print_every,
            hidden_layers=hidden_layers, neurons=neurons,
            stress_scale=stress_scale, seed=seed)
    if formulation == 'energy_xpinn':
        from atlas.energy_xpinn import train_energy_xpinn_forward_solver
        return train_energy_xpinn_forward_solver(
            benchmark, device=device, epochs=epochs,
            lbfgs_iters=(lbfgs_steps * lbfgs_max_iter_per_step) if use_lbfgs else 0,
            hidden_layers=hidden_layers, neurons=neurons, seed=seed)
    if formulation != 'residual':
        raise ValueError(f"Unknown formulation: {formulation!r}")
    if standard_pinn and (warm_start_eb or use_singular_features or
                          warm_start_kirsch or use_mpe):
        raise ValueError("standard_pinn is the plain single-network baseline; it "
                         "cannot be combined with warm_start_eb / use_singular_features "
                         "/ warm_start_kirsch / use_mpe")
    if standard_pinn:
        import math as _math
        n_sub = {'cantilever': 2, 'l_bracket': 3, 'plate_hole': 3}.get(benchmark, 3)
        neurons = int(round(neurons * _math.sqrt(n_sub)))

    print("=" * 70)
    print(f"FORWARD SOLVE — {benchmark.upper()}")
    print("=" * 70)

    E = 210e9
    nu = 0.3
    physics_real = PlaneStressElasticity(E=E, nu=nu)
    physics_train = PlaneStressElasticity(E=E / stress_scale, nu=nu)

    u_s = None
    resultants = None
    trainer_anchor = None

    if use_mpe and benchmark != 'plate_hole':
        raise ValueError(
            "use_mpe is currently only implemented for benchmark='plate_hole'")
    if use_mpe and warm_start_kirsch:
        raise ValueError("use_mpe and warm_start_kirsch are mutually exclusive "
                         "(the residual-based Kirsch prefit is meaningless under "
                         "a potential-energy objective)")

    if benchmark == 'cantilever':
        sampler = CantileverBeamSampler(
            L=4.0, H=1.0,
            n_interior=n_interior, n_boundary=n_boundary,
            n_interface=n_interface, device=device
        )
        I_c = sampler.H ** 3 / 12.0
        u_s = (1e6 * sampler.L ** 3) / (3.0 * E * I_c)
        model = create_xpinn_for_benchmark('cantilever', hidden_layers=hidden_layers, neurons=neurons, device=device,
                                           output_scale=u_s, monolith=(warm_start_eb or standard_pinn))
        P = -1e6 / stress_scale
        subdomain_data = sampler.sample_interior()
        boundary_data = sampler.sample_boundary(P=P)
        interface_data = sampler.sample_interface()
        loss_weights = {
            'pde': 1.0, 'dirichlet': 100.0, 'neumann': 10.0,
            'interface_u': 10.0, 'interface_t': 10.0
        }
        tau_ref = np.sqrt(1.2) * abs(P) / sampler.H
        loss_scales = _loss_scales(u_s, sampler.L, physics_train.E, tau_ref)
        y_w = np.linspace(0, sampler.H, sampler.n_boundary).reshape(-1, 1)
        resultants = [{
            'subdomain': 0,
            'x': sampler.to_tensor(np.zeros_like(y_w)),
            'y': sampler.to_tensor(y_w),
            'y_center': sampler.H / 2.0,
            'targets': {
                'sxy': (P / sampler.H, tau_ref),
                'moment': (P * sampler.L, tau_ref * sampler.L),
            }
        }]
        if warm_start_eb:
            print("=" * 70)
            print("WARM START — EB/Timoshenko prefit (monolith network)")
            print("=" * 70)
            _eb_prefit_cantilever(model, sampler, E=E, P=-1e6,
                                  device=device, epochs=prefit_epochs)
            loss_weights = {
                'pde': 10.0, 'dirichlet': 100.0, 'neumann': 10.0,
                'interface_u': 0.0, 'interface_t': 0.0
            }
            resultants[0]['weight'] = 100.0
            I_a = sampler.H ** 3 / 12.0
            rng_a = np.random.default_rng(7)
            xa = rng_a.uniform(0.1, sampler.L - 0.05, (300, 1))
            ya = rng_a.uniform(0, sampler.H, (300, 1))
            eta_a = ya - sampler.H / 2.0
            ua = -(-1e6) * eta_a * xa * (2 * sampler.L - xa) / (2.0 * E * I_a)
            va = (-1e6) * xa ** 2 * (3 * sampler.L - xa) / (6.0 * E * I_a)
            trainer_anchor = (
                sampler.to_tensor(np.hstack([xa, ya])),
                sampler.to_tensor(np.hstack([ua, va])),
                cantilever_anchor_weight,
            )
        else:
            trainer_anchor = None

    elif benchmark == 'l_bracket':
        sampler = LBracketSampler(
            size=1.0,
            n_interior=n_interior, n_boundary=n_boundary,
            n_interface=n_interface, device=device,
            fillet_init=l_bracket_fillet_init,
        )
        I_c = sampler.size ** 3 / 12.0
        u_s = (1e6 * sampler.size ** 3) / (3.0 * E * I_c)
        model = create_xpinn_for_benchmark('l_bracket', hidden_layers=hidden_layers, neurons=neurons, device=device,
                                           output_scale=u_s, monolith=standard_pinn)
        if use_singular_features:
            from atlas.networks import SubdomainNetwork
            for idx in range(3):
                model.subnets[idx] = SubdomainNetwork(
                    hidden_layers=hidden_layers, neurons_per_layer=neurons,
                    output_scale=u_s,
                    singular_center=(sampler.size, sampler.size),
                    singular_harmonics=2,
                ).to(device)
        P = 1e6 / stress_scale
        subdomain_data = sampler.sample_interior()
        boundary_data = sampler.sample_boundary(P=P)
        interface_data = sampler.sample_interface()
        loss_weights = {
            'pde': 1.0, 'dirichlet': 100.0, 'neumann': 10.0,
            'interface_u': 10.0, 'interface_t': 20.0
        }
        I_c = sampler.size ** 3 / 12.0
        u_s = (1e6 * sampler.size ** 3) / (3.0 * E * I_c)
        tau_ref = np.sqrt(1.2) * abs(P)
        loss_scales = _loss_scales(u_s, sampler.size, physics_train.E, tau_ref)
        yc = sampler.size / 2.0
        loaded_face = max(
            (b for b in boundary_data
             if b.get('type') == 'neumann' and b.get('subdomain') == 2
             and 't_x' in b),
            key=lambda b: float(b['t_x'].abs().max().item()),
        )
        applied_moment = torch.mean(
            (loaded_face['y'] - yc) * loaded_face['t_x']
        ).item()
        y_r = np.linspace(0.0, sampler.size,
                          sampler.n_boundary).reshape(-1, 1)
        resultants = [{
            'subdomain': 0,
            'x': sampler.to_tensor(np.zeros_like(y_r)),
            'y': sampler.to_tensor(y_r),
            'y_center': yc,
            'targets': {
                'sxx': (-P, tau_ref),
                'moment': (applied_moment, tau_ref * sampler.size),
            },
            'weight': 100.0,
        }]

    elif benchmark == 'plate_hole':
        sampler = PlateWithHoleSampler(
            W=2.0, H=2.0, r_init=0.3,
            n_interior=n_interior, n_boundary=n_boundary,
            n_interface=n_interface, device=device
        )
        u_s = (1e6 * sampler.W) / E
        sigma_inf = 1e6 / stress_scale
        if use_mpe:
            loss_weights = {
                'pde': 0.0, 'dirichlet': 0.0, 'neumann': 0.0,
                'interface_u': 0.0, 'interface_t': 0.0,
            }
            tau_ref = abs(sigma_inf)
            loss_scales = _loss_scales(u_s, sampler.W, physics_train.E, tau_ref)
            model = create_xpinn_for_benchmark(
                'plate_hole', hidden_layers=hidden_layers, neurons=neurons,
                device=device, output_scale=u_s, monolith=True,
                essential_mask=_plate_hole_essential_mask,
                fourier_freqs=default_fourier_freqs(sampler.W),
            )
            subdomain_data = sampler.sample_interior()
            boundary_data = sampler.sample_boundary(sigma_inf=sigma_inf)
            interface_data = sampler.sample_interface()
            resultants = None
        else:
            model = create_xpinn_for_benchmark('plate_hole', hidden_layers=hidden_layers, neurons=neurons, device=device,
                                               output_scale=u_s, monolith=standard_pinn)
            if use_singular_features:
                from atlas.networks import SubdomainNetwork
                model.subnets[0] = SubdomainNetwork(
                    hidden_layers=hidden_layers, neurons_per_layer=neurons,
                    output_scale=u_s, singular_center=(0.0, 0.0),
                    singular_harmonics=2,
                ).to(device)
            subdomain_data = sampler.sample_interior()
            boundary_data = sampler.sample_boundary(sigma_inf=sigma_inf)
            interface_data = sampler.sample_interface()
            loss_weights = {
                'pde': 1.0, 'dirichlet': 100.0, 'neumann': 10.0,
                'interface_u': 10.0, 'interface_t': 20.0
            }
            tau_ref = abs(sigma_inf)
            loss_scales = _loss_scales(u_s, sampler.W, physics_train.E, tau_ref)
            y_r = np.linspace(0.0, sampler.H, sampler.n_boundary).reshape(-1, 1)
            resultants = [{
                'subdomain': 2,
                'x': sampler.to_tensor(np.full_like(y_r, sampler.W)),
                'y': sampler.to_tensor(y_r),
                'targets': {'sxx': (sigma_inf, tau_ref)},
            }]
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    if warm_start_kirsch:
        if benchmark != 'plate_hole':
            raise ValueError("warm_start_kirsch is only valid for benchmark='plate_hole'")
        _kirsch_prefit_plate(model, sampler, E=E, nu=nu, sigma_inf=1e6,
                             device=device)

    trainer = XPINNTrainer(
        xpinn_model=model,
        physics=physics_train,
        learning_rate=1e-3,
        loss_weights=loss_weights,
        loss_scales=loss_scales
    )

    if use_mpe:
        trainer.loss_fn.energy_context = _plate_hole_energy_context(
            sampler, sigma_inf, device)
        print("=" * 70)
        print("MPE OBJECTIVE — minimizing signed total potential energy "
              "PI = 1/2 int(sig:eps) - int(t.u)")
        interior_ctx = trainer.loss_fn.energy_context[0]
        desc = ', '.join("%dpts+%.4fA" % (int(p['x'].shape[0]), float(p['area'].sum())) for p in interior_ctx)
        print(f"  structured quadrature (per-point weights): "
              f"{len(interior_ctx)} panels, weights=[{desc}]")
        print(f"  loaded edge x=W length: "
              f"{trainer.loss_fn.energy_context[1][0]['length']}")
        print("=" * 70)

    if probe is not None:
        trainer.probe = lambda it: probe(it, model, sampler)
        trainer.probe_every = probe_every
    if resultants is not None:
        trainer.loss_fn.resultants = resultants
    if benchmark == 'cantilever' and warm_start_eb:
        trainer.loss_fn.anchor_data = trainer_anchor

    trainer.train(
        subdomain_data=subdomain_data,
        interface_data=interface_data,
        boundary_data=boundary_data,
        epochs=epochs,
        print_every=print_every,
        use_lbfgs=use_lbfgs,
        lbfgs_steps=lbfgs_steps,
        lbfgs_max_iter_per_step=lbfgs_max_iter_per_step,
        adaptive_weights=adaptive_weights,
        adapt_every=adapt_every,
    )

    return model, trainer, sampler, physics_real


def run_inverse_design_on_plate(model, sampler, physics, device: str,
                                n_outer: int = 12, n_inner: int = 300,
                                stress_scale: float = 1e11):
    print("=" * 70)
    print("INVERSE DESIGN — PLATE WITH HOLE")
    print("=" * 70)

    from atlas.loss_functions import XPINNLoss
    from atlas.physics import PlaneStressElasticity
    from atlas.inverse_design import hole_region_sensitivity_objective

    physics_train = PlaneStressElasticity(E=physics.E / stress_scale, nu=physics.nu)
    sigma_inf_scaled = 1e6 / stress_scale

    u_s = (1e6 * sampler.W) / physics.E
    loss_scales = _loss_scales(u_s, sampler.W, physics_train.E, abs(1e6 / stress_scale))

    config = InverseDesignConfig(
        initial_radius=float(sampler.r.item()),
        min_radius=0.1,
        max_radius=0.8,
        lr_geometry=1e-3,
        lr_network=1e-3,
        n_outer_iterations=n_outer,
        n_inner_iterations=n_inner
    )

    loss_fn = XPINNLoss(physics_train, scales=loss_scales)

    from atlas.energy_solver import EnergyNet, make_plate_energy_inner_solver
    energy_net = model.get_subdomain_network(0)
    use_energy_inner = isinstance(energy_net, EnergyNet)
    inner_solve_fn = None
    if use_energy_inner:
        inner_solve_fn = make_plate_energy_inner_solver(
            energy_net, physics_train, sampler, sigma_inf_scaled, u_s,
            lbfgs_iters=n_inner, device=device)

    inverse_optimizer = InverseDesignOptimizer(
        xpinn_model=model,
        physics=physics_train,
        sampler=sampler,
        config=config,
        design_param=sampler.r,
        param_name="hole radius",
        resample_fn=lambda: (
            sampler.sample_interior(),
            sampler.sample_interface(),
            sampler.sample_boundary(sigma_inf=sigma_inf_scaled),
        ),
        sensitivity_fn=lambda: hole_region_sensitivity_objective(
            model, physics_train, sampler, delta=0.05
        ),
    )

    if use_energy_inner:
        inverse_optimizer.loss_label = 'Negative potential energy (-Pi)'
    inverse_optimizer.optimize(loss_fn=loss_fn, verbose=True,
                               inner_solve_fn=inner_solve_fn)
    if use_energy_inner:
        energy_net.float()

    os.makedirs('results', exist_ok=True)
    inverse_optimizer.plot_history(str(RESULTS_DIR / 'inverse_design_history.png'))
    return inverse_optimizer


def run_inverse_design_on_l_bracket(model, sampler, physics, device: str,
                                    n_outer: int = 12, n_inner: int = 300,
                                    stress_scale: float = 1e11):
    if not getattr(sampler, 'fillet_enabled', False):
        raise ValueError(
            "This LBracketSampler was built without fillet_init set, so it has "
            "no trainable fillet parameter to optimize. Retrain with "
            "train_forward_solver(benchmark='l_bracket', l_bracket_fillet_init=<value>) "
            "first, then run inverse design on THAT model."
        )

    print("=" * 70)
    print("INVERSE DESIGN — L-BRACKET FILLET")
    print("=" * 70)

    from atlas.loss_functions import XPINNLoss
    from atlas.physics import PlaneStressElasticity
    from atlas.inverse_design import fillet_region_sensitivity_objective

    physics_train = PlaneStressElasticity(E=physics.E / stress_scale, nu=physics.nu)
    P_scaled = 1e6 / stress_scale

    I_c = sampler.size ** 3 / 12.0
    u_s = (1e6 * sampler.size ** 3) / (3.0 * physics.E * I_c)
    loss_scales = _loss_scales(u_s, sampler.size, physics_train.E, np.sqrt(1.2) * abs(P_scaled))

    config = InverseDesignConfig(
        initial_radius=float(sampler.fillet.item()),
        min_radius=1e-4,
        max_radius=sampler.max_fillet,
        lr_geometry=1e-3,
        lr_network=1e-3,
        n_outer_iterations=n_outer,
        n_inner_iterations=n_inner
    )

    loss_fn = XPINNLoss(physics_train, scales=loss_scales)
    inverse_optimizer = InverseDesignOptimizer(
        xpinn_model=model,
        physics=physics_train,
        sampler=sampler,
        config=config,
        design_param=sampler.fillet,
        param_name="fillet radius",
        resample_fn=lambda: (
            sampler.sample_interior(),
            sampler.sample_interface(),
            sampler.sample_boundary(P=P_scaled),
        ),
        sensitivity_fn=lambda: fillet_region_sensitivity_objective(
            model, physics_train, sampler, delta=0.05
        ),
    )

    inverse_optimizer.optimize(loss_fn=loss_fn, verbose=True)

    os.makedirs('results', exist_ok=True)
    inverse_optimizer.plot_history(str(RESULTS_DIR / 'l_bracket_inverse_design_history.png'))
    return inverse_optimizer


def design_result_to_json(res: Dict) -> Dict:
    def num(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        return float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
    out = {k: num(v) for k, v in res.items() if k not in ('case', 'net', 'history', 'metrics',
                                                          'baseline', 'bounds')}
    out['bounds'] = [float(b) for b in res['bounds']]
    out['baseline'] = {k: num(v) for k, v in res['baseline'].items()}
    out['metrics'] = {k: num(v) for k, v in res['metrics'].items()}
    out['history'] = [{k: num(v) for k, v in row.items()} for row in res['history']]
    out['param_baseline'] = float(res['case'].p0)
    return out


def run_inverse_design_all(benchmark: str, device: str, metric: Optional[str] = None,
                           target_ratio: Optional[float] = None,
                           target_value: Optional[float] = None, net=None,
                           out_dir=None, verbose: bool = True, decomposed: bool = False,
                           **solver_kw) -> Dict:
    import json
    from pathlib import Path
    from atlas.energy_design import run_inverse_design
    from atlas.design_plots import plot_design_history, plot_design_result

    res = run_inverse_design(benchmark, metric=metric, target_ratio=target_ratio,
                             target_value=target_value, device=device, net=net,
                             verbose=verbose, decomposed=decomposed, **solver_kw)
    default_out = (RESULTS_DIR / 'atlas_xpinn_energy' / 'inverse_design' if decomposed
                   else RESULTS_DIR / 'atlas_energy' / 'inverse_design')
    out = Path(out_dir) if out_dir else default_out
    out.mkdir(parents=True, exist_ok=True)
    plot_design_history(res, str(out / f'{benchmark}_history.png'))
    plot_design_result(res, str(out / f'{benchmark}_design.png'))
    with open(out / f'{benchmark}_inverse.json', 'w') as f:
        json.dump(design_result_to_json(res), f, indent=2)
    import torch
    torch.save({k: v.float() if v.is_floating_point() else v
                for k, v in res['net'].state_dict().items()},
               out / f'{benchmark}_design_net.pt')
    res['out_dir'] = str(out)
    print(f"Inverse design ({benchmark}): {res['param_name']} = {res['p_star']:.4f} "
          f"[{res['status']}]; artifacts in {out}")
    return res


def visualize_results(model, sampler, physics, trainer, benchmark: str):
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    prefix = RESULTS_DIR / benchmark

    plot_displacement_field(model, sampler, physics, save_path=f'{prefix}_displacement.png')
    plot_von_mises_stress(model, sampler, physics, save_path=f'{prefix}_stress.png')
    plot_training_convergence(
        trainer.history, save_path=f'{prefix}_convergence.png',
        adam_epoch_count=trainer.adam_epoch_count,
    )

    plot_deformed_shape(model, sampler, n_grid=50 if benchmark == 'cantilever' else 70,
                        save_path=f'{prefix}_deformed.png')

    print(f"Plots saved under {RESULTS_DIR}/ ({benchmark})")


def main():
    print("=" * 70)
    print("ATLAS — Adaptive Multi-Scale Framework for Structural Design")
    print("XPINN Forward Solver + Inverse Design + FEM Validation")
    print("=" * 70)
    print()

    device = setup_device()
    print()

    print("Benchmarks:")
    print("  1. Cantilever Beam")
    print("  2. L-Shaped Bracket")
    print("  3. Plate with Circular Hole")
    print()
    print("After training you can:")
    print("  V  → run FEM validation")
    print("  I  → run inverse design (cantilever: beam height; plate-with-hole: hole radius; l-bracket: corner fillet)")
    print("  B  → both")
    print("  N  → neither, just train + plot")
    print()

    choice = input("Select benchmark (1-3): ").strip()
    post = input("After training, run [V/I/B/N]? ").strip().upper() or 'N'

    benchmark_map = {'1': 'cantilever', '2': 'l_bracket', '3': 'plate_hole'}
    benchmark = benchmark_map.get(choice, 'cantilever')
    if choice not in benchmark_map:
        print("Invalid choice, defaulting to cantilever.")

    l_bracket_fillet_init = None

    if l_bracket_fillet_init is None:
        budget = dict(epochs=2000, lbfgs_steps=15, lbfgs_max_iter_per_step=100)
    else:
        budget = dict(epochs=11000)
    model, trainer, sampler, physics = train_forward_solver(
        benchmark=benchmark,
        device=device,
        print_every=500,
        l_bracket_fillet_init=l_bracket_fillet_init,
        **budget,
    )

    if post in ('V', 'B'):
        run_validation(
            model=model,
            sampler=sampler,
            physics=physics,
            benchmark=benchmark,
            device=device,
            use_mock=True
        )
        os.makedirs(str(RESULTS_DIR), exist_ok=True)
        plot_fem_comparison(
            model, sampler, physics, benchmark=benchmark,
            use_mock=True,
            save_path=str(RESULTS_DIR / f'{benchmark}_fem_comparison.png')
        )

    if post in ('I', 'B'):
        from atlas.energy_solver import EnergyNet
        net = model.get_subdomain_network(0)
        design_net = net.double() if (benchmark == 'plate_hole' and isinstance(net, EnergyNet)) else None
        try:
            raw = input("Target as a ratio of the baseline metric (Enter = default): ").strip()
            ratio = float(raw) if raw else None
        except ValueError:
            ratio = None
        run_inverse_design_all(benchmark, device, target_ratio=ratio, net=design_net)
        if design_net is not None:
            design_net.float()

    visualize_results(model, sampler, physics, trainer, benchmark)

    print()
    print("Done. Check the results/ folder.")
    print("=" * 70)


if __name__ == "__main__":
    main()