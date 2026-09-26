"""energy_xpinn.py - Domain-decomposed (XPINN) deep-energy solver.

    displacement:  (gamma_u E/L) * int_Gamma |u_A - u_B|^2 ds
    traction:      (gamma_t L/E) * int_Gamma |sigma_A n - sigma_B n|^2 ds

    cantilever : 2 subnetworks, interface x = L/2
    plate_hole : 3 subnetworks, concentric rings rho in [r, r+0.3], [r+0.3, r+0.6],
                 and the far field (interfaces are the two circles)
    l_bracket  : 3 subnetworks, squares [0,s]^2, [s,2s]x[0,s], [s,2s]^2
                 (each carries the Williams corner enrichment)
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from atlas.energy_solver import (EnergyNet, _gauss_on_edges, graded_nodes,
                                 l_bracket_blocks, line_quadrature, rect_quadrature,
                                 train_energy, williams_features)
from atlas.networks import create_xpinn_for_benchmark

import math as _m

MID_MARGIN, FAR_MARGIN = 0.3, 0.6      # plate ring radii offsets (as in PlateWithHoleSampler)



class DecomposedEnergyProblem:
    """Potential energy summed over subdomains + interface continuity penalties."""

    def __init__(self, physics, blocks, loads, interfaces, energy_scale: float,
                 u_scale: float, length: float, gamma_u: float = 100.0,
                 gamma_t: float = 1.0, device: str = 'cpu', dtype=torch.float64):
        self.physics, self.device, self.dtype = physics, device, dtype
        self.energy_scale = float(energy_scale)
        self.u_scale, self.length = float(u_scale), float(length)
        self.gamma_u, self.gamma_t = float(gamma_u), float(gamma_t)
        at = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
        # a block is (sub, pts, w) or (sub, pts, w, physics) - the latter for a multi-material body
        self.blocks = [(b[0], at(b[1]), at(b[2]).reshape(-1), b[3] if len(b) > 3 else None)
                       for b in blocks]
        self.loads = [(i, at(p), at(w).reshape(-1), at(t)) for i, p, w, t in loads]
        # optional 'phys_a' / 'phys_b': the material on each side (traction continuity)
        self.interfaces = [dict(a=d['a'], b=d['b'], pts=at(d['pts']),
                                w=at(d['w']).reshape(-1), n=at(d['n']),
                                phys_a=d.get('phys_a'), phys_b=d.get('phys_b')) for d in interfaces]
        self.n_points = sum(p.shape[0] for _, p, _, _ in self.blocks)
        # interface weights are scaled by the STIFFEST material present (for one material this is
        # physics.E); scaling by a softer one inflates the traction term by the stiffness contrast
        self.E_ref = max([self.physics.E] + [ph.E for _, _, _, ph in self.blocks if ph is not None]
                         + [d[k].E for d in self.interfaces for k in ('phys_a', 'phys_b') if d[k] is not None])
        self.last: Dict[str, float] = {}
        self.lam = [torch.zeros_like(d['pts']) for d in self.interfaces]

    # ---- pieces -------------------------------------------------------------
    def _strains(self, net, pts):
        c = pts.clone().requires_grad_(True)
        uv = net(c)
        gu = torch.autograd.grad(uv[:, 0].sum(), c, create_graph=True)[0]
        gv = torch.autograd.grad(uv[:, 1].sum(), c, create_graph=True)[0]
        return uv, gu[:, 0], gv[:, 1], 0.5 * (gu[:, 1] + gv[:, 0])

    def strain_energy(self, model) -> torch.Tensor:
        total = 0.0
        for i, pts, w, blk_ph in self.blocks:
            ph = blk_ph or self.physics
            _, exx, eyy, exy = self._strains(model.get_subdomain_network(i), pts)
            dens = 0.5 * (ph.C11 * exx * exx + 2.0 * ph.C12 * exx * eyy
                          + ph.C11 * eyy * eyy + 4.0 * ph.C33 * exy * exy)
            total = total + torch.sum(w * dens)
        return total

    def external_work(self, model) -> torch.Tensor:
        work = 0.0
        for i, pts, w, t in self.loads:
            uv = model.get_subdomain_network(i)(pts)
            work = work + torch.sum(w * (t[:, 0] * uv[:, 0] + t[:, 1] * uv[:, 1]))
        return work

    def interface_terms(self, model) -> Tuple[torch.Tensor, torch.Tensor]:
        pen_u, pen_t, lam_w = 0.0, 0.0, 0.0
        for k, d in enumerate(self.interfaces):
            uv_a, exx_a, eyy_a, exy_a = self._strains(model.get_subdomain_network(d['a']), d['pts'])
            uv_b, exx_b, eyy_b, exy_b = self._strains(model.get_subdomain_network(d['b']), d['pts'])
            pen_u = pen_u + torch.sum(d['w'] * ((uv_a - uv_b) ** 2).sum(dim=1))
            lam_w = lam_w + torch.sum(d['w'] * (self.lam[k] * (uv_a - uv_b)).sum(dim=1))
            if self.gamma_t > 0.0:
                n = d['n']

                def trac(exx, eyy, exy, ph):
                    sxx = ph.C11 * exx + ph.C12 * eyy
                    syy = ph.C12 * exx + ph.C11 * eyy
                    sxy = 2.0 * ph.C33 * exy
                    return sxx * n[:, 0] + sxy * n[:, 1], sxy * n[:, 0] + syy * n[:, 1]

                ta = trac(exx_a, eyy_a, exy_a, d['phys_a'] or self.physics)
                tb = trac(exx_b, eyy_b, exy_b, d['phys_b'] or self.physics)
                pen_t = pen_t + torch.sum(d['w'] * ((ta[0] - tb[0]) ** 2 + (ta[1] - tb[1]) ** 2))
        self._lam_w = lam_w
        return pen_u, pen_t

    def __call__(self, model):
        se = self.strain_energy(model)
        we = self.external_work(model)
        pu, pt = self.interface_terms(model)
        E = self.E_ref
        loss_u = self.gamma_u * (E / self.length) * pu
        loss_t = self.gamma_t * (self.length / E) * pt if self.gamma_t > 0.0 else 0.0 * pu
        s = self.energy_scale
        self.last = dict(jump_u=float(pu.detach()), pen_u=float(loss_u.detach()) / s,
                         pen_t=float(loss_t.detach()) / s)
        return (se - we + loss_u + loss_t + self._lam_w) / s, se / s, we / s

    @torch.no_grad()
    def update_multipliers(self, model) -> float:
        """Method-of-multipliers step lam <- lam + rho [u], rho = 2 gamma_u E / L (so the
        quadratic term is exactly (rho/2) int |[u]|^2). At convergence lam equals the
        interface traction and the displacement jump -> 0 for a FIXED, moderate gamma_u,
        which removes the penalty's consistency error (the spurious interface compliance
        the pure penalty leaves behind). Returns the relative interface jump before the step.
        """
        rho = 2.0 * self.gamma_u * self.E_ref / self.length
        num, den = 0.0, 0.0
        for k, d in enumerate(self.interfaces):
            ua = model.get_subdomain_network(d['a'])(d['pts'])
            ub = model.get_subdomain_network(d['b'])(d['pts'])
            self.lam[k] = self.lam[k] + rho * (ua - ub)
            num += float((d['w'] * ((ua - ub) ** 2).sum(1)).sum())
            den += float((d['w'] * (ua ** 2).sum(1)).sum())
        return (num / max(den, 1e-300)) ** 0.5



def _line(p0, p1, n_cells=10, ng=4):
    pts, w, _ = line_quadrature(p0, p1, n_cells, ng)
    return pts, w


def _polar_block(a, b, th_lo, th_hi, n_th, n_xi, power, toward, ng):
    """Polar block: rho = a + xi*(b-a), a/b scalars or callables of theta."""
    th_p, th_w = _gauss_on_edges(np.linspace(th_lo, th_hi, n_th + 1), ng)
    xi_p, xi_w = _gauss_on_edges(graded_nodes(0.0, 1.0, n_xi, power, toward), ng)
    TH, XI = np.meshgrid(th_p, xi_p, indexing='ij')
    A = a(TH) if callable(a) else np.full_like(TH, a)
    B = b(TH) if callable(b) else np.full_like(TH, b)
    RHO = A + XI * (B - A)
    jac = RHO * (B - A)
    pts = np.stack([(RHO * np.cos(TH)).reshape(-1), (RHO * np.sin(TH)).reshape(-1)], axis=1)
    return pts, (np.outer(th_w, xi_w) * jac).reshape(-1)


def cantilever_geometry(P: float, L: float = 4.0, H: float = 1.0, ny: int = 20,
                        nx_half: int = 30, ng: int = 3):
    ye = graded_nodes(0.0, H, ny, 1.6, 'both')
    blocks = [(0, *rect_quadrature(graded_nodes(0.0, L / 2, nx_half, 2.0, 'a'), ye, ng)),
              (1, *rect_quadrature(graded_nodes(L / 2, L, nx_half, 1.0, 'none'), ye, ng))]
    lp, lw, s = line_quadrature((L, 0.0), (L, H), 10, 4)
    tr = np.stack([np.zeros_like(s), 6.0 * (P / H) * s * (1.0 - s)], axis=1)
    ip, iw = _line((L / 2, 0.0), (L / 2, H), 10, 4)
    interfaces = [dict(a=0, b=1, pts=ip, w=iw, n=np.tile([1.0, 0.0], (ip.shape[0], 1)))]
    return blocks, [(1, lp, lw, tr)], interfaces


def plate_geometry(sigma_inf: float, r: float = 0.3, W: float = 2.0, H: float = 2.0,
                   ng: int = 3, n_th: int = 20, n_xi=(12, 8, 14)):
    r_mid, r_far = r + MID_MARGIN, r + FAR_MARGIN
    split = math.atan2(H, W)
    blocks = []
    for th_lo, th_hi, rho_sq in [(0.0, split, lambda th: W / np.cos(th)),
                                 (split, 0.5 * math.pi, lambda th: H / np.sin(th))]:
        blocks.append((0, *_polar_block(r, r_mid, th_lo, th_hi, n_th, n_xi[0], 1.5, 'a', ng)))
        blocks.append((1, *_polar_block(r_mid, r_far, th_lo, th_hi, n_th, n_xi[1], 1.0, 'none', ng)))
        blocks.append((2, *_polar_block(r_far, rho_sq, th_lo, th_hi, n_th, n_xi[2], 1.0, 'none', ng)))
    lp, lw, _ = line_quadrature((W, 0.0), (W, H), 12, 4)
    tr = np.stack([np.full(lp.shape[0], sigma_inf), np.zeros(lp.shape[0])], axis=1)
    interfaces = []
    th_p, th_w = _gauss_on_edges(np.linspace(0.0, 0.5 * math.pi, 25), 4)
    for a, b, rad in ((0, 1, r_mid), (1, 2, r_far)):
        interfaces.append(dict(a=a, b=b,
                               pts=np.stack([rad * np.cos(th_p), rad * np.sin(th_p)], axis=1),
                               w=rad * th_w, n=np.stack([np.cos(th_p), np.sin(th_p)], axis=1)))
    return blocks, [(2, lp, lw, tr)], interfaces


def l_bracket_geometry(P: float, s: float = 1.0):
    base = l_bracket_blocks(s)
    blocks = [(i, p, w) for i, (p, w) in enumerate(base)]
    lp, lw, t = line_quadrature((2 * s, s), (2 * s, 2 * s), 12, 4)
    tr = np.stack([6.0 * P * t * (1.0 - t), np.zeros_like(t)], axis=1)
    i01p, i01w = _line((s, 0.0), (s, s), 12, 4)              # x = s, y in [0,s]   (0 -> 1)
    i12p, i12w = _line((s, s), (2 * s, s), 12, 4)            # y = s, x in [s,2s]  (1 -> 2)
    interfaces = [dict(a=0, b=1, pts=i01p, w=i01w, n=np.tile([1.0, 0.0], (i01p.shape[0], 1))),
                  dict(a=1, b=2, pts=i12p, w=i12w, n=np.tile([0.0, 1.0], (i12p.shape[0], 1)))]
    return blocks, [(2, lp, lw, tr)], interfaces


def l_bracket_corner_geometry(P: float, s: float = 1.0, a: float = 0.125):
    """'concentration' layout: the paper's three squares plus a fourth subdomain, the
    re-entrant-corner patch |x-s| <= a, |y-s| <= a (three quadrants, the void one excluded).
    The singular corner field then lives inside ONE subnetwork instead of on the junction
    of three interfaces. The quadrature is exactly the single-network one (points are only
    relabelled; a = (12/24)^3 lies on a cell edge of every graded block, checked by area).
    """
    base = l_bracket_blocks(s)
    blocks = []
    for i, (p, w) in enumerate(base):
        inp = (np.abs(p[:, 0] - s) < a) & (np.abs(p[:, 1] - s) < a)
        blocks.append((i, p[~inp], w[~inp]))
        blocks.append((3, p[inp], w[inp]))
    lp, lw, t = line_quadrature((2 * s, s), (2 * s, 2 * s), 12, 4)
    tr = np.stack([6.0 * P * t * (1.0 - t), np.zeros_like(t)], axis=1)
    segs = [(3, 0, (s - a, s - a), (s - a, s), (-1.0, 0.0), 6), (3, 0, (s - a, s - a), (s, s - a), (0.0, -1.0), 6),
            (3, 1, (s, s - a), (s + a, s - a), (0.0, -1.0), 6), (3, 1, (s + a, s - a), (s + a, s), (1.0, 0.0), 6),
            (3, 2, (s + a, s), (s + a, s + a), (1.0, 0.0), 6), (3, 2, (s, s + a), (s + a, s + a), (0.0, 1.0), 6),
            (0, 1, (s, 0.0), (s, s - a), (1.0, 0.0), 12), (1, 2, (s + a, s), (2 * s, s), (0.0, 1.0), 12)]
    interfaces = []
    for ia, ib, p0, p1, n, nc in segs:
        ip, iw = _line(p0, p1, nc, 4)
        interfaces.append(dict(a=ia, b=ib, pts=ip, w=iw, n=np.tile(n, (ip.shape[0], 1))))
    return blocks, [(2, lp, lw, tr)], interfaces


def l_bracket_corner_router(s: float = 1.0, a: float = 0.125):
    def route(c: torch.Tensor) -> torch.Tensor:
        x, y = c[:, 0], c[:, 1]
        idx = torch.zeros(c.shape[0], dtype=torch.long, device=c.device)
        idx[(x > s) & (y <= s)] = 1
        idx[(x >= s) & (y > s)] = 2
        idx[((x - s).abs() <= a) & ((y - s).abs() <= a)] = 3
        return idx
    return route


class RoutedEnergyXPINN(nn.Module):
    """Subnetworks + a geometric router; evaluates every point with the network that owns
    it. `self_routing` tells visualization/validation to call the model directly instead of
    routing with the benchmark sampler (whose layouts only know the paper's subdomains).
    """
    self_routing = True
    monolith = False

    def __init__(self, subnets, router):
        super().__init__()
        self.subnets = nn.ModuleList(subnets)
        self.num_subdomains = len(subnets)
        self._router = router

    def get_subdomain_network(self, i: int) -> nn.Module:
        return self.subnets[i]

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        idx = self._router(coords)
        out = torch.zeros(coords.shape[0], 2, dtype=coords.dtype, device=coords.device)
        for i in range(self.num_subdomains):
            m = idx == i
            if bool(m.any()):
                out[m] = self.subnets[i](coords[m]).to(out.dtype)
        return out



E_MOD, NU, STRESS_SCALE = 210e9, 0.3, 1e11


def build_xpinn_energy(benchmark: str, physics_train, hidden_layers: int = 4,
                       neurons: int = 64, gamma_u: float = 30.0, gamma_t: float = 0.1,
                       device: str = 'cpu', dtype=torch.float64, layout: str = 'paper'):
    """(model, problem, meta). model is an XPINNModel whose subnetworks are
    EnergyNets with exact essential BCs; problem is the DecomposedEnergyProblem.

    layout 'paper'         : the decompositions listed in the module docstring.
    layout 'concentration' : L-bracket only - adds a 4th subdomain around the re-entrant
                             corner (see l_bracket_corner_geometry). On the cantilever and
                             the plate the stress concentrations are already inside one
                             subdomain, so it is identical to 'paper' there.
    """
    if benchmark == 'cantilever':
        L, H, P = 4.0, 1.0, -1e6 / STRESS_SCALE
        u_s = 1e6 * L ** 3 / (3.0 * E_MOD * (H ** 3 / 12.0))
        blocks, loads, ifaces = cantilever_geometry(P, L, H)
        escale, length = abs(P) * u_s * H, L
        mk = lambda: EnergyNet((0.0, 0.0), (L, H), hidden_layers, neurons, u_s, 'clamp_x0', L)
    elif benchmark == 'plate_hole':
        W = H = 2.0
        sigma = 1e6 / STRESS_SCALE
        u_s = 1e6 * W / E_MOD
        blocks, loads, ifaces = plate_geometry(sigma)
        escale, length = abs(sigma) * u_s * H, W
        mk = lambda: EnergyNet((0.0, 0.0), (W, H), hidden_layers, neurons, u_s, 'sym', 1.0)
    elif benchmark == 'l_bracket':
        s, P = 1.0, 1e6 / STRESS_SCALE
        u_s = 1e6 * s ** 3 / (3.0 * E_MOD * (s ** 3 / 12.0))
        blocks, loads, ifaces = l_bracket_geometry(P, s)
        escale, length = abs(P) * u_s * s, s
        fn, nf = williams_features((s, s), -math.pi, 0.5 * math.pi)
        mk = lambda: EnergyNet((0.0, 0.0), (2 * s, 2 * s), hidden_layers, neurons, u_s,
                               'clamp_x0', 2 * s, enrich_fn=fn, n_enrich=nf)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    if benchmark == 'l_bracket' and layout == 'concentration':
        blocks, loads, ifaces = l_bracket_corner_geometry(P, s)
        model = RoutedEnergyXPINN([mk() for _ in range(4)], l_bracket_corner_router(s))
        model.to(device).to(dtype)
        problem = DecomposedEnergyProblem(physics_train, blocks, loads, ifaces, escale, u_s, length,
                                          gamma_u, gamma_t, device, dtype)
        return model, problem, dict(u_s=u_s, length=length, n_sub=4)
    model = create_xpinn_for_benchmark(benchmark, hidden_layers=hidden_layers,
                                       neurons=neurons, device='cpu', monolith=False)
    n_sub = model.num_subdomains
    model.subnets = nn.ModuleList([mk() for _ in range(n_sub)])
    model.to(device).to(dtype)
    problem = DecomposedEnergyProblem(physics_train, blocks, loads, ifaces, escale, u_s, length,
                                      gamma_u, gamma_t, device, dtype)
    return model, problem, dict(u_s=u_s, length=length, n_sub=n_sub)


def train_energy_xpinn_forward_solver(benchmark: str, device: str = 'cpu', epochs: int = 2000,
                                      lbfgs_iters: int = 2000, seed: Optional[int] = 42,
                                      hidden_layers: int = 4, neurons: int = 64,
                                      gamma_u: float = 30.0, gamma_t: float = 0.1,
                                      verbose: bool = True, dtype=torch.float64):
    """Train the domain-decomposed energy solver. Returns (model, trainer, sampler,
    physics_real) like atlas_main.train_forward_solver (model is an XPINNModel in float32
    on `device`, so validation, plots and routing work unchanged).
    """
    from atlas.energy_solver import EnergyTrainer, history_for_plot
    from atlas.physics import PlaneStressElasticity
    from atlas.sampling import CantileverBeamSampler, LBracketSampler, PlateWithHoleSampler
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    model, problem, _ = build_xpinn_energy(
        benchmark, PlaneStressElasticity(E=E_MOD / STRESS_SCALE, nu=NU), hidden_layers,
        neurons, gamma_u, gamma_t, device, dtype)
    hist = train_decomposed(model, problem, epochs=epochs, lbfgs_per_stage=lbfgs_iters,
                            gammas=(gamma_u,), verbose=verbose)
    model = model.float().to(device)
    sampler = {'cantilever': CantileverBeamSampler, 'plate_hole': PlateWithHoleSampler,
               'l_bracket': LBracketSampler}[benchmark](device=device)
    trainer = EnergyTrainer()
    trainer.history = history_for_plot(hist)
    trainer.adam_epoch_count = epochs
    return model, trainer, sampler, PlaneStressElasticity(E=E_MOD, nu=NU)


def train_decomposed(model, problem: DecomposedEnergyProblem, epochs: int = 2000,
                     lr: float = 2e-3, lbfgs_per_stage: int = 2000,
                     gammas: Sequence[float] = (30.0,),
                     verbose: bool = True, probe=None,
                     al_stages: int = 3, warm_start: int = 1000,
                     al_keep_memory: bool = False, warm_start_lbfgs: int = 0) -> Dict[str, List[float]]:
    """Adam at the first interface weight, then L-BFGS; extra entries in `gammas`
    add stages with a stiffer interface penalty (continuation).

    Measured on the plate (docs/BUGFIXES.md): gamma_u = 30 already matches the
    interfaces to ~1e-8 relative, and extra stiff stages (300, 3000) only made
    L-BFGS ill-conditioned (2.8% -> 0.85% displacement error when replaced by a
    single stage with more iterations). gamma_u = 10 is too soft (3.6%).

    Default recipe (measured on the L-bracket, results/comparison/xpinn_fix_study):
    al_stages = 3 and warm_start = 1000 (1.44% -> 0.30% displacement error, 5.30% -> 4.55%
    von Mises at the same budget). Pass al_stages=0, warm_start=0 for the old plain-penalty run.
    """
    hist = {'total': [], 'strain': [], 'work': [], 'iter': []}
    offset = 0
    warm_start = min(int(warm_start), epochs // 2)
    if warm_start > 0:
        import copy
        from atlas.energy_solver import EnergyProblem
        whole = EnergyProblem(problem.physics, [(p, w, ph) for _, p, w, ph in problem.blocks],
                              [(p, w, t) for _, p, w, t in problem.loads],
                              problem.energy_scale, device=problem.device, dtype=problem.dtype)
        one = copy.deepcopy(model.get_subdomain_network(0))
        warm_start_lbfgs = min(int(warm_start_lbfgs), lbfgs_per_stage // 2)
        h = train_energy(one, whole, epochs=warm_start, lr=lr, lbfgs_iters=warm_start_lbfgs,
                         lbfgs_chunk=100, print_every=500, verbose=verbose, probe=None)
        for i in range(len(model.subnets)):
            sd = dict(one.state_dict())
            # keep each subnetwork's OWN normalisation box (local_norm); only the weights are shared
            for k in ('_lo', '_hi'):
                if k in sd and k in model.subnets[i].state_dict():
                    sd[k] = model.subnets[i].state_dict()[k]
            model.subnets[i].load_state_dict(sd)
        for key in ('total', 'strain', 'work'):
            hist[key] += h[key]
        hist['iter'] += h['iter']
        offset = warm_start + warm_start_lbfgs
        epochs = epochs - warm_start
        lbfgs_per_stage = lbfgs_per_stage - warm_start_lbfgs
        if probe is not None:
            probe(offset)
    if al_stages > 0 and al_keep_memory:
        problem.gamma_u = float(gammas[0])
        n_part = lbfgs_per_stage // (al_stages + 1)

        def on_chunk(done):
            if done % n_part == 0 and done < lbfgs_per_stage:
                rel = problem.update_multipliers(model)
                if verbose:
                    print(f"  [AL update, L-BFGS memory kept] relative jump {rel:.3e}", flush=True)
                return True
            return False

        off = offset
        h = train_energy(model, problem, epochs=epochs, lr=lr, lbfgs_iters=lbfgs_per_stage,
                         lbfgs_chunk=100, print_every=500, verbose=verbose, on_chunk=on_chunk,
                         probe=None if probe is None else (lambda it, off=off: probe(off + it)))
        for key in ('total', 'strain', 'work'):
            hist[key] += h[key]
        hist['iter'] += [offset + i for i in h['iter']]
        return hist
    if al_stages > 0:
        assert len(gammas) == 1, "augmented Lagrangian uses one fixed gamma_u"
        n_part = lbfgs_per_stage // (al_stages + 1)
        stages = [(0, epochs, n_part)] + [(k, 0, n_part) for k in range(1, al_stages + 1)]
        # (epochs already reduced by warm_start above)
    else:
        stages = [(k, epochs if k == 0 else 0, lbfgs_per_stage) for k in range(len(gammas))]
    for k, ep, n_lb in stages:
        problem.gamma_u = float(gammas[0] if al_stages > 0 else gammas[k])
        g = problem.gamma_u
        if al_stages > 0 and k > 0:
            rel = problem.update_multipliers(model)
            if verbose:
                print(f"  [AL update {k}] relative interface jump before update = {rel:.3e}", flush=True)
        off = offset
        h = train_energy(model, problem, epochs=ep, lr=lr,
                         lbfgs_iters=n_lb, lbfgs_chunk=100, print_every=500,
                         verbose=verbose,
                         probe=None if probe is None else (lambda it, off=off: probe(off + it)))
        for key in ('total', 'strain', 'work'):
            hist[key] += h[key]
        hist['iter'] += [offset + i for i in h['iter']]
        offset = hist['iter'][-1]
        if verbose:
            print(f"  [stage {k + 1}] gamma_u={g:g}  interface jump^2 (int) = "
                  f"{problem.last.get('jump_u', float('nan')):.3e}", flush=True)
    return hist
