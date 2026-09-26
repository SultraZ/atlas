"""energy_solver.py - Deep-energy (Ritz) forward solver for the ATLAS benchmarks.

    Pi[u] = 1/2 int sigma(u):eps(u) dA  -  int_loaded t.u ds

  * The functional is convex/quadratic in the field, so the load amplitude is
    pinned (no collapse into the trivial zero field) and only FIRST
    derivatives of u are needed.
  * Essential BCs (clamp, symmetry rollers) are imposed EXACTLY through a
    multiplicative mask u = m(x) * u_hat, so no soft-penalty balancing.
    Natural BCs (traction-free edges, the applied load) are built into Pi.
  * Quadrature is deterministic and graded toward the known singular
    points, so the optimizer sees a fixed smooth objective (L-BFGS-friendly).
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn



def graded_nodes(a: float, b: float, n: int, power: float = 1.0,
                 toward: str = 'none') -> np.ndarray:
    """n+1 cell edges on [a, b]. power>1 clusters cells toward the chosen end
    ('a', 'b', or 'both'); power=1 (or toward='none') is uniform.
    """
    s = np.linspace(0.0, 1.0, n + 1)
    if toward == 'a':
        t = s ** power
    elif toward == 'b':
        t = 1.0 - (1.0 - s) ** power
    elif toward == 'both':
        h = 0.5 * np.where(s < 0.5, (2 * s) ** power, 2.0 - (2 * (1 - s)) ** power)
        t = h
    else:
        t = s
    return a + (b - a) * t


def _gauss_on_edges(edges: np.ndarray, ng: int) -> Tuple[np.ndarray, np.ndarray]:
    xi, wi = np.polynomial.legendre.leggauss(ng)
    lo, hi = edges[:-1], edges[1:]
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    pts = (mid[:, None] + half[:, None] * xi[None, :]).reshape(-1)
    wts = (half[:, None] * wi[None, :]).reshape(-1)
    return pts, wts


def rect_quadrature(x_edges: np.ndarray, y_edges: np.ndarray,
                    ng: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Tensor-product composite Gauss rule on the grid x_edges x y_edges."""
    xp, xw = _gauss_on_edges(x_edges, ng)
    yp, yw = _gauss_on_edges(y_edges, ng)
    X, Y = np.meshgrid(xp, yp, indexing='ij')
    W = np.outer(xw, yw)
    return np.stack([X.reshape(-1), Y.reshape(-1)], axis=1), W.reshape(-1)


def line_quadrature(p0: Tuple[float, float], p1: Tuple[float, float],
                    n_cells: int = 8, ng: int = 4,
                    power: float = 1.0, toward: str = 'none'
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composite Gauss rule along the segment p0->p1. Returns (pts, weights
    in arc length, curve parameter s in [0,1]).
    """
    edges = graded_nodes(0.0, 1.0, n_cells, power, toward)
    sp, sw = _gauss_on_edges(edges, ng)
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    pts = p0[None, :] + sp[:, None] * (p1 - p0)[None, :]
    return pts, sw * float(np.linalg.norm(p1 - p0)), sp



class EnergyNet(nn.Module):
    """u(x,y) = mask(x,y) * [ MLP(normalized coords) + W_enrich . feats(x,y) ] * u_s.

    * `lo`/`hi` normalise the input to [-1, 1] per axis (conditioning).
    * `mask_kind`:  'none' | 'clamp_x0' (u=v=0 on x=0) | 'clamp_y0' (u=v=0 on y=0)
                    | 'clamp_x1' (u=v=0 on x = mask_x1, default hi[0])
                    | 'sym' (u=0 on x=0, v=0 on y=0)
    * `enrich_fn`:  optional callable (N,2)->(N,F) of FIXED (non-trainable)
      basis functions; their coefficients are learned linearly and start at 0.
      Used for classical corner-singularity (Williams) terms.
    """

    def __init__(self, lo: Tuple[float, float], hi: Tuple[float, float],
                 hidden_layers: int = 4, neurons: int = 64, out_scale: float = 1.0,
                 mask_kind: str = 'none', mask_len: float = 1.0, mask_x1: Optional[float] = None,
                 enrich_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 n_enrich: int = 0, activation: str = 'tanh'):
        super().__init__()
        self.register_buffer('_lo', torch.tensor(lo, dtype=torch.float32))
        self.register_buffer('_hi', torch.tensor(hi, dtype=torch.float32))
        self.output_scale = float(out_scale)
        self.mask_kind = mask_kind
        self.mask_len = float(mask_len)
        self.mask_x1 = float(hi[0] if mask_x1 is None else mask_x1)
        self.enrich_fn = enrich_fn
        self.n_enrich = int(n_enrich)
        act = {'tanh': nn.Tanh, 'silu': nn.SiLU}[activation]
        layers: List[nn.Module] = [nn.Linear(2, neurons), act()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons, neurons), act()]
        layers.append(nn.Linear(neurons, 2))
        self.network = nn.Sequential(*layers)
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        if self.n_enrich > 0:
            self.enrich_weight = nn.Parameter(torch.zeros(self.n_enrich, 2))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.to(self._lo.dtype)
        xn = 2.0 * (coords - self._lo) / (self._hi - self._lo) - 1.0
        out = self.network(xn)
        if self.n_enrich > 0:
            out = out + self.enrich_fn(coords) @ self.enrich_weight
        out = out * self.output_scale
        x, y = coords[:, 0:1], coords[:, 1:2]
        if self.mask_kind == 'clamp_x0':
            m = x / self.mask_len
            out = out * torch.cat([m, m], dim=1)
        elif self.mask_kind == 'clamp_x1':            # u = v = 0 on x = hi_x
            m = (self.mask_x1 - x) / self.mask_len
            out = out * torch.cat([m, m], dim=1)
        elif self.mask_kind == 'clamp_y0':
            m = y / self.mask_len
            out = out * torch.cat([m, m], dim=1)
        elif self.mask_kind == 'sym':
            out = out * torch.cat([x / self.mask_len, y / self.mask_len], dim=1)
        return out


# ---- Williams corner-singularity enrichment ---------------------------------

WILLIAMS_270 = (0.5444837367, 0.9085291898)


def williams_features(center: Tuple[float, float], sector_lo: float,
                      sector_hi: float, lambdas=WILLIAMS_270,
                      eps: float = 1e-4) -> Tuple[Callable, int]:
    """Fixed enrichment basis  r^lam * {cos,sin}(k*phi), k in {lam-2, lam, lam+2}
    for each lam. Cartesian displacement components of a Williams mode are
    linear combinations of exactly these terms. phi is measured inside the
    material sector [sector_lo, sector_hi] (branch cut placed in the void, so
    the basis is continuous over the whole material, including the two free
    edges). r is regularised by eps and the angle is gradient-safe at r=0.
    """
    cx, cy = center
    ks_of = lambda lam: (lam - 2.0, lam, lam + 2.0)
    n_feat = sum(2 * len(ks_of(l)) for l in lambdas)

    def fn(coords: torch.Tensor) -> torch.Tensor:
        dx = coords[:, 0:1] - cx
        dy = coords[:, 1:2] - cy
        r2 = dx * dx + dy * dy
        safe = r2 > 1e-24
        dx_s = torch.where(safe, dx, torch.ones_like(dx))
        dy_s = torch.where(safe, dy, torch.zeros_like(dy))
        phi = torch.atan2(dy_s, dx_s)
        phi = torch.where(safe, phi, torch.zeros_like(phi))
        # move the branch cut into the void: material sector is
        # [sector_lo, sector_hi]; anything above sector_hi wraps by -2*pi.
        phi = torch.where(phi > sector_hi + 1e-9, phi - 2.0 * math.pi, phi)
        r_reg = torch.sqrt(r2 + eps * eps)
        feats = []
        for lam in lambdas:
            rl = r_reg ** lam
            for k in ks_of(lam):
                feats.append(rl * torch.cos(k * phi))
                feats.append(rl * torch.sin(k * phi))
        return torch.cat(feats, dim=1)

    return fn, n_feat



class EnergyProblem:
    """Discrete total potential energy on a fixed quadrature."""

    def __init__(self, physics, interior, loads, energy_scale: float,
                 device: str = 'cpu', dtype=torch.float64):
        self.physics = physics
        self.device = device
        self.dtype = dtype
        self.energy_scale = float(energy_scale)
        at = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
        # a block is (pts, w) or (pts, w, physics) - the latter for a multi-material body
        self.interior = [(at(b[0]), at(b[1]).reshape(-1), b[2] if len(b) > 2 else None)
                         for b in interior]
        self.loads = [(at(p), at(w).reshape(-1), at(t)) for p, w, t in loads]
        self.n_points = sum(p.shape[0] for p, _, _ in self.interior)

    def strain_energy(self, net: nn.Module) -> torch.Tensor:
        total = 0.0
        for pts, w, blk_ph in self.interior:
            ph = blk_ph or self.physics
            c = pts.clone()
            if not c.requires_grad:
                c.requires_grad_(True)
            uv = net(c)
            gu = torch.autograd.grad(uv[:, 0].sum(), c, create_graph=True)[0]
            gv = torch.autograd.grad(uv[:, 1].sum(), c, create_graph=True)[0]
            exx, eyy = gu[:, 0], gv[:, 1]
            exy = 0.5 * (gu[:, 1] + gv[:, 0])
            dens = 0.5 * (ph.C11 * exx * exx + 2.0 * ph.C12 * exx * eyy
                          + ph.C11 * eyy * eyy + 4.0 * ph.C33 * exy * exy)
            total = total + torch.sum(w * dens)
        return total

    def external_work(self, net: nn.Module) -> torch.Tensor:
        work = 0.0
        for pts, w, t in self.loads:
            uv = net(pts)
            work = work + torch.sum(w * (t[:, 0] * uv[:, 0] + t[:, 1] * uv[:, 1]))
        return work

    def __call__(self, net: nn.Module):
        se = self.strain_energy(net)
        we = self.external_work(net)
        return (se - we) / self.energy_scale, se / self.energy_scale, we / self.energy_scale



def train_energy(net: nn.Module, problem: EnergyProblem, epochs: int = 3000,
                 lr: float = 2e-3, lbfgs_iters: int = 3000,
                 lbfgs_chunk: int = 100, print_every: int = 500,
                 verbose: bool = True, keep_best: bool = False,
                 probe=None, probe_every: int = 100, on_chunk=None) -> Dict[str, List[float]]:
    """Adam (cosine-decayed) warm-up, then L-BFGS (strong Wolfe) polish."""
    hist = {'total': [], 'strain': [], 'work': [], 'iter': []}
    params = [p for p in net.parameters() if p.requires_grad]

    if epochs > 0:
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.02)
        for ep in range(epochs):
            opt.zero_grad()
            loss, se, we = problem(net)
            loss.backward()
            opt.step()
            sched.step()
            hist['total'].append(loss.item())
            hist['strain'].append(se.item())
            hist['work'].append(we.item())
            hist['iter'].append(ep + 1)
            if probe is not None and (ep + 1) % probe_every == 0:
                probe(ep + 1)
            if verbose and (ep + 1) % print_every == 0:
                print(f"  [adam] {ep + 1:6d}  Pi={loss.item():+.8f}  "
                      f"U={se.item():.6f}  W={we.item():.6f}")

    if lbfgs_iters > 0:
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=lbfgs_chunk,
                                history_size=100, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-12, tolerance_change=1e-16)

        def closure():
            opt.zero_grad()
            loss, se, we = problem(net)
            loss.backward()
            closure.last = (loss.item(), se.item(), we.item())
            return loss

        done = 0
        prev = None
        best_l, best_state = float('inf'), None
        while done < lbfgs_iters:
            opt.step(closure)
            done += lbfgs_chunk
            l, s, w = closure.last
            if keep_best and np.isfinite(l) and l < best_l:
                best_l = l
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            hist['total'].append(l)
            hist['strain'].append(s)
            hist['work'].append(w)
            hist['iter'].append(max(epochs, 0) + done)
            if probe is not None:
                probe(max(epochs, 0) + done)
            if verbose and (done // lbfgs_chunk) % max(1, print_every // lbfgs_chunk) == 0:
                print(f"  [lbfgs] {done:6d}  Pi={l:+.10f}  U={s:.6f}  W={w:.6f}")
            if on_chunk is not None and on_chunk(done):
                prev = None
                continue
            if prev is not None and abs(prev - l) < 1e-13 * max(1.0, abs(l)):
                break
            prev = l
        if keep_best and best_state is not None:
            net.load_state_dict(best_state)
    return hist



def build_cantilever(physics_train, P: float, u_s: float, L: float = 4.0,
                     H: float = 1.0, nx: int = 60, ny: int = 20, ng: int = 3,
                     hidden_layers: int = 4, neurons: int = 64,
                     device: str = 'cpu', dtype=torch.float64):
    """Clamp at x=0 (exact, via mask), parabolic tip shear on x=L."""
    xe = graded_nodes(0.0, L, nx, 2.0, 'a')
    ye = graded_nodes(0.0, H, ny, 1.6, 'both')
    pts, w = rect_quadrature(xe, ye, ng)
    lp, lw, s = line_quadrature((L, 0.0), (L, H), 10, 4)
    tr = np.stack([np.zeros_like(s), 6.0 * (P / H) * s * (1.0 - s)], axis=1)
    prob = EnergyProblem(physics_train, [(pts, w)], [(lp, lw, tr)],
                         energy_scale=abs(P) * u_s * H, device=device, dtype=dtype)
    net = EnergyNet((0.0, 0.0), (L, H), hidden_layers, neurons, u_s,
                    'clamp_x0', L).to(device).to(dtype)
    return net, prob


def l_bracket_blocks(s: float = 1.0, n: int = 24, ng: int = 3, power: float = 3.0):
    """Quadrature blocks (numpy) tiling L = [0,2s]x[0,s] U [s,2s]x[s,2s], graded
    toward the re-entrant corner (s,s) and mildly toward the clamp x=0.
    """
    blocks = []
    # sub0 [0,s]^2 : refine toward x=s and y=s (corner) and mildly toward x=0
    xe0 = np.unique(np.concatenate([graded_nodes(0, s, n // 3, 1.5, 'a'),
                                    graded_nodes(0, s, n, power, 'b')]))
    ye0 = graded_nodes(0, s, n, power, 'b')
    ye0 = np.unique(np.concatenate([ye0, graded_nodes(0, s, n // 3, 1.5, 'a')]))
    blocks.append(rect_quadrature(xe0, ye0, ng))
    # sub1 [s,2s]x[0,s] : refine toward x=s, y=s
    blocks.append(rect_quadrature(graded_nodes(s, 2 * s, n, power, 'a'),
                                  graded_nodes(0, s, n, power, 'b'), ng))
    # sub2 [s,2s]^2 : refine toward x=s, y=s
    blocks.append(rect_quadrature(graded_nodes(s, 2 * s, n, power, 'a'),
                                  graded_nodes(s, 2 * s, n, power, 'a'), ng))
    return blocks


def build_l_bracket(physics_train, P: float, u_s: float, s: float = 1.0,
                    n: int = 24, ng: int = 3, power: float = 3.0,
                    hidden_layers: int = 4, neurons: int = 64,
                    enrich: bool = True, device: str = 'cpu',
                    dtype=torch.float64):
    """L = [0,2s]x[0,s] U [s,2s]x[s,2s]; clamp x=0; parabolic t_x on x=2s,
    y in [s,2s]; re-entrant corner (s,s) enriched with Williams terms and
    quadrature graded toward it.
    """
    blocks = l_bracket_blocks(s, n, ng, power)
    lp, lw, t = line_quadrature((2 * s, s), (2 * s, 2 * s), 12, 4)
    tr = np.stack([6.0 * P * t * (1.0 - t), np.zeros_like(t)], axis=1)
    prob = EnergyProblem(physics_train, blocks, [(lp, lw, tr)],
                         energy_scale=abs(P) * u_s * s, device=device, dtype=dtype)
    if enrich:
        fn, nf = williams_features((s, s), -math.pi, 0.5 * math.pi)
    else:
        fn, nf = None, 0
    net = EnergyNet((0.0, 0.0), (2 * s, 2 * s), hidden_layers, neurons, u_s,
                    'clamp_x0', 2 * s, enrich_fn=fn, n_enrich=nf).to(device).to(dtype)
    return net, prob


def plate_hole_problem(physics_train, sigma_inf: float, u_s: float, r: float = 0.3,
                       W: float = 2.0, H: float = 2.0, n_xi: int = 24,
                       n_th: int = 24, ng: int = 3, device: str = 'cpu',
                       dtype=torch.float64) -> "EnergyProblem":
    """Energy problem (quadrature + load) for the quarter plate with a quarter
    hole of radius r. Structured O-grid quadrature: x = rho(xi,theta)
    (cos th, sin th), rho = r + xi*(rho_sq-r), rho_sq the square boundary along
    the ray (two blocks split at the diagonal). Traction sigma_inf on x=W.
    Only the quadrature depends on r, so a network trained for one radius can
    be re-equilibrated for another by rebuilding this problem.
    """
    th_split = math.atan2(H, W)
    xi_edges = graded_nodes(0.0, 1.0, n_xi, 1.5, 'a')
    blocks = []
    for th_lo, th_hi, rho_sq in [
            (0.0, th_split, lambda th: W / np.cos(th)),
            (th_split, 0.5 * math.pi, lambda th: H / np.sin(th))]:
        th_p, th_w = _gauss_on_edges(np.linspace(th_lo, th_hi, n_th + 1), ng)
        xi_p, xi_w = _gauss_on_edges(xi_edges, ng)
        TH, XI = np.meshgrid(th_p, xi_p, indexing='ij')
        WT = np.outer(th_w, xi_w)
        RS = rho_sq(TH)
        RHO = r + XI * (RS - r)
        jac = RHO * (RS - r)
        pts = np.stack([(RHO * np.cos(TH)).reshape(-1),
                        (RHO * np.sin(TH)).reshape(-1)], axis=1)
        blocks.append((pts, (WT * jac).reshape(-1)))
    lp, lw, _ = line_quadrature((W, 0.0), (W, H), 12, 4)
    tr = np.stack([np.full(lp.shape[0], sigma_inf), np.zeros(lp.shape[0])], axis=1)
    return EnergyProblem(physics_train, blocks, [(lp, lw, tr)],
                         energy_scale=abs(sigma_inf) * u_s * H, device=device, dtype=dtype)


def build_plate_hole(physics_train, sigma_inf: float, u_s: float, r: float = 0.3,
                     W: float = 2.0, H: float = 2.0, n_xi: int = 24,
                     n_th: int = 24, ng: int = 3, hidden_layers: int = 4,
                     neurons: int = 64, device: str = 'cpu', dtype=torch.float64):
    """Quarter plate with quarter hole at the origin: (net, problem). Symmetry
    rollers are exact via the mask; see plate_hole_problem for the quadrature.
    """
    prob = plate_hole_problem(physics_train, sigma_inf, u_s, r, W, H, n_xi, n_th,
                              ng, device, dtype)
    net = EnergyNet((0.0, 0.0), (W, H), hidden_layers, neurons, u_s,
                    'sym', 1.0).to(device).to(dtype)
    return net, prob


def make_plate_energy_inner_solver(net: nn.Module, physics_train, sampler,
                                   sigma_inf: float, u_s: float, lbfgs_iters: int,
                                   device: str = 'cpu', dtype=torch.float64):
    """Inner forward-solve callback for inverse design on an energy-trained
    plate model. Each call re-equilibrates `net` (warm start) at the sampler's
    CURRENT hole radius by rebuilding the quadrature and running L-BFGS, and
    returns the positive scalar -Pi (= half the external work at equilibrium).
    The network is kept in float64 for the whole inverse run; the caller casts
    it back to float32 afterwards.
    """
    net.to(dtype)

    def inner_solve() -> float:
        r = float(sampler.r.detach().cpu().item())
        prob = plate_hole_problem(physics_train, sigma_inf, u_s, r, sampler.W,
                                  sampler.H, device=device, dtype=dtype)
        hist = train_energy(net, prob, epochs=0, lbfgs_iters=lbfgs_iters,
                            lbfgs_chunk=max(1, min(lbfgs_iters, 50)), verbose=False)
        return -float(hist['total'][-1])

    return inner_solve



def history_for_plot(hist: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Training history in the format visualization.plot_training_convergence expects
    for energy solvers: 'total' is the relative energy gap to the final value (the
    final point is that reference, so it is dropped from every array), 'strain' and
    'work' are the normalised energies, 'iter' the iteration axis.
    """
    pi = np.asarray(hist['total'], dtype=np.float64)
    gap = (pi - pi[-1]) / abs(pi[-1])
    keep = slice(0, len(pi) - 1)
    return {'energy': pi.tolist(),
            'total': np.maximum(gap[keep], 1e-12).tolist(),
            'strain': list(hist['strain'][keep]), 'work': list(hist['work'][keep]),
            'iter': list(hist['iter'][keep])}


class EnergyTrainer:
    """Minimal stand-in for XPINNTrainer: carries the training history in the
    keys visualization.plot_training_convergence expects. 'total' is the energy
    GAP (Pi - Pi_final, strictly positive) so it can be drawn on a log axis;
    the raw signed potential energy is kept under 'energy'.
    """

    def __init__(self):
        self.history: Dict[str, List[float]] = {'total': [], 'energy': [],
                                                'strain': [], 'work': [], 'iter': []}
        self.adam_epoch_count: Optional[int] = None


def train_energy_forward_solver(benchmark: str, device: str = 'cpu',
                                epochs: int = 2000, lbfgs_iters: int = 1500,
                                lbfgs_chunk: int = 100, lr: float = 2e-3,
                                print_every: int = 500,
                                hidden_layers: int = 4, neurons: int = 64,
                                stress_scale: float = 1e11, seed: Optional[int] = 42,
                                dtype=torch.float64):
    """Train a benchmark with the deep-energy method. Returns
    (model, trainer, sampler, physics_real) exactly like the residual path in
    atlas_main.train_forward_solver, so validation/plots/inverse design consume
    it unchanged. The model is a monolith XPINNModel whose single network is an
    EnergyNet (exact hard BCs); it is returned in float32 on `device`.
    """
    from atlas.networks import create_xpinn_for_benchmark
    from atlas.physics import PlaneStressElasticity
    from atlas.sampling import (CantileverBeamSampler, LBracketSampler,
                                PlateWithHoleSampler)

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    E, nu = 210e9, 0.3
    physics_real = PlaneStressElasticity(E=E, nu=nu)
    physics_train = PlaneStressElasticity(E=E / stress_scale, nu=nu)
    train_dev = device

    print("=" * 70)
    print(f"FORWARD SOLVE (deep-energy / Ritz) - {benchmark.upper()}")
    print("=" * 70)

    if benchmark == 'cantilever':
        L, H = 4.0, 1.0
        u_s = 1e6 * L ** 3 / (3.0 * E * (H ** 3 / 12.0))
        net, prob = build_cantilever(physics_train, -1e6 / stress_scale, u_s, L, H,
                                     hidden_layers=hidden_layers, neurons=neurons,
                                     device=train_dev, dtype=dtype)
        sampler = CantileverBeamSampler(L=L, H=H, device=device)
    elif benchmark == 'l_bracket':
        s = 1.0
        u_s = 1e6 * s ** 3 / (3.0 * E * (s ** 3 / 12.0))
        net, prob = build_l_bracket(physics_train, 1e6 / stress_scale, u_s, s,
                                    hidden_layers=hidden_layers, neurons=neurons,
                                    device=train_dev, dtype=dtype)
        sampler = LBracketSampler(size=s, device=device)
    elif benchmark == 'plate_hole':
        W = H = 2.0
        u_s = 1e6 * W / E
        net, prob = build_plate_hole(physics_train, 1e6 / stress_scale, u_s, 0.3, W, H,
                                     hidden_layers=hidden_layers, neurons=neurons,
                                     device=train_dev, dtype=dtype)
        sampler = PlateWithHoleSampler(W=W, H=H, r_init=0.3, device=device)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    print(f"  quadrature points: {prob.n_points}, "
          f"parameters: {sum(p.numel() for p in net.parameters())}, dtype: {dtype}")
    hist = train_energy(net, prob, epochs=epochs, lr=lr, lbfgs_iters=lbfgs_iters,
                        lbfgs_chunk=lbfgs_chunk, print_every=print_every)

    net = net.float().to(device)
    model = create_xpinn_for_benchmark(benchmark, device=device, monolith=True)
    model.subnets[0] = net

    trainer = EnergyTrainer()
    pi = np.asarray(hist['total'], dtype=np.float64)
    # Relative energy gap to the final value. The final point is that
    # reference (gap 0), so it is dropped from the plotted arrays.
    gap = (pi - pi[-1]) / abs(pi[-1])
    keep = slice(0, len(pi) - 1)
    trainer.history['energy'] = pi.tolist()
    trainer.history['total'] = np.maximum(gap[keep], 1e-12).tolist()
    trainer.history['strain'] = list(hist['strain'][keep])
    trainer.history['work'] = list(hist['work'][keep])
    trainer.history['iter'] = list(hist['iter'][keep])
    trainer.adam_epoch_count = epochs if lbfgs_iters > 0 else None
    print(f"  final potential energy Pi = {pi[-1]:+.10f} (normalized)")
    return model, trainer, sampler, physics_real


def load_energy_model(benchmark: str, path: str, device: str = 'cpu',
                      hidden_layers: int = 4, neurons: int = 64,
                      stress_scale: float = 1e11):
    """Rebuild the (model, sampler, physics_real) triple for a benchmark and load
    a state dict saved by scripts/run_benchmark.py (model.state_dict()).
    """
    from atlas.networks import create_xpinn_for_benchmark
    from atlas.physics import PlaneStressElasticity
    from atlas.sampling import (CantileverBeamSampler, LBracketSampler,
                                PlateWithHoleSampler)
    E, nu = 210e9, 0.3
    pt = PlaneStressElasticity(E=E / stress_scale, nu=nu)
    kw = dict(hidden_layers=hidden_layers, neurons=neurons, device='cpu')
    # output_scale (u_s) is a plain float, not part of the state dict, so it
    # must be rebuilt with exactly the value used in training.
    if benchmark == 'cantilever':
        u_s = 1e6 * 4.0 ** 3 / (3.0 * E * (1.0 / 12.0))
        net, _ = build_cantilever(pt, -1e6 / stress_scale, u_s, **kw)
        sampler = CantileverBeamSampler(L=4.0, H=1.0, device=device)
    elif benchmark == 'l_bracket':
        u_s = 1e6 / (3.0 * E * (1.0 / 12.0))
        net, _ = build_l_bracket(pt, 1e6 / stress_scale, u_s, **kw)
        sampler = LBracketSampler(size=1.0, device=device)
    elif benchmark == 'plate_hole':
        u_s = 1e6 * 2.0 / E
        net, _ = build_plate_hole(pt, 1e6 / stress_scale, u_s, **kw)
        sampler = PlateWithHoleSampler(W=2.0, H=2.0, r_init=0.3, device=device)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    model = create_xpinn_for_benchmark(benchmark, device='cpu', monolith=True)
    model.subnets[0] = net.float()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.to(device)
    return model, sampler, PlaneStressElasticity(E=E, nu=nu)
