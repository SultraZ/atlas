import math
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from atlas.energy_solver import (EnergyNet, EnergyProblem, _gauss_on_edges,
                                 graded_nodes, l_bracket_blocks, line_quadrature,
                                 train_energy)
from atlas.physics import PlaneStressElasticity

E_MOD, NU, STRESS_SCALE = 210e9, 0.3, 1e11


def von_mises_at(net, pts: np.ndarray, physics_real, device, dtype) -> np.ndarray:
    c = torch.tensor(pts, dtype=dtype, device=device, requires_grad=True)
    uv = net(c)
    gu = torch.autograd.grad(uv[:, 0].sum(), c, retain_graph=True)[0]
    gv = torch.autograd.grad(uv[:, 1].sum(), c)[0]
    exx, eyy, exy = gu[:, 0], gv[:, 1], 0.5 * (gu[:, 1] + gv[:, 0])
    sxx = physics_real.C11 * exx + physics_real.C12 * eyy
    syy = physics_real.C12 * exx + physics_real.C11 * eyy
    sxy = 2.0 * physics_real.C33 * exy
    vm = torch.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2)
    return vm.detach().cpu().numpy()


class DesignCase:
    benchmark = ''
    param_name = ''
    param_unit = 'm'
    lo = 0.0
    hi = 1.0
    p0 = 0.5
    default_metric = 'compliance'
    metrics_available = ('compliance', 'volume')
    hint = ''

    def __init__(self, device: str = 'cpu', dtype=torch.float64,
                 hidden_layers: int = 4, neurons: int = 64):
        self.device = device
        self.dtype = dtype
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        self.physics_train = PlaneStressElasticity(E=E_MOD / STRESS_SCALE, nu=NU)
        self.physics_real = PlaneStressElasticity(E=E_MOD, nu=NU)

    def geometry(self, p):
        raise NotImplementedError

    def make_net(self) -> EnergyNet:
        raise NotImplementedError

    def energy_scale(self) -> float:
        raise NotImplementedError

    def amplitude(self, p):
        return self._u_s

    def stress_points(self, p: float) -> Optional[np.ndarray]:
        return None

    def outline(self, p: float) -> np.ndarray:
        raise NotImplementedError

    def inside(self, X: np.ndarray, Y: np.ndarray, p: float) -> np.ndarray:
        raise NotImplementedError

    def _t(self, a):
        return torch.as_tensor(np.asarray(a) if not torch.is_tensor(a) else a,
                               dtype=self.dtype, device=self.device)

    def _pt(self, p, grad=False):
        return torch.tensor(float(p), dtype=self.dtype, device=self.device,
                            requires_grad=grad)

    def problem(self, p: float) -> EnergyProblem:
        with torch.no_grad():
            interior, loads, _ = self.geometry(self._pt(p))
        return EnergyProblem(self.physics_train, interior, loads, self.energy_scale(),
                             self.device, self.dtype)

    def new_net(self) -> EnergyNet:
        net = self.make_net().to(self.device).to(self.dtype)
        self.prepare(net, self.p0)
        return net

    def prepare(self, net, p: float):
        net.output_scale = float(self.amplitude(p))

    def volume(self, p: float) -> float:
        with torch.no_grad():
            return float(self.geometry(self._pt(p))[2])

    def compliance(self, net, p: float) -> float:
        prob = self.problem(p)
        self.prepare(net, p)
        with torch.no_grad():
            we = prob.external_work(net)
        return float(we) * STRESS_SCALE

    def peak_vm(self, net, p: float) -> Optional[float]:
        pts = self.stress_points(p)
        if pts is None:
            return None
        self.prepare(net, p)
        return float(von_mises_at(net, pts, self.physics_real, self.device, self.dtype).max())

    def metrics(self, net, p: float) -> Dict[str, float]:
        m = {'param': float(p), 'compliance': self.compliance(net, p),
             'volume': self.volume(p)}
        pk = self.peak_vm(net, p)
        if pk is not None:
            m['peak_vm'] = pk
        return m

    def dcompliance_dp(self, net, p: float) -> float:
        pt = self._pt(p, grad=True)
        interior, loads, _ = self.geometry(pt)
        prob = EnergyProblem(self.physics_train, interior, loads, self.energy_scale(),
                             self.device, self.dtype)
        old = net.output_scale
        net.output_scale = self.amplitude(pt)
        try:
            with torch.enable_grad():
                pi, _, _ = prob(net)
                (g,) = torch.autograd.grad(pi, pt)
        finally:
            net.output_scale = old
        return float(-2.0 * g) * self.energy_scale() * STRESS_SCALE

    def resolve(self, net, p: float, iters: int = 300, chunk: int = 50):
        self.prepare(net, p)
        train_energy(net, self.problem(p), epochs=0, lbfgs_iters=iters,
                     lbfgs_chunk=chunk, verbose=False, keep_best=True)


class CantileverDesign(DesignCase):
    benchmark = 'cantilever'
    param_name = 'beam height'
    lo, hi, p0 = 0.5, 1.6, 1.0
    default_metric = 'compliance'
    metrics_available = ('compliance', 'volume')
    hint = 'compliance falls as the beam gets taller (about 1/H^3)'
    L = 4.0

    def __init__(self, *a, nx: int = 60, ny: int = 20, ng: int = 3, **kw):
        super().__init__(*a, **kw)
        self.P = -1e6 / STRESS_SCALE
        self._u_s = 1e6 * self.L ** 3 / (3.0 * E_MOD * (self.p0 ** 3 / 12.0))
        xi_p, xi_w = _gauss_on_edges(graded_nodes(0, 1, nx, 2.0, 'a'), ng)
        eta_p, eta_w = _gauss_on_edges(graded_nodes(0, 1, ny, 1.6, 'both'), ng)
        XI, ETA = np.meshgrid(xi_p, eta_p, indexing='ij')
        self._XI, self._ETA = self._t(XI.reshape(-1)), self._t(ETA.reshape(-1))
        self._W = self._t(np.outer(xi_w, eta_w).reshape(-1))
        s_p, s_w = _gauss_on_edges(graded_nodes(0, 1, 10, 1.0, 'none'), 4)
        self._S, self._SW = self._t(s_p), self._t(s_w)

    def amplitude(self, p):
        return self._u_s * (self.p0 / p) ** 3

    def geometry(self, p):
        H = p if torch.is_tensor(p) else self._pt(p)
        pts = torch.stack([self.L * self._XI, H * self._ETA], dim=1)
        w = self._W * self.L * H
        lp = torch.stack([torch.full_like(self._S, self.L), H * self._S], dim=1)
        lw = H * self._SW
        ty = 6.0 * (self.P / H) * self._S * (1.0 - self._S)
        tr = torch.stack([torch.zeros_like(ty), ty], dim=1)
        return [(pts, w)], [(lp, lw, tr)], self.L * H

    def energy_scale(self):
        return abs(self.P) * self._u_s * self.p0

    def make_net(self):
        return EnergyNet((0.0, 0.0), (self.L, self.hi), self.hidden_layers, self.neurons,
                         self._u_s, 'clamp_x0', self.L)

    def outline(self, p):
        return np.array([[0, 0], [self.L, 0], [self.L, p], [0, p], [0, 0]], dtype=float)

    def inside(self, X, Y, p):
        return (Y <= p + 1e-12) & (X >= 0) & (X <= self.L) & (Y >= 0)


class PlateHoleDesign(DesignCase):
    benchmark = 'plate_hole'
    param_name = 'hole radius'
    lo, hi, p0 = 0.1, 0.8, 0.3
    default_metric = 'compliance'
    metrics_available = ('compliance', 'peak_vm', 'volume')
    hint = 'compliance grows with the hole size (less material)'
    W = H = 2.0

    def __init__(self, *a, n_xi: int = 24, n_th: int = 24, ng: int = 3, **kw):
        super().__init__(*a, **kw)
        self.sigma = 1e6 / STRESS_SCALE
        self._u_s = 1e6 * self.W / E_MOD
        th_split = math.atan2(self.H, self.W)
        xi_edges = graded_nodes(0.0, 1.0, n_xi, 1.5, 'a')
        self._blocks = []
        for th_lo, th_hi, rho_sq in [
                (0.0, th_split, lambda th: self.W / np.cos(th)),
                (th_split, 0.5 * math.pi, lambda th: self.H / np.sin(th))]:
            th_p, th_w = _gauss_on_edges(np.linspace(th_lo, th_hi, n_th + 1), ng)
            xi_p, xi_w = _gauss_on_edges(xi_edges, ng)
            TH, XI = np.meshgrid(th_p, xi_p, indexing='ij')
            self._blocks.append((self._t(TH), self._t(XI), self._t(rho_sq(TH)),
                                 self._t(np.outer(th_w, xi_w))))
        lp, lw, _ = line_quadrature((self.W, 0.0), (self.W, self.H), 12, 4)
        self._load = (self._t(lp), self._t(lw),
                      self._t(np.stack([np.full(lp.shape[0], self.sigma),
                                        np.zeros(lp.shape[0])], axis=1)))

    def geometry(self, p):
        r = p if torch.is_tensor(p) else self._pt(p)
        interior = []
        for TH, XI, RS, WT in self._blocks:
            RHO = r + XI * (RS - r)
            jac = RHO * (RS - r)
            pts = torch.stack([(RHO * torch.cos(TH)).reshape(-1),
                               (RHO * torch.sin(TH)).reshape(-1)], dim=1)
            interior.append((pts, (WT * jac).reshape(-1)))
        return interior, [self._load], self.W * self.H - math.pi * r ** 2 / 4.0

    def energy_scale(self):
        return abs(self.sigma) * self._u_s * self.H

    def make_net(self):
        return EnergyNet((0.0, 0.0), (self.W, self.H), self.hidden_layers, self.neurons,
                         self._u_s, 'sym', 1.0)

    def stress_points(self, p):
        th = np.linspace(0.0, 0.5 * math.pi, 181)
        return np.stack([p * np.cos(th), p * np.sin(th)], axis=1)

    def outline(self, p):
        th = np.linspace(0.5 * math.pi, 0.0, 60)
        arc = np.stack([p * np.cos(th), p * np.sin(th)], axis=1)
        return np.concatenate([arc, [[self.W, 0.0], [self.W, self.H], [0.0, self.H], [0.0, p]]])

    def inside(self, X, Y, p):
        return np.hypot(X, Y) >= p


class LBracketFilletDesign(DesignCase):
    benchmark = 'l_bracket'
    param_name = 'fillet radius'
    lo, hi, p0 = 0.05, 0.4, 0.1
    default_metric = 'peak_vm'
    metrics_available = ('peak_vm', 'compliance', 'volume')
    hint = 'peak stress falls as the fillet grows (compliance barely changes)'
    s = 1.0

    def __init__(self, *a, n: int = 24, n_th: int = 16, n_xi: int = 16, ng: int = 3, **kw):
        super().__init__(*a, **kw)
        s = self.s
        self.P = 1e6 / STRESS_SCALE
        self._u_s = 1e6 * s ** 3 / (3.0 * E_MOD * (s ** 3 / 12.0))
        self._base = [(self._t(p), self._t(w)) for p, w in l_bracket_blocks(s, n, ng, 3.0)]
        lp, lw, t = line_quadrature((2 * s, s), (2 * s, 2 * s), 12, 4)
        self._load = (self._t(lp), self._t(lw),
                      self._t(np.stack([6.0 * self.P * t * (1.0 - t), np.zeros_like(t)], axis=1)))
        xi_p, xi_w = _gauss_on_edges(graded_nodes(0.0, 1.0, n_xi, 1.5, 'a'), ng)
        self._patch = []
        for th_lo, th_hi, kind in [(-0.25 * math.pi, 0.0, 'x'), (-0.5 * math.pi, -0.25 * math.pi, 'y')]:
            th_p, th_w = _gauss_on_edges(np.linspace(th_lo, th_hi, n_th + 1), ng)
            TH, XI = np.meshgrid(th_p, xi_p, indexing='ij')
            k = 1.0 / np.cos(TH) if kind == 'x' else 1.0 / (-np.sin(TH))
            self._patch.append((self._t(TH), self._t(XI), self._t(k),
                                self._t(np.outer(th_w, xi_w))))

    def geometry(self, p):
        f = p if torch.is_tensor(p) else self._pt(p)
        s = self.s
        interior = list(self._base)
        cx, cy = s - f, s + f
        for TH, XI, K, WT in self._patch:
            rho = f + XI * (K * f - f)
            jac = rho * (K * f - f)
            pts = torch.stack([(cx + rho * torch.cos(TH)).reshape(-1),
                               (cy + rho * torch.sin(TH)).reshape(-1)], dim=1)
            interior.append((pts, (WT * jac).reshape(-1)))
        return interior, [self._load], 3.0 * s * s + f * f * (1.0 - math.pi / 4.0)

    def energy_scale(self):
        return abs(self.P) * self._u_s * self.s

    def make_net(self):
        s = self.s
        return EnergyNet((0.0, 0.0), (2 * s, 2 * s), self.hidden_layers, self.neurons,
                         self._u_s, 'clamp_x0', 2 * s)

    def stress_points(self, p):
        s = self.s
        th = np.linspace(-0.5 * math.pi, 0.0, 181)
        cx, cy = s - p, s + p
        arc = np.stack([cx + p * np.cos(th), cy + p * np.sin(th)], axis=1)
        e1 = np.stack([np.linspace(s - 0.5, s - p, 60), np.full(60, s)], axis=1)
        e2 = np.stack([np.full(60, s), np.linspace(s + p, s + 0.5, 60)], axis=1)
        return np.concatenate([arc, e1, e2], axis=0)

    def outline(self, p):
        s = self.s
        th = np.linspace(0.0, -0.5 * math.pi, 60)
        arc = np.stack([s - p + p * np.cos(th), s + p + p * np.sin(th)], axis=1)
        head = [[0, 0], [2 * s, 0], [2 * s, 2 * s], [s, 2 * s]]
        return np.concatenate([head, arc, [[0, s], [0, 0]]])

    def inside(self, X, Y, p):
        s = self.s
        base = (X >= 0) & (Y >= 0) & (X <= 2 * s) & (Y <= 2 * s) & ~((X < s) & (Y > s))
        patch = ((X >= s - p) & (X <= s) & (Y >= s) & (Y <= s + p)
                 & (np.hypot(X - (s - p), Y - (s + p)) >= p))
        return base | patch


CASES = {'cantilever': CantileverDesign, 'plate_hole': PlateHoleDesign,
         'l_bracket': LBracketFilletDesign}


def make_case(benchmark: str, device: str = 'cpu', **kw) -> DesignCase:
    if benchmark not in CASES:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return CASES[benchmark](device=device, **kw)


def train_forward(case: DesignCase, epochs: int = 2000, lbfgs_iters: int = 1500,
                  lr: float = 2e-3, p: Optional[float] = None, net: Optional[EnergyNet] = None,
                  seed: Optional[int] = 42, verbose: bool = True) -> EnergyNet:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    p = case.p0 if p is None else p
    net = net if net is not None else case.new_net()
    case.prepare(net, p)
    if hasattr(case, 'fit'):
        case.fit(net, p, epochs=epochs, lbfgs_iters=lbfgs_iters, lr=lr, verbose=verbose)
    else:
        train_energy(net, case.problem(p), epochs=epochs, lr=lr, lbfgs_iters=lbfgs_iters,
                     lbfgs_chunk=100, print_every=500, verbose=verbose)
    return net


def solve_inverse(case: DesignCase, net: EnergyNet, metric: str, target: float,
                  p_init: Optional[float] = None, tol: Optional[float] = None, max_iter: int = 14,
                  resolve_iters: int = 300, polish_iters: int = 1000,
                  use_autodiff: bool = True, verbose: bool = True,
                  callback: Optional[Callable[[Dict], None]] = None) -> Dict:
    if metric not in case.metrics_available:
        raise ValueError(f"metric '{metric}' not available for {case.benchmark}; "
                         f"choose from {case.metrics_available}")
    if tol is None:
        tol = 1e-2 if metric == 'peak_vm' else 2e-3
    lo, hi = case.lo, case.hi
    p = float(case.p0 if p_init is None else p_init)
    history: List[Dict] = []

    def evaluate(p_, iters, tag):
        case.resolve(net, p_, iters)
        m = case.metrics(net, p_)
        g = m[metric] / target - 1.0
        row = dict(m, iter=len(history), g=g, target=target, tag=tag)
        history.append(row)
        if verbose:
            print(f"  [{tag:>7s}] {case.param_name}={p_:.5f}  {metric}={m[metric]:.6e}  "
                  f"g={g:+.4e}", flush=True)
        if callback:
            callback(row)
        return m, g

    m, g = evaluate(p, max(resolve_iters, 100), 'start')
    bracket = {'neg': None, 'pos': None}

    def note(p_, g_):
        bracket['neg' if g_ < 0 else 'pos'] = (p_, g_)

    note(p, g)
    prev = None
    status = 'max_iter'
    slope = None
    for k in range(max_iter):
        if abs(g) < tol:
            status = 'converged'
            break
        source = 'secant'
        if metric == 'compliance' and use_autodiff:
            slope = case.dcompliance_dp(net, p) / target
            source = 'autodiff'
        elif prev is not None and abs(prev[0] - p) > 1e-9:
            slope = (g - prev[1]) / (p - prev[0])
        else:
            step = 0.03 * (hi - lo)
            p_probe = min(max(p + step, lo), hi)
            if abs(p_probe - p) < 1e-9:
                p_probe = p - step
            _, g_probe = evaluate(p_probe, resolve_iters, 'probe')
            slope = (g_probe - g) / (p_probe - p)
            note(p_probe, g_probe)
            source = 'fd'
            m, g = evaluate(p, resolve_iters, 'return')
        if not np.isfinite(slope) or abs(slope) < 1e-12:
            status = 'flat'
            break
        p_new = p - g / slope
        max_step = 0.35 * (hi - lo)
        p_new = p + float(np.clip(p_new - p, -max_step, max_step))
        p_new = float(np.clip(p_new, lo, hi))
        if bracket['neg'] is not None and bracket['pos'] is not None:
            a, b = sorted([bracket['neg'][0], bracket['pos'][0]])
            if not (a < p_new < b):
                p_new = 0.5 * (a + b)
        if abs(p_new - p) < 1e-7:
            status = 'stalled_at_bound' if p_new in (lo, hi) else 'converged'
            break
        prev = (p, g)
        p = p_new
        m, g = evaluate(p, resolve_iters, source)
        history[-1]['grad_source'] = source
        note(p, g)
        if p in (lo, hi) and abs(g) >= tol and prev is not None and (g * prev[1] > 0):
            status = 'target_not_reachable'
            break
    else:
        status = 'converged' if abs(g) < tol else 'max_iter'
    if abs(g) < tol:
        status = 'converged'

    m, g = evaluate(p, polish_iters, 'polish')
    best = (abs(g), p, m, g)
    slope_last = slope
    for _ in range(2):
        if abs(g) < tol or not slope_last or not np.isfinite(slope_last) \
                or abs(slope_last) < 1e-12 or status in ('target_not_reachable', 'flat'):
            break
        step = float(np.clip(-g / slope_last, -0.04 * (hi - lo), 0.04 * (hi - lo)))
        p_new = float(np.clip(p + step, lo, hi))
        if abs(p_new - p) < 1e-7:
            break
        p = p_new
        m, g = evaluate(p, polish_iters, 'refine')
        if abs(g) < best[0]:
            best = (abs(g), p, m, g)
    if best[1] != p:
        p = best[1]
        m, g = evaluate(p, polish_iters, 'restore')
    if abs(g) < tol:
        status = 'converged'
    elif status == 'converged':
        status = 'max_iter'
    return dict(benchmark=case.benchmark, param_name=case.param_name, metric=metric,
                target=target, p_star=p, metrics=m, g=g, status=status,
                converged=abs(g) < tol, history=history, bounds=(lo, hi), tol=tol)


def run_inverse_design(benchmark: str, metric: Optional[str] = None,
                       target_ratio: Optional[float] = None,
                       target_value: Optional[float] = None,
                       p_init: Optional[float] = None, device: str = 'cpu',
                       net: Optional[EnergyNet] = None, forward_epochs: int = 2000,
                       forward_lbfgs: Optional[int] = None, seed: int = 42, verbose: bool = True,
                       decomposed: bool = False, **solver_kw) -> Dict:
    if decomposed:
        from atlas.energy_xpinn_design import make_xpinn_case
        case = make_xpinn_case(benchmark, device=device)
    else:
        case = make_case(benchmark, device=device)
    if forward_lbfgs is None:
        forward_lbfgs = 2000 if decomposed else 1500
    metric = metric or case.default_metric
    if net is None:
        net = train_forward(case, forward_epochs, forward_lbfgs, seed=seed, verbose=verbose)
    base = case.metrics(net, case.p0)
    if target_value is None:
        target_ratio = {'cantilever': 0.6, 'plate_hole': 1.10, 'l_bracket': 0.75}[benchmark] \
            if target_ratio is None else target_ratio
        target_value = target_ratio * base[metric]
    if verbose:
        print(f"baseline {case.param_name}={case.p0}: {base}")
        print(f"target {metric} = {target_value:.6e}")
    if p_init is not None and abs(p_init - case.p0) > 1e-12:
        case.resolve(net, p_init, 1500)
    out = solve_inverse(case, net, metric, target_value, p_init=p_init,
                        verbose=verbose, **solver_kw)
    out.update(case=case, net=net, baseline=base)
    return out


def design_fields(case: DesignCase, net: EnergyNet, p: float, n: int = 140) -> Dict:
    case.prepare(net, p)
    if case.benchmark == 'cantilever':
        xs, ys = np.linspace(0, case.L, n), np.linspace(0, p, max(12, n // 4))
    elif case.benchmark == 'plate_hole':
        xs, ys = np.linspace(0, case.W, n), np.linspace(0, case.H, n)
    else:
        xs, ys = np.linspace(0, 2 * case.s, n), np.linspace(0, 2 * case.s, n)
    X, Y = np.meshgrid(xs, ys)
    inside = case.inside(X, Y, p)
    pts = np.stack([X[inside], Y[inside]], axis=1)
    c = torch.tensor(pts, dtype=case.dtype, device=case.device)
    with torch.no_grad():
        uv = net(c).cpu().numpy()
    vm = von_mises_at(net, pts, case.physics_real, case.device, case.dtype)
    out = {}
    for name, val in (('u', uv[:, 0]), ('v', uv[:, 1]),
                      ('disp', np.hypot(uv[:, 0], uv[:, 1])), ('vm', vm)):
        full = np.full(X.shape, np.nan)
        full[inside] = val
        out[name] = full
    out.update(X=X, Y=Y)
    return out
