import math

import numpy as np
import torch
import torch.nn as nn

from atlas.energy_design import (CantileverDesign, DesignCase, LBracketFilletDesign,
                                 PlateHoleDesign, STRESS_SCALE, von_mises_at)
from atlas.energy_solver import (EnergyNet, _gauss_on_edges, graded_nodes,
                                 l_bracket_blocks, line_quadrature)
from atlas.energy_xpinn import (DecomposedEnergyProblem, FAR_MARGIN, MID_MARGIN)
from atlas.energy_solver import train_energy
from atlas.physics import PlaneStressElasticity


class RoutedXPINN(nn.Module):
    def __init__(self, subnets, router):
        super().__init__()
        self.subnets = nn.ModuleList(subnets)
        self.num_subdomains = len(subnets)
        self._router = router
        self._p = None

    def set_p(self, p: float):
        self._p = float(p)

    def get_subdomain_network(self, i: int):
        return self.subnets[i]

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        idx = self._router(coords, self._p)
        out = self.subnets[0](coords)
        for i in range(1, self.num_subdomains):
            out = torch.where((idx == i).unsqueeze(1), self.subnets[i](coords), out)
        return out


class InclusionDesign(DesignCase):
    param_unit = 'm'
    metrics_available = ('compliance', 'peak_vm', 'volume')
    W = H = 2.0
    E1, NU1, E2, NU2 = 210e9, 0.3, 210e9, 0.3

    def __init__(self, *a, E1=None, nu1=None, E2=None, nu2=None, sigma: float = 1e6,
                 p0: float = None, **kw):
        E1 = self.E1 if E1 is None else E1
        nu1 = self.NU1 if nu1 is None else nu1
        E2 = self.E2 if E2 is None else E2
        nu2 = self.NU2 if nu2 is None else nu2
        super().__init__(*a, E=E1, nu=nu1, **kw)
        self.E1, self.nu1, self.E2, self.nu2 = E1, nu1, E2, nu2
        self.inc_train = PlaneStressElasticity(E=E2 / STRESS_SCALE, nu=nu2)
        self.inc_real = PlaneStressElasticity(E=E2, nu=nu2)
        if p0 is not None:
            self.p0 = float(p0)
        self.sigma = sigma / STRESS_SCALE
        self._u_s = sigma * self.W / E1

    def axes(self, p):
        return p, p

    def r_e(self, th, p):
        a, b = self.axes(p)
        lib = torch if torch.is_tensor(th) else np
        return a * b / lib.sqrt((b * lib.cos(th)) ** 2 + (a * lib.sin(th)) ** 2)

    def energy_scale(self):
        return abs(self.sigma) * self._u_s * self.H

    def make_net(self):
        return EnergyNet((0.0, 0.0), (self.W, self.H), self.hidden_layers, self.neurons,
                         self._u_s, 'sym', 1.0)

    def stress_points(self, p):
        th = np.linspace(0.0, 0.5 * math.pi, 181)
        r = self.r_e(th, float(p)) * (1.0 + 2e-3)
        return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)

    def outline(self, p):
        th = np.linspace(0.0, 0.5 * math.pi, 90)
        r = self.r_e(th, float(p))
        arc = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
        return np.concatenate([arc, [[0.0, self.H], [self.W, self.H], [self.W, 0.0], arc[0]]])

    def inside(self, X, Y, p):
        return np.ones(np.shape(X), dtype=bool)

    def in_inclusion(self, X, Y, p):
        a, b = self.axes(float(p))
        return (np.asarray(X) / a) ** 2 + (np.asarray(Y) / b) ** 2 < 1.0

    def volume(self, p: float) -> float:
        a, b = self.axes(float(p))
        return math.pi * a * b / 4.0


class SteelInsertDesign(InclusionDesign):
    benchmark = 'steel_insert'
    param_name = 'insert radius'
    lo, hi, p0 = 0.15, 0.75, 0.3
    default_metric = 'compliance'
    hint = 'a larger steel insert makes the aluminium plate stiffer'
    E1, NU1, E2, NU2 = 70e9, 0.33, 210e9, 0.30


class SoftInclusionDesign(InclusionDesign):
    benchmark = 'bimaterial'
    param_name = 'inclusion radius'
    lo, hi, p0 = 0.15, 0.75, 0.3
    default_metric = 'compliance'
    hint = 'a larger soft inclusion makes the steel plate more flexible'
    E1, NU1, E2, NU2 = 210e9, 0.30, 14e9, 0.30


class FilledSlotDesign(InclusionDesign):
    benchmark = 'filled_slot'
    param_name = 'slot semi-axis along the load'
    lo, hi, p0 = 0.08, 0.6, 0.2
    default_metric = 'peak_vm'
    hint = 'a wider, rounder slot lowers the peak stress at its tip'
    E1, NU1, E2, NU2 = 210e9, 0.30, 3.5e9, 0.35

    def __init__(self, *a, b: float = 0.4, **kw):
        self.b = float(b)
        super().__init__(*a, **kw)

    def axes(self, p):
        return p, self.b


class XPINNDesign(DesignCase):
    base_cls = None
    n_sub = 0

    def __init__(self, device: str = 'cpu', dtype=torch.float64, hidden_layers: int = 4,
                 neurons: int = 64, gamma_u: float = 30.0, gamma_t: float = 0.1,
                 al_stages: int = 3, warm_start: int = 1000, **kw):
        super().__init__(device, dtype, hidden_layers, neurons)
        self.al_stages, self.warm_start = int(al_stages), int(warm_start)
        self.base = self.base_cls(device=device, dtype=dtype, hidden_layers=hidden_layers,
                                  neurons=neurons, **kw)
        self.E, self.nu = self.base.E, self.base.nu
        self.physics_train, self.physics_real = self.base.physics_train, self.base.physics_real
        for a in ('benchmark', 'param_name', 'param_unit', 'lo', 'hi', 'p0',
                  'default_metric', 'metrics_available', 'hint'):
            setattr(self, a, getattr(self.base, a))
        self._u_s = self.base._u_s
        self.gamma_u, self.gamma_t = gamma_u, gamma_t
        self.length = self._length()

    def __getattr__(self, name):
        base = self.__dict__.get('base')
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def amplitude(self, p):
        return self.base.amplitude(p)

    def energy_scale(self):
        return self.base.energy_scale()

    def stress_points(self, p):
        return self.base.stress_points(p)

    def outline(self, p):
        return self.base.outline(p)

    def inside(self, X, Y, p):
        return self.base.inside(X, Y, p)

    def volume(self, p: float) -> float:
        return self.base.volume(p)

    def parts(self, p):
        raise NotImplementedError

    def _length(self) -> float:
        raise NotImplementedError

    def _router(self, coords, p):
        raise NotImplementedError

    def _mk_net(self) -> EnergyNet:
        return self.base.make_net()

    def _problem_from(self, p):
        blocks, loads, ifaces, _ = self.parts(p)
        return DecomposedEnergyProblem(self.physics_train, blocks, loads, ifaces,
                                       self.energy_scale(), self._u_s, self.length,
                                       self.gamma_u, self.gamma_t, self.device, self.dtype)

    def problem(self, p: float):
        with torch.no_grad():
            return self._problem_from(self._pt(p))

    def make_net(self):
        return RoutedXPINN([self._mk_net() for _ in range(self.n_sub)], self._router)

    def fit(self, net, p: float, epochs: int = 2000, lbfgs_iters: int = 1500, lr: float = 2e-3,
            verbose: bool = True):
        from atlas.energy_xpinn import train_decomposed
        self.prepare(net, p)
        prob = self.problem(p)
        train_decomposed(net, prob, epochs=epochs, lr=lr, lbfgs_per_stage=lbfgs_iters,
                         gammas=(self.gamma_u,), verbose=verbose, al_stages=self.al_stages,
                         warm_start=self.warm_start)
        self._lam = [l.detach().clone() for l in prob.lam]

    def resolve(self, net, p: float, iters: int = 300, chunk: int = 50):
        self.prepare(net, p)
        prob = self.problem(p)
        n = max(chunk, iters // (self.al_stages + 1))
        for k in range(self.al_stages + 1):
            if k:
                prob.update_multipliers(net)
            train_energy(net, prob, epochs=0, lbfgs_iters=n, lbfgs_chunk=chunk,
                         verbose=False, keep_best=True)
        self._lam = [l.detach().clone() for l in prob.lam]

    def _shape_problem(self, pt):
        return self._problem_from(pt)

    def prepare(self, net, p: float):
        net.set_p(p)
        amp = float(self.amplitude(p))
        for sub in net.subnets:
            sub.output_scale = amp

    def compliance(self, net, p: float) -> float:
        prob = self.problem(p)
        self.prepare(net, p)
        with torch.no_grad():
            we = prob.external_work(net)
        return float(we) * STRESS_SCALE

    def dcompliance_dp(self, net, p: float) -> float:
        pt = self._pt(p, grad=True)
        prob = self._shape_problem(pt)
        old = [s.output_scale for s in net.subnets]
        amp = self.amplitude(pt)
        for s in net.subnets:
            s.output_scale = amp
        try:
            with torch.enable_grad():
                phi, _, _ = prob(net)
                (g,) = torch.autograd.grad(phi, pt)
        finally:
            for s, o in zip(net.subnets, old):
                s.output_scale = o
        return float(-2.0 * g) * self.energy_scale() * STRESS_SCALE


def _ref(edges, ng=3):
    return _gauss_on_edges(edges, ng)


class CantileverXPINNDesign(XPINNDesign):
    base_cls = CantileverDesign
    n_sub = 2

    def __init__(self, *a, nx_half: int = 30, ny: int = 20, ng: int = 3, **kw):
        super().__init__(*a, **kw)
        L = self.base.L
        eta_p, eta_w = _ref(graded_nodes(0, 1, ny, 1.6, 'both'), ng)
        self._grids = []
        for lo_, hi_, power, toward in ((0.0, L / 2, 2.0, 'a'), (L / 2, L, 1.0, 'none')):
            xi_p, xi_w = _ref(graded_nodes(lo_, hi_, nx_half, power, toward), ng)
            XI, ETA = np.meshgrid(xi_p, eta_p, indexing='ij')
            self._grids.append((self._t(XI.reshape(-1)), self._t(ETA.reshape(-1)),
                                self._t(np.outer(xi_w, eta_w).reshape(-1))))
        s_p, s_w = _gauss_on_edges(graded_nodes(0, 1, 10, 1.0, 'none'), 4)
        self._S, self._SW = self._t(s_p), self._t(s_w)

    def _length(self):
        return self.base.L

    def _router(self, coords, p):
        return (coords[:, 0] > self.base.L / 2).long()

    def parts(self, p):
        H = p if torch.is_tensor(p) else self._pt(p)
        L, P = self.base.L, self.base.P
        blocks = [(i, torch.stack([XI, H * ETA], dim=1), W * H)
                  for i, (XI, ETA, W) in enumerate(self._grids)]
        ip = torch.stack([torch.full_like(self._S, L / 2), H * self._S], dim=1)
        n = torch.stack([torch.ones_like(self._S), torch.zeros_like(self._S)], dim=1)
        ifaces = [dict(a=0, b=1, pts=ip, w=H * self._SW, n=n)]
        lp = torch.stack([torch.full_like(self._S, L), H * self._S], dim=1)
        ty = 6.0 * (P / H) * self._S * (1.0 - self._S)
        loads = [(1, lp, H * self._SW, torch.stack([torch.zeros_like(ty), ty], dim=1))]
        return blocks, loads, ifaces, L * H


class PlateHoleXPINNDesign(XPINNDesign):
    base_cls = PlateHoleDesign
    n_sub = 3

    def __init__(self, *a, n_th: int = 20, n_xi=(12, 8, 14), ng: int = 3, **kw):
        super().__init__(*a, **kw)
        W, H = self.base.W, self.base.H
        split = math.atan2(H, W)
        specs = [(0.0, split, lambda th: W / np.cos(th)),
                 (split, 0.5 * math.pi, lambda th: H / np.sin(th))]
        self._rings = []
        for th_lo, th_hi, rho_sq in specs:
            th_p, th_w = _ref(np.linspace(th_lo, th_hi, n_th + 1), ng)
            for ring, (power, toward) in enumerate(((1.5, 'a'), (1.0, 'none'), (1.0, 'none'))):
                xi_p, xi_w = _ref(graded_nodes(0.0, 1.0, n_xi[ring], power, toward), ng)
                TH, XI = np.meshgrid(th_p, xi_p, indexing='ij')
                self._rings.append((ring, self._t(TH), self._t(XI),
                                    self._t(rho_sq(TH)) if ring == 2 else None,
                                    self._t(np.outer(th_w, xi_w))))
        th_p, th_w = _gauss_on_edges(np.linspace(0.0, 0.5 * math.pi, 25), 4)
        self._ith, self._itw = self._t(th_p), self._t(th_w)
        lp, lw, _ = line_quadrature((W, 0.0), (W, H), 12, 4)
        self._load = (2, self._t(lp), self._t(lw),
                      self._t(np.stack([np.full(lp.shape[0], self.base.sigma),
                                        np.zeros(lp.shape[0])], axis=1)))

    def _length(self):
        return self.base.W

    def _router(self, coords, p):
        d = torch.hypot(coords[:, 0], coords[:, 1])
        return (d > p + MID_MARGIN).long() + (d > p + FAR_MARGIN).long()

    def parts(self, p):
        r = p if torch.is_tensor(p) else self._pt(p)
        r_mid, r_far = r + MID_MARGIN, r + FAR_MARGIN
        blocks = []
        for ring, TH, XI, RS, WT in self._rings:
            a, b = ((r, r_mid), (r_mid, r_far), (r_far, RS))[ring]
            RHO = a + XI * (b - a)
            jac = RHO * (b - a)
            pts = torch.stack([(RHO * torch.cos(TH)).reshape(-1),
                               (RHO * torch.sin(TH)).reshape(-1)], dim=1)
            blocks.append((ring, pts, (WT * jac).reshape(-1)))
        ifaces = []
        n = torch.stack([torch.cos(self._ith), torch.sin(self._ith)], dim=1)
        for a, b, rad in ((0, 1, r_mid), (1, 2, r_far)):
            pts = torch.stack([rad * torch.cos(self._ith), rad * torch.sin(self._ith)], dim=1)
            ifaces.append(dict(a=a, b=b, pts=pts, w=rad * self._itw, n=n))
        W, H = self.base.W, self.base.H
        return blocks, [self._load], ifaces, W * H - math.pi * r ** 2 / 4.0


class LBracketXPINNDesign(XPINNDesign):
    base_cls = LBracketFilletDesign
    n_sub = 3

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        s = self.base.s
        self._base_blocks = [(i, self._t(p), self._t(w))
                             for i, (p, w) in enumerate(l_bracket_blocks(s))]
        self._patch = self.base._patch
        self._load = (2, self.base._load[0], self.base._load[1], self.base._load[2])
        i01p, i01w, _ = line_quadrature((s, 0.0), (s, s), 12, 4)
        i12p, i12w, _ = line_quadrature((s, s), (2 * s, s), 12, 4)
        one, zero = np.tile([1.0, 0.0], (i01p.shape[0], 1)), np.tile([0.0, 1.0], (i12p.shape[0], 1))
        self._ifaces = [dict(a=0, b=1, pts=self._t(i01p), w=self._t(i01w), n=self._t(one)),
                        dict(a=1, b=2, pts=self._t(i12p), w=self._t(i12w), n=self._t(zero))]
        t_p, t_w = _gauss_on_edges(graded_nodes(0, 1, 8, 1.0, 'none'), 4)
        self._T, self._TW = self._t(t_p), self._t(t_w)

    def _length(self):
        return self.base.s

    def _mk_net(self):
        return self.base.make_net()

    def _router(self, coords, p):
        s = self.base.s
        x, y = coords[:, 0], coords[:, 1]
        idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        idx = torch.where((x > s) & (y <= s), torch.ones_like(idx), idx)
        idx = torch.where((x >= s) & (y > s), 2 * torch.ones_like(idx), idx)
        return idx

    def parts(self, p):
        f = p if torch.is_tensor(p) else self._pt(p)
        s = self.base.s
        blocks = list(self._base_blocks)
        cx, cy = s - f, s + f
        for TH, XI, K, WT in self._patch:
            rho = f + XI * (K * f - f)
            jac = rho * (K * f - f)
            pts = torch.stack([(cx + rho * torch.cos(TH)).reshape(-1),
                               (cy + rho * torch.sin(TH)).reshape(-1)], dim=1)
            blocks.append((0, pts, (WT * jac).reshape(-1)))
        ifaces = list(self._ifaces)
        pp = torch.stack([torch.full_like(self._T, s), s + f * self._T], dim=1)
        nn_ = torch.stack([torch.ones_like(self._T), torch.zeros_like(self._T)], dim=1)
        ifaces.append(dict(a=0, b=2, pts=pp, w=f * self._TW, n=nn_))
        area = 3.0 * s * s + f * f * (1.0 - math.pi / 4.0)
        return blocks, [self._load], ifaces, area



class InclusionXPINNDesign(XPINNDesign):
    n_sub = 2

    def __init__(self, *a, n_th: int = 24, n_xi=(12, 24), n_if: int = 32, ng: int = 3, **kw):
        super().__init__(*a, **kw)
        W, H = self.base.W, self.base.H
        split = math.atan2(H, W)
        th_p, th_w = _ref(np.linspace(0.0, 0.5 * math.pi, n_th + 1), ng)
        xi_p, xi_w = _ref(graded_nodes(0.0, 1.0, n_xi[0], 1.5, 'b'), ng)
        TH, XI = np.meshgrid(th_p, xi_p, indexing='ij')
        self._inc = (self._t(TH), self._t(XI), self._t(np.outer(th_w, xi_w)))
        self._mat = []
        for th_lo, th_hi, rho_sq in [(0.0, split, lambda th: W / np.cos(th)),
                                     (split, 0.5 * math.pi, lambda th: H / np.sin(th))]:
            th_p, th_w = _ref(np.linspace(th_lo, th_hi, n_th + 1), ng)
            xi_p, xi_w = _ref(graded_nodes(0.0, 1.0, n_xi[1], 1.5, 'a'), ng)
            TH, XI = np.meshgrid(th_p, xi_p, indexing='ij')
            self._mat.append((self._t(TH), self._t(XI), self._t(rho_sq(TH)), self._t(np.outer(th_w, xi_w))))
        th_p, th_w = _gauss_on_edges(np.linspace(0.0, 0.5 * math.pi, n_if + 1), 4)
        self._ith, self._itw = self._t(th_p), self._t(th_w)
        lp, lw, _ = line_quadrature((W, 0.0), (W, H), 12, 4)
        self._load = (1, self._t(lp), self._t(lw),
                      self._t(np.stack([np.full(lp.shape[0], self.base.sigma),
                                        np.zeros(lp.shape[0])], axis=1)))

    def _length(self):
        return self.base.W

    def _router(self, coords, p):
        a, b = self.base.axes(p)
        return ((coords[:, 0] / a) ** 2 + (coords[:, 1] / b) ** 2 >= 1.0).long()

    def parts(self, p):
        pt = p if torch.is_tensor(p) else self._pt(p)
        a, b = self.base.axes(pt)
        r_e = lambda th: self.base.r_e(th, pt)
        TH, XI, WT = self._inc
        R = r_e(TH)
        RHO = XI * R
        pts = torch.stack([(RHO * torch.cos(TH)).reshape(-1), (RHO * torch.sin(TH)).reshape(-1)], dim=1)
        blocks = [(0, pts, (WT * RHO * R).reshape(-1), self.base.inc_train)]
        for TH, XI, RS, WT in self._mat:
            R = r_e(TH)
            RHO = R + XI * (RS - R)
            pts = torch.stack([(RHO * torch.cos(TH)).reshape(-1), (RHO * torch.sin(TH)).reshape(-1)], dim=1)
            blocks.append((1, pts, (WT * RHO * (RS - R)).reshape(-1), self.physics_train))
        th = self._ith
        c, s_ = torch.cos(th), torch.sin(th)
        q = (b * c) ** 2 + (a * s_) ** 2
        r = a * b / torch.sqrt(q)
        dr = a * b * (b * b - a * a) * s_ * c / q ** 1.5
        pts = torch.stack([r * c, r * s_], dim=1)
        g = torch.stack([pts[:, 0] / a ** 2, pts[:, 1] / b ** 2], dim=1)
        ifaces = [dict(a=0, b=1, pts=pts, w=self._itw * torch.sqrt(r * r + dr * dr),
                       n=g / torch.linalg.norm(g, dim=1, keepdim=True),
                       phys_a=self.base.inc_train, phys_b=self.physics_train)]
        return blocks, [self._load], ifaces, math.pi * a * b / 4.0

    def _shape_problem(self, pt):
        prob = self._problem_from(pt)
        lam = self.__dict__.get('_lam')
        if lam is not None:
            prob.lam = [l.clone() for l in lam]
        return prob

    def peak_vm(self, net, p: float) -> float:
        self.prepare(net, p)
        return float(von_mises_at(net.subnets[1], self.stress_points(p), self.physics_real,
                                  self.device, self.dtype).max())

    def field_vm(self, net, pts: np.ndarray, p: float) -> np.ndarray:
        ins = self.base.in_inclusion(pts[:, 0], pts[:, 1], p)
        vm = np.empty(len(pts))
        for k, (mask, ph) in enumerate(((ins, self.base.inc_real), (~ins, self.physics_real))):
            if mask.any():
                vm[mask] = von_mises_at(net.subnets[k], pts[mask], ph, self.device, self.dtype)
        return vm


class SteelInsertXPINNDesign(InclusionXPINNDesign):
    base_cls = SteelInsertDesign


class SoftInclusionXPINNDesign(InclusionXPINNDesign):
    base_cls = SoftInclusionDesign


class FilledSlotXPINNDesign(InclusionXPINNDesign):
    base_cls = FilledSlotDesign

XPINN_CASES = {'cantilever': CantileverXPINNDesign, 'plate_hole': PlateHoleXPINNDesign,
               'l_bracket': LBracketXPINNDesign, 'steel_insert': SteelInsertXPINNDesign,
               'bimaterial': SoftInclusionXPINNDesign, 'filled_slot': FilledSlotXPINNDesign}


def make_xpinn_case(benchmark: str, device: str = 'cpu', **kw) -> XPINNDesign:
    if benchmark not in XPINN_CASES:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return XPINN_CASES[benchmark](device=device, **kw)
