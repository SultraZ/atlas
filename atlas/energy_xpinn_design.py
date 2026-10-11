import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from atlas.energy_design import (CantileverDesign, DesignCase, LBracketFilletDesign,
                                 PlateHoleDesign, STRESS_SCALE)
from atlas.energy_solver import (EnergyNet, _gauss_on_edges, graded_nodes,
                                 l_bracket_blocks, line_quadrature)
from atlas.energy_xpinn import (DecomposedEnergyProblem, FAR_MARGIN, MID_MARGIN)
from atlas.energy_solver import train_energy


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
        train_decomposed(net, self.problem(p), epochs=epochs, lr=lr, lbfgs_per_stage=lbfgs_iters,
                         gammas=(self.gamma_u,), verbose=verbose, al_stages=self.al_stages,
                         warm_start=self.warm_start)

    def resolve(self, net, p: float, iters: int = 300, chunk: int = 50):
        self.prepare(net, p)
        prob = self.problem(p)
        n = max(chunk, iters // (self.al_stages + 1))
        for k in range(self.al_stages + 1):
            if k:
                prob.update_multipliers(net)
            train_energy(net, prob, epochs=0, lbfgs_iters=n, lbfgs_chunk=chunk,
                         verbose=False, keep_best=True)

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
        prob = self._problem_from(pt)
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


XPINN_CASES = {'cantilever': CantileverXPINNDesign, 'plate_hole': PlateHoleXPINNDesign,
               'l_bracket': LBracketXPINNDesign}


def make_xpinn_case(benchmark: str, device: str = 'cpu', **kw) -> XPINNDesign:
    if benchmark not in XPINN_CASES:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return XPINN_CASES[benchmark](device=device, **kw)
