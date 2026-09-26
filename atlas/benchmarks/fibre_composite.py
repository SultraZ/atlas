import math

import numpy as np
import torch
import torch.nn as nn

from atlas.energy_solver import EnergyNet, EnergyProblem, _gauss_on_edges, line_quadrature
from atlas.energy_xpinn import DecomposedEnergyProblem, _polar_block
from atlas.physics import PlaneStressElasticity

E_FIB, NU_FIB = 72e9, 0.22
E_MAT, NU_MAT = 3.5e9, 0.35
STRESS_SCALE = 1e10
SIGMA = 1e6
W, H, S, RF = 2.0, 1.0, 0.5, 0.15
CENTRES = [((i + 0.5) * S, (j + 0.5) * S) for j in range(int(round(H / S))) for i in range(int(round(W / S)))]


def fibre_index(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    out = np.full(x.shape, -1)
    for k, (cx, cy) in enumerate(CENTRES):
        out[np.hypot(x - cx, y - cy) < RF] = k
    return out


def in_fibre(x, y):
    return fibre_index(x, y) >= 0


def material_arrays(x, y):
    f = in_fibre(x, y)
    return np.where(f, E_FIB, E_MAT), np.where(f, NU_FIB, NU_MAT)


def materials():
    return (PlaneStressElasticity(E=E_MAT / STRESS_SCALE, nu=NU_MAT),
            PlaneStressElasticity(E=E_FIB / STRESS_SCALE, nu=NU_FIB))


def blocks(n_th_fib: int = 24, n_xi_fib: int = 6, n_th_side: int = 6, n_xi_mat: int = 8, ng: int = 3):
    mat, fib = materials()
    h = 0.5 * S
    sides = [(-0.25 * math.pi, 0.25 * math.pi, lambda th: h / np.cos(th)),
             (0.25 * math.pi, 0.75 * math.pi, lambda th: h / np.sin(th)),
             (0.75 * math.pi, 1.25 * math.pi, lambda th: -h / np.cos(th)),
             (1.25 * math.pi, 1.75 * math.pi, lambda th: -h / np.sin(th))]
    out = []
    for c in CENTRES:
        p, w = _polar_block(0.0, RF, 0.0, 2 * math.pi, n_th_fib, n_xi_fib, 1.5, 'b', ng)
        out.append((0, p + np.array(c), w, fib))
        for lo, hi, outer in sides:
            p, w = _polar_block(RF, outer, lo, hi, n_th_side, n_xi_mat, 1.5, 'a', ng)
            out.append((1, p + np.array(c), w, mat))
    return out


def loads(sigma: float):
    lp, lw, _ = line_quadrature((W, 0.0), (W, H), 16, 4)
    tr = np.stack([np.full(lp.shape[0], sigma), np.zeros(lp.shape[0])], axis=1)
    return lp, lw, tr


def interfaces(n_cells: int = 32, ng: int = 4):
    mat, fib = materials()
    th, w = _gauss_on_edges(np.linspace(0.0, 2 * math.pi, n_cells + 1), ng)
    n = np.stack([np.cos(th), np.sin(th)], axis=1)
    pts = np.concatenate([np.array(c) + RF * n for c in CENTRES])
    return [dict(a=0, b=1, pts=pts, w=np.tile(RF * w, len(CENTRES)), n=np.tile(n, (len(CENTRES), 1)),
                 phys_a=fib, phys_b=mat)]


def characteristic_displacement() -> float:
    return SIGMA * W / E_MAT


def _setup():
    sigma = SIGMA / STRESS_SCALE
    u_s = characteristic_displacement()
    return sigma, u_s, abs(sigma) * u_s * H


def build_single(hidden_layers: int = 4, neurons: int = 64, device: str = 'cpu', dtype=torch.float64, **quad):
    sigma, u_s, escale = _setup()
    mat, _ = materials()
    lp, lw, tr = loads(sigma)
    prob = EnergyProblem(mat, [(p, w, ph) for _, p, w, ph in blocks(**quad)], [(lp, lw, tr)],
                         energy_scale=escale, device=device, dtype=dtype)
    net = EnergyNet((0.0, 0.0), (W, H), hidden_layers, neurons, u_s, 'sym', 1.0).to(device).to(dtype)
    return net, prob


class RoutedFibreModel(nn.Module):
    def __init__(self, sub0: nn.Module, sub1: nn.Module):
        super().__init__()
        self.subnets = nn.ModuleList([sub0, sub1])
        self.register_buffer('_centres', torch.tensor(CENTRES, dtype=torch.float64))

    def get_subdomain_network(self, i: int) -> nn.Module:
        return self.subnets[i]

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(pts.detach().to(self._centres.dtype), self._centres)
        in0 = (d < RF).any(dim=1)
        out = torch.zeros(pts.shape[0], 2, dtype=pts.dtype, device=pts.device)
        if bool(in0.any()):
            out[in0] = self.subnets[0](pts[in0]).to(out.dtype)
        if bool((~in0).any()):
            out[~in0] = self.subnets[1](pts[~in0]).to(out.dtype)
        return out


def build_decomposed(hidden_layers: int = 4, neurons: int = 64, gamma_u: float = 30.0, gamma_t: float = 0.1,
                     device: str = 'cpu', dtype=torch.float64, **quad):
    sigma, u_s, escale = _setup()
    mat, _ = materials()
    lp, lw, tr = loads(sigma)
    mk = lambda: EnergyNet((0.0, 0.0), (W, H), hidden_layers, neurons, u_s, 'sym', 1.0)
    model = RoutedFibreModel(mk(), mk()).to(device).to(dtype)
    prob = DecomposedEnergyProblem(mat, blocks(**quad), [(1, lp, lw, tr)], interfaces(), escale, u_s, W,
                                   gamma_u, gamma_t, device, dtype)
    return model, prob


def predict(net_or_model, coords: np.ndarray, device: str = 'cpu'):
    dt = next(net_or_model.parameters()).dtype
    X = torch.tensor(coords, dtype=dt, device=device, requires_grad=True)
    uv = net_or_model(X)
    gu = torch.autograd.grad(uv[:, 0].sum(), X, create_graph=False, retain_graph=True)[0]
    gv = torch.autograd.grad(uv[:, 1].sum(), X, create_graph=False)[0]
    f = lambda t: t.detach().cpu().numpy()
    exx, eyy, gxy = f(gu[:, 0]), f(gv[:, 1]), f(gu[:, 1] + gv[:, 0])
    E, nu = material_arrays(coords[:, 0], coords[:, 1])
    c11, c12, c33 = E / (1 - nu ** 2), nu * E / (1 - nu ** 2), E / (2 * (1 + nu))
    sx, sy, tau = c11 * exx + c12 * eyy, c12 * exx + c11 * eyy, c33 * gxy
    return f(uv[:, 0]), f(uv[:, 1]), np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)


def compliance(net_or_model, device: str = 'cpu') -> float:
    lp, lw, tr = loads(SIGMA)
    dt = next(net_or_model.parameters()).dtype
    with torch.no_grad():
        uv = net_or_model(torch.tensor(lp, dtype=dt, device=device)).cpu().numpy()
    return float(np.sum(lw * (tr[:, 0] * uv[:, 0] + tr[:, 1] * uv[:, 1])))
