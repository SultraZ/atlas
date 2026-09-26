import math

import numpy as np
import torch
import torch.nn as nn

from atlas.energy_solver import EnergyNet, EnergyProblem, _gauss_on_edges, graded_nodes, line_quadrature
from atlas.energy_xpinn import DecomposedEnergyProblem
from atlas.physics import PlaneStressElasticity

E_ST, NU_ST = 210e9, 0.30
E_AL, NU_AL = 70e9, 0.33
STRESS_SCALE = 1e11
SIGMA = 1e6
W, H = 2.0, 1.0
X0, A, LAM = 1.0, 0.1, 0.5
SINGULAR = [(X0 + A * math.cos(2 * math.pi * H / LAM), H)]


def f(y):
    return X0 + A * np.cos(2 * np.pi * np.asarray(y) / LAM)


def df(y):
    return -A * (2 * np.pi / LAM) * np.sin(2 * np.pi * np.asarray(y) / LAM)


def in_steel(x, y):
    return np.asarray(x) < f(y)


def material_arrays(x, y):
    s = in_steel(x, y)
    return np.where(s, E_ST, E_AL), np.where(s, NU_ST, NU_AL)


def materials():
    return (PlaneStressElasticity(E=E_ST / STRESS_SCALE, nu=NU_ST),
            PlaneStressElasticity(E=E_AL / STRESS_SCALE, nu=NU_AL))


def blocks(ns: int = 20, ny: int = 40, ng: int = 3):
    steel, alu = materials()
    yp, yw = _gauss_on_edges(np.linspace(0.0, H, ny + 1), ng)
    out = []
    for sub, toward in ((0, 'b'), (1, 'a')):
        sp, sw = _gauss_on_edges(graded_nodes(0.0, 1.0, ns, 1.5, toward), ng)
        S, Y = np.meshgrid(sp, yp, indexing='ij')
        F = f(Y)
        if sub == 0:
            X, J = S * F, F
        else:
            X, J = F + S * (W - F), W - F
        pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        out.append((sub, pts, (np.outer(sw, yw) * J).ravel(), steel if sub == 0 else alu))
    return out


def loads(sigma: float):
    lp, lw, _ = line_quadrature((W, 0.0), (W, H), 12, 4)
    tr = np.stack([np.full(lp.shape[0], sigma), np.zeros(lp.shape[0])], axis=1)
    return lp, lw, tr


def interfaces(n_cells: int = 40, ng: int = 4):
    steel, alu = materials()
    y, wy = _gauss_on_edges(np.linspace(0.0, H, n_cells + 1), ng)
    g = df(y)
    ds = np.sqrt(1.0 + g * g)
    n = np.stack([1.0 / ds, -g / ds], axis=1)
    return [dict(a=0, b=1, pts=np.stack([f(y), y], axis=1), w=wy * ds, n=n, phys_a=steel, phys_b=alu)]


def characteristic_displacement() -> float:
    return SIGMA * (X0 / E_ST + (W - X0) / E_AL)


def _setup():
    sigma = SIGMA / STRESS_SCALE
    u_s = characteristic_displacement()
    return sigma, u_s, abs(sigma) * u_s * H


def build_single(hidden_layers: int = 4, neurons: int = 64, device: str = 'cpu', dtype=torch.float64, **quad):
    sigma, u_s, escale = _setup()
    steel, _ = materials()
    lp, lw, tr = loads(sigma)
    prob = EnergyProblem(steel, [(p, w, ph) for _, p, w, ph in blocks(**quad)], [(lp, lw, tr)],
                         energy_scale=escale, device=device, dtype=dtype)
    net = EnergyNet((0.0, 0.0), (W, H), hidden_layers, neurons, u_s, 'sym', 1.0).to(device).to(dtype)
    return net, prob


class RoutedJointModel(nn.Module):
    def __init__(self, sub0: nn.Module, sub1: nn.Module):
        super().__init__()
        self.subnets = nn.ModuleList([sub0, sub1])

    def get_subdomain_network(self, i: int) -> nn.Module:
        return self.subnets[i]

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        x, y = pts[:, 0].detach(), pts[:, 1].detach()
        in0 = x < X0 + A * torch.cos(2 * math.pi * y / LAM)
        out = torch.zeros(pts.shape[0], 2, dtype=pts.dtype, device=pts.device)
        if bool(in0.any()):
            out[in0] = self.subnets[0](pts[in0]).to(out.dtype)
        if bool((~in0).any()):
            out[~in0] = self.subnets[1](pts[~in0]).to(out.dtype)
        return out


def build_decomposed(hidden_layers: int = 4, neurons: int = 64, gamma_u: float = 30.0, gamma_t: float = 0.1,
                     device: str = 'cpu', dtype=torch.float64, **quad):
    sigma, u_s, escale = _setup()
    steel, _ = materials()
    lp, lw, tr = loads(sigma)
    mk = lambda: EnergyNet((0.0, 0.0), (W, H), hidden_layers, neurons, u_s, 'sym', 1.0)
    model = RoutedJointModel(mk(), mk()).to(device).to(dtype)
    prob = DecomposedEnergyProblem(steel, blocks(**quad), [(1, lp, lw, tr)], interfaces(), escale, u_s, W,
                                   gamma_u, gamma_t, device, dtype)
    return model, prob


def predict(net_or_model, coords: np.ndarray, device: str = 'cpu'):
    dt = next(net_or_model.parameters()).dtype
    X = torch.tensor(coords, dtype=dt, device=device, requires_grad=True)
    uv = net_or_model(X)
    gu = torch.autograd.grad(uv[:, 0].sum(), X, create_graph=False, retain_graph=True)[0]
    gv = torch.autograd.grad(uv[:, 1].sum(), X, create_graph=False)[0]
    fn = lambda t: t.detach().cpu().numpy()
    exx, eyy, gxy = fn(gu[:, 0]), fn(gv[:, 1]), fn(gu[:, 1] + gv[:, 0])
    E, nu = material_arrays(coords[:, 0], coords[:, 1])
    c11, c12, c33 = E / (1 - nu ** 2), nu * E / (1 - nu ** 2), E / (2 * (1 + nu))
    sx, sy, tau = c11 * exx + c12 * eyy, c12 * exx + c11 * eyy, c33 * gxy
    return fn(uv[:, 0]), fn(uv[:, 1]), np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)


def compliance(net_or_model, device: str = 'cpu') -> float:
    lp, lw, tr = loads(SIGMA)
    dt = next(net_or_model.parameters()).dtype
    with torch.no_grad():
        uv = net_or_model(torch.tensor(lp, dtype=dt, device=device)).cpu().numpy()
    return float(np.sum(lw * (tr[:, 0] * uv[:, 0] + tr[:, 1] * uv[:, 1])))
