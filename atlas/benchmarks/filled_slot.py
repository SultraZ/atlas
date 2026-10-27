import math

import numpy as np
import torch
import torch.nn as nn

from atlas.energy_solver import EnergyNet, EnergyProblem, _gauss_on_edges, line_quadrature
from atlas.energy_xpinn import DecomposedEnergyProblem, _polar_block
from atlas.physics import PlaneStressElasticity

E_ST, NU_ST = 210e9, 0.30
E_EP, NU_EP = 3.5e9, 0.35
STRESS_SCALE, SIGMA = 1e11, 1e6
W = H = 2.0
AX, BY = 0.2, 0.4


def r_e(th):
    c, s = np.cos(th), np.sin(th)
    return AX * BY / np.sqrt((BY * c) ** 2 + (AX * s) ** 2)


def dr_e(th):
    c, s = np.cos(th), np.sin(th)
    return AX * BY * (BY ** 2 - AX ** 2) * s * c / ((BY * c) ** 2 + (AX * s) ** 2) ** 1.5


def in_filler(x, y):
    return (np.asarray(x) / AX) ** 2 + (np.asarray(y) / BY) ** 2 < 1.0


def radial_distance(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return np.abs(np.hypot(x, y) - r_e(np.arctan2(y, x)))


def materials():
    return (PlaneStressElasticity(E=E_ST / STRESS_SCALE, nu=NU_ST),
            PlaneStressElasticity(E=E_EP / STRESS_SCALE, nu=NU_EP))


def blocks(n_th: int = 24, n_xi_fill: int = 12, n_xi_plate: int = 24, ng: int = 3):
    plate, filler = materials()
    out = [(0, *_polar_block(0.0, r_e, 0.0, 0.5 * math.pi, n_th, n_xi_fill, 1.5, 'b', ng), filler)]
    split = math.atan2(H, W)
    for lo, hi, rho_sq in [(0.0, split, lambda th: W / np.cos(th)),
                           (split, 0.5 * math.pi, lambda th: H / np.sin(th))]:
        out.append((1, *_polar_block(r_e, rho_sq, lo, hi, n_th, n_xi_plate, 1.5, 'a', ng), plate))
    return out


def loads(sigma: float):
    lp, lw, _ = line_quadrature((W, 0.0), (W, H), 12, 4)
    tr = np.stack([np.full(lp.shape[0], sigma), np.zeros(lp.shape[0])], axis=1)
    return lp, lw, tr


def interfaces(n_cells: int = 32, ng: int = 4):
    plate, filler = materials()
    th, wth = _gauss_on_edges(np.linspace(0.0, 0.5 * math.pi, n_cells + 1), ng)
    r, dr = r_e(th), dr_e(th)
    pts = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    g = np.stack([pts[:, 0] / AX ** 2, pts[:, 1] / BY ** 2], axis=1)
    return [dict(a=0, b=1, pts=pts, w=wth * np.sqrt(r * r + dr * dr),
                 n=g / np.linalg.norm(g, axis=1, keepdims=True), phys_a=filler, phys_b=plate)]


def characteristic_displacement() -> float:
    return SIGMA * W / E_ST


def _setup():
    sigma = SIGMA / STRESS_SCALE
    u_s = characteristic_displacement()
    return sigma, u_s, abs(sigma) * u_s * H


def build_single(hidden_layers: int = 4, neurons: int = 64, device: str = 'cpu', dtype=torch.float64, **quad):
    sigma, u_s, escale = _setup()
    plate, _ = materials()
    lp, lw, tr = loads(sigma)
    prob = EnergyProblem(plate, [(p, w, ph) for _, p, w, ph in blocks(**quad)], [(lp, lw, tr)],
                         energy_scale=escale, device=device, dtype=dtype)
    net = EnergyNet((0.0, 0.0), (W, H), hidden_layers, neurons, u_s, 'sym', 1.0).to(device).to(dtype)
    return net, prob


class RoutedSlotModel(nn.Module):
    def __init__(self, sub0: nn.Module, sub1: nn.Module):
        super().__init__()
        self.subnets = nn.ModuleList([sub0, sub1])

    def get_subdomain_network(self, i: int) -> nn.Module:
        return self.subnets[i]

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        x, y = pts[:, 0].detach(), pts[:, 1].detach()
        in0 = (x / AX) ** 2 + (y / BY) ** 2 < 1.0
        out = torch.zeros(pts.shape[0], 2, dtype=pts.dtype, device=pts.device)
        if bool(in0.any()):
            out[in0] = self.subnets[0](pts[in0]).to(out.dtype)
        if bool((~in0).any()):
            out[~in0] = self.subnets[1](pts[~in0]).to(out.dtype)
        return out


def build_decomposed(hidden_layers: int = 4, neurons: int = 64, gamma_u: float = 30.0, gamma_t: float = 0.1,
                     device: str = 'cpu', dtype=torch.float64, **quad):
    sigma, u_s, escale = _setup()
    plate, _ = materials()
    lp, lw, tr = loads(sigma)
    mk = lambda: EnergyNet((0.0, 0.0), (W, H), hidden_layers, neurons, u_s, 'sym', 1.0)
    model = RoutedSlotModel(mk(), mk()).to(device).to(dtype)
    prob = DecomposedEnergyProblem(plate, blocks(**quad), [(1, lp, lw, tr)], interfaces(), escale, u_s, W,
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
    fil = in_filler(coords[:, 0], coords[:, 1])
    E, nu = np.where(fil, E_EP, E_ST), np.where(fil, NU_EP, NU_ST)
    c11, c12, c33 = E / (1 - nu ** 2), nu * E / (1 - nu ** 2), E / (2 * (1 + nu))
    sx, sy, tau = c11 * exx + c12 * eyy, c12 * exx + c11 * eyy, c33 * gxy
    return f(uv[:, 0]), f(uv[:, 1]), np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)


def compliance(net_or_model, device: str = 'cpu') -> float:
    lp, lw, tr = loads(SIGMA)
    dt = next(net_or_model.parameters()).dtype
    with torch.no_grad():
        uv = net_or_model(torch.tensor(lp, dtype=dt, device=device)).cpu().numpy()
    return float(np.sum(lw * (tr[:, 0] * uv[:, 0] + tr[:, 1] * uv[:, 1])))
