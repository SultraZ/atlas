"""bimaterial.py - two-material benchmark: the plate-with-hole geometry with the hole filled by
a softer second material, used to test whether ATLAS-XPINN (subdomain interface on the
material boundary) beats the single-network ATLAS energy solver. Protocol:
results/multi_material/bimaterial/PROTOCOL.md.

    plate      quarter [0, W] x [0, H], W = H = 2, symmetry rollers on x = 0 and y = 0
    inclusion  quarter disc rho < r = 0.3 at the origin, E_inc = E / contrast, same nu
    load       uniform tension sigma on x = W; y = H traction-free; perfect bonding

Quadrature (identical for both solvers, only the subdomain labels differ):
  inclusion (sub 0)  polar block rho in [0, r], graded toward the interface
  matrix    (sub 1)  the plate-with-hole O-grid, graded toward the interface
"""

import math

import numpy as np
import torch
import torch.nn as nn

from atlas.energy_solver import EnergyNet, EnergyProblem, _gauss_on_edges, line_quadrature
from atlas.energy_xpinn import DecomposedEnergyProblem, _polar_block
from atlas.physics import PlaneStressElasticity

E_MOD, NU, STRESS_SCALE = 210e9, 0.3, 1e11
SIGMA = 1e6                      # applied traction [Pa]
W = H = 2.0
R_INC = 0.3


def e_inclusion(contrast: float) -> float:
    return E_MOD / float(contrast)


def in_inclusion(x, y):
    return np.hypot(np.asarray(x), np.asarray(y)) < R_INC


def materials(contrast: float):
    """(matrix, inclusion) plane-stress models in training units (stress / STRESS_SCALE)."""
    return (PlaneStressElasticity(E=E_MOD / STRESS_SCALE, nu=NU),
            PlaneStressElasticity(E=e_inclusion(contrast) / STRESS_SCALE, nu=NU))


def blocks(contrast: float, n_th: int = 24, n_xi_inc: int = 12, n_xi_mat: int = 24, ng: int = 3):
    """[(sub, pts, w, physics)]: sub 0 = inclusion, sub 1 = matrix (two O-grid blocks)."""
    mat, inc = materials(contrast)
    out = [(0, *_polar_block(0.0, R_INC, 0.0, 0.5 * math.pi, n_th, n_xi_inc, 1.5, 'b', ng), inc)]
    split = math.atan2(H, W)
    for th_lo, th_hi, rho_sq in [(0.0, split, lambda th: W / np.cos(th)),
                                 (split, 0.5 * math.pi, lambda th: H / np.sin(th))]:
        out.append((1, *_polar_block(R_INC, rho_sq, th_lo, th_hi, n_th, n_xi_mat, 1.5, 'a', ng), mat))
    return out


def loads(sigma: float):
    lp, lw, _ = line_quadrature((W, 0.0), (W, H), 12, 4)
    tr = np.stack([np.full(lp.shape[0], sigma), np.zeros(lp.shape[0])], axis=1)
    return lp, lw, tr


def interfaces(contrast: float, n_cells: int = 24, ng: int = 4):
    """The material interface rho = r, normal from the inclusion (a=0) into the matrix (b=1)."""
    mat, inc = materials(contrast)
    th, w = _gauss_on_edges(np.linspace(0.0, 0.5 * math.pi, n_cells + 1), ng)
    n = np.stack([np.cos(th), np.sin(th)], axis=1)
    return [dict(a=0, b=1, pts=R_INC * n, w=R_INC * w, n=n, phys_a=inc, phys_b=mat)]


def characteristic_displacement() -> float:
    return SIGMA * W / E_MOD


def build_single(contrast: float, hidden_layers: int = 4, neurons: int = 64, device: str = 'cpu',
                 dtype=torch.float64, **quad):
    """ATLAS energy: ONE network over both materials. Returns (net, problem)."""
    sigma = SIGMA / STRESS_SCALE
    u_s = characteristic_displacement()
    mat, _ = materials(contrast)
    lp, lw, tr = loads(sigma)
    prob = EnergyProblem(mat, [(p, w, ph) for _, p, w, ph in blocks(contrast, **quad)],
                         [(lp, lw, tr)], energy_scale=abs(sigma) * u_s * H, device=device, dtype=dtype)
    net = EnergyNet((0.0, 0.0), (W, H), hidden_layers, neurons, u_s, 'sym', 1.0).to(device).to(dtype)
    return net, prob


class RoutedBimaterialModel(nn.Module):
    """Two subnetworks, routed by the material: sub 0 for rho < r, sub 1 otherwise."""

    def __init__(self, sub0: nn.Module, sub1: nn.Module):
        super().__init__()
        self.subnets = nn.ModuleList([sub0, sub1])

    def get_subdomain_network(self, i: int) -> nn.Module:
        return self.subnets[i]

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        in0 = torch.hypot(pts[:, 0], pts[:, 1]).detach() < R_INC
        out = torch.zeros(pts.shape[0], 2, dtype=pts.dtype, device=pts.device)
        if bool(in0.any()):
            out[in0] = self.subnets[0](pts[in0]).to(out.dtype)
        if bool((~in0).any()):
            out[~in0] = self.subnets[1](pts[~in0]).to(out.dtype)
        return out


def build_decomposed(contrast: float, hidden_layers: int = 4, neurons: int = 64,
                     gamma_u: float = 30.0, gamma_t: float = 0.1, device: str = 'cpu',
                     dtype=torch.float64, **quad):
    """ATLAS-XPINN: subnetwork 0 on the inclusion, subnetwork 1 on the matrix, tied on
    the material interface. Both keep the global input box and the symmetry mask (the
    inclusion also touches x = 0 and y = 0), as in the default plate-with-hole decomposition.
    """
    sigma = SIGMA / STRESS_SCALE
    u_s = characteristic_displacement()
    mat, _ = materials(contrast)
    lp, lw, tr = loads(sigma)
    mk = lambda: EnergyNet((0.0, 0.0), (W, H), hidden_layers, neurons, u_s, 'sym', 1.0)
    model = RoutedBimaterialModel(mk(), mk()).to(device).to(dtype)
    prob = DecomposedEnergyProblem(mat, blocks(contrast, **quad), [(1, lp, lw, tr)],
                                   interfaces(contrast), abs(sigma) * u_s * H, u_s, W,
                                   gamma_u, gamma_t, device, dtype)
    return model, prob


def predict(net_or_model, coords: np.ndarray, contrast: float, device: str = 'cpu'):
    """(u, v, von Mises [Pa]) at coords; the stress uses each point's own material."""
    dt = next(net_or_model.parameters()).dtype
    X = torch.tensor(coords, dtype=dt, device=device, requires_grad=True)
    uv = net_or_model(X)
    gu = torch.autograd.grad(uv[:, 0].sum(), X, create_graph=False, retain_graph=True)[0]
    gv = torch.autograd.grad(uv[:, 1].sum(), X, create_graph=False)[0]
    f = lambda t: t.detach().cpu().numpy()
    exx, eyy, gxy = f(gu[:, 0]), f(gv[:, 1]), f(gu[:, 1] + gv[:, 0])
    E = np.where(in_inclusion(coords[:, 0], coords[:, 1]), e_inclusion(contrast), E_MOD)
    c11, c12, c33 = E / (1 - NU ** 2), NU * E / (1 - NU ** 2), E / (2 * (1 + NU))
    sx, sy, tau = c11 * exx + c12 * eyy, c12 * exx + c11 * eyy, c33 * gxy
    vm = np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)
    return f(uv[:, 0]), f(uv[:, 1]), vm


def compliance(net_or_model, device: str = 'cpu') -> float:
    """Load-edge work int t.u ds [N m per unit thickness] of the predicted field."""
    lp, lw, tr = loads(SIGMA)
    dt = next(net_or_model.parameters()).dtype
    with torch.no_grad():
        uv = net_or_model(torch.tensor(lp, dtype=dt, device=device)).cpu().numpy()
    return float(np.sum(lw * (tr[:, 0] * uv[:, 0] + tr[:, 1] * uv[:, 1])))
