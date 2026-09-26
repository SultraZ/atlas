import numpy as np
import torch

from atlas.benchmarks import bimaterial as bm
from atlas.energy_solver import EnergyNet, EnergyProblem
from atlas.energy_xpinn import DecomposedEnergyProblem
from atlas.physics import PlaneStressElasticity

E_AL, NU_AL = 70e9, 0.33
E_ST, NU_ST = 210e9, 0.30
STRESS_SCALE, SIGMA, W, H, R_INC = bm.STRESS_SCALE, bm.SIGMA, bm.W, bm.H, bm.R_INC
in_inclusion = bm.in_inclusion


def materials():
    return (PlaneStressElasticity(E=E_AL / STRESS_SCALE, nu=NU_AL),
            PlaneStressElasticity(E=E_ST / STRESS_SCALE, nu=NU_ST))


def blocks(**quad):
    plate, insert = materials()
    return [(s, p, w, insert if s == 0 else plate) for s, p, w, _ in bm.blocks(1.0, **quad)]


def interfaces():
    plate, insert = materials()
    return [dict(d, phys_a=insert, phys_b=plate) for d in bm.interfaces(1.0)]


def characteristic_displacement() -> float:
    return SIGMA * W / E_AL


def _setup():
    sigma = SIGMA / STRESS_SCALE
    u_s = characteristic_displacement()
    return sigma, u_s, abs(sigma) * u_s * H


def build_single(hidden_layers: int = 4, neurons: int = 64, device: str = 'cpu', dtype=torch.float64, **quad):
    sigma, u_s, escale = _setup()
    plate, _ = materials()
    lp, lw, tr = bm.loads(sigma)
    prob = EnergyProblem(plate, [(p, w, ph) for _, p, w, ph in blocks(**quad)], [(lp, lw, tr)],
                         energy_scale=escale, device=device, dtype=dtype)
    net = EnergyNet((0.0, 0.0), (W, H), hidden_layers, neurons, u_s, 'sym', 1.0).to(device).to(dtype)
    return net, prob


def build_decomposed(hidden_layers: int = 4, neurons: int = 64, gamma_u: float = 30.0, gamma_t: float = 0.1,
                     device: str = 'cpu', dtype=torch.float64, **quad):
    sigma, u_s, escale = _setup()
    plate, _ = materials()
    lp, lw, tr = bm.loads(sigma)
    mk = lambda: EnergyNet((0.0, 0.0), (W, H), hidden_layers, neurons, u_s, 'sym', 1.0)
    model = bm.RoutedBimaterialModel(mk(), mk()).to(device).to(dtype)
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
    ins = in_inclusion(coords[:, 0], coords[:, 1])
    E, nu = np.where(ins, E_ST, E_AL), np.where(ins, NU_ST, NU_AL)
    c11, c12, c33 = E / (1 - nu ** 2), nu * E / (1 - nu ** 2), E / (2 * (1 + nu))
    sx, sy, tau = c11 * exx + c12 * eyy, c12 * exx + c11 * eyy, c33 * gxy
    return f(uv[:, 0]), f(uv[:, 1]), np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)


compliance = bm.compliance
