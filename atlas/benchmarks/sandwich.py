import numpy as np
import torch
import torch.nn as nn

from atlas.energy_solver import EnergyNet, EnergyProblem, _gauss_on_edges, graded_nodes, line_quadrature, rect_quadrature
from atlas.energy_xpinn import DecomposedEnergyProblem
from atlas.physics import PlaneStressElasticity

E_FACE, NU_FACE = 210e9, 0.3
E_CORE, NU_CORE = 3e9, 0.35
STRESS_SCALE = 1e11
Q = 1e6
L, H, T = 4.0, 1.0, 0.1
LAYERS = [(0.0, T), (T, H - T), (H - T, H)]
SINGULAR = [(0.0, 0.0), (0.0, H), (0.0, T), (0.0, H - T)]


def layer_of(y):
    y = np.asarray(y)
    return np.where(y < T, 0, np.where(y > H - T, 2, 1))


def material_arrays(y):
    core = layer_of(y) == 1
    return np.where(core, E_CORE, E_FACE), np.where(core, NU_CORE, NU_FACE)


def materials():
    return (PlaneStressElasticity(E=E_FACE / STRESS_SCALE, nu=NU_FACE),
            PlaneStressElasticity(E=E_CORE / STRESS_SCALE, nu=NU_CORE))


def blocks(nx: int = 60, ny_face: int = 6, ny_core: int = 16, ng: int = 3):
    face, core = materials()
    xe = graded_nodes(0.0, L, nx, 2.0, 'a')
    out = []
    for sub, (y0, y1) in enumerate(LAYERS):
        ye = graded_nodes(y0, y1, ny_core if sub == 1 else ny_face, 1.6, 'both')
        out.append((sub, *rect_quadrature(xe, ye, ng), core if sub == 1 else face))
    return out


def loads(q: float):
    lp, lw, _ = line_quadrature((0.0, H), (L, H), 40, 4)
    tr = np.stack([np.zeros(lp.shape[0]), np.full(lp.shape[0], -q)], axis=1)
    return lp, lw, tr


def interfaces(n_cells: int = 40, ng: int = 4):
    face, core = materials()
    xp, xw = _gauss_on_edges(graded_nodes(0.0, L, n_cells, 2.0, 'a'), ng)
    out = []
    for a, b, y, pa, pb in ((0, 1, T, face, core), (1, 2, H - T, core, face)):
        out.append(dict(a=a, b=b, pts=np.stack([xp, np.full_like(xp, y)], axis=1), w=xw,
                        n=np.tile([0.0, 1.0], (len(xp), 1)), phys_a=pa, phys_b=pb))
    return out


def characteristic_displacement() -> float:
    d, c = H - T, H - 2 * T
    D = E_FACE * T * d ** 2 / 2
    S = E_CORE / (2 * (1 + NU_CORE)) * d ** 2 / c
    return Q * L ** 4 / (8 * D) + Q * L ** 2 / (2 * S)


def _setup():
    q = Q / STRESS_SCALE
    u_s = characteristic_displacement()
    return q, u_s, q * u_s * L


def build_single(hidden_layers: int = 4, neurons: int = 64, device: str = 'cpu', dtype=torch.float64, **quad):
    q, u_s, escale = _setup()
    face, _ = materials()
    lp, lw, tr = loads(q)
    prob = EnergyProblem(face, [(p, w, ph) for _, p, w, ph in blocks(**quad)], [(lp, lw, tr)],
                         energy_scale=escale, device=device, dtype=dtype)
    net = EnergyNet((0.0, 0.0), (L, H), hidden_layers, neurons, u_s, 'clamp_x0', L).to(device).to(dtype)
    return net, prob


class RoutedSandwichModel(nn.Module):
    def __init__(self, subnets):
        super().__init__()
        self.subnets = nn.ModuleList(subnets)

    def get_subdomain_network(self, i: int) -> nn.Module:
        return self.subnets[i]

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        y = pts[:, 1].detach()
        lab = torch.where(y < T, 0, torch.where(y > H - T, 2, 1))
        out = torch.zeros(pts.shape[0], 2, dtype=pts.dtype, device=pts.device)
        for i in range(3):
            m = lab == i
            if bool(m.any()):
                out[m] = self.subnets[i](pts[m]).to(out.dtype)
        return out


def build_decomposed(hidden_layers: int = 4, neurons: int = 64, gamma_u: float = 30.0, gamma_t: float = 0.1,
                     device: str = 'cpu', dtype=torch.float64, **quad):
    q, u_s, escale = _setup()
    face, _ = materials()
    lp, lw, tr = loads(q)
    mk = lambda: EnergyNet((0.0, 0.0), (L, H), hidden_layers, neurons, u_s, 'clamp_x0', L)
    model = RoutedSandwichModel([mk() for _ in range(3)]).to(device).to(dtype)
    prob = DecomposedEnergyProblem(face, blocks(**quad), [(2, lp, lw, tr)], interfaces(), escale, u_s, L,
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
    E, nu = material_arrays(coords[:, 1])
    c11, c12, c33 = E / (1 - nu ** 2), nu * E / (1 - nu ** 2), E / (2 * (1 + nu))
    sx, sy, tau = c11 * exx + c12 * eyy, c12 * exx + c11 * eyy, c33 * gxy
    return f(uv[:, 0]), f(uv[:, 1]), np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)


def compliance(net_or_model, device: str = 'cpu') -> float:
    lp, lw, tr = loads(Q)
    dt = next(net_or_model.parameters()).dtype
    with torch.no_grad():
        uv = net_or_model(torch.tensor(lp, dtype=dt, device=device)).cpu().numpy()
    return float(np.sum(lw * (tr[:, 0] * uv[:, 0] + tr[:, 1] * uv[:, 1])))
