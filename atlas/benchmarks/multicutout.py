import math
import os

import numpy as np
import torch
import torch.nn as nn

E_MOD, NU, STRESS_SCALE = 210e9, 0.3, 1e11
SIGMA = 1e6

ARM = 0.4
LEG = 1.0
CORNER = (ARM, ARM)
HOLES = [(0.20, 0.82, 0.065), (0.20, 0.53, 0.045), (0.80, 0.20, 0.070)]
SLOT_PATH = [(0.21, 0.30), (0.21, 0.17), (0.37, 0.17)]
SLOT_HALF = 0.04
Y_CUT, X_CUT = 0.68, 0.60
CLAMP_CORNERS = [(LEG, 0.0), (LEG, ARM)]
QUAD_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                         "multicutout_quadrature.npz")


def _seg_dist(px, py, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    return np.hypot(px - ax - t * dx, py - ay - t * dy)


def in_body(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return (x >= 0) & (y >= 0) & (((x <= LEG) & (y <= ARM)) | ((x <= ARM) & (y <= LEG)))


def in_slot(x, y):
    d = np.minimum(_seg_dist(x, y, SLOT_PATH[0], SLOT_PATH[1]), _seg_dist(x, y, SLOT_PATH[1], SLOT_PATH[2]))
    return d < SLOT_HALF


def material(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    m = in_body(x, y) & ~in_slot(x, y)
    for cx, cy, r in HOLES:
        m &= np.hypot(x - cx, y - cy) >= r
    return m


def subdomain(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return np.where(y >= Y_CUT, 0, np.where(x >= X_CUT, 2, 1))


def cutout_zones(coords, band=0.03):
    z = {}
    for name, (cx, cy, r) in zip("ABC", HOLES):
        z[f"hole_{name}"] = np.hypot(coords[:, 0] - cx, coords[:, 1] - cy) < r + band
    ds = np.minimum(_seg_dist(coords[:, 0], coords[:, 1], SLOT_PATH[0], SLOT_PATH[1]),
                    _seg_dist(coords[:, 0], coords[:, 1], SLOT_PATH[1], SLOT_PATH[2]))
    z["slot"] = ds < SLOT_HALF + band
    return z


def characteristic_displacement():
    P = SIGMA * ARM
    I = ARM ** 3 / 12.0
    return P * LEG ** 3 / (3.0 * E_MOD * I)


def singular_points():
    return [CORNER] + CLAMP_CORNERS


def load_quadrature(path: str = QUAD_FILE):
    q = np.load(path)
    blocks = [(i, q[f"pts{i}"], q[f"w{i}"]) for i in range(3)]
    return blocks


def load_check_quadrature(path: str = QUAD_FILE):
    q = np.load(path)
    return [(q[f"chk_pts{i}"], q[f"chk_w{i}"]) for i in range(3)]


def _line(p0, p1, n_cells, ng=4):
    from atlas.energy_solver import line_quadrature
    pts, w, _ = line_quadrature(p0, p1, n_cells, ng)
    return pts, w


def loads(sigma):
    lp, lw = _line((0.0, LEG), (ARM, LEG), 12)
    tr = np.stack([np.zeros(len(lp)), np.full(len(lp), sigma)], axis=1)
    return lp, lw, tr


def interfaces():
    out = []
    for a, b, p0, p1, n in [(0, 1, (0.0, Y_CUT), (ARM, Y_CUT), (0.0, -1.0)),
                            (1, 2, (X_CUT, 0.0), (X_CUT, ARM), (1.0, 0.0))]:
        pts, w = _line(p0, p1, 16)
        out.append(dict(a=a, b=b, pts=pts, w=w, n=np.tile(n, (len(pts), 1))))
    return out


LOCAL_BOXES = [((0.0, Y_CUT), (ARM, LEG)), ((0.0, 0.0), (X_CUT, Y_CUT)), ((X_CUT, 0.0), (LEG, ARM))]


def _net(hidden_layers, neurons, u_s, box=None):
    from atlas.energy_solver import EnergyNet
    lo, hi = box if box is not None else ((0.0, 0.0), (LEG, LEG))
    return EnergyNet(lo, hi, hidden_layers, neurons, u_s, 'clamp_x1', LEG, mask_x1=LEG)


def build_single(physics_train, hidden_layers=4, neurons=64, device='cpu', dtype=torch.float64):
    from atlas.energy_solver import EnergyProblem
    sigma = SIGMA / STRESS_SCALE
    u_s = characteristic_displacement()
    blocks = [(p, w) for _, p, w in load_quadrature()]
    lp, lw, tr = loads(sigma)
    prob = EnergyProblem(physics_train, blocks, [(lp, lw, tr)], energy_scale=abs(sigma) * u_s * ARM,
                         device=device, dtype=dtype)
    return _net(hidden_layers, neurons, u_s).to(device).to(dtype), prob


def router(c: torch.Tensor) -> torch.Tensor:
    x, y = c[:, 0], c[:, 1]
    idx = torch.ones(c.shape[0], dtype=torch.long, device=c.device)
    idx[x >= X_CUT] = 2
    idx[y >= Y_CUT] = 0
    return idx


def build_decomposed(physics_train, hidden_layers=4, neurons=64, gamma_u=30.0, gamma_t=0.1,
                     device='cpu', dtype=torch.float64, local_norm=False):
    from atlas.energy_xpinn import DecomposedEnergyProblem, RoutedEnergyXPINN
    sigma = SIGMA / STRESS_SCALE
    u_s = characteristic_displacement()
    lp, lw, tr = loads(sigma)
    boxes = LOCAL_BOXES if local_norm else [None] * 3
    model = RoutedEnergyXPINN([_net(hidden_layers, neurons, u_s, b) for b in boxes], router)
    model.to(device).to(dtype)
    prob = DecomposedEnergyProblem(physics_train, load_quadrature(), [(0, lp, lw, tr)], interfaces(),
                                   abs(sigma) * u_s * ARM, u_s, ARM, gamma_u, gamma_t, device, dtype)
    return model, prob


def predict(model, coords: np.ndarray, device: str = 'cpu'):
    dt = next(model.parameters()).dtype
    X = torch.tensor(coords, dtype=dt, device=device, requires_grad=True)
    uv = model(X)
    gu = torch.autograd.grad(uv[:, 0].sum(), X, retain_graph=True)[0]
    gv = torch.autograd.grad(uv[:, 1].sum(), X)[0]
    exx, eyy, gxy = gu[:, 0], gv[:, 1], gu[:, 1] + gv[:, 0]
    c11, c12, c33 = E_MOD / (1 - NU ** 2), NU * E_MOD / (1 - NU ** 2), E_MOD / (2 * (1 + NU))
    sx, sy, tau = c11 * exx + c12 * eyy, c12 * exx + c11 * eyy, c33 * gxy
    vm = torch.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)
    f = lambda t: t.detach().cpu().numpy()
    return f(uv[:, 0]), f(uv[:, 1]), f(vm)
