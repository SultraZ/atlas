import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from atlas.energy_solver import (EnergyNet, EnergyProblem, _gauss_on_edges, graded_nodes,
                                 line_quadrature, rect_quadrature)
from atlas.energy_xpinn import DecomposedEnergyProblem, _polar_block

E_MOD, NU, STRESS_SCALE = 210e9, 0.3, 1e11
SIGMA = 1e6


class NotchGeometry:
    def __init__(self, W: float = 1.0, H: float = 2.0, h: float = 0.25, r: float = 0.025,
                 R: float = 0.15):
        self.W, self.H, self.h, self.r, self.R = W, H, h, r, R
        self.y0 = 0.5 * H
        self.cx = h - r
        self.C = (self.cx, self.y0)
        self.apex = (h, self.y0)
        assert R > 2 * r and self.cx + R < W and self.y0 - R > 0 and self.y0 + R < H

    def material(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        inside = (x >= 0) & (x <= self.W) & (y >= 0) & (y <= self.H)
        slot = (x < self.cx) & (np.abs(y - self.y0) < self.r)
        disc = np.hypot(x - self.cx, y - self.y0) < self.r
        return inside & ~slot & ~disc

    def in_patch(self, x, y):
        return ((np.asarray(x) >= self.cx) & (np.asarray(x) <= self.cx + self.R)
                & (np.abs(np.asarray(y) - self.y0) <= self.R))

    @property
    def area(self):
        return self.W * self.H - self.cx * 2 * self.r - 0.5 * math.pi * self.r ** 2


def notch_blocks(g: NotchGeometry, n_th: int = 8, n_xi: int = 14, power: float = 2.0,
                 ng: int = 3, n_rect: Tuple[int, int] = (10, 14)) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    cx, y0, r, R, W, H = g.cx, g.y0, g.r, g.R, g.W, g.H
    blocks = []
    wedges = [(-0.25 * math.pi, 0.25 * math.pi, lambda th: R / np.cos(th)),
              (0.25 * math.pi, 0.5 * math.pi, lambda th: R / np.sin(th)),
              (-0.5 * math.pi, -0.25 * math.pi, lambda th: -R / np.sin(th))]
    for lo, hi, outer in wedges:
        p, w = _polar_block(r, outer, lo, hi, n_th, n_xi, power, 'a', ng)
        blocks.append((0, p + np.array([cx, y0]), w))
    nx, ny = n_rect
    gx = lambda a, b, toward, pw=1.6: graded_nodes(a, b, nx, pw, toward)
    gy = lambda a, b, toward, pw=1.6: graded_nodes(a, b, ny, pw, toward)
    rects = [((0.0, cx, 'b'), (y0 + r, H, 'a')),
             ((0.0, cx, 'b'), (0.0, y0 - r, 'both')),
             ((cx, W, 'a'), (y0 + R, H, 'a')),
             ((cx, W, 'a'), (0.0, y0 - R, 'both')),
             ((cx + R, W, 'a'), (y0 - R, y0 + R, 'none'))]
    for (xa, xb, tx), (ya, yb, ty) in rects:
        p, w = rect_quadrature(gx(xa, xb, tx), gy(ya, yb, ty), ng)
        blocks.append((1, p, w))
    return blocks


def notch_loads(g: NotchGeometry, sigma: float):
    lp, lw, _ = line_quadrature((0.0, g.H), (g.W, g.H), 12, 4)
    tr = np.stack([np.zeros(lp.shape[0]), np.full(lp.shape[0], sigma)], axis=1)
    return lp, lw, tr


def notch_interfaces(g: NotchGeometry):
    cx, y0, r, R = g.cx, g.y0, g.r, g.R
    segs = [((cx + R, y0 - R), (cx + R, y0 + R), (1.0, 0.0)),
            ((cx, y0 + R), (cx + R, y0 + R), (0.0, 1.0)),
            ((cx, y0 - R), (cx + R, y0 - R), (0.0, -1.0)),
            ((cx, y0 + r), (cx, y0 + R), (-1.0, 0.0)),
            ((cx, y0 - R), (cx, y0 - r), (-1.0, 0.0))]
    out = []
    for p0, p1, n in segs:
        pts, w, _ = line_quadrature(p0, p1, 10, 4)
        out.append(dict(a=0, b=1, pts=pts, w=w, n=np.tile(n, (pts.shape[0], 1))))
    return out


def characteristic_displacement(g: NotchGeometry) -> float:
    return SIGMA * g.H / E_MOD


def build_single(physics_train, g: NotchGeometry = None, hidden_layers: int = 4,
                 neurons: int = 64, device: str = 'cpu', dtype=torch.float64, **quad):
    g = g or NotchGeometry()
    sigma = SIGMA / STRESS_SCALE
    u_s = characteristic_displacement(g)
    blocks = [(p, w) for _, p, w in notch_blocks(g, **quad)]
    lp, lw, tr = notch_loads(g, sigma)
    prob = EnergyProblem(physics_train, blocks, [(lp, lw, tr)],
                         energy_scale=abs(sigma) * u_s * g.H, device=device, dtype=dtype)
    net = EnergyNet((0.0, 0.0), (g.W, g.H), hidden_layers, neurons, u_s,
                    'clamp_y0', g.H).to(device).to(dtype)
    return net, prob


def split_interfaces(g: NotchGeometry, n_cells: int = 24, ng: int = 4):
    p, w = _gauss_on_edges(graded_nodes(g.h, g.W, n_cells, 2.0, 'a'), ng)
    pts = np.stack([p, np.full_like(p, g.y0)], axis=1)
    return [dict(a=0, b=1, pts=pts, w=w, n=np.tile([0.0, -1.0], (len(p), 1)))]


class RoutedNotchModel(nn.Module):
    def __init__(self, g: NotchGeometry, sub0: nn.Module, sub1: nn.Module, layout: str = 'patch'):
        super().__init__()
        self.g = g
        self.layout = layout
        self.subnets = nn.ModuleList([sub0, sub1])

    def get_subdomain_network(self, i: int) -> nn.Module:
        return self.subnets[i]

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        c = pts.detach().cpu().numpy()
        m0 = c[:, 1] >= self.g.y0 if self.layout == 'split' else self.g.in_patch(c[:, 0], c[:, 1])
        in0 = torch.as_tensor(m0, device=pts.device)
        out = torch.zeros(pts.shape[0], 2, dtype=pts.dtype, device=pts.device)
        if bool(in0.any()):
            out[in0] = self.subnets[0](pts[in0]).to(out.dtype)
        if bool((~in0).any()):
            out[~in0] = self.subnets[1](pts[~in0]).to(out.dtype)
        return out


def build_decomposed(physics_train, g: NotchGeometry = None, hidden_layers: int = 4,
                     neurons: int = 64, gamma_u: float = 30.0, gamma_t: float = 0.1,
                     device: str = 'cpu', dtype=torch.float64, layout: str = 'patch', **quad):
    g = g or NotchGeometry()
    sigma = SIGMA / STRESS_SCALE
    u_s = characteristic_displacement(g)
    lp, lw, tr = notch_loads(g, sigma)
    if layout == 'split':
        blocks = []
        for _, p, w in notch_blocks(g, **quad):
            top = p[:, 1] >= g.y0
            for sub, m in ((0, top), (1, ~top)):
                if m.any():
                    blocks.append((sub, p[m], w[m]))
        sub0 = EnergyNet((0.0, g.y0), (g.W, g.H), hidden_layers, neurons, u_s, 'none', 1.0)
        sub1 = EnergyNet((0.0, 0.0), (g.W, g.y0), hidden_layers, neurons, u_s, 'clamp_y0', g.y0)
        model = RoutedNotchModel(g, sub0, sub1, 'split').to(device).to(dtype)
        prob = DecomposedEnergyProblem(physics_train, blocks, [(0, lp, lw, tr)],
                                       split_interfaces(g), abs(sigma) * u_s * g.H, u_s, g.W,
                                       gamma_u, gamma_t, device, dtype)
        return model, prob
    blocks = notch_blocks(g, **quad)
    sub0 = EnergyNet((g.cx, g.y0 - g.R), (g.cx + g.R, g.y0 + g.R), hidden_layers, neurons,
                     u_s, 'none', 1.0)
    sub1 = EnergyNet((0.0, 0.0), (g.W, g.H), hidden_layers, neurons, u_s, 'clamp_y0', g.H)
    model = RoutedNotchModel(g, sub0, sub1).to(device).to(dtype)
    prob = DecomposedEnergyProblem(physics_train, blocks, [(1, lp, lw, tr)],
                                   notch_interfaces(g), abs(sigma) * u_s * g.H, u_s, g.W,
                                   gamma_u, gamma_t, device, dtype)
    return model, prob


def predict(net_or_model, coords: np.ndarray, device: str = 'cpu'):
    dt = next(net_or_model.parameters()).dtype
    X = torch.tensor(coords, dtype=dt, device=device, requires_grad=True)
    uv = net_or_model(X)
    gu = torch.autograd.grad(uv[:, 0].sum(), X, create_graph=False, retain_graph=True)[0]
    gv = torch.autograd.grad(uv[:, 1].sum(), X, create_graph=False)[0]
    exx, eyy, gxy = gu[:, 0], gv[:, 1], gu[:, 1] + gv[:, 0]
    c11, c12, c33 = E_MOD / (1 - NU ** 2), NU * E_MOD / (1 - NU ** 2), E_MOD / (2 * (1 + NU))
    sx, sy, tau = c11 * exx + c12 * eyy, c12 * exx + c11 * eyy, c33 * gxy
    vm = torch.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)
    f = lambda t: t.detach().cpu().numpy()
    return f(uv[:, 0]), f(uv[:, 1]), f(vm)
