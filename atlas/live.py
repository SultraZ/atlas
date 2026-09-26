import math

import numpy as np
import torch

from atlas.energy_solver import EnergyNet, _gauss_on_edges, line_quadrature
from atlas.energy_xpinn import (FAR_MARGIN, MID_MARGIN, DecomposedEnergyProblem, RoutedEnergyXPINN,
                                _polar_block, cantilever_geometry, l_bracket_geometry, plate_geometry,
                                williams_features)
from atlas.physics import PlaneStressElasticity

STRESS_SCALE = 1e11
GPA, MPA, KNM = 1e9, 1e6, 1e3
PLATE_W = PLATE_H = 2.0
L_BRACKET_S = 1.0
INCLUSION = ("steel_insert", "bimaterial", "filled_slot")


def _p(key, label, unit, value, lo, hi, step, group, si=1.0):
    return dict(key=key, label=label, unit=unit, value=value, min=lo, max=hi, step=step, group=group, si=si)


def _mat(group, e, nu, suffix=""):
    return [_p("E" + suffix, "Young's modulus", "GPa", e, 1, 400, 1, group, GPA),
            _p("nu" + suffix, "Poisson's ratio", "", nu, 0.0, 0.45, 0.01, group)]


PARAMS = {
    "cantilever": _mat("Material", 210, 0.30) + [
        _p("load", "Tip load", "kN/m", 1000, 50, 5000, 50, "Load", KNM),
        _p("L", "Length", "m", 4.0, 2.0, 8.0, 0.5, "Geometry"),
        _p("H", "Height", "m", 1.0, 0.5, 2.0, 0.1, "Geometry")],
    "l_bracket": _mat("Material", 210, 0.30) + [
        _p("load", "Load on the arm", "kN/m", 1000, 50, 5000, 50, "Load", KNM)],
    "plate_hole": _mat("Material", 210, 0.30) + [
        _p("sigma", "Tension", "MPa", 1.0, 0.1, 50.0, 0.1, "Load", MPA),
        _p("r", "Hole radius", "m", 0.3, 0.1, 0.8, 0.05, "Geometry")],
    "steel_insert": _mat("Plate", 70, 0.33, "1") + _mat("Insert", 210, 0.30, "2") + [
        _p("sigma", "Tension", "MPa", 1.0, 0.1, 50.0, 0.1, "Load", MPA),
        _p("r", "Insert radius", "m", 0.3, 0.1, 0.7, 0.05, "Geometry")],
    "bimaterial": _mat("Plate", 210, 0.30, "1") + _mat("Inclusion", 14, 0.30, "2") + [
        _p("sigma", "Tension", "MPa", 1.0, 0.1, 50.0, 0.1, "Load", MPA),
        _p("r", "Inclusion radius", "m", 0.3, 0.1, 0.7, 0.05, "Geometry")],
    "filled_slot": _mat("Plate", 210, 0.30, "1") + _mat("Filler", 3.5, 0.35, "2") + [
        _p("sigma", "Tension", "MPa", 1.0, 0.1, 50.0, 0.1, "Load", MPA),
        _p("a", "Semi-axis along the load", "m", 0.2, 0.05, 0.6, 0.05, "Geometry"),
        _p("b", "Semi-axis across the load", "m", 0.4, 0.05, 0.9, 0.05, "Geometry")],
}


def values(bench, params=None):
    params = params or {}
    out = {}
    for spec in PARAMS[bench]:
        v = params.get(spec["key"], spec["value"])
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = spec["value"]
        out[spec["key"]] = min(max(v, spec["min"]), spec["max"])
    return out


def is_default(bench, params=None):
    v = values(bench, params)
    return all(abs(v[s["key"]] - s["value"]) <= 1e-9 * max(1.0, abs(s["value"])) for s in PARAMS[bench])


def si(bench, params=None):
    v = values(bench, params)
    return {s["key"]: v[s["key"]] * s["si"] for s in PARAMS[bench]}


def _semi_axes(bench, p):
    return (p["a"], p["b"]) if bench == "filled_slot" else (p["r"], p["r"])


def _r_e(a, b):
    return lambda th: a * b / np.sqrt((b * np.cos(th)) ** 2 + (a * np.sin(th)) ** 2)


def _dr_e(a, b):
    return lambda th: a * b * (b * b - a * a) * np.sin(th) * np.cos(th) / ((b * np.cos(th)) ** 2 + (a * np.sin(th)) ** 2) ** 1.5


def _inclusion_problem(bench, p):
    a, b = _semi_axes(bench, p)
    r_e, dr_e = _r_e(a, b), _dr_e(a, b)
    mat = PlaneStressElasticity(E=p["E1"] / STRESS_SCALE, nu=p["nu1"])
    inc = PlaneStressElasticity(E=p["E2"] / STRESS_SCALE, nu=p["nu2"])
    W, H = PLATE_W, PLATE_H
    blocks = [(0, *_polar_block(0.0, r_e, 0.0, 0.5 * math.pi, 24, 12, 1.5, 'b', 3), inc)]
    split = math.atan2(H, W)
    for lo, hi, rho_sq in [(0.0, split, lambda th: W / np.cos(th)), (split, 0.5 * math.pi, lambda th: H / np.sin(th))]:
        blocks.append((1, *_polar_block(r_e, rho_sq, lo, hi, 24, 24, 1.5, 'a', 3), mat))
    sigma = p["sigma"] / STRESS_SCALE
    lp, lw, _ = line_quadrature((W, 0.0), (W, H), 12, 4)
    loads = [(1, lp, lw, np.stack([np.full(lp.shape[0], sigma), np.zeros(lp.shape[0])], axis=1))]
    th, wth = _gauss_on_edges(np.linspace(0.0, 0.5 * math.pi, 33), 4)
    r, dr = r_e(th), dr_e(th)
    pts = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    g = np.stack([pts[:, 0] / a ** 2, pts[:, 1] / b ** 2], axis=1)
    ifaces = [dict(a=0, b=1, pts=pts, w=wth * np.sqrt(r * r + dr * dr), n=g / np.linalg.norm(g, axis=1, keepdims=True),
                   phys_a=inc, phys_b=mat)]
    u_s = p["sigma"] * W / p["E1"]
    router = lambda c: ((c[:, 0] / a) ** 2 + (c[:, 1] / b) ** 2 >= 1.0).long()
    return mat, blocks, loads, ifaces, u_s, abs(sigma) * u_s * H, W, 2, router, ((0.0, 0.0), (W, H), 'sym', 1.0), None


def build(bench, params=None, device='cpu', hidden_layers=4, neurons=64, dtype=torch.float64):
    p = si(bench, params)
    if bench in INCLUSION:
        phys, blocks, loads, ifaces, u_s, escale, length, n_sub, router, box, enrich = _inclusion_problem(bench, p)
    else:
        phys = PlaneStressElasticity(E=p["E"] / STRESS_SCALE, nu=p["nu"])
        enrich = None
        if bench == "cantilever":
            L, H = p["L"], p["H"]
            P = p["load"] / STRESS_SCALE
            u_s = p["load"] * L ** 3 / (3.0 * p["E"] * (H ** 3 / 12.0))
            blocks, loads, ifaces = cantilever_geometry(-P, L, H)
            escale, length, n_sub = P * u_s * H, L, 2
            router = lambda c: (c[:, 0] >= 0.5 * L).long()
            box = ((0.0, 0.0), (L, H), 'clamp_x0', L)
        elif bench == "plate_hole":
            r, W, H = p["r"], PLATE_W, PLATE_H
            sigma = p["sigma"] / STRESS_SCALE
            u_s = p["sigma"] * W / p["E"]
            blocks, loads, ifaces = plate_geometry(sigma, r, W, H)
            escale, length, n_sub = abs(sigma) * u_s * H, W, 3
            r1, r2 = r + MID_MARGIN, r + FAR_MARGIN

            def router(c):
                rho = torch.hypot(c[:, 0], c[:, 1])
                return (rho >= r1).long() + (rho >= r2).long()
            box = ((0.0, 0.0), (W, H), 'sym', 1.0)
        elif bench == "l_bracket":
            s = L_BRACKET_S
            P = p["load"] / STRESS_SCALE
            u_s = p["load"] * s ** 3 / (3.0 * p["E"] * (s ** 3 / 12.0))
            blocks, loads, ifaces = l_bracket_geometry(P, s)
            escale, length, n_sub = abs(P) * u_s * s, s, 3

            def router(c):
                x, y = c[:, 0], c[:, 1]
                idx = torch.zeros(c.shape[0], dtype=torch.long, device=c.device)
                idx[(x > s) & (y <= s)] = 1
                idx[(x >= s) & (y > s)] = 2
                return idx
            box = ((0.0, 0.0), (2 * s, 2 * s), 'clamp_x0', 2 * s)
            enrich = williams_features((s, s), -math.pi, 0.5 * math.pi)
        else:
            raise ValueError(f"unknown benchmark {bench!r}")
    lo, hi, mask, mask_len = box
    kw = dict(enrich_fn=enrich[0], n_enrich=enrich[1]) if enrich else {}
    subnets = [EnergyNet(lo, hi, hidden_layers, neurons, u_s, mask, mask_len, **kw) for _ in range(n_sub)]
    model = RoutedEnergyXPINN(subnets, router).to(device).to(dtype)
    problem = DecomposedEnergyProblem(phys, blocks, loads, ifaces, escale, u_s, length, 30.0, 0.1, device, dtype)
    return model, problem


def materials_at(bench, params, x, y):
    p = si(bench, params)
    if bench not in INCLUSION:
        return np.full(np.shape(x), p["E"]), np.full(np.shape(x), p["nu"])
    ins = inside(bench, params, x, y)
    return np.where(ins, p["E2"], p["E1"]), np.where(ins, p["nu2"], p["nu1"])


def inside(bench, params, x, y):
    a, b = _semi_axes(bench, si(bench, params))
    return (np.asarray(x) / a) ** 2 + (np.asarray(y) / b) ** 2 < 1.0


def interface_distance(bench, params, x, y):
    a, b = _semi_axes(bench, si(bench, params))
    x, y = np.asarray(x), np.asarray(y)
    return np.abs(np.hypot(x, y) - _r_e(a, b)(np.arctan2(y, x)))


def singular_points(bench, params=None):
    p = si(bench, params)
    if bench == "cantilever":
        return [(0.0, 0.0), (0.0, p["H"])]
    if bench == "l_bracket":
        return [(L_BRACKET_S, L_BRACKET_S)]
    return []


def predict(model, coords, bench, params, device='cpu'):
    dt = next(model.parameters()).dtype
    X = torch.tensor(coords, dtype=dt, device=device, requires_grad=True)
    uv = model(X)
    gu = torch.autograd.grad(uv[:, 0].sum(), X, retain_graph=True)[0]
    gv = torch.autograd.grad(uv[:, 1].sum(), X)[0]
    f = lambda t: t.detach().cpu().numpy()
    exx, eyy, gxy = f(gu[:, 0]), f(gv[:, 1]), f(gu[:, 1] + gv[:, 0])
    E, nu = materials_at(bench, params, coords[:, 0], coords[:, 1])
    c11, c12, c33 = E / (1 - nu ** 2), nu * E / (1 - nu ** 2), E / (2 * (1 + nu))
    sx, sy, tau = c11 * exx + c12 * eyy, c12 * exx + c11 * eyy, c33 * gxy
    return f(uv[:, 0]), f(uv[:, 1]), np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)


def fem_spec(bench, params=None):
    p = si(bench, params)
    if bench in INCLUSION:
        a, b = _semi_axes(bench, p)
        return dict(kind="inclusion", a=a, b=b, W=PLATE_W, H=PLATE_H, sigma=p["sigma"],
                    E1=p["E1"], nu1=p["nu1"], E2=p["E2"], nu2=p["nu2"])
    if bench == "cantilever":
        return dict(kind="cantilever", L=p["L"], H=p["H"], P=p["load"], E=p["E"], nu=p["nu"])
    if bench == "plate_hole":
        return dict(kind="plate_hole", r=p["r"], W=PLATE_W, H=PLATE_H, sigma=p["sigma"], E=p["E"], nu=p["nu"])
    return dict(kind="l_bracket", s=L_BRACKET_S, P=p["load"], E=p["E"], nu=p["nu"])


def compliance(model, bench, params, device='cpu'):
    p = si(bench, params)
    dt = next(model.parameters()).dtype
    if bench == "cantilever":
        lp, lw, t = line_quadrature((p["L"], 0.0), (p["L"], p["H"]), 20, 4)
        tr = np.stack([np.zeros_like(t), -6.0 * (p["load"] / p["H"]) * t * (1.0 - t)], axis=1)
    elif bench == "l_bracket":
        s = L_BRACKET_S
        lp, lw, t = line_quadrature((2 * s, s), (2 * s, 2 * s), 20, 4)
        tr = np.stack([6.0 * p["load"] * t * (1.0 - t), np.zeros_like(t)], axis=1)
    else:
        lp, lw, _ = line_quadrature((PLATE_W, 0.0), (PLATE_W, PLATE_H), 20, 4)
        tr = np.stack([np.full(lp.shape[0], p["sigma"]), np.zeros(lp.shape[0])], axis=1)
    with torch.no_grad():
        uv = model(torch.tensor(lp, dtype=dt, device=device)).cpu().numpy()
    return float(np.sum(lw * (tr[:, 0] * uv[:, 0] + tr[:, 1] * uv[:, 1])))
