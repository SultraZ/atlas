"""Run from the project root:
    python tests/test_energy_solver.py
"""

import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from atlas.energy_solver import (build_cantilever, build_l_bracket, build_plate_hole,
                                 graded_nodes, line_quadrature, rect_quadrature,
                                 train_energy, williams_features)
from atlas.physics import PlaneStressElasticity

ok, bad = [], []


def check(name, fn):
    try:
        fn()
        ok.append(name)
        print(f"PASS  {name}")
    except Exception as e:
        bad.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


E, SS = 210e9, 1e11
PHYS = PlaneStressElasticity(E=E / SS, nu=0.3)
KW = dict(device='cpu', dtype=torch.float64)


def area(prob):
    return sum(float(w.sum()) for _, w, _ in prob.interior)


def t_quadrature():
    pts, w = rect_quadrature(graded_nodes(0, 2, 7, 3.0, 'b'), graded_nodes(0, 1, 5, 2.0, 'both'), 3)
    assert abs(w.sum() - 2.0) < 1e-12
    assert abs((w * pts[:, 0] ** 2).sum() - 8.0 / 3.0) < 1e-12
    lp, lw, _ = line_quadrature((3, 0), (3, 2), 5, 4)
    assert abs(lw.sum() - 2.0) < 1e-12


def t_areas():
    _, p = build_cantilever(PHYS, -1e-5, 1e-3, **KW)
    assert abs(area(p) - 4.0) < 1e-9
    _, p = build_l_bracket(PHYS, 1e-5, 1e-4, **KW)
    assert abs(area(p) - 3.0) < 1e-9
    _, p = build_plate_hole(PHYS, 1e-5, 1e-5, **KW)
    assert abs(area(p) - (4.0 - math.pi * 0.3 ** 2 / 4.0)) < 1e-6


def t_hard_bcs():
    net, _ = build_cantilever(PHYS, -1e-5, 1e-3, **KW)
    y = torch.linspace(0, 1, 9, dtype=torch.float64).reshape(-1, 1)
    out = net(torch.cat([torch.zeros_like(y), y], 1))
    assert torch.all(out == 0), "clamp x=0 must give exactly u=v=0"
    net, _ = build_plate_hole(PHYS, 1e-5, 1e-5, **KW)
    s = torch.linspace(0.3, 2.0, 9, dtype=torch.float64).reshape(-1, 1)
    assert torch.all(net(torch.cat([torch.zeros_like(s), s], 1))[:, 0] == 0)
    assert torch.all(net(torch.cat([s, torch.zeros_like(s)], 1))[:, 1] == 0)


def t_williams_finite_at_corner():
    fn, nf = williams_features((1.0, 1.0), -math.pi, 0.5 * math.pi)
    c = torch.tensor([[1.0, 1.0], [1.0, 1.5], [0.5, 1.0], [0.7, 0.7]], dtype=torch.float64, requires_grad=True)
    f = fn(c)
    assert f.shape == (4, nf) and torch.isfinite(f).all()
    g = torch.autograd.grad(f.sum(), c)[0]
    assert torch.isfinite(g).all(), "gradient must be finite even exactly at the corner"
    assert not torch.allclose(f[1], f[2])


def t_energy_decreases_and_positive_strain():
    torch.manual_seed(0)
    net, prob = build_cantilever(PHYS, -1e-5, 1.2e-3, nx=8, ny=4, **KW)
    p0 = prob(net)[0].item()
    train_energy(net, prob, epochs=40, lr=2e-3, lbfgs_iters=0, verbose=False)
    loss, se, we = prob(net)
    assert loss.item() < p0, "potential energy must decrease"
    assert se.item() > 0.0


def test_energy_solver_suite():
    ok.clear()
    bad.clear()
    check("quadrature exact for polynomials / lengths", t_quadrature)
    check("quadrature areas match the three geometries", t_areas)
    check("hard BCs exact (clamp, symmetry rollers)", t_hard_bcs)
    check("Williams enrichment finite (value+gradient) at the corner", t_williams_finite_at_corner)
    check("energy decreases under training", t_energy_decreases_and_positive_strain)
    print(f"passed: {len(ok)}  failed: {len(bad)}")
    assert not bad, f"Failed energy solver checks: {bad}"


if __name__ == "__main__":
    test_energy_solver_suite()
    sys.exit(0)
