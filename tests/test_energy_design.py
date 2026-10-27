"""Run from the project root:
    python tests/test_energy_design.py
"""

import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from atlas.energy_design import (CantileverDesign, LBracketFilletDesign,
                                 PlateHoleDesign, make_case, solve_inverse, train_forward)

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


def quad_area(case, p):
    with torch.no_grad():
        interior, _, area = case.geometry(case._pt(p))
    return sum(float(w.sum()) for _, w in interior), float(area)


def t_areas():
    for H in (0.6, 1.0, 1.5):
        q, a = quad_area(CantileverDesign(), H)
        assert abs(q - 4.0 * H) < 1e-9 and abs(a - 4.0 * H) < 1e-12
    for r in (0.1, 0.3, 0.8):
        q, a = quad_area(PlateHoleDesign(), r)
        assert abs(q - (4.0 - math.pi * r * r / 4.0)) < 1e-6, (r, q)
    for f in (0.05, 0.1, 0.4):
        q, a = quad_area(LBracketFilletDesign(), f)
        assert abs(q - (3.0 + f * f * (1.0 - math.pi / 4.0))) < 1e-8, (f, q)
        assert abs(a - q) < 1e-8


def t_fillet_is_a_real_fillet():
    f = 0.2
    case = LBracketFilletDesign()
    with torch.no_grad():
        interior, _, _ = case.geometry(case._pt(f))
    pts = torch.cat([p for p, _ in interior[3:]]).numpy()
    assert pts[:, 0].min() >= 1.0 - f - 1e-9 and pts[:, 0].max() <= 1.0 + 1e-9
    assert pts[:, 1].min() >= 1.0 - 1e-9 and pts[:, 1].max() <= 1.0 + f + 1e-9
    assert (np.hypot(pts[:, 0] - (1.0 - f), pts[:, 1] - (1.0 + f)) >= f - 1e-9).all()
    near_corner = np.hypot(pts[:, 0] - 1.0, pts[:, 1] - 1.0) < 0.1 * f
    assert near_corner.any(), "material must reach right up to the old corner"


def t_autodiff_gradient_matches_fd_and_beam_theory():
    torch.manual_seed(0)
    case = CantileverDesign(nx=14, ny=8)
    net = train_forward(case, epochs=500, lbfgs_iters=500, verbose=False)
    p = 1.0
    g_ad = case.dcompliance_dp(net, p)
    vals = []
    for pp in (p - 0.05, p + 0.05):
        case.resolve(net, pp, 500)
        vals.append(case.compliance(net, pp))
    g_fd = (vals[1] - vals[0]) / 0.1
    print(f"    dC/dH: autodiff {g_ad:.2f}  finite-difference {g_fd:.2f}")
    assert g_ad < 0 and abs(g_ad - g_fd) / abs(g_fd) < 0.08, (g_ad, g_fd)
    case.resolve(net, p, 500)
    c = case.compliance(net, p)
    assert abs(g_ad - (-3.0 * c / p)) / abs(3.0 * c / p) < 0.12, (g_ad, -3.0 * c / p)


def t_inverse_solver_finds_root():
    torch.manual_seed(0)
    case = CantileverDesign(nx=14, ny=8)
    net = train_forward(case, epochs=500, lbfgs_iters=500, verbose=False)
    base = case.compliance(net, case.p0)
    out = solve_inverse(case, net, 'compliance', 0.7 * base, tol=5e-3, max_iter=8,
                        resolve_iters=250, polish_iters=400, verbose=False)
    print(f"    H* = {out['p_star']:.4f} (beam theory ~{(1 / 0.7) ** (1 / 3):.4f}), status {out['status']}")
    assert out['converged'] and abs(out['p_star'] - (1 / 0.7) ** (1 / 3)) < 0.06


def t_metric_validation():
    try:
        solve_inverse(make_case('cantilever'), None, 'peak_vm', 1.0)
    except ValueError:
        return
    raise AssertionError("cantilever has no peak_vm metric; must raise")


def test_energy_design_suite():
    ok.clear()
    bad.clear()
    check("quadrature areas for the three design cases", t_areas)
    check("fillet geometry is a real fillet (corner side, outside the disk)", t_fillet_is_a_real_fillet)
    check("autodiff shape derivative = finite differences ~ beam theory", t_autodiff_gradient_matches_fd_and_beam_theory)
    check("inverse solver recovers the beam height for a compliance target", t_inverse_solver_finds_root)
    check("unavailable metric is rejected", t_metric_validation)
    print(f"passed: {len(ok)}  failed: {len(bad)}")
    assert not bad, f"Failed design checks: {bad}"


if __name__ == "__main__":
    test_energy_design_suite()
    sys.exit(0)
