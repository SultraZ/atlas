"""Run from the project root:
    python tests/test_energy_xpinn_design.py
"""

import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from atlas.energy_design import solve_inverse, train_forward
from atlas.energy_xpinn_design import make_xpinn_case

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


def parts(case, p):
    with torch.no_grad():
        return case.parts(case._pt(p))


def t_areas_and_interfaces():
    for b, ps in (('cantilever', (0.6, 1.0, 1.5)), ('plate_hole', (0.1, 0.3, 0.8)),
                  ('l_bracket', (0.05, 0.1, 0.4))):
        case = make_xpinn_case(b)
        for p in ps:
            blocks, loads, ifaces, area = parts(case, p)
            total = sum(float(w.sum()) for _, _, w in blocks)
            assert abs(total - float(area)) < 1e-6, (b, p, total, float(area))
            assert abs(float(area) - case.volume(p)) < 1e-9
            lens = [float(d['w'].sum()) for d in ifaces]
            if b == 'cantilever':
                assert abs(lens[0] - p) < 1e-9
            elif b == 'plate_hole':
                assert abs(lens[0] - 0.5 * math.pi * (p + 0.3)) < 1e-6
                assert abs(lens[1] - 0.5 * math.pi * (p + 0.6)) < 1e-6
            else:
                assert abs(lens[0] - 1.0) < 1e-9 and abs(lens[1] - 1.0) < 1e-9
                assert abs(lens[2] - p) < 1e-9, "fillet-patch interface has length f"


def t_router_matches_blocks():
    for b, ps in (('cantilever', (0.7, 1.4)), ('plate_hole', (0.15, 0.7)), ('l_bracket', (0.06, 0.35))):
        case = make_xpinn_case(b)
        for p in ps:
            blocks, _, ifaces, _ = parts(case, p)
            for sub, pts, _ in blocks:
                idx = case._router(pts, p)
                assert bool((idx == sub).all()), (b, p, sub, idx.unique().tolist())


def t_gradient_matches_fd():
    torch.manual_seed(0)
    case = make_xpinn_case('cantilever', nx_half=8, ny=6)
    net = train_forward(case, epochs=400, lbfgs_iters=500, verbose=False)
    p = 1.0
    g_ad = case.dcompliance_dp(net, p)
    vals = []
    for pp in (p - 0.05, p + 0.05):
        case.resolve(net, pp, 500)
        vals.append(case.compliance(net, pp))
    g_fd = (vals[1] - vals[0]) / 0.1
    print(f"    dC/dH: autodiff {g_ad:.1f}  finite-difference {g_fd:.1f}")
    assert g_ad < 0 and abs(g_ad - g_fd) / abs(g_fd) < 0.12, (g_ad, g_fd)
    case.resolve(net, p, 500)
    c = case.compliance(net, p)
    assert abs(g_ad - (-3.0 * c / p)) / (3.0 * c / p) < 0.15, (g_ad, -3.0 * c / p)
    return case, net


def t_solver_recovers_beam_height():
    torch.manual_seed(0)
    case = make_xpinn_case('cantilever', nx_half=8, ny=6)
    net = train_forward(case, epochs=400, lbfgs_iters=500, verbose=False)
    base = case.compliance(net, case.p0)
    out = solve_inverse(case, net, 'compliance', 0.7 * base, tol=8e-3, max_iter=8,
                        resolve_iters=250, polish_iters=400, verbose=False)
    print(f"    H* = {out['p_star']:.4f} (beam theory ~{(1 / 0.7) ** (1 / 3):.4f}), status {out['status']}")
    assert out['converged'] and abs(out['p_star'] - (1 / 0.7) ** (1 / 3)) < 0.07


check("areas, subdomain totals and interface lengths (all three cases)", t_areas_and_interfaces)
check("router assigns every quadrature point to its own subnetwork", t_router_matches_blocks)
check("decomposed autodiff shape derivative = finite differences ~ beam theory", t_gradient_matches_fd)
check("decomposed solver recovers the beam height for a compliance target", t_solver_recovers_beam_height)

print(f"passed: {len(ok)}  failed: {len(bad)}")
sys.exit(1 if bad else 0)
