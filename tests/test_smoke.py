"""Run from the project root:
    python tests/test_smoke.py
"""

import os, sys, traceback
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "scripts"))

import torch
import numpy as np

ok = []
bad = []
def check(name, fn):
    try:
        fn()
        ok.append(name)
        print(f"PASS  {name}")
    except Exception as e:
        bad.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()

def make_sampler_data(bench, device):
    from atlas.sampling import CantileverBeamSampler, LBracketSampler, PlateWithHoleSampler
    from atlas.networks import create_xpinn_for_benchmark
    from atlas.physics import PlaneStressElasticity
    if bench == "cantilever":
        sam = CantileverBeamSampler(L=4.0, H=1.0, n_interior=200, n_boundary=40, n_interface=20, device=device)
        P = -1e6
        return sam, create_xpinn_for_benchmark("cantilever", device=device), PlaneStressElasticity(E=2.1e-2, nu=0.3)
    if bench == "l_bracket":
        sam = LBracketSampler(size=1.0, n_interior=150, n_boundary=30, n_interface=15, device=device)
        P = 1e6
        return sam, create_xpinn_for_benchmark("l_bracket", device=device), PlaneStressElasticity(E=2.1e-2, nu=0.3)
    if bench == "plate_hole":
        sam = PlateWithHoleSampler(W=2.0, H=2.0, r_init=0.3, n_interior=150, n_boundary=30, n_interface=15, device=device)
        P = 1e6
        return sam, create_xpinn_for_benchmark("plate_hole", device=device), PlaneStressElasticity(E=2.1e-2, nu=0.3)

def run_loss(bench):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampler, model, phys = make_sampler_data(bench, device)
    from atlas.loss_functions import XPINNLoss
    loss_fn = XPINNLoss(phys)
    if bench == "cantilever":
        sd = sampler.sample_interior(); id_ = sampler.sample_interface(); bd = sampler.sample_boundary(P=-1e6)
    elif bench == "l_bracket":
        sd = sampler.sample_interior(); id_ = sampler.sample_interface(); bd = sampler.sample_boundary(P=1e6)
    else:
        sd = sampler.sample_interior(); id_ = sampler.sample_interface(); bd = sampler.sample_boundary(sigma_inf=1e6)
    total, comps = loss_fn.compute_total_loss(model, sd, id_, bd)
    total.backward()
    assert np.isfinite(total.item()), f"non-finite total loss {total.item()}"
    for k, v in comps.items():
        assert np.isfinite(v.item()), f"non-finite {k} loss {v.item()}"
    return total.item()

check("import all top modules", lambda: [
    __import__("atlas.atlas_main"), __import__("atlas.physics"), __import__("atlas.networks"),
    __import__("atlas.sampling"), __import__("atlas.loss_functions"), __import__("atlas.inverse_design"),
    __import__("atlas.validation"), __import__("atlas.visualization"),
    __import__("run_repeatability_study")])
check("cantilever forward loss+backward", lambda: run_loss("cantilever"))
check("l_bracket forward loss+backward", lambda: run_loss("l_bracket"))
check("plate_hole forward loss+backward", lambda: run_loss("plate_hole"))

from atlas.sampling import CantileverBeamSampler, LBracketSampler, PlateWithHoleSampler
from atlas.networks import create_xpinn_for_benchmark
from atlas.physics import PlaneStressElasticity
d = "cuda" if torch.cuda.is_available() else "cpu"
def _cantilever_counts():
    sam = CantileverBeamSampler(n_interior=100, n_boundary=50, n_interface=25, device=d)
    sd = sam.sample_interior(); id_ = sam.sample_interface(); bd = sam.sample_boundary(P=-1)
    assert len(sd) == 2 and sum(s['x'].shape[0] for s in sd) == 100
    assert len(id_) == 1 and id_[0]['x'].shape[0] == 25
    assert len(bd) == 6, f"got {len(bd)} BC entries"
check("cantilever sampler geometry", lambda: _cantilever_counts())
def _l_bracket_check():
    for f in (None, 0.15):
        sam = LBracketSampler(n_interior=150, n_boundary=30, n_interface=15, device=d, fillet_init=f)
        sd = sam.sample_interior(); bd = sam.sample_boundary(P=1e6); id_ = sam.sample_interface()
        assert len(sd) == 3 and len(id_) == 2
        if f is not None:
            xa, ya = sam.sample_fillet_arc_differentiable()
            assert xa.requires_grad and ya.requires_grad
check("l_bracket sampler + fillet diff geometry", lambda: _l_bracket_check())
def _plate_check():
    sam = PlateWithHoleSampler(r_init=0.3, n_interior=150, n_boundary=30, n_interface=15, device=d)
    sd = sam.sample_interior(); id_ = sam.sample_interface(); bd = sam.sample_boundary(sigma_inf=1e6)
    x0 = sd[0]['x']; assert not x0.requires_grad, "interior sub0 should be detached"
    xh, yh = sam.sample_hole_boundary_differentiable()
    assert xh.requires_grad

check("plate_hole sampler diff radius", lambda: _plate_check())

print(f"\npassed: {len(ok)}  failed: {len(bad)}  {bad if bad else ''}")
print("DONE")