"""Inverse design smoke check for the plate hole radius, for BOTH forward
formulations: the deep-energy default (inner loop = energy re-solve at the new
radius) and the original residual path (inner loop = residual training).
Passes if the radius moves and everything stays finite.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLBACKEND", "Agg")
import torch, numpy as np
from atlas.atlas_main import train_forward_solver, run_inverse_design_on_plate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ok = True
for formulation, kw in [
    ('energy', dict(epochs=300, use_lbfgs=True, lbfgs_steps=4, lbfgs_max_iter_per_step=50)),
    ('residual', dict(epochs=400, use_lbfgs=False, n_interior=600, n_boundary=150, n_interface=80)),
]:
    torch.manual_seed(42); np.random.seed(42)
    model, trainer, sampler, phy = train_forward_solver(
        'plate_hole', device, print_every=200, hidden_layers=4, neurons=64,
        formulation=formulation, **kw)
    r0 = float(sampler.r.item())
    opt = run_inverse_design_on_plate(model, sampler, phy, device, n_outer=3, n_inner=40)
    r1 = float(sampler.r.item())
    finite = bool(np.isfinite(opt.history['radius']).all() and np.isfinite(opt.history['loss']).all())
    moved = abs(r1 - r0) > 1e-6
    print(f"INVERSE [{formulation}]: radius {r0:.4f} -> {r1:.4f}, moved={moved}, finite={finite}")
    ok &= finite and moved
sys.exit(0 if ok else 1)
