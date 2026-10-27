"""Run from the project root:
    python tests/test_physics.py
"""

import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from atlas.physics import PlaneStressElasticity

E, nu = 210e9, 0.3
p = PlaneStressElasticity(E=E, nu=nu)

expected = {
    "C11": E / (1 - nu**2),
    "C12": nu * E / (1 - nu**2),
    "C33": E / (2 * (1 + nu)),
}
for k, v in expected.items():
    got = getattr(p, k)
    assert abs(got - v) < 1e-6 * v, f"{k}: {got} != {v}"
print("PASS  Hooke constants match closed form")

sig = torch.tensor([[1e6, 2e6, 3e5]])
vm = p.compute_von_mises(sig[:, 0:1], sig[:, 1:2], sig[:, 2:3])
expected_vm = math.sqrt(1e12 - 2e12 + 4e12 + 3 * 9e10)
assert abs(float(vm.item()) - expected_vm) < 1e-5 * expected_vm, float(vm.item())
print("PASS  von Mises formula")

a, c, b = 1e-3, -3e-4, 2e-5
x = torch.linspace(0, 4, 50, requires_grad=True).reshape(-1, 1)
y = torch.linspace(0, 1, 50, requires_grad=True).reshape(-1, 1)
u = a * x + b * y
v = b * x + c * y
eps_xx, eps_yy, eps_xy = p.compute_strains(u, v, x, y)
assert torch.allclose(eps_xx, torch.full_like(eps_xx, a), atol=1e-12), eps_xx.abs().max()
assert torch.allclose(eps_yy, torch.full_like(eps_yy, c), atol=1e-12)
assert torch.allclose(eps_xy, torch.full_like(eps_xy, b), atol=1e-12)
print("PASS  strains from linear displacement field")
sig_xx, sig_yy, sig_xy = p.compute_stresses(eps_xx, eps_yy, eps_xy)
sxx_e = E/(1-nu**2)*(a + nu*c)
syy_e = E/(1-nu**2)*(c + nu*a)
sxy_e = 2 * E/(2*(1+nu)) * b
assert torch.allclose(sig_xx, torch.full_like(sig_xx, sxx_e), atol=1e-6*abs(sxx_e)) 
assert torch.allclose(sig_yy, torch.full_like(sig_yy, syy_e), atol=1e-6*abs(syy_e))
assert torch.allclose(sig_xy, torch.full_like(sig_xy, sxy_e), atol=1e-6*abs(sxy_e))
print("PASS  stresses from Hooke plane stress closed form")

torch.manual_seed(0)
xt = torch.linspace(0.1, 3.9, 200, requires_grad=True).reshape(-1, 1)
yt = torch.linspace(0.1, 0.9, 200, requires_grad=True).reshape(-1, 1)
def sxx_f(x, y): return 1e3 * torch.sin(1.5 * x) * torch.cos(2.0 * y)
def syy_f(y, x): return 1e3 * torch.cos(1.1 * x) * torch.sin(0.8 * y)
def sxy_f(x, y):
    return 0.3 * torch.sin(1.5*x)*torch.cos(2.0*y)*torch.cos(1.1*x)*torch.sin(0.8*y)*torch.sin(x)*torch.cos(y)
rx, ry = p.compute_equilibrium_residuals(sxx_f(xt, yt), syy_f(yt, xt), sxy_f(xt, yt), xt, yt)
dsxxdx = 1e3 * 1.5 * torch.cos(1.5*xt) * torch.cos(2.0*yt)
dsyydy = 1e3 * torch.cos(1.1*xt) * 0.8 * torch.cos(0.8*yt)
d_siny_terms = (-2.0*torch.sin(2.0*yt)*torch.sin(0.8*yt)*torch.cos(yt)
                + 0.8*torch.cos(2.0*yt)*torch.cos(0.8*yt)*torch.cos(yt)
                - torch.cos(2.0*yt)*torch.sin(0.8*yt)*torch.sin(yt))
d_sinx_terms = (1.5*torch.cos(1.5*xt)*torch.cos(1.1*xt)*torch.sin(xt)
                - 1.1*torch.sin(1.5*xt)*torch.sin(1.1*xt)*torch.sin(xt)
                + torch.sin(1.5*xt)*torch.cos(1.1*xt)*torch.cos(xt))
dsxydx = 0.3 * torch.cos(2.0*yt)*torch.sin(0.8*yt)*torch.cos(yt) * d_sinx_terms
dsxydy = 0.3 * torch.sin(1.5*xt)*torch.cos(1.1*xt)*torch.sin(xt) * d_siny_terms
scale = torch.abs(sxx_f(xt, yt)).max().item()
assert torch.allclose(rx.detach(), (dsxxdx + dsxydy), atol=1e-3 * scale), (rx.detach() - (dsxxdx+dsxydy)).abs().max()
assert torch.allclose(ry.detach(), (dsxydx + dsyydy), atol=1e-3 * scale), (ry.detach() - (dsxydx+dsyydy)).abs().max()
print("PASS  equilibrium residuals match analytic dσ/dx + dσ/dy")

n_x = torch.tensor([[1/math.sqrt(2)]])
n_y = torch.tensor([[1/math.sqrt(2)]])
tx, ty = p.compute_surface_traction(sig_xx[0:1], sig_yy[0:1], sig_xy[0:1], n_x, n_y)
exp_tx = (sxx_e + sxy_e) / math.sqrt(2)
exp_ty = (sxy_e + syy_e) / math.sqrt(2)
assert abs(float(tx.item()) - exp_tx) < 1e-5 * abs(exp_tx)
assert abs(float(ty.item()) - exp_ty) < 1e-5 * abs(exp_ty)
print("PASS  traction t = sigma . n")

from atlas.physics import compute_full_physics_step
import torch.nn as nn
class SmoothNet(nn.Module):
    def forward(self, coords):
        x = coords[:, 0:1]; y = coords[:, 1:2]
        u = 2e-3 * torch.tanh(1.3 * x) * torch.cos(1.1 * y)
        v = -1.5e-3 * torch.sin(1.2 * x) * torch.tanh(0.9 * y)
        return torch.cat([u, v], dim=1)
xr = torch.linspace(0, 1, 20, requires_grad=True).reshape(-1, 1)
yr = torch.linspace(0, 1, 20, requires_grad=True).reshape(-1, 1)
out = compute_full_physics_step(SmoothNet(), xr, yr, p)
for k, v in out.items():
    assert v.numel() > 0 and bool(torch.isfinite(v).all()), f"{k} non-finite"
assert out['von_mises'].min() >= 0
print("PASS  compute_full_physics_step")

print("\nALL PHYSICS CHECKS PASSED")