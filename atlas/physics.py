import torch
import torch.nn as nn
from typing import Tuple, Dict

class PlaneStressElasticity:
    def __init__(self, E: float = 210e9, nu: float = 0.3):
        self.E = E
        self.nu = nu
        
        self.C11 = E / (1 - nu**2)
        self.C12 = nu * E / (1 - nu**2)
        self.C33 = E / (2 * (1 + nu))
    
    def compute_strains(
        self, 
        u: torch.Tensor, 
        v: torch.Tensor, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        du_dx = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        du_dy = torch.autograd.grad(
            u, y, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        dv_dx = torch.autograd.grad(
            v, x, grad_outputs=torch.ones_like(v),
            create_graph=True, retain_graph=True
        )[0]
        
        dv_dy = torch.autograd.grad(
            v, y, grad_outputs=torch.ones_like(v),
            create_graph=True, retain_graph=True
        )[0]
        
        eps_xx = du_dx
        eps_yy = dv_dy
        eps_xy = 0.5 * (du_dy + dv_dx)
        
        return eps_xx, eps_yy, eps_xy
    
    def compute_stresses(
        self,
        eps_xx: torch.Tensor,
        eps_yy: torch.Tensor,
        eps_xy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sig_xx = self.C11 * eps_xx + self.C12 * eps_yy
        sig_yy = self.C12 * eps_xx + self.C11 * eps_yy
        sig_xy = 2 * self.C33 * eps_xy
        
        return sig_xx, sig_yy, sig_xy
    
    def compute_von_mises(
        self,
        sig_xx: torch.Tensor,
        sig_yy: torch.Tensor,
        sig_xy: torch.Tensor
    ) -> torch.Tensor:
        von_mises = torch.sqrt(
            sig_xx**2 - sig_xx * sig_yy + sig_yy**2 + 3 * sig_xy**2
        )
        return von_mises
    
    def compute_equilibrium_residuals(
        self,
        sig_xx: torch.Tensor,
        sig_yy: torch.Tensor,
        sig_xy: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dsig_xx_dx = torch.autograd.grad(
            sig_xx, x, grad_outputs=torch.ones_like(sig_xx),
            create_graph=True, retain_graph=True
        )[0]
        
        dsig_xy_dy = torch.autograd.grad(
            sig_xy, y, grad_outputs=torch.ones_like(sig_xy),
            create_graph=True, retain_graph=True
        )[0]
        
        dsig_xy_dx = torch.autograd.grad(
            sig_xy, x, grad_outputs=torch.ones_like(sig_xy),
            create_graph=True, retain_graph=True
        )[0]
        
        dsig_yy_dy = torch.autograd.grad(
            sig_yy, y, grad_outputs=torch.ones_like(sig_yy),
            create_graph=True, retain_graph=True
        )[0]
        
        res_x = dsig_xx_dx + dsig_xy_dy
        res_y = dsig_xy_dx + dsig_yy_dy
        
        return res_x, res_y
    
    def compute_surface_traction(
        self,
        sig_xx: torch.Tensor,
        sig_yy: torch.Tensor,
        sig_xy: torch.Tensor,
        n_x: torch.Tensor,
        n_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_x = sig_xx * n_x + sig_xy * n_y
        t_y = sig_xy * n_x + sig_yy * n_y
        
        return t_x, t_y


def compute_full_physics_step(
    network: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    physics: PlaneStressElasticity
) -> Dict[str, torch.Tensor]:
    coords = torch.cat([x, y], dim=1)
    uv = network(coords)
    u, v = uv[:, 0:1], uv[:, 1:2]
    
    eps_xx, eps_yy, eps_xy = physics.compute_strains(u, v, x, y)
    
    sig_xx, sig_yy, sig_xy = physics.compute_stresses(eps_xx, eps_yy, eps_xy)
    
    von_mises = physics.compute_von_mises(sig_xx, sig_yy, sig_xy)
    
    res_x, res_y = physics.compute_equilibrium_residuals(
        sig_xx, sig_yy, sig_xy, x, y
    )
    
    return {
        'u': u, 'v': v,
        'eps_xx': eps_xx, 'eps_yy': eps_yy, 'eps_xy': eps_xy,
        'sig_xx': sig_xx, 'sig_yy': sig_yy, 'sig_xy': sig_xy,
        'von_mises': von_mises,
        'res_x': res_x, 'res_y': res_y
    }
