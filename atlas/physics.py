

import torch
import torch.nn as nn
from typing import Tuple, Dict

class PlaneStressElasticity:
    """    Plane Stress Linear Elasticity Formulation

    Governing Equations (Navier-Cauchy Static Equilibrium):
        ∂σ_xx/∂x + ∂σ_xy/∂y = 0
        ∂σ_xy/∂x + ∂σ_yy/∂y = 0

    Constitutive Law (Hooke's Law for Plane Stress):
        σ_xx = (E / (1 - ν²)) * (ε_xx + ν * ε_yy)
        σ_yy = (E / (1 - ν²)) * (ε_yy + ν * ε_xx)
        σ_xy = (E / (2 * (1 + ν))) * (2 * ε_xy)
    """
    
    def __init__(self, E: float = 210e9, nu: float = 0.3):
        """        Initialize material properties

        Args:
            E: Young's modulus (Pa) - default: Steel ~210 GPa
            nu: Poisson's ratio (dimensionless) - default: 0.3
        """
        self.E = E
        self.nu = nu
        
        # Pre-compute elastic constants for efficiency
        self.C11 = E / (1 - nu**2)          # Normal stress coefficient
        self.C12 = nu * E / (1 - nu**2)     # Coupling coefficient
        self.C33 = E / (2 * (1 + nu))       # Shear stress coefficient (G)
    
    def compute_strains(
        self, 
        u: torch.Tensor, 
        v: torch.Tensor, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """        Compute strain tensor components using automatic differentiation

        Kinematics (Small Strain-Displacement Relations):
            ε_xx = ∂u/∂x
            ε_yy = ∂v/∂y
            2*ε_xy = ∂u/∂y + ∂v/∂x  (engineering shear strain)

        Args:
            u: x-displacement field
            v: y-displacement field
            x: x-coordinates (requires_grad=True)
            y: y-coordinates (requires_grad=True)

        Returns:
            eps_xx, eps_yy, eps_xy: Strain tensor components
        """
        # Compute displacement gradients using autograd
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
        
        # Strain components
        eps_xx = du_dx                          # Normal strain in x
        eps_yy = dv_dy                          # Normal strain in y
        eps_xy = 0.5 * (du_dy + dv_dx)          # Shear strain (tensor form)
        
        return eps_xx, eps_yy, eps_xy
    
    def compute_stresses(
        self,
        eps_xx: torch.Tensor,
        eps_yy: torch.Tensor,
        eps_xy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """        Compute stress tensor using Hooke's Law (Plane Stress)

        Args:
            eps_xx, eps_yy, eps_xy: Strain tensor components

        Returns:
            sig_xx, sig_yy, sig_xy: Stress tensor components
        """
        sig_xx = self.C11 * eps_xx + self.C12 * eps_yy
        sig_yy = self.C12 * eps_xx + self.C11 * eps_yy
        sig_xy = 2 * self.C33 * eps_xy  # Factor of 2 for engineering strain
        
        return sig_xx, sig_yy, sig_xy
    
    def compute_von_mises(
        self,
        sig_xx: torch.Tensor,
        sig_yy: torch.Tensor,
        sig_xy: torch.Tensor
    ) -> torch.Tensor:
        """        Compute von Mises equivalent stress for plane stress

        Returns:
            von_mises: Equivalent stress scalar field
        """
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
        """        Compute Navier-Cauchy equilibrium equation residuals

        Returns:
            res_x, res_y: Equilibrium residuals (should be zero)
        """
        # Stress gradients
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
        
        # Equilibrium residuals
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
        """        Compute surface traction vector from stress tensor

        Args:
            sig_xx, sig_yy, sig_xy: Stress components
            n_x, n_y: Unit normal vector components

        Returns:
            t_x, t_y: Traction vector components
        """
        t_x = sig_xx * n_x + sig_xy * n_y
        t_y = sig_xy * n_x + sig_yy * n_y
        
        return t_x, t_y


def compute_full_physics_step(
    network: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    physics: PlaneStressElasticity
) -> Dict[str, torch.Tensor]:
    """    Complete physics computation pipeline for a neural network

    Args:
        network: Neural network that outputs [u, v] displacements
        x, y: Coordinate tensors (requires_grad=True)
        physics: PlaneStressElasticity instance

    Returns:
        Dictionary containing all computed fields
    """
    # Get displacement predictions
    coords = torch.cat([x, y], dim=1)
    uv = network(coords)
    u, v = uv[:, 0:1], uv[:, 1:2]
    
    # Compute strains
    eps_xx, eps_yy, eps_xy = physics.compute_strains(u, v, x, y)
    
    # Compute stresses
    sig_xx, sig_yy, sig_xy = physics.compute_stresses(eps_xx, eps_yy, eps_xy)
    
    # Compute von Mises stress
    von_mises = physics.compute_von_mises(sig_xx, sig_yy, sig_xy)
    
    # Compute equilibrium residuals
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
