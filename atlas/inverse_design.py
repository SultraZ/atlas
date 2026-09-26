"""inverse_design.py - Parametric Shape Optimization using Automatic Differentiation
Optimizes geometric parameters (e.g., hole radius) through differentiable XPINN solver
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class InverseDesignConfig:
    """Configuration for inverse design optimization"""
    initial_radius: float = 0.3
    target_compliance: Optional[float] = None
    max_radius: float = 0.8
    min_radius: float = 0.1
    lr_geometry: float = 1e-3
    lr_network: float = 1e-3
    n_outer_iterations: int = 50
    n_inner_iterations: int = 500
    

class ComplianceObjective:
    """    Structural Compliance Objective Function"""
    
    def __init__(self, physics):
        """        Args:
            physics: PlaneStressElasticity instance
        """
        self.physics = physics
    
    def compute(
        self,
        network: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        t_x: torch.Tensor,
        t_y: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """        Compute compliance approximation using collocation points

        Args:
            network: Neural network predicting displacements
            x, y: Collocation point coordinates
            t_x, t_y: Applied traction at collocation points
            weights: Integration weights (optional)

        Returns:
            Compliance value (scalar tensor)
        """
        coords = torch.cat([x, y], dim=1)
        uv = network(coords)
        u, v = uv[:, 0:1], uv[:, 1:2]
        
        # Work done by external forces
        work = t_x * u + t_y * v
        
        if weights is not None:
            compliance = torch.sum(work * weights)
        else:
            compliance = torch.mean(work) * x.shape[0]  # Scale by number of points
        
        return compliance


class InverseDesignOptimizer:
    """    Inverse Shape Design using Differentiable XPINN - generalized to drive
    ANY single scalar geometric parameter that a sampler exposes as a
    trainable tensor (currently: PlateWithHoleSampler.r, or
    LBracketSampler.fillet when fillet_init was set). See __init__ for how
    a benchmark plugs itself in via design_param/resample_fn/sensitivity_fn
    rather than this class hardcoding "radius"/"plate" anywhere.

    Approach:
        1. Trainable geometric parameter (design_param) drives collocation
           point positions in a differentiable way
        2. Train XPINN to solve forward problem for the current geometry
        3. Compute a differentiable sensitivity objective (sensitivity_fn)
        4. Backpropagate to get its gradient w.r.t. design_param
        5. Update design_param to (heuristically) reduce that objective
    """
    
    def __init__(
        self,
        xpinn_model,
        physics,
        sampler,
        config: InverseDesignConfig,
        design_param: torch.Tensor,
        resample_fn,
        sensitivity_fn,
        param_name: str = "parameter",
    ):
        """        Args:
            xpinn_model: XPINN model instance
            physics: PlaneStressElasticity instance (should match whatever
                the model was actually trained against - e.g. the scaled
                physics_train if non-dimensionalization was used)
            sampler: the sampler owning design_param (kept for reference/
                history access, e.g. sampler.W/H in diagnostics)
            config: optimization configuration (min_radius/max_radius are
                used as generic bounds regardless of what design_param
                conceptually represents)
            design_param: the trainable geometry tensor being optimized -
                MUST be the exact same tensor object the sampler's own
                sample_* methods read from (e.g. sampler.r or
                sampler.fillet), so resample_fn() picks up updates to it
            resample_fn: zero-arg callable returning
                (subdomain_data, interface_data, boundary_data) for the
                CURRENT geometry. Kept as a callback (rather than this
                class calling sampler.sample_boundary(...) directly)
                because different benchmarks' sample_boundary() take
                different keyword args (P= vs sigma_inf=) - the caller
                bakes in whichever one applies via a closure.
            sensitivity_fn: zero-arg callable returning a scalar tensor
                differentiable w.r.t. design_param (e.g.
                hole_region_sensitivity_objective or
                fillet_region_sensitivity_objective below - see their
                docstrings for what they do and don't guarantee).
            param_name: only used for print/plot labels.
        """
        self.model = xpinn_model
        self.physics = physics
        self.sampler = sampler
        self.config = config
        self.design_param = design_param
        self.resample_fn = resample_fn
        self.sensitivity_fn = sensitivity_fn
        self.param_name = param_name
        
        # Geometry optimizer (for design_param)
        self.geom_optimizer = torch.optim.Adam(
            [design_param],
            lr=config.lr_geometry
        )
        
        # Network optimizer
        self.net_optimizer = torch.optim.Adam(
            xpinn_model.parameters(),
            lr=config.lr_network
        )
        
        self.history = {
            'radius': [],
            'compliance': [],
            'loss': []
        }
    
    def forward_solve_step(
        self,
        subdomain_data: List[Dict],
        interface_data: List[Dict],
        boundary_data: List[Dict],
        loss_fn
    ) -> float:
        """        One step of forward XPINN solve

        Returns:
            Total loss value
        """
        self.net_optimizer.zero_grad()
        
        total_loss, _ = loss_fn.compute_total_loss(
            self.model, subdomain_data, interface_data, boundary_data
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.net_optimizer.step()
        
        return total_loss.item()
    
    def geometry_update_step(self) -> Tuple[float, float]:
        """        Update design_param based on sensitivity_fn's gradient.

        Returns:
            (objective_value, new_design_param_value)
        """
        self.geom_optimizer.zero_grad()

        objective = self.sensitivity_fn()

        objective.backward()

        if self.design_param.grad is not None:
            print(f"  {self.param_name} gradient: {self.design_param.grad.item():.6e}")
            self.geom_optimizer.step()
            with torch.no_grad():
                self.design_param.clamp_(
                    self.config.min_radius,
                    self.config.max_radius
                )
        else:
            print(f"  [Warning] {self.param_name}.grad is None - it did not update this iteration.")

        return objective.item(), self.design_param.item()
    
    def optimize(
        self,
        loss_fn=None,
        verbose: bool = True,
        inner_solve_fn=None,
    ):
        """        Main inverse design optimization loop

        Args:
            loss_fn: XPINNLoss instance (residual formulation); may be None
                when inner_solve_fn is given
            verbose: Print progress
            inner_solve_fn: optional zero-argument callable that re-solves the
                forward problem at the design parameter's CURRENT value and
                returns a scalar loss to log. When given it replaces the
                residual-loss inner loop and the collocation re-sampling (used
                for models trained with the deep-energy solver, whose
                geometry enters through the quadrature instead).
        """
        print("=" * 70)
        print("INVERSE SHAPE DESIGN OPTIMIZATION")
        print("=" * 70)
        print(f"Initial {self.param_name}: {self.design_param.item():.4f}")
        print(f"{self.param_name} bounds: [{self.config.min_radius}, {self.config.max_radius}]")
        print()
        
        for outer_iter in range(self.config.n_outer_iterations):
            if verbose:
                print(f"Outer Iteration {outer_iter + 1}/{self.config.n_outer_iterations}")
                print(f"  Current {self.param_name}: {self.design_param.item():.4f}")
            
            if inner_solve_fn is not None:
                loss = inner_solve_fn()
                if verbose:
                    print(f"    Inner solve: loss = {loss:.6e}")
            else:
                # Re-sample collocation points for current geometry
                subdomain_data, interface_data, boundary_data = self.resample_fn()

                # Inner loop: Solve forward problem
                for inner_iter in range(self.config.n_inner_iterations):
                    loss = self.forward_solve_step(
                        subdomain_data, interface_data, boundary_data, loss_fn
                    )

                    if (inner_iter + 1) % 100 == 0 and verbose:
                        print(f"    Inner iter {inner_iter + 1}: Loss = {loss:.6e}")
            
            self.history['loss'].append(loss)
            
            # Geometry update
            objective_val, new_val = self.geometry_update_step()
            
            self.history['radius'].append(new_val)
            self.history['compliance'].append(objective_val)
            
            if verbose:
                print(f"  Objective (sensitivity proxy): {objective_val:.6e}")
                print(f"  New {self.param_name}: {new_val:.4f}")
                print()
        
        print("=" * 70)
        print("OPTIMIZATION COMPLETE")
        print(f"Final {self.param_name}: {self.design_param.item():.4f}")
        print(f"Final objective: {self.history['compliance'][-1]:.6e}")
        print("=" * 70)
    
    def plot_history(self, save_path: Optional[str] = None):
        """Plot optimization history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        label = self.param_name.capitalize()

        # Design parameter evolution
        axes[0].plot(self.history['radius'], 'b-o', markersize=4)
        axes[0].set_xlabel('Outer Iteration')
        axes[0].set_ylabel(label)
        axes[0].set_title(f'{label} Evolution')
        axes[0].grid(True, alpha=0.3)
        
        # Objective evolution
        axes[1].semilogy(self.history['compliance'], 'r-o', markersize=4)
        axes[1].set_xlabel('Outer Iteration')
        axes[1].set_ylabel('Sensitivity Objective')
        axes[1].set_title('Objective Evolution')
        axes[1].grid(True, alpha=0.3)
        
        # Loss evolution
        axes[2].semilogy(self.history['loss'], 'g-', linewidth=0.5)
        axes[2].set_xlabel('Outer Iteration')
        axes[2].set_ylabel(getattr(self, 'loss_label', 'XPINN Loss'))
        axes[2].set_title('Forward Solve Loss')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if plt.get_backend().lower() != 'agg':
            plt.show()
        plt.close(fig)


def compute_compliance_for_loaded_boundary(
    model, sampler, sigma_inf: float = 1e6
) -> torch.Tensor:
    """    Compute compliance on the loaded (right, x=W) boundary of a
    PlateWithHoleSampler problem.
    """
    compliance_fn = ComplianceObjective(None)
    n_pts = 100
    y = torch.linspace(0, sampler.H, n_pts, device=sampler.device).reshape(-1, 1)
    x = torch.full_like(y, sampler.W)
    t_x = torch.full_like(x, sigma_inf)
    t_y = torch.zeros_like(x)
    network = model.get_subdomain_network(2)  # far-field network
    return compliance_fn.compute(network, x, y, t_x, t_y)


def hole_region_sensitivity_objective(model, physics, sampler,
                                      delta: float = 0.05, n_theta: int = 100) -> torch.Tensor:
    """    Differentiable proxy objective for PlateWithHoleSampler's outer
    (geometry) loop.
    """
    x_hole, y_hole = sampler.sample_hole_boundary_differentiable(n_theta=n_theta)

    theta = torch.atan2(y_hole, x_hole)
    rho = sampler.r + delta
    x = (rho * torch.cos(theta)).requires_grad_(True)
    y = (rho * torch.sin(theta)).requires_grad_(True)

    network = model.get_subdomain_network(0)
    coords = torch.cat([x, y], dim=1)
    uv = network(coords)
    u, v = uv[:, 0:1], uv[:, 1:2]

    eps_xx, eps_yy, eps_xy = physics.compute_strains(u, v, x, y)
    sig_xx, sig_yy, sig_xy = physics.compute_stresses(eps_xx, eps_yy, eps_xy)
    energy_density = 0.5 * (sig_xx * eps_xx + sig_yy * eps_yy + 2 * sig_xy * eps_xy)

    scale = energy_density.detach().abs().mean() + 1e-12
    return torch.mean(energy_density) / scale


def fillet_region_sensitivity_objective(model, physics, sampler,
                                        delta: float = 0.05, n_theta: int = 60) -> torch.Tensor:
    """    Differentiable proxy objective for LBracketSampler's outer (geometry)
    loop, when fillet_init was set - the exact same idea and the exact same
    caveats as hole_region_sensitivity_objective above (pragmatic strain-
    energy proxy, not a rigorous shape derivative - validate with a
    finite-difference check before trusting it for a paper result), just
    applied to the fillet arc instead of the hole boundary.
    """
    x_arc, y_arc = sampler.sample_fillet_arc_differentiable(n_theta=n_theta)
    Cx, Cy = sampler._fillet_center()
    theta = torch.atan2(y_arc - Cy, x_arc - Cx)
    rho = sampler.fillet + delta
    x = (Cx + rho * torch.cos(theta)).requires_grad_(True)
    y = (Cy + rho * torch.sin(theta)).requires_grad_(True)

    network = model.get_subdomain_network(0)  # fillet patch belongs to sub0
    coords = torch.cat([x, y], dim=1)
    uv = network(coords)
    u, v = uv[:, 0:1], uv[:, 1:2]

    eps_xx, eps_yy, eps_xy = physics.compute_strains(u, v, x, y)
    sig_xx, sig_yy, sig_xy = physics.compute_stresses(eps_xx, eps_yy, eps_xy)
    energy_density = 0.5 * (sig_xx * eps_xx + sig_yy * eps_yy + 2 * sig_xy * eps_xy)

    scale = energy_density.detach().abs().mean() + 1e-12
    return torch.mean(energy_density) / scale


def differentiable_hole_boundary_sampling(
    r: torch.Tensor,
    n_theta: int = 100,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """    Generate hole boundary points as differentiable function of radius

    Args:
        r: Trainable radius tensor (scalar with requires_grad=True)
        n_theta: Number of angular samples
        device: Compute device

    Returns:
        x, y: Boundary point coordinates (differentiable w.r.t. r)
    """
    # Angles (fixed, not trainable)
    theta = torch.linspace(0, np.pi/2, n_theta, device=device)
    
    # Boundary points (differentiable w.r.t. r)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    
    return x.reshape(-1, 1), y.reshape(-1, 1)


class ComplianceSensitivity:
    """    Compute sensitivity of compliance with respect to design parameters
    using adjoint method or direct differentiation.
    """
    
    @staticmethod
    def compute_shape_sensitivity(
        compliance: torch.Tensor,
        design_params: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """        Compute ∂C/∂p for each design parameter p

        Args:
            compliance: Compliance objective (scalar tensor)
            design_params: List of trainable geometric parameters

        Returns:
            List of gradient tensors
        """
        sensitivities = torch.autograd.grad(
            compliance,
            design_params,
            create_graph=False,
            allow_unused=True
        )
        
        return list(sensitivities)


# Example usage in main training script:
"""
from atlas.inverse_design import InverseDesignOptimizer, InverseDesignConfig
from atlas.sampling import PlateWithHoleSampler
from atlas.networks import XPINNModel
from atlas.physics import PlaneStressElasticity
from atlas.loss_functions import XPINNLoss

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
physics = PlaneStressElasticity(E=210e9, nu=0.3)

# Create sampler with trainable radius
sampler = PlateWithHoleSampler(
    W=2.0, H=2.0, r_init=0.3,
    n_interior=500, device=device
)

# Create XPINN model
model = XPINNModel(num_subdomains=3, device=device)

# Create loss function
loss_fn = XPINNLoss(physics)

# Configure inverse design
config = InverseDesignConfig(
    initial_radius=0.3,
    min_radius=0.1,
    max_radius=0.8,
    n_outer_iterations=20,
    n_inner_iterations=200
)

# Run optimization
optimizer = InverseDesignOptimizer(model, physics, sampler, config)
optimizer.optimize(loss_fn, sigma_inf=1e6)

# Plot results
optimizer.plot_history('inverse_design_history.png')
"""
