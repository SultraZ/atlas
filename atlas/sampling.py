"""sampling.py - Collocation Point Generation for All Benchmarks"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


class CollocationSampler:
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32, device=self.device)



class CantileverBeamSampler(CollocationSampler):
    def __init__(self, L: float = 4.0, H: float = 1.0,
                 n_interior: int = 1000, n_boundary: int = 200,
                 n_interface: int = 100, device: str = 'cpu'):
        super().__init__(device)
        self.L = L
        self.H = H
        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.n_interface = n_interface
        self.interface_x = L / 2

    def sample_interior(self) -> List[Dict]:
        n_per_sub = self.n_interior // 2
        x0 = np.random.uniform(0, self.interface_x, (n_per_sub, 1))
        y0 = np.random.uniform(0, self.H, (n_per_sub, 1))
        x1 = np.random.uniform(self.interface_x, self.L, (n_per_sub, 1))
        y1 = np.random.uniform(0, self.H, (n_per_sub, 1))
        return [
            {'x': self.to_tensor(x0), 'y': self.to_tensor(y0)},
            {'x': self.to_tensor(x1), 'y': self.to_tensor(y1)}
        ]

    def sample_boundary(self, P: float = -1e6) -> List[Dict]:
        bcs = []
        n = self.n_boundary

        # Left edge – fixed
        y_left = np.linspace(0, self.H, n).reshape(-1, 1)
        x_left = np.zeros_like(y_left)
        bcs.append({
            'type': 'dirichlet', 'subdomain': 0,
            'x': self.to_tensor(x_left), 'y': self.to_tensor(y_left),
            'u_prescribed': self.to_tensor(np.zeros_like(x_left)),
            'v_prescribed': self.to_tensor(np.zeros_like(x_left))
        })

        y_right = np.linspace(0, self.H, n).reshape(-1, 1)
        x_right = np.full_like(y_right, self.L)
        tau_y = 6.0 * (P / self.H) * (y_right / self.H) * (1.0 - y_right / self.H)
        bcs.append({
            'type': 'neumann', 'subdomain': 1,
            'x': self.to_tensor(x_right), 'y': self.to_tensor(y_right),
            'n_x': self.to_tensor(np.ones_like(x_right)),
            'n_y': self.to_tensor(np.zeros_like(x_right)),
            't_x': self.to_tensor(np.zeros_like(x_right)),
            't_y': self.to_tensor(tau_y)
        })

        # Top & bottom traction-free
        for y_val, n_y_sign in [(self.H, 1.0), (0.0, -1.0)]:
            x_edge = np.linspace(0, self.L, 2*n).reshape(-1, 1)
            y_edge = np.full_like(x_edge, y_val)
            mask_left = x_edge.flatten() <= self.interface_x
            for subdomain, mask in [(0, mask_left), (1, ~mask_left)]:
                if mask.any():
                    bcs.append({
                        'type': 'neumann', 'subdomain': subdomain,
                        'x': self.to_tensor(x_edge[mask]),
                        'y': self.to_tensor(y_edge[mask]),
                        'n_x': self.to_tensor(np.zeros((mask.sum(), 1))),
                        'n_y': self.to_tensor(n_y_sign * np.ones((mask.sum(), 1))),
                        't_x': self.to_tensor(np.zeros((mask.sum(), 1))),
                        't_y': self.to_tensor(np.zeros((mask.sum(), 1)))
                    })
        return bcs

    def sample_interface(self) -> List[Dict]:
        y_int = np.linspace(0, self.H, self.n_interface).reshape(-1, 1)
        x_int = np.full_like(y_int, self.interface_x)
        return [{
            'subdomain_A': 0, 'subdomain_B': 1,
            'x': self.to_tensor(x_int), 'y': self.to_tensor(y_int),
            'n_x': self.to_tensor(np.ones_like(x_int)),
            'n_y': self.to_tensor(np.zeros_like(x_int))
        }]



class LBracketSampler(CollocationSampler):
    def __init__(self, size: float = 1.0, n_interior: int = 500,
                 n_boundary: int = 100, n_interface: int = 50, device: str = 'cpu',
                 fillet_init: Optional[float] = None):
        """        fillet_init: opt-in re-entrant-corner fillet radius. Leave as None
        (default) to get the exact original sharp-corner L-bracket geometry,
        byte-for-byte identical to before this parameter existed - meaning
        any FEM reference (.npy files) already generated for this benchmark
        stays valid with zero changes needed. Only setting fillet_init to a
        number activates the extra geometry described below, intended for
        InverseDesignOptimizer to grow/shrink via gradient descent.
        """
        super().__init__(device)
        self.size = size
        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.n_interface = n_interface

        self.fillet_enabled = fillet_init is not None
        if self.fillet_enabled:
            max_fillet = 0.4 * size  # keeps tangent points safely within [0,s]/[s,2s]
            init = min(max(fillet_init, 1e-4), max_fillet)
            self.fillet = torch.tensor([init], requires_grad=True, device=device)
            self.max_fillet = max_fillet

    def _fillet_center(self):
        """Center of the fillet disk, as a differentiable tensor (used only
        by the outer/inverse-design loop) - see class docstring for the
        geometric derivation.
        """
        s = self.size
        return (s - self.fillet), (s + self.fillet)

    def sample_fillet_arc_differentiable(self, n_theta: int = 60):
        """Fillet arc points, kept differentiable w.r.t. self.fillet - use
        ONLY for the outer geometry-sensitivity computation (mirrors
        sample_hole_boundary_differentiable in PlateWithHoleSampler and the
        same reasoning: reusing this inside the inner forward-solve
        training loop's repeated .backward() calls would eventually hit
        "Trying to backward through the graph a second time").
        """
        device = self.device
        theta = torch.linspace(-np.pi/2, 0.0, n_theta, device=device).reshape(-1, 1)
        Cx, Cy = self._fillet_center()
        x = Cx + self.fillet * torch.cos(theta)
        y = Cy + self.fillet * torch.sin(theta)
        return x, y

    def sample_interior(self) -> List[Dict]:
        s = self.size
        n = self.n_interior // 3
        x0 = np.random.uniform(0, s, (n, 1))
        y0 = np.random.uniform(0, s, (n, 1))
        x1 = np.random.uniform(s, 2*s, (n, 1))
        y1 = np.random.uniform(0, s, (n, 1))
        x2 = np.random.uniform(s, 2*s, (n, 1))
        y2 = np.random.uniform(s, 2*s, (n, 1))

        x0_t, y0_t = self.to_tensor(x0), self.to_tensor(y0)

        if self.fillet_enabled:
            n_fillet = max(20, self.n_interior // 20)
            f_val = float(self.fillet.detach().cpu().item())
            Cx_val, Cy_val = s - f_val, s + f_val
            theta_f = np.random.uniform(-np.pi/2, 0.0, (n_fillet, 1))
            rho_f = f_val * np.sqrt(np.random.uniform(0, 1, (n_fillet, 1)))
            x_fillet = Cx_val + rho_f * np.cos(theta_f)
            y_fillet = Cy_val + rho_f * np.sin(theta_f)
            x0_t = torch.cat([x0_t, self.to_tensor(x_fillet)], dim=0)
            y0_t = torch.cat([y0_t, self.to_tensor(y_fillet)], dim=0)

        return [
            {'x': x0_t, 'y': y0_t},
            {'x': self.to_tensor(x1), 'y': self.to_tensor(y1)},
            {'x': self.to_tensor(x2), 'y': self.to_tensor(y2)}
        ]

    def sample_boundary(self, P: float = 1e6) -> List[Dict]:
        bcs = []
        s = self.size
        n = self.n_boundary
        f_val = float(self.fillet.detach().cpu().item()) if self.fillet_enabled else 0.0

        # Fixed left face of Sub0
        y_fix = np.linspace(0, s, n).reshape(-1, 1)
        x_fix = np.zeros_like(y_fix)
        bcs.append({
            'type': 'dirichlet', 'subdomain': 0,
            'x': self.to_tensor(x_fix), 'y': self.to_tensor(y_fix),
            'u_prescribed': self.to_tensor(np.zeros_like(x_fix)),
            'v_prescribed': self.to_tensor(np.zeros_like(x_fix))
        })

        y_load = np.linspace(s, 2*s, n).reshape(-1, 1)
        x_load = np.full_like(y_load, 2*s)
        u_load = (y_load - s) / s
        tau_x = 6.0 * P * u_load * (1.0 - u_load)
        bcs.append({
            'type': 'neumann', 'subdomain': 2,
            'x': self.to_tensor(x_load), 'y': self.to_tensor(y_load),
            'n_x': self.to_tensor(np.ones_like(x_load)),
            'n_y': self.to_tensor(np.zeros_like(x_load)),
            't_x': self.to_tensor(tau_x),
            't_y': self.to_tensor(np.zeros_like(x_load))
        })

        # Bottom edges (traction-free)
        for subdomain, x_range in [(0, (0, s)), (1, (s, 2*s))]:
            x_bot = np.linspace(x_range[0], x_range[1], n).reshape(-1, 1)
            y_bot = np.zeros_like(x_bot)
            bcs.append({
                'type': 'neumann', 'subdomain': subdomain,
                'x': self.to_tensor(x_bot), 'y': self.to_tensor(y_bot),
                'n_x': self.to_tensor(np.zeros_like(x_bot)),
                'n_y': self.to_tensor(-np.ones_like(x_bot)),
                't_x': self.to_tensor(np.zeros_like(x_bot)),
                't_y': self.to_tensor(np.zeros_like(x_bot))
            })

        # Top of Sub2
        x_top = np.linspace(s, 2*s, n).reshape(-1, 1)
        y_top = np.full_like(x_top, 2*s)
        bcs.append({
            'type': 'neumann', 'subdomain': 2,
            'x': self.to_tensor(x_top), 'y': self.to_tensor(y_top),
            'n_x': self.to_tensor(np.zeros_like(x_top)),
            'n_y': self.to_tensor(np.ones_like(x_top)),
            't_x': self.to_tensor(np.zeros_like(x_top)),
            't_y': self.to_tensor(np.zeros_like(x_top))
        })

        # Right face of Sub1 (free)
        y_right1 = np.linspace(0, s, n).reshape(-1, 1)
        x_right1 = np.full_like(y_right1, 2*s)
        bcs.append({
            'type': 'neumann', 'subdomain': 1,
            'x': self.to_tensor(x_right1), 'y': self.to_tensor(y_right1),
            'n_x': self.to_tensor(np.ones_like(x_right1)),
            'n_y': self.to_tensor(np.zeros_like(x_right1)),
            't_x': self.to_tensor(np.zeros_like(x_right1)),
            't_y': self.to_tensor(np.zeros_like(x_right1))
        })

        y_inner_v = np.linspace(s + f_val, 2*s, n).reshape(-1, 1)
        x_inner_v = np.full_like(y_inner_v, s)
        bcs.append({
            'type': 'neumann', 'subdomain': 2,
            'x': self.to_tensor(x_inner_v), 'y': self.to_tensor(y_inner_v),
            'n_x': self.to_tensor(-np.ones_like(x_inner_v)),
            'n_y': self.to_tensor(np.zeros_like(x_inner_v)),
            't_x': self.to_tensor(np.zeros_like(x_inner_v)),
            't_y': self.to_tensor(np.zeros_like(x_inner_v))
        })

        # Inner horizontal face of re-entrant corner (y=s, x<s) – free.
        # Shortened to end at x=s-fillet (not x=s) for the same reason.
        x_inner_h = np.linspace(0, s - f_val, n).reshape(-1, 1)
        y_inner_h = np.full_like(x_inner_h, s)
        bcs.append({
            'type': 'neumann', 'subdomain': 0,
            'x': self.to_tensor(x_inner_h), 'y': self.to_tensor(y_inner_h),
            'n_x': self.to_tensor(np.zeros_like(x_inner_h)),
            'n_y': self.to_tensor(np.ones_like(x_inner_h)),
            't_x': self.to_tensor(np.zeros_like(x_inner_h)),
            't_y': self.to_tensor(np.zeros_like(x_inner_h))
        })

        if self.fillet_enabled:
            n_arc = max(15, n // 10)
            theta_arc = np.linspace(-np.pi/2, 0.0, n_arc).reshape(-1, 1)
            Cx_val, Cy_val = s - f_val, s + f_val
            x_arc = Cx_val + f_val * np.cos(theta_arc)
            y_arc = Cy_val + f_val * np.sin(theta_arc)
            bcs.append({
                'type': 'neumann', 'subdomain': 0,
                'x': self.to_tensor(x_arc), 'y': self.to_tensor(y_arc),
                'n_x': self.to_tensor(np.cos(theta_arc)),
                'n_y': self.to_tensor(np.sin(theta_arc)),
                't_x': self.to_tensor(np.zeros_like(x_arc)),
                't_y': self.to_tensor(np.zeros_like(x_arc))
            })

        return bcs

    def sample_interface(self) -> List[Dict]:
        s = self.size
        n = self.n_interface
        interfaces = []

        y_01 = np.linspace(0, s, n).reshape(-1, 1)
        x_01 = np.full_like(y_01, s)
        interfaces.append({
            'subdomain_A': 0, 'subdomain_B': 1,
            'x': self.to_tensor(x_01), 'y': self.to_tensor(y_01),
            'n_x': self.to_tensor(np.ones_like(x_01)),
            'n_y': self.to_tensor(np.zeros_like(x_01))
        })

        x_12 = np.linspace(s, 2*s, n).reshape(-1, 1)
        y_12 = np.full_like(x_12, s)
        interfaces.append({
            'subdomain_A': 1, 'subdomain_B': 2,
            'x': self.to_tensor(x_12), 'y': self.to_tensor(y_12),
            'n_x': self.to_tensor(np.zeros_like(x_12)),
            'n_y': self.to_tensor(np.ones_like(x_12))
        })
        return interfaces



class PlateWithHoleSampler(CollocationSampler):
    def __init__(self, W: float = 2.0, H: float = 2.0, r_init: float = 0.3,
                 n_interior: int = 500, n_boundary: int = 100,
                 n_interface: int = 50, device: str = 'cpu'):
        super().__init__(device)
        self.W = W
        self.H = H
        self.n_interior = n_interior
        self.n_boundary = n_boundary
        self.n_interface = n_interface
        self.r = torch.tensor([r_init], requires_grad=True, device=device)
        self.mid_margin = 0.3
        self.far_margin = 0.6
        self._update_subdomain_radii()

    def _update_subdomain_radii(self):
        """Keep r_mid/r_far a safe, fixed distance ahead of the current hole
        radius so the subdomain bands never invert or run off the domain.
        Called at the top of every sample_* method so callers that read
        sampler.r_mid / sampler.r_far (visualization.py, validation.py)
        always see values consistent with the current r.
        """
        r_val = float(self.r.detach().cpu().item())
        self.r_mid = r_val + self.mid_margin
        self.r_far = r_val + self.far_margin

    def sample_interior(self) -> List[Dict]:
        self._update_subdomain_radii()
        n = self.n_interior // 3
        device = self.device

        theta0 = torch.rand(n, 1, device=device) * (np.pi / 2)
        xi0 = torch.rand(n, 1, device=device)
        rho0 = torch.sqrt(self.r**2 + xi0 * (self.r_mid**2 - self.r**2))
        x0 = (rho0 * torch.cos(theta0)).detach()
        y0 = (rho0 * torch.sin(theta0)).detach()

        # Sub 1 – middle (fixed)
        theta1 = np.random.uniform(0, np.pi/2, n)
        rho1 = np.sqrt(np.random.uniform(self.r_mid**2, self.r_far**2, n))
        x1 = (rho1 * np.cos(theta1)).reshape(-1, 1)
        y1 = (rho1 * np.sin(theta1)).reshape(-1, 1)

        # Sub 2 – far field
        x2_list, y2_list = [], []
        while len(x2_list) < n:
            x_cand = np.random.uniform(0, self.W, n)
            y_cand = np.random.uniform(0, self.H, n)
            dist = np.sqrt(x_cand**2 + y_cand**2)
            valid = dist > self.r_far
            x2_list.extend(x_cand[valid].tolist())
            y2_list.extend(y_cand[valid].tolist())
        x2 = np.array(x2_list[:n]).reshape(-1, 1)
        y2 = np.array(y2_list[:n]).reshape(-1, 1)

        return [
            {'x': x0, 'y': y0},
            {'x': self.to_tensor(x1), 'y': self.to_tensor(y1)},
            {'x': self.to_tensor(x2), 'y': self.to_tensor(y2)}
        ]

    def sample_hole_boundary(self, n_theta: int = 100) -> Dict:
        """Hole-boundary traction-free BC for the forward solve. Detached
        from r for the same reason as sample_interior's sub-0 points (see
        note there) - avoids re-backwarding through a freed r-graph across
        the inner training loop.
        """
        device = self.device
        theta = torch.linspace(0, np.pi/2, n_theta, device=device).reshape(-1, 1)
        x_hole = (self.r * torch.cos(theta)).detach()
        y_hole = (self.r * torch.sin(theta)).detach()
        n_x = torch.cos(theta)
        n_y = torch.sin(theta)
        zeros = torch.zeros_like(x_hole)
        return {
            'type': 'neumann', 'subdomain': 0,
            'x': x_hole, 'y': y_hole,
            'n_x': n_x, 'n_y': n_y,
            't_x': zeros, 't_y': zeros
        }

    def sample_hole_boundary_differentiable(self, n_theta: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Same hole-boundary points as sample_hole_boundary(), but *keeps*
        the graph back to self.r. Use this only for the outer-loop geometry
        sensitivity computation (InverseDesignOptimizer), on a frozen network,
        never inside the inner forward-solve training loop.
        """
        device = self.device
        theta = torch.linspace(0, np.pi/2, n_theta, device=device).reshape(-1, 1)
        x_hole = self.r * torch.cos(theta)
        y_hole = self.r * torch.sin(theta)
        return x_hole, y_hole

    def sample_boundary(self, sigma_inf: float = 1e6) -> List[Dict]:
        self._update_subdomain_radii()
        bcs = []
        n = self.n_boundary
        bcs.append(self.sample_hole_boundary())

        # Right edge tension
        y_right = np.linspace(0, self.H, n).reshape(-1, 1)
        x_right = np.full_like(y_right, self.W)
        bcs.append({
            'type': 'neumann', 'subdomain': 2,
            'x': self.to_tensor(x_right), 'y': self.to_tensor(y_right),
            'n_x': self.to_tensor(np.ones_like(x_right)),
            'n_y': self.to_tensor(np.zeros_like(x_right)),
            't_x': self.to_tensor(np.full_like(x_right, sigma_inf)),
            't_y': self.to_tensor(np.zeros_like(x_right))
        })

        # Top free
        x_top = np.linspace(0, self.W, n).reshape(-1, 1)
        y_top = np.full_like(x_top, self.H)
        bcs.append({
            'type': 'neumann', 'subdomain': 2,
            'x': self.to_tensor(x_top), 'y': self.to_tensor(y_top),
            'n_x': self.to_tensor(np.zeros_like(x_top)),
            'n_y': self.to_tensor(np.ones_like(x_top)),
            't_x': self.to_tensor(np.zeros_like(x_top)),
            't_y': self.to_tensor(np.zeros_like(x_top))
        })

        # Symmetry rollers (only normal component constrained)
        r_val = float(self.r.detach().cpu().item())

        y_sym_x = np.linspace(0, self.H, n).reshape(-1, 1)
        x_sym_x = np.zeros_like(y_sym_x)
        valid = y_sym_x.flatten() > r_val
        n_valid = int(valid.sum())
        if n_valid > 0:
            bcs.append({
                'type': 'dirichlet', 'subdomain': 2,
                'x': self.to_tensor(x_sym_x[valid]),
                'y': self.to_tensor(y_sym_x[valid]),
                'u_prescribed': self.to_tensor(np.zeros((n_valid, 1))),
                'v_prescribed': self.to_tensor(np.zeros((n_valid, 1))),
                'constrain_u': True, 'constrain_v': False
            })

        x_sym_y = np.linspace(0, self.W, n).reshape(-1, 1)
        y_sym_y = np.zeros_like(x_sym_y)
        valid = x_sym_y.flatten() > r_val
        n_valid = int(valid.sum())
        if n_valid > 0:
            bcs.append({
                'type': 'dirichlet', 'subdomain': 2,
                'x': self.to_tensor(x_sym_y[valid]),
                'y': self.to_tensor(y_sym_y[valid]),
                'u_prescribed': self.to_tensor(np.zeros((n_valid, 1))),
                'v_prescribed': self.to_tensor(np.zeros((n_valid, 1))),
                'constrain_u': False, 'constrain_v': True
            })
        return bcs

    def sample_interface(self) -> List[Dict]:
        self._update_subdomain_radii()
        n = self.n_interface
        interfaces = []

        theta_01 = np.linspace(0, np.pi/2, n).reshape(-1, 1)
        x_01 = self.r_mid * np.cos(theta_01)
        y_01 = self.r_mid * np.sin(theta_01)
        interfaces.append({
            'subdomain_A': 0, 'subdomain_B': 1,
            'x': self.to_tensor(x_01), 'y': self.to_tensor(y_01),
            'n_x': self.to_tensor(np.cos(theta_01)),
            'n_y': self.to_tensor(np.sin(theta_01))
        })

        theta_12 = np.linspace(0, np.pi/2, n).reshape(-1, 1)
        x_12 = self.r_far * np.cos(theta_12)
        y_12 = self.r_far * np.sin(theta_12)
        interfaces.append({
            'subdomain_A': 1, 'subdomain_B': 2,
            'x': self.to_tensor(x_12), 'y': self.to_tensor(y_12),
            'n_x': self.to_tensor(np.cos(theta_12)),
            'n_y': self.to_tensor(np.sin(theta_12))
        })
        return interfaces