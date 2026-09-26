"""networks.py - Neural Network Architectures for XPINN
Implements subdomain-specific networks for domain decomposition
"""

import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple


class SubdomainNetwork(nn.Module):
    """    Fully-connected neural network for a single subdomain

    Architecture:
        Input (x, y) -> Hidden Layers -> Output (u, v)
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
        hidden_layers: int = 4,
        neurons_per_layer: int = 64,
        activation: nn.Module = nn.Tanh(),
        output_scale: Optional[float] = None,
        essential_mask: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        fourier_freqs: Optional[List[float]] = None,
        singular_center: Optional[Tuple[float, float]] = None,
        singular_harmonics: int = 2,
    ):
        """        Initialize the subdomain network

        Args:
            input_dim: Number of input features (default: 2 for x, y)
            output_dim: Number of outputs (default: 2 for u, v)
            hidden_layers: Number of hidden layers (default: 4)
            neurons_per_layer: Neurons in each hidden layer (default: 64)
            activation: Activation function (default: Tanh)
            output_scale: Optional characteristic displacement (meters) the
                network output is multiplied by. Random-init outputs (~0.05
                from Xavier) are then ~u_s-scaled instead of 10-100x u_s, so
                the dominant gradient from initialization is shaping the
                displacement field rather than simply shrinking its amplitude
                (which stalls the optimizer on the near-zero "trivial" field).
            essential_mask: Optional callable (x) -> (m_u, m_v) of coordinate
                tensors. The raw network output is reparameterized as
                u = m_u(x) * u_hat, v = m_v(x) * v_hat, turning hard essential
                boundary conditions (symmetry rollers) into EXACT constraints:
                e.g. m_u(x)=x enforces u=0 on the x=0 face, m_v(x)=y enforces
                v=0 on the y=0 face, at every evaluation point with zero
                soft-penalty cost.
            fourier_freqs: Optional list of spatial frequencies (cycles per
                unit length). When set, the network input becomes
                [x, y, sin/cos(2*pi*f_k*x), sin/cos(2*pi*f_k*y) for each f_k];
                the raw coordinates are kept as-is so the field is still a
                smooth function of (x, y) but the MLP sees multiple spatial
                wavelengths directly.
            singular_center: Optional (cx, cy) - when set, the network input
                is enriched with a local POLAR basis around this center:
                [rho, 1/(rho+eps), cos(k*theta), sin(k*theta) for k=1..H],
                where rho=|p-center|, theta=atan2(y-cy, x-cx), H is
                singular_harmonics. This targets the classical elasticity
                near-hole/near-crack-tip solution structure (e.g. the Kirsch
                solution for a hole in a plate: sigma ~ sigma_inf*(1 - a^2/r^2
                + ...*cos(2*theta))), which a plain-coordinate MLP has to
                approximate purely through depth/width - diagnosed (see
                BUGFIXES.md-style notes) as the actual bottleneck for the
                plate_hole near-hole subdomain: correlation between predicted
                and FEM v-displacement there was only ~0.26 even after 8000
                Adam + 60 L-BFGS steps and 4x denser sampling, i.e. a
                representational gap, not an optimization/data gap. Giving
                the network the 1/rho and cos/sin(k*theta) terms directly as
                INPUT FEATURES (not a hard constraint - the MLP still learns
                how much of each to use) lets it represent that decay/angular
                structure without needing many extra layers to approximate it
                from raw (x, y). eps is a small constant scaled to rho's
                typical range at this center, not machine epsilon, since this
                is meant to stay well-conditioned across the whole subdomain,
                not just exactly at the singularity.
            singular_harmonics: number of angular harmonics (cos/sin(k*theta),
                k=1..singular_harmonics) added when singular_center is set.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_scale = output_scale
        self.essential_mask = essential_mask

        # Sinusoidal position-encoding bank (fixed, non-trainable).
        self.fourier_freqs = None
        feat_dim = input_dim
        if fourier_freqs:
            self.register_buffer('_fourier_freqs',
                                 torch.tensor(fourier_freqs, dtype=torch.float32))
            self.fourier_freqs = list(fourier_freqs)
            # Each frequency adds sin/cos of BOTH x and y -> 4 features each.
            feat_dim = input_dim + 4 * len(fourier_freqs)

        # Local polar/singular feature basis (fixed, non-trainable) - see
        # docstring above for why this exists.
        self.singular_center = None
        self.singular_harmonics = singular_harmonics
        if singular_center is not None:
            self.singular_center = tuple(float(c) for c in singular_center)
            self.register_buffer(
                '_singular_center',
                torch.tensor(self.singular_center, dtype=torch.float32))
            # 2 (rho, 1/rho) + 2 per harmonic (cos k*theta, sin k*theta)
            feat_dim += 2 + 2 * singular_harmonics

        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(feat_dim, neurons_per_layer))
        layers.append(activation)
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(activation)
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(neurons_per_layer, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Expand raw (x, y) coordinates with sinusoidal Fourier features
        and/or a local polar/singular basis, whichever are configured;
        otherwise pass-through.
        """
        out = x
        f = getattr(self, '_fourier_freqs', None)
        if f is not None:
            fx = 2.0 * torch.pi * f * x[:, 0:1]  # (N, K)
            fy = 2.0 * torch.pi * f * x[:, 1:2]  # (N, K)
            out = torch.cat([out,
                             torch.sin(fx), torch.cos(fx),
                             torch.sin(fy), torch.cos(fy)], dim=1)

        c = getattr(self, '_singular_center', None)
        if c is not None:
            dx = x[:, 0:1] - c[0]
            dy = x[:, 1:2] - c[1]
            delta = 0.02 * max(abs(c[0]), abs(c[1]), 1.0)
            rho = torch.sqrt(dx * dx + dy * dy + delta * delta)
            inv_rho = 1.0 / rho
            c1 = dx * inv_rho
            s1 = dy * inv_rho
            harmonics = []
            ck, sk = c1, s1
            for k in range(1, self.singular_harmonics + 1):
                if k > 1:
                    ck, sk = ck * c1 - sk * s1, sk * c1 + ck * s1
                harmonics.append(ck)
                harmonics.append(sk)
            out = torch.cat([out, rho, inv_rho] + harmonics, dim=1)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """        Forward pass

        Args:
            x: Input coordinates tensor of shape (N, 2)

        Returns:
            Displacement tensor of shape (N, 2) containing [u, v]
        """
        out = self.network(self._encode(x))
        if self.output_scale is not None:
            out = out * self.output_scale
        if self.essential_mask is not None:
            m_u, m_v = self.essential_mask(x)
            out = out * torch.cat([m_u, m_v], dim=1)
        return out


class XPINNModel(nn.Module):
    """    Extended Physics-Informed Neural Network with Domain Decomposition"""
    
    def __init__(
        self,
        num_subdomains: int,
        hidden_layers: int = 4,
        neurons_per_layer: int = 64,
        device: str = 'cpu',
        output_scale: Optional[float] = None,
        monolith: bool = False,
        essential_mask: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        fourier_freqs: Optional[List[float]] = None,
    ):
        """        Initialize XPINN model with multiple subdomain networks

        Args:
            num_subdomains: Number of subdomains for domain decomposition
            hidden_layers: Hidden layers per subdomain network
            neurons_per_layer: Neurons per hidden layer
            device: Compute device ('cpu' or 'cuda')
            output_scale: Optional characteristic-displacement scale (meters)
                passed to each SubdomainNetwork (see SubdomainNetwork.__init__).
            monolith: If True, all subdomain slots share ONE network. Used for
                warm-started forward solves (e.g. the cantilever, whose
                subdomain interface terms were observed to drag the optimizer
                out of the true bending basin). The XPINN API is unchanged:
                get_subdomain_network(i) returns the same net for every i, and
                the interface loss terms become exactly zero by construction.
            essential_mask: Optional hard-essential-BC mask passed to every
                SubdomainNetwork (see SubdomainNetwork.__init__).
            fourier_freqs: Optional sinusoidal position-encoding bank passed
                to every SubdomainNetwork (see SubdomainNetwork.__init__).
        """
        super().__init__()
        
        self.num_subdomains = num_subdomains
        self.device = device
        self.monolith = monolith
        self.essential_mask = essential_mask
        self.fourier_freqs = fourier_freqs
        
        if monolith:
            self.subnets = nn.ModuleList([
                SubdomainNetwork(
                    hidden_layers=hidden_layers,
                    neurons_per_layer=neurons_per_layer,
                    output_scale=output_scale,
                    essential_mask=essential_mask,
                    fourier_freqs=fourier_freqs,
                )
            ])
        else:
            # Create subdomain networks
            self.subnets = nn.ModuleList([
                SubdomainNetwork(
                    hidden_layers=hidden_layers,
                    neurons_per_layer=neurons_per_layer,
                    output_scale=output_scale,
                    essential_mask=essential_mask,
                    fourier_freqs=fourier_freqs,
                )
                for _ in range(num_subdomains)
            ])
        
        # Move to device
        self.to(device)
    
    def forward_subdomain(
        self,
        subdomain_idx: int,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """        Forward pass for a specific subdomain

        Args:
            subdomain_idx: Index of the subdomain (0-indexed)
            x, y: Coordinate tensors

        Returns:
            Displacement tensor [u, v]
        """
        coords = torch.cat([x, y], dim=1)
        return self.subnets[0 if self.monolith else subdomain_idx](coords)
    
    def get_subdomain_network(self, idx: int) -> SubdomainNetwork:
        """Get a specific subdomain network (monolith: always the one net)"""
        return self.subnets[0 if self.monolith else idx]


class AdaptiveActivation(nn.Module):
    """    Adaptive activation function with learnable parameters

    Allows the network to learn optimal activation scaling:
        f(x) = activation(a * x) where 'a' is learnable
    """
    
    def __init__(self, activation: nn.Module = nn.Tanh(), init_scale: float = 1.0):
        super().__init__()
        self.activation = activation
        self.scale = nn.Parameter(torch.tensor([init_scale]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.scale * x)


class ResidualBlock(nn.Module):
    """    Residual block for deeper networks"""
    
    def __init__(self, features: int, activation: nn.Module = nn.Tanh()):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            activation,
            nn.Linear(features, features)
        )
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class EnhancedSubdomainNetwork(nn.Module):
    """    Enhanced network with residual connections and normalization"""
    
    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
        hidden_dim: int = 64,
        num_blocks: int = 4
    ):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim)
            for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        return self.output_layer(h)


# Utility function to create XPINN model for benchmarks
def create_xpinn_for_benchmark(
    benchmark: str,
    hidden_layers: int = 4,
    neurons: int = 64,
    device: str = 'cpu',
    output_scale: Optional[float] = None,
    monolith: bool = False,
    essential_mask: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    fourier_freqs: Optional[List[float]] = None,
) -> XPINNModel:
    """    Factory function to create XPINN model for specific benchmarks

    Args:
        benchmark: One of 'cantilever', 'l_bracket', 'plate_hole'
        hidden_layers: Number of hidden layers
        neurons: Neurons per layer
        device: Compute device
        output_scale: Optional characteristic-displacement scale (meters)
        monolith: Share one network across all subdomains
        essential_mask: Optional hard-essential-BC mask (see SubdomainNetwork)
        fourier_freqs: Optional sinusoidal position-encoding bank

    Returns:
        Configured XPINNModel instance
    """
    num_subdomains = {
        'cantilever': 2,
        'l_bracket': 3,
        'plate_hole': 3
    }
    
    if benchmark not in num_subdomains:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    return XPINNModel(
        num_subdomains=num_subdomains[benchmark],
        hidden_layers=hidden_layers,
        neurons_per_layer=neurons,
        device=device,
        output_scale=output_scale,
        monolith=monolith,
        essential_mask=essential_mask,
        fourier_freqs=fourier_freqs,
    )


def default_fourier_freqs(l_char: float, n_freqs: int = 8) -> List[float]:
    """Spatial frequency bank (cycles per unit length) spanning wavelengths
    l_char/1 ... l_char/n_freqs. For the 2x2 plate (l_char = W = 2) this
    resolves wavelengths 2 -> 0.25, i.e. domain-scale down to hole-scale
    features, without needing unreasonably dense collocation.
    """
    return [float(l_char) / k for k in range(1, n_freqs + 1)]
