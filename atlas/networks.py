import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple


class SubdomainNetwork(nn.Module):
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
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_scale = output_scale
        self.essential_mask = essential_mask

        self.fourier_freqs = None
        feat_dim = input_dim
        if fourier_freqs:
            self.register_buffer('_fourier_freqs',
                                 torch.tensor(fourier_freqs, dtype=torch.float32))
            self.fourier_freqs = list(fourier_freqs)
            feat_dim = input_dim + 4 * len(fourier_freqs)

        self.singular_center = None
        self.singular_harmonics = singular_harmonics
        if singular_center is not None:
            self.singular_center = tuple(float(c) for c in singular_center)
            self.register_buffer(
                '_singular_center',
                torch.tensor(self.singular_center, dtype=torch.float32))
            feat_dim += 2 + 2 * singular_harmonics

        layers = []
        
        layers.append(nn.Linear(feat_dim, neurons_per_layer))
        layers.append(activation)
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(activation)
        
        layers.append(nn.Linear(neurons_per_layer, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        f = getattr(self, '_fourier_freqs', None)
        if f is not None:
            fx = 2.0 * torch.pi * f * x[:, 0:1]
            fy = 2.0 * torch.pi * f * x[:, 1:2]
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
        out = self.network(self._encode(x))
        if self.output_scale is not None:
            out = out * self.output_scale
        if self.essential_mask is not None:
            m_u, m_v = self.essential_mask(x)
            out = out * torch.cat([m_u, m_v], dim=1)
        return out


class XPINNModel(nn.Module):
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
        
        self.to(device)
    
    def forward_subdomain(
        self,
        subdomain_idx: int,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        coords = torch.cat([x, y], dim=1)
        return self.subnets[0 if self.monolith else subdomain_idx](coords)
    
    def get_subdomain_network(self, idx: int) -> SubdomainNetwork:
        return self.subnets[0 if self.monolith else idx]


class AdaptiveActivation(nn.Module):
    def __init__(self, activation: nn.Module = nn.Tanh(), init_scale: float = 1.0):
        super().__init__()
        self.activation = activation
        self.scale = nn.Parameter(torch.tensor([init_scale]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.scale * x)


class ResidualBlock(nn.Module):
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
    return [float(l_char) / k for k in range(1, n_freqs + 1)]
