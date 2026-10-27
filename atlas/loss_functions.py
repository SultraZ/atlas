import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional
from atlas.physics import PlaneStressElasticity


class XPINNLoss:
    def __init__(self, physics: PlaneStressElasticity,
                 weights: Optional[Dict[str, float]] = None,
                 scales: Optional[Dict[str, float]] = None):
        self.physics = physics
        self.weights = weights or {
            'pde': 1.0,
            'dirichlet': 100.0,
            'neumann': 10.0,
            'interface_u': 10.0,
            'interface_t': 10.0
        }
        self.scales = {
            'pde': 1.0, 'dirichlet': 1.0, 'neumann': 1.0,
            'interface_u': 1.0, 'interface_t': 1.0, 'energy': 1.0
        }
        if scales:
            self.scales.update(scales)

        self.resultants: Optional[List[Dict]] = None

        self.energy_context: Optional[Tuple[List[Dict], List[Dict]]] = None

        self.anchor_data: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None

    def anchor_loss(self, network: nn.Module,
                    coords: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.mean((network(coords) - targets) ** 2) / self.scales['dirichlet']

    def resultant_loss(self, network: nn.Module, data: Dict,
                       weight: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = data['x'].detach().requires_grad_(True)
        y = data['y'].detach().requires_grad_(True)
        coords = torch.cat([x, y], dim=1)
        uv = network(coords)
        u, v = uv[:, 0:1], uv[:, 1:2]
        eps_xx, eps_yy, eps_xy = self.physics.compute_strains(u, v, x, y)
        sig_xx, sig_yy, sig_xy = self.physics.compute_stresses(eps_xx, eps_yy, eps_xy)

        total = torch.tensor(0.0, device=x.device)
        losses = {}
        loss_weight = data.get('weight', weight)
        for label, (target, scale) in data['targets'].items():
            if label == 'sxx':
                pred = torch.mean(sig_xx)
            elif label == 'sxy':
                pred = torch.mean(sig_xy)
            elif label == 'moment':
                pred = torch.mean((y - data['y_center']) * sig_xx)
            else:
                raise ValueError(f"Unknown resultant target: {label}")
            term = ((pred - target) ** 2) / (scale ** 2)
            total = total + loss_weight * term
            losses[label] = term
        return total, losses

    def pde_loss(self, network: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        coords = torch.cat([x, y], dim=1)
        uv = network(coords)
        u, v = uv[:, 0:1], uv[:, 1:2]

        eps_xx, eps_yy, eps_xy = self.physics.compute_strains(u, v, x, y)
        sig_xx, sig_yy, sig_xy = self.physics.compute_stresses(eps_xx, eps_yy, eps_xy)
        res_x, res_y = self.physics.compute_equilibrium_residuals(sig_xx, sig_yy, sig_xy, x, y)
        return torch.mean(res_x**2 + res_y**2) / self.scales['pde']

    def energy_loss(self, network: nn.Module, interior: List[Dict],
                    loaded_edges: List[Dict]) -> torch.Tensor:
        strain = torch.tensor(0.0, device=self.device_or(interior))
        for data in interior:
            x = data['x'].requires_grad_(True)
            y = data['y'].requires_grad_(True)
            coords = torch.cat([x, y], dim=1)
            uv = network(coords)
            u, v = uv[:, 0:1], uv[:, 1:2]
            eps_xx, eps_yy, eps_xy = self.physics.compute_strains(u, v, x, y)
            sig_xx, sig_yy, sig_xy = self.physics.compute_stresses(eps_xx, eps_yy, eps_xy)
            energy_density = 0.5 * (sig_xx * eps_xx + sig_yy * eps_yy + 2.0 * sig_xy * eps_xy)
            area = data.get('area', 1.0)
            if isinstance(area, torch.Tensor):
                strain = strain + torch.mean(energy_density * area)
            else:
                strain = strain + float(area) * torch.mean(energy_density)

        work = torch.tensor(0.0, device=strain.device)
        for bc in loaded_edges:
            x = bc['x'].requires_grad_(True)
            y = bc['y'].requires_grad_(True)
            coords = torch.cat([x, y], dim=1)
            uv = network(coords)
            length = float(bc.get('length', 1.0))
            t_dot_u = bc['t_x'] * uv[:, 0:1] + bc['t_y'] * uv[:, 1:2]
            work = work + length * torch.mean(t_dot_u)

        return (strain - work) / self.scales['energy']

    @staticmethod
    def device_or(data) -> torch.device:
        for d in data:
            return d['x'].device
        return torch.device('cpu')

    def dirichlet_loss(self, network: nn.Module, x: torch.Tensor, y: torch.Tensor,
                       u_prescribed: torch.Tensor, v_prescribed: torch.Tensor,
                       constrain_u: bool = True, constrain_v: bool = True) -> torch.Tensor:
        coords = torch.cat([x, y], dim=1)
        uv = network(coords)
        u_pred, v_pred = uv[:, 0:1], uv[:, 1:2]

        loss = torch.tensor(0.0, device=x.device)
        if constrain_u:
            loss = loss + torch.mean((u_pred - u_prescribed)**2)
        if constrain_v:
            loss = loss + torch.mean((v_pred - v_prescribed)**2)
        return loss / self.scales['dirichlet']

    def neumann_loss(self, network: nn.Module, x: torch.Tensor, y: torch.Tensor,
                     n_x: torch.Tensor, n_y: torch.Tensor,
                     t_prescribed_x: torch.Tensor, t_prescribed_y: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        coords = torch.cat([x, y], dim=1)
        uv = network(coords)
        u, v = uv[:, 0:1], uv[:, 1:2]

        eps_xx, eps_yy, eps_xy = self.physics.compute_strains(u, v, x, y)
        sig_xx, sig_yy, sig_xy = self.physics.compute_stresses(eps_xx, eps_yy, eps_xy)
        t_x, t_y = self.physics.compute_surface_traction(sig_xx, sig_yy, sig_xy, n_x, n_y)
        return torch.mean((t_x - t_prescribed_x)**2 + (t_y - t_prescribed_y)**2) / self.scales['neumann']

    def interface_displacement_loss(self, network_A: nn.Module, network_B: nn.Module,
                                    x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        coords = torch.cat([x, y], dim=1)
        uv_A = network_A(coords)
        uv_B = network_B(coords)
        return torch.mean((uv_A - uv_B)**2) / self.scales['interface_u']

    def interface_traction_loss(self, network_A: nn.Module, network_B: nn.Module,
                                x: torch.Tensor, y: torch.Tensor,
                                n_x: torch.Tensor, n_y: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        coords = torch.cat([x, y], dim=1)

        uv_A = network_A(coords)
        u_A, v_A = uv_A[:, 0:1], uv_A[:, 1:2]
        eps_xx_A, eps_yy_A, eps_xy_A = self.physics.compute_strains(u_A, v_A, x, y)
        sig_xx_A, sig_yy_A, sig_xy_A = self.physics.compute_stresses(eps_xx_A, eps_yy_A, eps_xy_A)
        t_x_A, t_y_A = self.physics.compute_surface_traction(sig_xx_A, sig_yy_A, sig_xy_A, n_x, n_y)

        x = x.detach().requires_grad_(True)
        y = y.detach().requires_grad_(True)
        coords = torch.cat([x, y], dim=1)
        uv_B = network_B(coords)
        u_B, v_B = uv_B[:, 0:1], uv_B[:, 1:2]
        eps_xx_B, eps_yy_B, eps_xy_B = self.physics.compute_strains(u_B, v_B, x, y)
        sig_xx_B, sig_yy_B, sig_xy_B = self.physics.compute_stresses(eps_xx_B, eps_yy_B, eps_xy_B)
        t_x_B, t_y_B = self.physics.compute_surface_traction(sig_xx_B, sig_yy_B, sig_xy_B, -n_x, -n_y)

        return torch.mean((t_x_A + t_x_B)**2 + (t_y_A + t_y_B)**2) / self.scales['interface_t']

    def compute_total_loss(self, model, subdomain_data: List[Dict],
                           interface_data: List[Dict],
                           boundary_data: List[Dict]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = model.device
        losses = {
            'pde': torch.tensor(0.0, device=device),
            'dirichlet': torch.tensor(0.0, device=device),
            'neumann': torch.tensor(0.0, device=device),
            'interface_u': torch.tensor(0.0, device=device),
            'interface_t': torch.tensor(0.0, device=device),
            'energy': torch.tensor(0.0, device=device),
            'resultant': torch.tensor(0.0, device=device),
            'anchor': torch.tensor(0.0, device=device),
        }

        for i, data in enumerate(subdomain_data):
            network = model.get_subdomain_network(i)
            losses['pde'] += self.pde_loss(network, data['x'], data['y'])

        for bc in boundary_data:
            network = model.get_subdomain_network(bc['subdomain'])
            if bc['type'] == 'dirichlet':
                constrain_u = bc.get('constrain_u', True)
                constrain_v = bc.get('constrain_v', True)
                losses['dirichlet'] += self.dirichlet_loss(
                    network, bc['x'], bc['y'],
                    bc['u_prescribed'], bc['v_prescribed'],
                    constrain_u=constrain_u, constrain_v=constrain_v
                )
            elif bc['type'] == 'neumann':
                losses['neumann'] += self.neumann_loss(
                    network, bc['x'], bc['y'],
                    bc['n_x'], bc['n_y'], bc['t_x'], bc['t_y']
                )
        
        for interface in interface_data:
            net_A = model.get_subdomain_network(interface['subdomain_A'])
            net_B = model.get_subdomain_network(interface['subdomain_B'])
            losses['interface_u'] += self.interface_displacement_loss(
                net_A, net_B, interface['x'], interface['y']
            )
            losses['interface_t'] += self.interface_traction_loss(
                net_A, net_B, interface['x'], interface['y'],
                interface['n_x'], interface['n_y']
            )

        if getattr(model, 'monolith', False) and self.energy_context is not None:
            cuts, loaded = self.energy_context
            net_all = model.get_subdomain_network(0)
            losses['energy'] = self.energy_loss(net_all, cuts, loaded)

        if self.resultants is not None:
            for r in self.resultants:
                network = model.get_subdomain_network(r['subdomain'])
                rloss, rparts = self.resultant_loss(network, r, self.weights['neumann'])
                losses['resultant'] += rloss

        if self.anchor_data is not None:
            coords, targets, weight = self.anchor_data
            anet = model.get_subdomain_network(0)
            losses['anchor'] += weight * self.anchor_loss(anet, coords, targets)

        total = sum(self.weights[key] * losses[key]
                    for key in ('pde', 'dirichlet', 'neumann', 'interface_u', 'interface_t'))
        total = total + losses['energy'] + losses['resultant'] + losses['anchor']
        return total, losses


class XPINNTrainer:
    def __init__(self, xpinn_model, physics: PlaneStressElasticity,
                 learning_rate: float = 1e-3,
                 loss_weights: Optional[Dict[str, float]] = None,
                 loss_scales: Optional[Dict[str, float]] = None):
        self.model = xpinn_model
        self.physics = physics
        self.loss_fn = XPINNLoss(physics, loss_weights, scales=loss_scales)
        self._base_weights: Dict[str, float] = dict(self.loss_fn.weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=1000
        )
        self.history = {
            'total': [], 'pde': [], 'dirichlet': [],
            'neumann': [], 'interface_u': [], 'interface_t': [], 'energy': [],
            'resultant': [], 'anchor': [],
        }
        self.adam_epoch_count: Optional[int] = None
        self.probe = None
        self.probe_every = 100

    def compute_total_loss(self, subdomain_data: List[Dict],
                           interface_data: List[Dict],
                           boundary_data: List[Dict]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.loss_fn.compute_total_loss(
            self.model, subdomain_data, interface_data, boundary_data
        )

    def train_step(self, subdomain_data, interface_data, boundary_data) -> Dict[str, float]:
        self.optimizer.zero_grad()
        total_loss, loss_components = self.compute_total_loss(
            subdomain_data, interface_data, boundary_data
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.history['total'].append(total_loss.item())
        for key, val in loss_components.items():
            self.history[key].append(val.item())
        return {key: val.item() for key, val in loss_components.items()}

    def _component_grad_norms(self, subdomain_data, interface_data, boundary_data,
                              keys) -> Dict[str, float]:
        params = list(self.model.parameters())
        self.optimizer.zero_grad()
        _total, comps = self.compute_total_loss(
            subdomain_data, interface_data, boundary_data)
        norms: Dict[str, float] = {}
        for key in keys:
            val = comps.get(key)
            if val is None or not val.requires_grad:
                continue
            grads = torch.autograd.grad(
                val, params, grad_outputs=torch.ones_like(val),
                retain_graph=True, allow_unused=True)
            norm2 = torch.sum(torch.stack([
                (g if g is not None else torch.tensor(0.0, device=val.device)).pow(2).sum()
                for g in grads
            ]))
            norms[key] = norm2.sqrt().item()
        return norms

    def adapt_weights(self, subdomain_data, interface_data, boundary_data,
                      keys, lo: float = 0.2, hi: float = 5.0) -> Dict[str, float]:
        norms = self._component_grad_norms(
            subdomain_data, interface_data, boundary_data, keys)
        if not norms:
            return {}
        med = float(np.median(list(norms.values())))
        out: Dict[str, float] = {}
        for key, g in norms.items():
            if g <= 0.0 or key not in self._base_weights:
                continue
            base = self._base_weights[key]
            factor = float(np.clip(med / g, 0.5, 2.0))
            new_w = float(np.clip(base * factor, base * lo, base * hi))
            self.loss_fn.weights[key] = new_w
            out[key] = new_w
        return out

    def train(self, subdomain_data, interface_data, boundary_data,
              epochs: int = 10000, print_every: int = 500,
              use_lbfgs: bool = True, lbfgs_steps: int = 25,
              lbfgs_max_iter_per_step: int = 20, lbfgs_history_size: int = 50,
              lbfgs_print_every: int = 5,
              adaptive_weights: bool = False, adapt_every: int = 100):
        print("Starting XPINN Training (Stage 1: Adam)...")
        print("=" * 60)
        for epoch in range(epochs):
            losses = self.train_step(subdomain_data, interface_data, boundary_data)
            lr_before = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(self.history['total'][-1])
            if self.probe is not None and (epoch + 1) % self.probe_every == 0:
                self.probe(epoch + 1)
            lr_after = self.optimizer.param_groups[0]['lr']
            if lr_after < lr_before:
                print(f"  [scheduler] Epoch {epoch+1}: LR reduced {lr_before:.3e} -> {lr_after:.3e}")
            if adaptive_weights and (epoch + 1) % adapt_every == 0:
                adjusted = self.adapt_weights(
                    subdomain_data, interface_data, boundary_data,
                    list(self.loss_fn.weights.keys()))
                if adjusted:
                    print("  [adaptive] epoch {:5d}: ".format(epoch + 1)
                          + ", ".join(f"{k}={v:.4g}" for k, v in adjusted.items()))
            if (epoch + 1) % print_every == 0:
                total = self.history['total'][-1]
                print(f"Epoch {epoch+1:5d} | Total Loss: {total:.6e}")
                print(f"  PDE: {losses['pde']:.4e} | "
                      f"Dirichlet: {losses['dirichlet']:.4e} | "
                      f"Neumann: {losses['neumann']:.4e}")
                print(f"  Interface u: {losses['interface_u']:.4e} | "
                      f"Interface t: {losses['interface_t']:.4e}")
                energy_val = losses.get('energy')
                if energy_val is not None:
                    print(f"  Energy (PI, signed): {energy_val:.4e}")
                res_val, anc_val = losses.get('resultant'), losses.get('anchor')
                if (res_val is not None and float(res_val) != 0.0) or \
                   (anc_val is not None and float(anc_val) != 0.0):
                    print(f"  Resultant (already-weighted, added once): {res_val:.4e} | "
                          f"Anchor (already-weighted, added once): {anc_val:.4e}")
        print("=" * 60)
        print(f"Adam stage complete! ({epochs} epochs, final loss "
              f"{self.history['total'][-1]:.6e})")

        self.adam_epoch_count = len(self.history['total'])

        if use_lbfgs:
            self.train_lbfgs(
                subdomain_data, interface_data, boundary_data,
                n_lbfgs_steps=lbfgs_steps,
                max_iter_per_step=lbfgs_max_iter_per_step,
                history_size=lbfgs_history_size,
                print_every=lbfgs_print_every,
            )
        else:
            print("(use_lbfgs=False - skipping stage 2)")

        print("Training Complete!")

    def _lbfgs_closure_factory(self, subdomain_data, interface_data, boundary_data):
        def closure():
            self.lbfgs_optimizer.zero_grad()
            total_loss, loss_components = self.compute_total_loss(
                subdomain_data, interface_data, boundary_data
            )
            total_loss.backward()
            self._last_lbfgs_loss_components = loss_components
            return total_loss
        return closure

    def train_lbfgs(self, subdomain_data, interface_data, boundary_data,
                     n_lbfgs_steps: int = 25, max_iter_per_step: int = 20,
                     history_size: int = 50, line_search_fn: str = 'strong_wolfe',
                     tolerance_grad: float = 1e-8, tolerance_change: float = 1e-10,
                     print_every: int = 5):
        self.lbfgs_optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=max_iter_per_step,
            history_size=history_size,
            line_search_fn=line_search_fn,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
        )

        closure = self._lbfgs_closure_factory(subdomain_data, interface_data, boundary_data)

        print("=" * 60)
        print("Starting XPINN Training (Stage 2: L-BFGS fine-tuning)...")
        print("=" * 60)

        for step in range(n_lbfgs_steps):
            loss_val = self.lbfgs_optimizer.step(closure)
            total = loss_val.item() if torch.is_tensor(loss_val) else float(loss_val)
            comp = self._last_lbfgs_loss_components

            self.history['total'].append(total)
            for key, val in comp.items():
                self.history[key].append(val.item())
            if self.probe is not None:
                self.probe((self.adam_epoch_count or 0) + (step + 1) * max_iter_per_step)

            if (step + 1) % print_every == 0 or step == n_lbfgs_steps - 1:
                print(f"L-BFGS step {step+1:3d}/{n_lbfgs_steps} | Total Loss: {total:.6e}")
                print(f"  PDE: {comp['pde'].item():.4e} | "
                      f"Dirichlet: {comp['dirichlet'].item():.4e} | "
                      f"Neumann: {comp['neumann'].item():.4e}")
                print(f"  Interface u: {comp['interface_u'].item():.4e} | "
                      f"Interface t: {comp['interface_t'].item():.4e}")
                res_val, anc_val = comp.get('resultant'), comp.get('anchor')
                if (res_val is not None and float(res_val.detach()) != 0.0) or \
                   (anc_val is not None and float(anc_val.detach()) != 0.0):
                    print(f"  Resultant (already-weighted, added once): {res_val.item():.4e} | "
                          f"Anchor (already-weighted, added once): {anc_val.item():.4e}")

        print("=" * 60)
        print(f"L-BFGS stage complete! (final loss {self.history['total'][-1]:.6e})")