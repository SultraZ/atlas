# Two-material benchmark: protocol (fixed 2026-09-25, before any training run)

Question: when the solution has a kink a single smooth network cannot represent (the strain jump at a
material interface), does ATLAS-XPINN, with the subdomain interface placed on the material
boundary, beat the single-network ATLAS energy solver?

Precedent: Wang et al., "CENN: conservative energy method based on neural networks with subdomains",
CMAME 2022 (arXiv 2110.01359): on a two-phase problem with a circular interface and a 15:1 property
contrast, the single-network deep energy method reached 6.9 % relative L2 and the subdomain version
0.81 %. Diao et al., CMAME 413 (2023) 116120, report the same trend for strong-form PINNs in
multi-material solid mechanics.

## Problem
- Quarter plate [0, 2] x [0, 2] m with a circular inclusion of radius 0.3 m at the origin (the
  plate-with-hole benchmark with the hole filled by a second material).
- Plane stress. Matrix: E = 210 GPa, nu = 0.3. Inclusion: E = 210 GPa / contrast, nu = 0.3
  (the same nu, so only the stiffness jumps).
- Contrast 15 (primary, the CENN contrast) and 3 (secondary, about aluminium in steel).
- Symmetry rollers on x = 0 and y = 0 (exact via the output mask), uniform tension 1 MPa on x = 2,
  y = 2 traction-free. Perfect bonding.
- Reference: FEniCS P2 on a mesh that conforms to the interface, stresses taken element-wise
  (DG1, exact for P2), mesh-convergence check on two levels. Never used in training.

## Solvers (all seed 42 first, identical quadrature points, identical budget 2000 Adam + 2000 L-BFGS)
- atlas       : one 4 x 64 tanh network, per-point material in the energy (12.8k parameters)
- xpinn       : ATLAS-XPINN, 2 subnetworks (inclusion, matrix), interface = the material
                boundary, the default recipe unchanged (augmented Lagrangian 3 stages, 1000-step
                shared warm start, gamma_u 30, gamma_t 0.1, global input normalisation) (25.6k)
- atlas_wide  : one 4 x 91 network, parameter-matched to xpinn (control for capacity)

## Metrics (vs FEM)
- displacement: vector relative L2 on a 0.01 m grid over the whole quarter plate
- von Mises: relative L2 on the same grid (no singular points, so no exclusion)
- interface band (secondary): von Mises relative L2 on a 0.0025 m grid within 0.05 m of the
  interface, inclusion side and matrix side separately
- compliance (load-edge work) vs FEM

## Decision rule (decided now)
XPINN "wins" only if, at the primary contrast, it has lower displacement AND lower von Mises error
than BOTH single networks (atlas and atlas_wide). Only then: 5 seeds (42-46) for xpinn and the
better single network, and the claim stands if the 5-seed mean is lower for both metrics.
All runs are reported whatever the outcome, one GPU job at a time.
