# Steel-aluminium transition joint: protocol (fixed 2026-09-25, before any training run)

Replacement candidate after the sandwich panel and the fibre composite failed. Chosen to match what
made the inclusion benchmark work: one interface, no thin layers in bending, a problem a single
network otherwise solves well, so the interface kink is the dominant error.

Question: with a WAVY bonded interface between steel and aluminium (the interface shape produced by
explosion welding, the process used for the steel-aluminium transition joints that connect
aluminium superstructures to steel hulls), does ATLAS-XPINN with the subdomain interface on
the bond line beat the single-network ATLAS energy solver?

## Problem (idealised, plane stress)
- Plate [0, 2] x [0, 1] m (the upper half of a joint symmetric about y = 0). Steel for
  x < f(y), aluminium for x > f(y), f(y) = 1 + 0.1 cos(4 pi y): amplitude 0.1 m, wavelength 0.5 m.
- Steel E = 210 GPa, nu = 0.30; aluminium E = 70 GPa, nu = 0.33 (contrast 3). Perfect bonding.
- Symmetry rollers on x = 0 and y = 0 (exact via the output mask), uniform tension 1 MPa on
  x = 2, y = 1 traction-free.
- Singular point: where the bond line meets the free edge, (1.1, 1) (bimaterial free-edge corner).
- Reference: FEniCS P2 on a structured mesh mapped so that it follows the bond line, element-wise
  DG1 stresses, two mesh levels. Never used in training.

## Solvers (seed 42 first, identical quadrature, identical budget 2000 Adam + 2000 L-BFGS)
- atlas       : one 4 x 64 network, per-point material (12.8k parameters)
- xpinn       : ATLAS-XPINN, 2 subnetworks (steel, aluminium), interface = the bond line,
                default recipe unchanged, interface weights scaled by the stiffest material (25.6k)
- atlas_wide  : one 4 x 91 network, parameter-matched (25.6k)
Run order: atlas, then xpinn; if atlas beats xpinn on displacement or von Mises the benchmark
stops (reported as failed); otherwise atlas_wide.

## Metrics (vs FEM)
- displacement vector relative L2 and von Mises relative L2 (excluding the singular point) on a
  0.01 m grid
- secondary: von Mises L2 within 0.05 m of the bond line (steel side, aluminium side); compliance

## Decision rule
XPINN must have lower displacement AND lower von Mises error than BOTH single networks at seed 42;
then 5 seeds (42-46) for xpinn and the better single network, and the claim stands if the 5-seed
mean is lower for both metrics. All runs reported.
