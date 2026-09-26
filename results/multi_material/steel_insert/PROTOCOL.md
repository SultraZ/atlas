# Steel insert in an aluminium plate: protocol (fixed 2026-09-25, before any training run)

Candidate chosen after the sandwich panel, fibre composite and transition joint failed. Those
failures, and the inclusion benchmark's success, suggest ATLAS-XPINN wins when the interface is
closed and inside the part (one subdomain with a simple field, the other a plate-with-hole-like
field) and loses when the interface crosses the part or there are many high-contrast interfaces.
This benchmark tests that pattern on a STIFF inclusion, the opposite regime to the soft inclusion.

Engineering case: a bonded steel bushing / insert in an aluminium plate (steel bushings in
aluminium lugs and fittings are standard in aircraft structures).

## Problem (plane stress)
- Quarter plate [0, 2] x [0, 2] m, circular insert of radius 0.3 m at the origin (the same geometry
  and quadrature as the inclusion benchmark).
- Plate: aluminium, E = 70 GPa, nu = 0.33. Insert: steel, E = 210 GPa, nu = 0.30 (3x stiffer).
- Symmetry rollers on x = 0 and y = 0 (exact via the mask), uniform tension 1 MPa on x = 2,
  y = 2 traction-free. Perfect bonding.
- Reference: FEniCS P2 on an interface-conforming mesh, element-wise DG1 stresses, two levels.

## Solvers (seed 42 first, identical quadrature, identical budget 2000 Adam + 2000 L-BFGS)
- atlas (4 x 64), xpinn (2 subnetworks: insert, plate; default recipe; interface weights scaled
  by the stiffest material), atlas_wide (4 x 91, parameter-matched). Run order and stopping rule as
  for the other benchmarks.

## Metrics and decision rule
Displacement vector L2 and von Mises L2 on a 0.01 m grid (no singular points); secondary: von Mises
within 0.05 m of the interface (insert side, plate side), compliance. XPINN must beat BOTH single
networks on displacement AND von Mises at seed 42; then 5 seeds (42-46) for xpinn and the better
single network, claim stands if the 5-seed means are lower for both. All runs reported.
