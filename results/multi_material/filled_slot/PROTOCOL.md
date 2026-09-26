# Epoxy-filled elliptical slot in a steel plate: protocol (fixed 2026-09-25, before any run)

Third candidate in the class where ATLAS-XPINN has won (a closed material interface inside the
part), differing from the circular inclusion and the steel insert in shape (ellipse), stress
concentration (an open slot of this shape has Kt = 1 + 2 b/a = 5) and contrast (60:1).
Engineering case: a slot or cut-out filled with an epoxy filler.

## Problem (plane stress)
- Quarter plate [0, 2] x [0, 2] m, elliptical filler centred at the origin, semi-axes 0.2 m along
  the load (x) and 0.4 m across it (y).
- Plate: steel, E = 210 GPa, nu = 0.30. Filler: epoxy, E = 3.5 GPa, nu = 0.35 (contrast 60).
- Symmetry rollers on x = 0 and y = 0 (exact via the mask), uniform tension 1 MPa on x = 2,
  y = 2 traction-free. Perfect bonding. No singular points.
- Reference: FEniCS P2 on an interface-conforming mesh, element-wise DG1 stresses, two levels.

## Solvers (seed 42 first, identical quadrature, identical budget 2000 Adam + 2000 L-BFGS)
- atlas (4 x 64), xpinn (2 subnetworks: filler, plate; default recipe; interface weights scaled by
  the stiffest material), atlas_wide (4 x 91, parameter-matched). Run order and stopping rule as
  for the other benchmarks.

## Metrics and decision rule
Displacement vector L2 and von Mises L2 on a 0.01 m grid; secondary: von Mises within 0.05 m of the
interface (filler side, plate side), compliance. XPINN must beat BOTH single networks on
displacement AND von Mises at seed 42; then 5 seeds (42-46) for xpinn and the better single network;
the claim stands if the 5-seed means are lower for both. All runs reported.
