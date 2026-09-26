# Sandwich panel benchmark: protocol (fixed 2026-09-25, before any training run)

Question: with two long material interfaces and a 70:1 stiffness jump, where the panel deflects
mostly by shearing its soft core (so du/dy changes slope sharply at both interfaces), does
ATLAS-XPINN with one subnetwork per layer beat the single-network ATLAS energy solver?

Precedent: steel-polymer-steel sandwich plates (the "sandwich plate system", steel faces bonded to
an elastomer or polymer core) are used for ship and bridge decks; sandwich beam theory (Allen 1969;
Zenkert 1995) splits the deflection into face bending and core shear. Diao et al., CMAME 413 (2023)
116120 motivate domain decomposition for multi-material solids.

## Problem
- Cantilever panel [0, 4] x [0, 1] m, plane stress. Faces y in [0, 0.1] and [0.9, 1]: steel,
  E = 210 GPa, nu = 0.3. Core y in [0.1, 0.9]: structural polymer, E = 3 GPa, nu = 0.35
  (contrast 70). Perfect bonding.
- x = 0 clamped (u = v = 0, exact via the output mask); uniform pressure 1 MPa downward on the
  top face y = 1; all other edges traction-free.
- Reference: FEniCS P2 on a mesh aligned with both interfaces, element-wise DG1 stresses, two
  mesh levels. Never used in training.

## Solvers (seed 42 first, identical quadrature, identical budget 2000 Adam + 2000 L-BFGS)
- atlas       : one 4 x 64 network, per-point material (12.8k parameters)
- xpinn       : ATLAS-XPINN, 3 subnetworks (bottom face, core, top face), interfaces on the
                two material boundaries, default recipe unchanged (38.4k)
- atlas_wide  : one 4 x 112 network, parameter-matched to xpinn (38.5k)
Run order: atlas, then xpinn. If atlas beats xpinn on displacement or von Mises, the benchmark
stops there (reported as a failed candidate). Otherwise atlas_wide runs.

## Metrics (vs FEM)
- displacement vector relative L2 on a 0.01 m grid
- von Mises relative L2 on the same grid, excluding the four singular points on the clamp
  (the two corners and the two points where the interfaces meet the clamp), exclusion radius
  from the evaluation-grid spacing as in the other benchmarks
- secondary: von Mises L2 in the faces and in the core separately; tip deflection at (4, 0.5);
  compliance (work of the pressure)

## Decision rule
XPINN wins only if it has lower displacement AND lower von Mises error than BOTH atlas and
atlas_wide at seed 42. Then 5 seeds (42-46) for xpinn and the better single network; the claim
stands if the 5-seed mean is lower for both metrics. All runs reported whatever the outcome.
