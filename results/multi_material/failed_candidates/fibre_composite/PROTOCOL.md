# Fibre-composite benchmark: protocol (fixed 2026-09-25, before any training run)

Question: with many curved interfaces around STIFF inclusions (the opposite regime to the soft
inclusion of results/multi_material/bimaterial), does ATLAS-XPINN with a material-wise decomposition beat
the single-network ATLAS energy solver?

Precedent: transverse loading of a unidirectional fibre composite is the standard micromechanics
test (e.g. Henkes, Wessels, Mahnken, "Physics informed neural networks for continuum
micromechanics", CMAME 393 (2022) 114790). Wang et al. (CENN, CMAME 2022) decompose by material.

## Problem
- Plate [0, 2] x [0, 1] m, plane stress, 4 x 2 square cells of side 0.5 with a fibre of radius
  0.15 at each centre (fibre volume fraction 28 %).
- Fibres: E-glass, E = 72 GPa, nu = 0.22. Matrix: epoxy, E = 3.5 GPa, nu = 0.35 (contrast 20.6).
- Symmetry rollers on x = 0 and y = 0 (exact via the mask), uniform tension 1 MPa on x = 2,
  y = 1 traction-free. Perfect bonding.
- Reference: FEniCS P2 on a mesh that conforms to every fibre, element-wise DG1 stresses, two
  mesh levels. Never used in training.

## Solvers (seed 42 first, identical quadrature, identical budget 2000 Adam + 2000 L-BFGS)
- atlas       : one 4 x 64 network, per-point material (12.8k parameters)
- xpinn       : ATLAS-XPINN, 2 subnetworks: all fibres / matrix, interfaces on all eight
                fibre boundaries, default recipe unchanged (25.6k)
- atlas_wide  : one 4 x 91 network, parameter-matched (25.6k)
Run order: atlas, then xpinn; if atlas beats xpinn on displacement or von Mises the benchmark stops
there (reported as a failed candidate); otherwise atlas_wide.

## Metrics (vs FEM)
- displacement vector relative L2 and von Mises relative L2 on a 0.01 m grid (no singular points)
- secondary: von Mises L2 per phase, and within 0.03 m of the interfaces (fibre side, matrix
  side); compliance

## Decision rule
As for the other benchmarks: XPINN must have lower displacement AND lower von Mises error than
BOTH single networks at seed 42; then 5 seeds (42-46) for xpinn and the better single network, and
the claim stands if the 5-seed mean is lower for both metrics. All runs reported.

## Addendum (2026-09-25, after the first seed-42 runs)
First runs: ATLAS energy 39.2% / 86.4%, ATLAS-XPINN 97.9% / 90.9% (displacement / von Mises), so
by the rule above the benchmark stopped. Diagnosis of the XPINN collapse (error rose from 75.7% to
98% the moment the subnetworks separated): the interface penalty weights were scaled by the
problem's base material, which here is the SOFT matrix (3.5 GPa), not the stiffest material.
Measured at the warm-start state, the traction-continuity term was 60x the strain energy (0.011x on
the inclusion benchmark, whose base material is the stiff one), so the optimiser removed
deformation instead. Fix, chosen from dimensional reasoning and not from any comparison with FEM:
scale the interface terms by the stiffest material present (identical to before for every earlier
benchmark, checked bit-for-bit on all six). The first XPINN run is kept in
xpinn_first_run_soft_reference/ and reported. XPINN is rerun once with the fix under the same
decision rule; the single-network result is unaffected (it has no interface terms) and stands.
