# Fibre composite: failed candidate (ATLAS-XPINN did not beat ATLAS energy)

Epoxy plate (3.5 GPa) with eight E-glass fibres (72 GPa, r = 0.15 m, 28 % volume fraction),
transverse tension. Protocol and dated addendum: [PROTOCOL.md](PROTOCOL.md). FEM reference
converged (compliance 1.7e-6, interface-band von Mises 1.5e-3 between the last two levels).

## Seed 42 (2000 Adam + 2000 L-BFGS, same 24,192 quadrature points)

| run | train time | displacement L2 | von Mises L2 | band vM fibre side | band vM matrix side | compliance |
|---|---|---|---|---|---|---|
| ATLAS energy (single 4x64) | 1394 s | **39.2%** | 86.4% | 174% | 74.1% | -37.7% |
| ATLAS-XPINN, first run (interfaces scaled by the soft matrix) | 1434 s | 97.9% | 90.9% | 89.6% | 94.0% | -89.1% |
| ATLAS-XPINN, rerun (interfaces scaled by the stiffest material) | 1517 s | 91.9% | **69.9%** | 20.8% | 95.3% | -91.9% |

Neither solver comes close to the targets, and ATLAS energy has the lower displacement error, so
the benchmark is closed after the one disclosed rerun.

## What the numbers say
- The single network stalls: Adam does not move the error (75.7% from step 100 to 2000) and
  L-BFGS brings it to 39%. It sits near a uniform-strain ("Voigt") state: a smooth network cannot
  put the strain into the matrix between the fibres, so the plate comes out about 4x too stiff at
  first and the fibres carry far too much stress (174% fibre-side error).
- First XPINN run: the interface weights were scaled by the soft matrix, making the traction term
  60x the strain energy at the warm-start state; the optimiser removed deformation instead. Fixed
  (scale by the stiffest material; no earlier result changes).
- Rerun: the fibre stresses improve (fibre-side band error 90% -> 21%), but with the weights now
  scaled by the fibre, the displacement-continuity term is 20x stiff for the matrix and the matrix
  subnetwork barely deforms (compliance -92%). One set of interface weights cannot serve both
  sides of a 20:1 contrast around eight interfaces at this budget.
- Per-side interface weights (or no traction term: in an energy method traction continuity is a
  natural condition, which is how CENN formulates it) are the obvious next step. They were not
  tried here, because they would depart from the ATLAS-XPINN formulation and would be tuning after
  seeing the result.

Files: `atlas/`, `xpinn/` (rerun), `xpinn_first_run_soft_reference/`, `logs/`.
Code: `atlas/benchmarks/fibre_composite.py`, `fem/fenics_scripts/fibre_composite_fenics.py`, `scripts/multi_material/run_fibre_composite.py`.
