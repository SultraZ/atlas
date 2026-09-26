# Sandwich panel: failed candidate (ATLAS-XPINN did not beat ATLAS energy)

Steel faces (0.1 m, 210 GPa) on a polymer core (3 GPa, contrast 70), cantilever 4 x 1 m, clamped at
x = 0, 1 MPa pressure on the top face. Protocol fixed before the runs: [PROTOCOL.md](PROTOCOL.md).
FEM reference converged (compliance and displacement change < 2e-6 between the last two levels,
von Mises 1.2e-3 away from the clamp singularities); tip deflection 10.22 mm (sandwich beam theory
10.87 mm).

## Seed 42 (2000 Adam + 2000 L-BFGS, same quadrature)

| solver | params | train time | displacement L2 | von Mises L2 (excl.) | faces vM | core vM | tip deflection | compliance |
|---|---|---|---|---|---|---|---|---|
| ATLAS energy (single 4x64) | 12,802 | 370 s | 5.46% | **7.96%** | 7.10% | 21.0% | -4.8% | -5.67% |
| ATLAS-XPINN (3 layers) | 38,406 | 444 s | **4.84%** | 18.59% | 18.6% | 18.4% | -3.0% | -5.43% |

ATLAS energy has the lower von Mises error, so by the decision rule the benchmark stops here
(no parameter-matched control, no seeds).

## What the numbers say
- Neither solver converged within the fixed budget. Both are too stiff (compliance -5.7% and
  -5.4%) and their potential energies (-0.2481, -0.2502 in solver units) lie 5-6% above the FEM
  minimum (-0.264). The single network misses the 5% displacement target, the first benchmark where
  it does.
- ATLAS-XPINN's displacement error was still swinging between 4.8% and 13.9% over its last ten
  probes as the augmented-Lagrangian updates perturbed it; its interfaces were continuous
  (integrated squared jump 6e-11).
- The obstacle is the conditioning of the energy functional (thin stiff faces, contrast 70,
  deflection dominated by core shear), which decomposition does not remove at this budget. A
  longer budget was not tried: changing the budget after seeing the result is outside the protocol.

Files: `atlas/`, `xpinn/` (seed42.json with error-vs-time probes, .npz fields, model), `logs/`.
Code: `atlas/benchmarks/sandwich.py`, `fem/fenics_scripts/sandwich_fenics.py`, `scripts/multi_material/run_sandwich.py`.
