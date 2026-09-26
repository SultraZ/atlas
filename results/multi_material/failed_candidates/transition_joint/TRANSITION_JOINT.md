# Steel-aluminium transition joint: failed candidate (ATLAS-XPINN did not beat ATLAS energy)

Steel (210 GPa) bonded to aluminium (70 GPa) along a wavy line x = 1 + 0.1 cos(4 pi y), plate
2 x 1 m (upper half of a symmetric joint), 1 MPa tension. Protocol: [PROTOCOL.md](PROTOCOL.md).
FEM reference converged (compliance 4.5e-5, displacement 7.5e-5, von Mises 1.0e-3 between levels).

## Seed 42 (2000 Adam + 2000 L-BFGS, same 14,400 quadrature points)

| solver | params | train time | displacement L2 | von Mises L2 (excl.) | band vM steel side | band vM aluminium side | compliance |
|---|---|---|---|---|---|---|---|
| ATLAS energy (single 4x64) | 12,802 | 351 s | **0.69%** | 6.35% | 29.7% | 23.7% | -0.44% |
| ATLAS-XPINN (steel + aluminium) | 25,604 | 412 s | 1.98% | **4.84%** | 12.5% | 8.0% | -0.98% |

ATLAS energy has the lower displacement error, so by the decision rule the benchmark stops here.

## What the numbers say
- ATLAS-XPINN does what it should at the bond line: the stress error within 0.05 m of it drops from
  24-30% to 8-12%, and the global von Mises error from 6.35% to 4.84%.
- But it converges more slowly overall: its potential energy (-0.4901) is ABOVE the single
  network's (-0.4923), i.e. further from the minimum, and its displacement error was still falling
  when the budget ended (5.73% -> 2.95% -> 2.48% -> 1.98% at 2000 / 3000 / 3500 / 4000 iterations,
  against 2.53% -> 0.69% for the single network). The interface is continuous (jump^2 2e-15).
- Contrast with the inclusion benchmark, where ATLAS-XPINN ended at a LOWER energy: there the
  interface is closed and the inclusion's own field is simple (uniform strain); here the bond line
  crosses the whole part and both subdomains carry half of the global deformation, so the coupling
  through the augmented-Lagrangian stages costs iterations the single network does not spend.

Files: `atlas/`, `xpinn/` (seed42.json with error-vs-time probes, .npz fields, model), `logs/`.
Code: `atlas/benchmarks/transition_joint.py`, `fem/fenics_scripts/transition_joint_fenics.py`, `scripts/multi_material/run_transition_joint.py`.
