# Plate with a circular inclusion (two materials): ATLAS energy vs ATLAS-XPINN

The plate-with-hole geometry with the hole filled by a softer second material: quarter plate
[0, 2] x [0, 2] m, inclusion radius 0.3 m, steel matrix (210 GPa, nu 0.3), inclusion 210 GPa /
contrast (nu 0.3), symmetry rollers on x = 0 and y = 0, 1 MPa tension on x = 2. The protocol,
including the decision rule, was fixed before any run: [PROTOCOL.md](PROTOCOL.md).

**Result: ATLAS-XPINN beats the single-network solver, including a parameter-matched one,
on every metric and every seed.** This is the first of the benchmarks where domain decomposition
pays off; the other five (cantilever, plate with hole, L-bracket, notched plate, multi-cutout
bracket) all have a single material.

## Reference
FEniCS P2 on a mesh that conforms to the interface (110k cells), stresses element-wise (DG1, exact
for P2, so the stress jump is kept). Mesh convergence between the last two levels: compliance
1e-9, interface-band von Mises 1.8e-4 (contrast 15). Cross-check: the uniform stress inside the
inclusion is 0.183 sigma (contrast 15) and 0.610 sigma (contrast 3), against 0.175 and 0.597 from
Goodier's infinite-plate solution (a finite plate reads slightly higher).

## Contrast 15 (primary), seed 42
Same quadrature points (12,960), same budget (2000 Adam + 2000 L-BFGS) for every solver.

| solver | params | train time | displacement L2 | von Mises L2 | band vM, inclusion side | band vM, matrix side | compliance vs FEM |
|---|---|---|---|---|---|---|---|
| ATLAS energy (single 4x64) | 12,802 | 325 s | 0.196% | 0.704% | 21.64% | 4.08% | -0.071% |
| ATLAS energy (single 4x91, parameter-matched) | 25,573 | 545 s | 0.133% | 0.502% | 18.71% | 3.48% | -0.023% |
| **ATLAS-XPINN (inclusion + matrix)** | 25,604 | 352 s | **0.043%** | **0.318%** | **1.38%** | **2.24%** | **-0.000%** |

"Band" = the 0.0025 m grid within 0.05 m of the interface.

## Contrast 15, five seeds (42-46): ATLAS-XPINN vs the better single network

| solver | displacement L2 | von Mises L2 | band vM, inclusion side | band vM, matrix side | train time |
|---|---|---|---|---|---|
| ATLAS energy (single 4x91) | 0.153 +- 0.021% | 0.635 +- 0.114% | 20.5 +- 1.8% | 4.17 +- 0.48% | 549 s |
| **ATLAS-XPINN** | **0.047 +- 0.018%** | **0.325 +- 0.091%** | **1.31 +- 0.87%** | **2.11 +- 0.72%** | **360 s** |

ATLAS-XPINN has the lower displacement AND von Mises error in all five seeds (not just on
average): 3.3x lower displacement error, 2.0x lower von Mises error, 16x lower stress error on
the inclusion side of the interface, in two thirds of the time.

## Contrast 3 (secondary, about aluminium in steel), seed 42

| solver | params | train time | displacement L2 | von Mises L2 | band vM, inclusion side | band vM, matrix side |
|---|---|---|---|---|---|---|
| ATLAS energy (single 4x64) | 12,802 | 324 s | 0.097% | 0.960% | 11.80% | 8.17% |
| ATLAS energy (single 4x91) | 25,573 | 557 s | 0.146% | 1.168% | 14.98% | 9.47% |
| **ATLAS-XPINN** | 25,604 | 363 s | **0.049%** | **0.316%** | **0.98%** | **2.21%** |

The advantage persists at a mild jump. Single seed only (the protocol runs five seeds at the
primary contrast).

## Why it works, and checks that it is genuine
- A single tanh network is smooth, so it cannot have the kink in displacement (jump in strain) the
  exact solution has across the interface; it rounds it off and rings on both sides
  ([fields.png](c15/fields.png), [profiles.png](c15/profiles.png)). Doubling the parameters barely
  helps (21.6% -> 18.7% on the inclusion side), so the error is structural, not a capacity limit.
  With the subdomain interface on the material boundary each subnetwork only has to represent a
  smooth field.
- The decomposed solution is continuous: final integrated squared interface jump 6e-17 (the
  augmented-Lagrangian recipe). Its total potential energy
  (-0.52302) is below the single network's (-0.52281); for a conforming (continuous) field, lower
  energy means closer to the exact solution, so the improvement is not a penalty loophole of the
  kind seen on the notched plate before that fix.
- Same quadrature, same budget, same recipe as every other benchmark; nothing was tuned here.
- The error-vs-time curves ([convergence.png](c15/convergence.png)): the single 4x64 network
  stalls near 0.2% displacement error after ~300 s; ATLAS-XPINN passes it at ~240 s and is still
  improving when the budget ends.

## Limitation seen
Along y = 0 the matrix-side stress peak just outside the interface (FEM 0.88 MPa at x = 0.31) is
underestimated by ATLAS-XPINN (0.74 MPa, -16%), while the single networks overshoot it (4x64:
1.11 MPa) and oscillate behind it. The global maximum (2.5 MPa at the top of the inclusion) is
reproduced by all solvers.

## Files
- `c15/`, `c3/`: per-solver `seed*.json` (metrics, error-vs-time probes), `seed*.npz` (fields),
  `seed*_model.pt`; `table.md`, `convergence.png`, `fields.png`, `profiles.png`
- code: `atlas/benchmarks/bimaterial.py`, `fem/fenics_scripts/bimaterial_fenics.py`, `scripts/multi_material/run_bimaterial.py`,
  `scripts/multi_material/make_report.py`

Reproduce (one GPU job at a time):

    PATH=$CONDA_PREFIX/bin:$PATH python fem/fenics_scripts/bimaterial_fenics.py --contrast 15
    python scripts/multi_material/run_bimaterial.py --solver xpinn --contrast 15 --seed 42
    python scripts/multi_material/make_report.py
