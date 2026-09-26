# PINN (DeepXDE library) vs ATLAS-XPINN (energy) vs FEM

Errors are relative L2 against the FEniCS reference in `fem/reference/` (displacement magnitude, and von Mises with the singular region excluded, the same metrics `atlas.validation` reports). Targets: displacement < 5 %, von Mises < 10 %. No solver sees FEM data in training.

Compliance C = int t.u ds over the loaded edge, computed from each solver's displacement at the reference nodes; its error is relative to the FEM value (spec target: within 10 %).

Two displacement errors are shown. *Magnitude* is the metric `atlas.validation` reports (compares sqrt(u^2+v^2)); *vector* compares the (u, v) field itself and therefore catches a sign-flipped solution that the magnitude metric hides.

| benchmark | solver | displacement L2 (magnitude) | displacement L2 (vector) | von Mises L2 (excl.) | compliance [N m] | compliance vs FEM | parameters | train time |
|---|---|---|---|---|---|---|---|---|
| Cantilever beam | PINN (DeepXDE library) | 87.79% (fail) | 88.09% (fail) | 89.92% (fail) | 132 (FEM 1272) | -89.63% (fail) | 12,802 | 11.6 min |
| Cantilever beam | ATLAS-XPINN (energy) | 0.27% (PASS) | 0.27% (PASS) | 2.10% (PASS) | 1269 (FEM 1272) | -0.22% (PASS) | 25,604 | 5.0 min |
| Plate with circular hole | PINN (DeepXDE library) | 41.85% (fail) | 51.42% (fail) | 31.66% (fail) | 15.54 (FEM 20.14) | -22.83% (fail) | 12,802 | 11.1 min |
| Plate with circular hole | ATLAS-XPINN (energy) | 0.75% (PASS) | 0.81% (PASS) | 3.39% (PASS) | 20.09 (FEM 20.14) | -0.23% (PASS) | 38,406 | 7.3 min |
| L-shaped bracket | PINN (DeepXDE library) | 84.15% (fail) | 115.18% (fail) | 81.97% (fail) | -6.296 (FEM 96.52) | -106.52% (fail) | 12,802 | 12.4 min |
| L-shaped bracket | ATLAS-XPINN (energy) | 0.22% (PASS) | 0.30% (PASS) | 4.55% (PASS) | 96.89 (FEM 96.52) | +0.38% (PASS) | 38,478 | 14.5 min |

Compliance difference and pointwise maxima

Compliance difference is C - C_FEM. Maximum pointwise errors are the single worst node over the whole reference mesh, as a percentage of the FEM maximum of that field. The von Mises maximum usually sits at a stress concentration (hole edge, re-entrant corner, clamp corner), where the FEM reference itself is least reliable, so treat it as a worst-case indicator, not a pass/fail metric.

| benchmark | solver | compliance difference [N m] | max pointwise displacement error | max pointwise von Mises error | at (x, y) |
|---|---|---|---|---|---|
| Cantilever beam | PINN (DeepXDE library) | -1140 | 90.09% | 97.24% (4.11e+07 Pa) | (0.000, 1.000) |
| Cantilever beam | ATLAS-XPINN (energy) | -2.814 | 0.38% | 43.26% (1.83e+07 Pa) | (0.000, 1.000) |
| Plate with circular hole | PINN (DeepXDE library) | -4.597 | 65.99% | 89.09% (5e+06 Pa) | (0.000, 0.313) |
| Plate with circular hole | ATLAS-XPINN (energy) | -0.04641 | 5.39% | 53.60% (3.01e+06 Pa) | (0.000, 0.313) |
| L-shaped bracket | PINN (DeepXDE library) | -102.8 | 114.16% | 99.03% (2.63e+07 Pa) | (1.000, 1.000) |
| L-shaped bracket | ATLAS-XPINN (energy) | +0.3642 | 1.16% | 63.71% (1.69e+07 Pa) | (1.000, 1.000) |

Solver definitions

- **PINN (DeepXDE library)**: an independent baseline built with the open-source DeepXDE library (Lu et al., SIAM Review 2021): one tanh network, soft boundary conditions, library-default loss weights, Adam then L-BFGS; nothing from the ATLAS solver. Five cantilever settings (loss weights 1-100, float32/float64) were tried; none got below 88 % displacement error, so the library defaults are reported.
- **ATLAS-XPINN (energy)**: the same energy formulation and hard boundary conditions, domain-decomposed: one subnetwork per subdomain with interface displacement and traction continuity (augmented-Lagrangian coupling, single-network warm start; docs/BUGFIXES.md section 16).
