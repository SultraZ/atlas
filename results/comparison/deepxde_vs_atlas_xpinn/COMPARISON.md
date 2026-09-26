# PINN (DeepXDE, library defaults) vs PINN (DeepXDE, configured) vs ATLAS-XPINN vs FEM

Errors are relative L2 against the FEniCS reference in `fem/reference/` (displacement magnitude, and von Mises with the singular region excluded, the same metrics `atlas.validation` reports). Targets: displacement < 5 %, von Mises < 10 %. No solver sees FEM data in training.

Compliance C = int t.u ds over the loaded edge, computed from each solver's displacement at the reference nodes; its error is relative to the FEM value (spec target: within 10 %).

Two displacement errors are shown. *Magnitude* is the metric `atlas.validation` reports (compares sqrt(u^2+v^2)); *vector* compares the (u, v) field itself and therefore catches a sign-flipped solution that the magnitude metric hides.

| benchmark | solver | displacement L2 (magnitude) | displacement L2 (vector) | von Mises L2 (excl.) | compliance [N m] | compliance vs FEM | parameters | train time |
|---|---|---|---|---|---|---|---|---|
| Cantilever beam | PINN (DeepXDE, library defaults) | 87.79% (fail) | 88.09% (fail) | 89.92% (fail) | 132 (FEM 1272) | -89.63% (fail) | 12,802 | 11.6 min |
| Cantilever beam | PINN (DeepXDE, configured) | 97.16% (fail) | 102.57% (fail) | 93.30% (fail) | -32.25 (FEM 1272) | -102.54% (fail) | 12,802 | 58.0 min |
| Cantilever beam | ATLAS-XPINN | 0.27% (PASS) | 0.27% (PASS) | 2.10% (PASS) | 1269 (FEM 1272) | -0.22% (PASS) | 25,604 | 5.0 min |
| Plate with circular hole | PINN (DeepXDE, library defaults) | 41.85% (fail) | 51.42% (fail) | 31.66% (fail) | 15.54 (FEM 20.14) | -22.83% (fail) | 12,802 | 11.1 min |
| Plate with circular hole | PINN (DeepXDE, configured) | 44.22% (fail) | 50.50% (fail) | 33.60% (fail) | 15.26 (FEM 20.14) | -24.24% (fail) | 12,802 | 50.1 min |
| Plate with circular hole | ATLAS-XPINN | 0.75% (PASS) | 0.81% (PASS) | 3.39% (PASS) | 20.09 (FEM 20.14) | -0.23% (PASS) | 38,406 | 7.3 min |
| L-shaped bracket | PINN (DeepXDE, library defaults) | 84.15% (fail) | 115.18% (fail) | 81.97% (fail) | -6.296 (FEM 96.52) | -106.52% (fail) | 12,802 | 12.4 min |
| L-shaped bracket | PINN (DeepXDE, configured) | 98.88% (fail) | 100.52% (fail) | 94.45% (fail) | 2.394 (FEM 96.52) | -97.52% (fail) | 12,802 | 54.6 min |
| L-shaped bracket | ATLAS-XPINN | 0.22% (PASS) | 0.30% (PASS) | 4.55% (PASS) | 96.89 (FEM 96.52) | +0.38% (PASS) | 38,478 | 14.5 min |

Compliance difference and pointwise maxima

Compliance difference is C - C_FEM. Maximum pointwise errors are the single worst node over the whole reference mesh, as a percentage of the FEM maximum of that field. The von Mises maximum usually sits at a stress concentration (hole edge, re-entrant corner, clamp corner), where the FEM reference itself is least reliable, so treat it as a worst-case indicator, not a pass/fail metric.

| benchmark | solver | compliance difference [N m] | max pointwise displacement error | max pointwise von Mises error | at (x, y) |
|---|---|---|---|---|---|
| Cantilever beam | PINN (DeepXDE, library defaults) | -1140 | 90.09% | 97.24% (4.11e+07 Pa) | (0.000, 1.000) |
| Cantilever beam | PINN (DeepXDE, configured) | -1304 | 102.73% | 98.05% (4.14e+07 Pa) | (0.000, 1.000) |
| Cantilever beam | ATLAS-XPINN | -2.814 | 0.38% | 43.26% (1.83e+07 Pa) | (0.000, 1.000) |
| Plate with circular hole | PINN (DeepXDE, library defaults) | -4.597 | 65.99% | 89.09% (5e+06 Pa) | (0.000, 0.313) |
| Plate with circular hole | PINN (DeepXDE, configured) | -4.881 | 71.32% | 88.16% (4.95e+06 Pa) | (0.000, 0.313) |
| Plate with circular hole | ATLAS-XPINN | -0.04641 | 5.39% | 53.60% (3.01e+06 Pa) | (0.000, 0.313) |
| L-shaped bracket | PINN (DeepXDE, library defaults) | -102.8 | 114.16% | 99.03% (2.63e+07 Pa) | (1.000, 1.000) |
| L-shaped bracket | PINN (DeepXDE, configured) | -94.13 | 100.72% | 99.44% (2.64e+07 Pa) | (1.000, 1.000) |
| L-shaped bracket | ATLAS-XPINN | +0.3642 | 1.16% | 63.71% (1.69e+07 Pa) | (1.000, 1.000) |

Solver definitions

- **PINN (DeepXDE library)**: an independent baseline built with the open-source DeepXDE library (Lu et al., SIAM Review 2021): one tanh network, soft boundary conditions, library-default loss weights, Adam then L-BFGS; nothing from the ATLAS solver. Five cantilever settings (loss weights 1-100, float32/float64) were tried; none got below 88 % displacement error, so the library defaults are reported.
- **PINN (DeepXDE, configured)**: the same library with settings fixed in advance from standard DeepXDE practice and never chosen by the FEM error: the essential BCs imposed exactly through an output transform, float64, extra collocation points near each benchmark's stress concentration, and loss weights balanced so every term starts at 1 (from the initial training losses only).
- **ATLAS-XPINN**: the same energy formulation and hard boundary conditions, domain-decomposed: one subnetwork per subdomain with interface displacement and traction continuity (augmented-Lagrangian coupling, single-network warm start; docs/BUGFIXES.md section 16).
