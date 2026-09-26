# ATLAS vs standard PINN vs FEM

Errors are relative L2 against the FEniCS reference in `fem/reference/` (displacement magnitude, and von Mises with the singular region excluded, the same metrics `atlas.validation` reports). Targets: displacement < 5 %, von Mises < 10 %. No solver sees FEM data in training.

Compliance C = int t.u ds over the loaded edge, computed from each solver's displacement at the reference nodes; its error is relative to the FEM value (spec target: within 10 %).

Two displacement errors are shown. *Magnitude* is the metric `atlas.validation` reports (compares sqrt(u^2+v^2)); *vector* compares the (u, v) field itself and therefore catches a sign-flipped solution that the magnitude metric hides.

| benchmark | solver | displacement L2 (magnitude) | displacement L2 (vector) | von Mises L2 (excl.) | compliance [N m] | compliance vs FEM | parameters | train time |
|---|---|---|---|---|---|---|---|---|
| Cantilever beam | Standard PINN (in-house) | 9.51% (fail) | 191.03% (fail) | 20.91% (fail) | -1112 (FEM 1272) | -187.45% (fail) | 25,573 | 16.8 min |
| Cantilever beam | PINN (DeepXDE library) | 87.79% (fail) | 88.09% (fail) | 89.92% (fail) | 132 (FEM 1272) | -89.63% (fail) | 12,802 | 11.6 min |
| Cantilever beam | ATLAS (energy, single net) | 0.19% (PASS) | 0.19% (PASS) | 2.29% (PASS) | 1270 (FEM 1272) | -0.17% (PASS) | 12,802 | - |
| Cantilever beam | ATLAS-XPINN (energy) | 0.62% (PASS) | 0.64% (PASS) | 2.97% (PASS) | 1282 (FEM 1272) | +0.77% (PASS) | 25,604 | 5.9 min |
| Plate with circular hole | Standard PINN (in-house) | 40.58% (fail) | 47.12% (fail) | 30.89% (fail) | 15.81 (FEM 20.14) | -21.51% (fail) | 37,853 | 20.6 min |
| Plate with circular hole | PINN (DeepXDE library) | 41.85% (fail) | 51.42% (fail) | 31.66% (fail) | 15.54 (FEM 20.14) | -22.83% (fail) | 12,802 | 11.1 min |
| Plate with circular hole | ATLAS (energy, single net) | 0.19% (PASS) | 0.21% (PASS) | 2.58% (PASS) | 20.12 (FEM 20.14) | -0.06% (PASS) | 12,802 | - |
| Plate with circular hole | ATLAS-XPINN (energy) | 0.87% (PASS) | 0.95% (PASS) | 3.52% (PASS) | 20.24 (FEM 20.14) | +0.54% (PASS) | 38,406 | 8.5 min |
| L-shaped bracket | Standard PINN (in-house) | 51.73% (fail) | 56.63% (fail) | 70.82% (fail) | 29.08 (FEM 96.52) | -69.87% (fail) | 37,853 | 25.8 min |
| L-shaped bracket | PINN (DeepXDE library) | 84.15% (fail) | 115.18% (fail) | 81.97% (fail) | -6.296 (FEM 96.52) | -106.52% (fail) | 12,802 | 12.4 min |
| L-shaped bracket | ATLAS (energy, single net) | 0.19% (PASS) | 0.27% (PASS) | 4.20% (PASS) | 96.93 (FEM 96.52) | +0.42% (PASS) | 12,826 | - |
| L-shaped bracket | ATLAS-XPINN (energy) | 1.31% (PASS) | 1.44% (PASS) | 5.30% (PASS) | 98.67 (FEM 96.52) | +2.22% (PASS) | 38,478 | 32.0 min |

Compliance difference and pointwise maxima

Compliance difference is C - C_FEM. Maximum pointwise errors are the single worst node over the whole reference mesh, as a percentage of the FEM maximum of that field. The von Mises maximum usually sits at a stress concentration (hole edge, re-entrant corner, clamp corner), where the FEM reference itself is least reliable, so treat it as a worst-case indicator, not a pass/fail metric.

| benchmark | solver | compliance difference [N m] | max pointwise displacement error | max pointwise von Mises error | at (x, y) |
|---|---|---|---|---|---|
| Cantilever beam | Standard PINN (in-house) | -2385 | 187.21% | 42.41% (1.79e+07 Pa) | (0.000, 1.000) |
| Cantilever beam | PINN (DeepXDE library) | -1140 | 90.09% | 97.24% (4.11e+07 Pa) | (0.000, 1.000) |
| Cantilever beam | ATLAS (energy, single net) | -2.212 | 0.33% | 35.85% (1.52e+07 Pa) | (0.000, 1.000) |
| Cantilever beam | ATLAS-XPINN (energy) | +9.85 | 0.80% | 45.54% (1.92e+07 Pa) | (0.000, 1.000) |
| Plate with circular hole | Standard PINN (in-house) | -4.331 | 60.73% | 88.32% (4.96e+06 Pa) | (0.000, 0.313) |
| Plate with circular hole | PINN (DeepXDE library) | -4.597 | 65.99% | 89.09% (5e+06 Pa) | (0.000, 0.313) |
| Plate with circular hole | ATLAS (energy, single net) | -0.01232 | 5.54% | 50.60% (2.84e+06 Pa) | (0.007, 0.300) |
| Plate with circular hole | ATLAS-XPINN (energy) | +0.1078 | 5.19% | 51.37% (2.88e+06 Pa) | (0.000, 0.313) |
| L-shaped bracket | Standard PINN (in-house) | -67.44 | 61.72% | 99.69% (2.65e+07 Pa) | (1.000, 1.000) |
| L-shaped bracket | PINN (DeepXDE library) | -102.8 | 114.16% | 99.03% (2.63e+07 Pa) | (1.000, 1.000) |
| L-shaped bracket | ATLAS (energy, single net) | +0.4044 | 1.14% | 89.02% (2.36e+07 Pa) | (1.000, 1.000) |
| L-shaped bracket | ATLAS-XPINN (energy) | +2.144 | 1.55% | 84.93% (2.26e+07 Pa) | (1.000, 1.000) |

Solver definitions

- **Standard PINN**: one network over the whole domain, strong-form residual loss (same terms, scaling and load-amplitude constraints as ATLAS), no domain decomposition and no accuracy aids; parameter count matched to ATLAS-XPINN.
- **PINN (DeepXDE library)**: an independent baseline built with the open-source DeepXDE library (Lu et al., SIAM Review 2021): one tanh network, soft boundary conditions, library-default loss weights, Adam then L-BFGS; nothing from the ATLAS solver. Five cantilever settings (loss weights 1-100, float32/float64) were tried; none got below 88 % displacement error, so the library defaults are reported.
- **ATLAS (energy, single net)**: a deep-energy (Ritz) formulation with exact hard boundary conditions and ONE network (no domain decomposition).
- **ATLAS-XPINN (energy)**: the same energy formulation and hard boundary conditions, domain-decomposed: one subnetwork per subdomain with interface displacement and traction continuity.
