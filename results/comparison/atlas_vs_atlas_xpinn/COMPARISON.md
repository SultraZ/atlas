# ATLAS (energy, single net) vs ATLAS-XPINN vs FEM

Errors are relative L2 against the FEniCS reference in `fem/reference/` (displacement magnitude, and von Mises with the singular region excluded, the same metrics `atlas.validation` reports). Targets: displacement < 5 %, von Mises < 10 %. No solver sees FEM data in training.

Compliance C = int t.u ds over the loaded edge, computed from each solver's displacement at the reference nodes; its error is relative to the FEM value (spec target: within 10 %).

Two displacement errors are shown. *Magnitude* is the metric `atlas.validation` reports (compares sqrt(u^2+v^2)); *vector* compares the (u, v) field itself and therefore catches a sign-flipped solution that the magnitude metric hides.

| benchmark | solver | displacement L2 (magnitude) | displacement L2 (vector) | von Mises L2 (excl.) | compliance [N m] | compliance vs FEM | parameters | train time |
|---|---|---|---|---|---|---|---|---|
| Cantilever beam | ATLAS (energy, single net) | 0.19% (PASS) | 0.19% (PASS) | 2.29% (PASS) | 1270 (FEM 1272) | -0.17% (PASS) | 12,802 | - |
| Cantilever beam | ATLAS-XPINN | 0.27% (PASS) | 0.27% (PASS) | 2.10% (PASS) | 1269 (FEM 1272) | -0.22% (PASS) | 25,604 | 5.0 min |
| Plate with circular hole | ATLAS (energy, single net) | 0.19% (PASS) | 0.21% (PASS) | 2.58% (PASS) | 20.12 (FEM 20.14) | -0.06% (PASS) | 12,802 | - |
| Plate with circular hole | ATLAS-XPINN | 0.75% (PASS) | 0.81% (PASS) | 3.39% (PASS) | 20.09 (FEM 20.14) | -0.23% (PASS) | 38,406 | 7.3 min |
| L-shaped bracket | ATLAS (energy, single net) | 0.19% (PASS) | 0.27% (PASS) | 4.20% (PASS) | 96.93 (FEM 96.52) | +0.42% (PASS) | 12,826 | - |
| L-shaped bracket | ATLAS-XPINN | 0.22% (PASS) | 0.30% (PASS) | 4.55% (PASS) | 96.89 (FEM 96.52) | +0.38% (PASS) | 38,478 | 14.5 min |

Compliance difference and pointwise maxima

Compliance difference is C - C_FEM. Maximum pointwise errors are the single worst node over the whole reference mesh, as a percentage of the FEM maximum of that field. The von Mises maximum usually sits at a stress concentration (hole edge, re-entrant corner, clamp corner), where the FEM reference itself is least reliable, so treat it as a worst-case indicator, not a pass/fail metric.

| benchmark | solver | compliance difference [N m] | max pointwise displacement error | max pointwise von Mises error | at (x, y) |
|---|---|---|---|---|---|
| Cantilever beam | ATLAS (energy, single net) | -2.212 | 0.33% | 35.85% (1.52e+07 Pa) | (0.000, 1.000) |
| Cantilever beam | ATLAS-XPINN | -2.814 | 0.38% | 43.26% (1.83e+07 Pa) | (0.000, 1.000) |
| Plate with circular hole | ATLAS (energy, single net) | -0.01232 | 5.54% | 50.60% (2.84e+06 Pa) | (0.007, 0.300) |
| Plate with circular hole | ATLAS-XPINN | -0.04641 | 5.39% | 53.60% (3.01e+06 Pa) | (0.000, 0.313) |
| L-shaped bracket | ATLAS (energy, single net) | +0.4044 | 1.14% | 89.02% (2.36e+07 Pa) | (1.000, 1.000) |
| L-shaped bracket | ATLAS-XPINN | +0.3642 | 1.16% | 63.71% (1.69e+07 Pa) | (1.000, 1.000) |

Solver definitions

- **ATLAS (energy, single net)**: a deep-energy (Ritz) formulation with exact hard boundary conditions and ONE network (no domain decomposition).
- **ATLAS-XPINN**: the same energy formulation and hard boundary conditions, domain-decomposed: one subnetwork per subdomain with interface displacement and traction continuity (augmented-Lagrangian coupling, single-network warm start; docs/BUGFIXES.md section 16).
