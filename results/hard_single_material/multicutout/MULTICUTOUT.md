# Multi-cutout bracket (fourth benchmark): ATLAS energy vs ATLAS-XPINN

Geometry: L-bracket (arms 0.4 m wide, legs 1 m), holes A (0.20, 0.82) r 0.065, B (0.20, 0.53) r 0.045,
C (0.80, 0.20) r 0.070, rounded slot around (0.21,0.30)-(0.21,0.17)-(0.37,0.17), half-width 0.04.
Clamped at x = 1, uniform tension 1 MPa on y = 1 (loading corrected from the spec: with its bottom
clamp, holes A and B were almost unloaded). FEM reference: P2, 83,835 cells, compliance converged to
0.001 %, cutout peak stresses to 0.05-2.4 % between the last two mesh levels. The FEM result is used only
for scoring.

ATLAS-XPINN: 3 subdomains (upper arm y >= 0.68 / corner region / horizontal arm x >= 0.6), interfaces
through plain material away from every cutout and the re-entrant corner, augmented-Lagrangian
coupling + warm start. Both solvers use the same 72,612 integration points (6-point rule on a
triangle mesh of each subdomain; geometry only).

Energy-check guard (no FEM data, same rule for both): the strain energy is re-evaluated on an
independent 7-point rule; training stops when the two differ by > 0.3 % and the last weights within
0.15 % are restored. Needed because with too few integration points the network lowers the discrete
energy by bending between them (first attempt with a 3-point rule diverged; archived in
results/archive_phases/multicutout_quad3pt/).

## Seed 42, 2000 Adam + 8000 L-BFGS (same budget for every solver)

| solver | params | displacement L2 | von Mises L2 (excl.) | von Mises near cutouts | compliance vs FEM | train time |
|---|---|---|---|---|---|---|
| ATLAS energy (single network) | 12,802 | 0.47% | 6.97% | 23.8% | -0.53% | 3055 s (guard stopped at 9100, restored 8800) |
| ATLAS-XPINN (default recipe) | 38,406 | 9.46% | 18.30% | 45.3% | -7.96% | 3674 s (full budget) |
| ATLAS-XPINN + fix A: L-BFGS memory kept across AL updates | 38,406 | 11.09% | 19.65% | 46.2% | -9.79% | 3639 s (full budget) |
| ATLAS-XPINN + fixes A and B: coarse-to-fine warm start (2000 shared L-BFGS) | 38,406 | 6.21% | 15.04% | 40.0% | -4.95% | 3654 s (full budget) |
| ATLAS-XPINN + fix B + attempt C: per-subdomain input normalisation | 38,406 | 10.76% | 19.65% | 42.9% | -9.39% | 3640 s (full budget) |
| PINN (DeepXDE library, defaults) | 12,802 | 98.51% | 96.74% | 87.6% | - | 1041 s (full budget) |
| PINN (DeepXDE, configured: exact clamp, float64, extra points, balanced weights) | 12,802 | 100.61% | 98.02% | 90.4% | - | 5123 s (full budget) |

Displacement error by subdomain (ATLAS-XPINN, default recipe): upper arm 10.6 %, corner region 8.4 %,
horizontal arm 4.9 %; single network 0.6 / 0.3 / 0.5 %. The interfaces are continuous (jump^2 ~1e-16),
so this is not an interface leak: ATLAS-XPINN reaches a continuous but too-stiff field and converges
much more slowly. The negative compliance error says the same thing - it stops short of the energy
minimum, which the single network reaches.

Two convergence fixes were tried (both inside the paper's XPINN definition):
  * **fix A** - run ONE L-BFGS through all augmented-Lagrangian stages instead of restarting it at every
    multiplier update (`train_decomposed(al_keep_memory=True)`): no help (11.1 % vs 9.5 %).
  * **fix B** - the shared warm-start network also takes 2000 L-BFGS iterations (a coarse global solve)
    before it is copied into the subnetworks, out of the same total budget
    (`warm_start_lbfgs=2000`): a clear gain, 9.46 % -> 6.21 %, but still ~13x the single network's error.
  * **attempt C** - fix B plus per-subdomain input normalisation, so each subnetwork maps its own region
    to [-1, 1]^2 (`--local-norm`): worse again (10.76 %), as it was on the L-bracket.

The two PINN baselines (open-source DeepXDE, library defaults and a configured run whose settings were
fixed in advance from standard practice and never from the FEM error) both fail completely here
(98-101 % displacement error), as they do on the other benchmarks.

## Conclusion (one seed)

On this benchmark the single-network energy solver wins clearly and meets both targets (5 % / 10 %);
ATLAS-XPINN does not, at any of the three recipes tried. The 5-seed test was therefore not triggered.
Across the five benchmarks run so far (cantilever, plate with hole, L-bracket, notched plate,
multi-cutout bracket) domain decomposition has never beaten the single network with this solver; on the
three easy ones it comes close, on the two hard ones it is clearly behind, and it always costs more
training time. What ATLAS-XPINN does show is that it passes the targets where standard PINNs fail
badly - on the three easy benchmarks - and that its weakness here is optimisation speed, not the
representation of local features.
