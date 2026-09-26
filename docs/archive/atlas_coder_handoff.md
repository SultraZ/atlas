# ATLAS handoff for Claude Code

Purpose of this doc: everything a coder picking this up in Claude Code needs — what ATLAS is, exactly where each benchmark stands, the prioritized technical work to close the remaining error gap, and a cleanup pass on what this debugging session left behind. Companion doc: `atlas_error_reduction_findings.md` (same project) has the full diagnostic detail behind each finding below — read that for the "why," this doc is the "what to do."

## What ATLAS is

XPINN (Extended Physics-Informed Neural Network) forward solver for structural mechanics, with domain decomposition across subdomain networks per benchmark. PyTorch, CPU-only in this environment. Three benchmarks, each validated against real FEM (FEniCS) reference data:

- **cantilever** — beam, tip shear load, clamped at one end
- **l_bracket** — L-shaped bracket with a re-entrant corner (stress concentration)
- **plate_hole** — square plate with a circular hole (stress concentration)

Targets: **<5% displacement relative-L2 error**, **<10% von Mises relative-L2 error** (excluded-singularity region), against real FEM data.

## Current status (none meet target yet)

| benchmark | displacement L2 | von Mises L2 | vs. target |
|---|---|---|---|
| cantilever | 7.17% | 40.65% | disp close, VM 4x over |
| plate_hole | 21.89% | 19.78% | both ~2x over |
| l_bracket | 50.47% | 57.24% | both ~5-6x over |

All three numbers above are from this session's best reproducible runs (seed=42, see "how to reproduce" below).

## What was fixed this session (already done, don't redo)

1. **Loss double-counting bug** (`atlas/loss_functions.py`) — resultant/anchor loss terms were being added twice into the total. Fixed and verified via no-op regression against pre-fix baselines.
2. **Cantilever neutral-axis bug** (`atlas/atlas_main.py`, `_eb_prefit_cantilever` and the warm-start anchor block) — the analytic Euler-Bernoulli warm-start used raw `y` instead of `(y - H/2)`, the distance from the neutral axis. This shifted the whole `u`-field by a spurious offset. Fixing it took cantilever from 8.95%/88.89% to **7.17%/40.65%** (disp/VM). Confirmed reproducible.
3. **Singular/polar feature basis** (`atlas/networks.py`, `SubdomainNetwork._encode`) — added ρ, 1/ρ, cos(kθ), sin(kθ) (k=1,2) features centered on the known stress-concentration point, computed via a regularized `ρ = sqrt(dx²+dy²+δ²)` and angle-doubling recurrence (no `atan2`, which has a singular gradient at ρ=0 — l_bracket's boundary sampling puts an exact point there and NaN'd with atan2). Applied to plate_hole's near-hole subnet (centered at hole center) and all three l_bracket subnets (centered at the re-entrant corner). Took plate_hole from ~36% to 21.89% disp, l_bracket from ~58% to 50.47% disp.

## Priority next steps to reduce error further

Ranked by expected payoff. Each is a real architecture/loss change, not a hyperparameter sweep — hyperparameter sweeps (longer training, reweighting) were tried and plateaued on all three benchmarks this session, so don't spend time there first.

### 1. Cantilever: near-clamp boundary-layer representation (highest payoff, best understood)

Diagnosed root cause: the left edge is a **true pointwise clamp** (u=v=0 exactly). Near a real clamp, the exact 2D elasticity field has a Saint-Venant boundary layer — displacement/stress genuinely depart from simple beam-bending kinematics within about one beam-height of the support. Confirmed via x-binned error: **>380% local u-error for x∈[0, 0.5)** vs **<1% for x∈[3.5, 4.0)** (beam length 4, height 1). This single region is likely responsible for most of the 40.65% global von Mises error.

Tried and ruled out: more training (no change), lower anchor weight (worse), moving the anchor away from the clamp (no change — confirms this isn't the EB anchor fighting the true BC, it's that nothing currently represents the boundary layer correctly).

**What to try:**
- Denser interior collocation specifically in x < 1.0 (currently `n_interior` interior points are sampled uniformly across the whole beam — a boundary layer of relative width ~H/L = 0.25 of the domain is getting proportionally few points).
- A local feature enrichment analogous to the singular-feature trick, but shaped for a straight-edge boundary layer rather than a point singularity — e.g. a feature that decays with `x` (something like `exp(-x/H)` or low-order polynomials in `x` restricted to a subnet/region near x=0), instead of the polar ρ/θ basis used for the hole/corner.
- Increase `pde` loss weight specifically in that near-wall training pass, or add a second warm-start stage using a plane-elasticity correction term (Timoshenko's classical correction to EB theory near a clamped end has closed-form extra terms in `y³` — worth deriving and adding to the prefit target if this becomes the priority).

### 2. L-bracket: match the singular basis to the actual corner exponent

Current singular features use a generic 2-harmonic ρ/θ basis, the same one that worked well for the plate_hole's circular hole. But the re-entrant corner's stress singularity order depends on the corner's actual interior angle (a 270° re-entrant corner has a different singularity exponent than a circular hole, per the classical Williams corner-singularity solution — the exponent is not 1/√r as it is near a crack tip, it's a specific irrational power determined by the corner angle). Using a generic basis instead of one matched to that exponent is likely why l_bracket's improvement (58%→50%) was smaller than plate_hole's (36%→22%) despite the same technique.

**What to try:**
- Derive or look up the Williams eigenvalue for a 270° corner and add `ρ^λ` (and `ρ^λ * cos/sin(λθ)`) terms with that specific λ to the singular basis, rather than generic integer harmonics.
- Denser near-corner collocation (same idea as the cantilever boundary layer — check whether the l_bracket sampler is under-sampling right at the corner).
- Consider more harmonics only after the exponent-matched term is in — the earlier attempt to just widen the generic-harmonic network regressed (28% vs 22% on plate_hole), so "more capacity" isn't the lever; "correct basis" is.

### 3. Plate with hole: closest to target, likely needs refinement not redesign

Already at 21.89%/19.78%, both roughly 2x over target. A wider/more-harmonic version regressed (28%/23%), and a Kirsch-analytic warm start didn't help (24%/21%, attributed to infinite-plate solution not matching the finite plate). This suggests the architecture is close to right and what's missing is refinement:
- Denser near-hole sampling without widening the network.
- One additional harmonic (k=3) *combined with* denser sampling, rather than widening neurons/layers (the earlier regression widened everything at once, confounding the harmonic increase with the capacity increase — worth isolating).
- Longer L-BFGS budget specifically (the Adam stage may be undertrained relative to L-BFGS refinement — check the loss curve shape).

## Cosmetic / cleanup items

None of these affect correctness — they're leftover from an interactive debugging session and worth tidying before this goes into a real repo/PR.

- **Debug/scratch scripts at the repo root** (`run_cantilever*.py`, `run_plate_hole_*.py`, `run_lbracket_*.py`, `diagnose_*.py`, and their `.log` files) — about 25 one-off scripts and logs from this session's experiments. Worth consolidating into a single parameterized `scripts/run_benchmark.py --benchmark X --config Y` plus deleting the dead ends (MPE, Kirsch-warm-start, widened-network, low-anchor — all negative results, already captured in the findings doc, don't need to live on as code).
- **Model checkpoint proliferation** — 9 `.pt` files at the repo root (`cantilever_model.pt`, `cantilever_model_long.pt`, `cantilever_model_lowanchor.pt`, `cantilever_model_smoke.pt`, `lbracket_singular_model.pt`, `lbracket_singular_long_model.pt`, `plate_hole_*` x4). Only one per benchmark is the "current best" (`cantilever_model.pt`, `lbracket_singular_long_model.pt`, `plate_hole_singular_long_model.pt`) — the rest are intermediate/negative-result artifacts safe to delete.
- **`cantilever_anchor_weight` parameter** added to `train_forward_solver` during debugging (default 3000.0, matches prior hardcoded behavior) — harmless and worth keeping as a knob, but the docstring/signature should get a one-line comment explaining what it's for (currently only explained in an inline comment near its use, not at the parameter declaration).
- **`atlas/test_smoke.py` has a pre-existing broken import**: `__import__("run_repeatability_study")` (line 62) fails with `ModuleNotFoundError` because it's a bare import instead of `atlas.run_repeatability_study`. Unrelated to anything fixed this session but it's a real test failure sitting in the suite — one-line fix.
- **`atlas/loss_functions.py` line ~645**: `UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior` from `float(res_val)` in the L-BFGS print block. Harmless (it's just for a print-frequency check) but easy to silence with `.detach()` before `float(...)`.
- **`atlas/validation.py` geometry-metadata warning** fires on every validation run against the FEM references used this session ("no geometry metadata found alongside this FEM reference... cannot verify it matches the current run's geometry"). The reference `.npy` files used this session are the ones the user uploaded directly rather than ones generated by `fenics_scripts/*.py`, so this warning is expected, not a bug — but it's worth either (a) regenerating the references through the provided FEniCS scripts so they carry geometry metadata and the warning goes away, or (b) at minimum documenting in the repo that this is expected for hand-supplied reference data.
- **Reverted negative-result code left as comments**: the plate_hole widened-network attempt and the l_bracket boundary-layer-anchor-exclusion attempt are both documented as inline comments in `atlas_main.py` rather than as dead code paths — intentional (so the next person doesn't retry the same dead end), but worth double-checking these comments read as accurate history and not confusing to a future editor.

## How to reproduce the current best numbers

All runs use `SEED = 42` (random/numpy/torch) for determinism. CPU-only; each full run takes roughly 10-20 minutes depending on epoch/L-BFGS-step count.

```python
# cantilever — 7.17% disp / 40.65% VM
train_forward_solver(benchmark='cantilever', device='cpu', epochs=8000,
                      use_lbfgs=True, lbfgs_steps=60, lbfgs_max_iter_per_step=20,
                      warm_start_eb=True, prefit_epochs=8000)

# plate_hole — 21.89% disp / 19.78% VM
train_forward_solver(benchmark='plate_hole', device='cpu', epochs=8000,
                      use_lbfgs=True, lbfgs_steps=60, lbfgs_max_iter_per_step=20,
                      use_singular_features=True)

# l_bracket — 50.47% disp / 57.24% VM
train_forward_solver(benchmark='l_bracket', device='cpu', epochs=8000,
                      use_lbfgs=True, lbfgs_steps=60, lbfgs_max_iter_per_step=20,
                      use_singular_features=True)
```

FEM reference data goes in `fem/reference/` as `{benchmark}_{coords,u,v,vonmises}.npy`. Validate with `atlas.validation.run_validation(model, sampler, physics_real, benchmark=..., device='cpu', use_mock=True)`.
