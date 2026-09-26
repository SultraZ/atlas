# What was actually wrong, and what I changed

## 1. Inverse design never moved the radius (sampling.py, inverse_design.py)

Two compounding bugs meant `sampler.r` was, in practice, frozen:

- `compute_compliance_for_loaded_boundary()` evaluated compliance on the
  **right edge (x=W)** using the far-field network. Those points never
  depend on the hole radius `r` at all, so `compliance.backward()` never
  reached `r` — `sampler.r.grad` stayed `None` every single outer
  iteration, `if self.sampler.r.grad is not None` was always false, and
  `geom_optimizer.step()` never ran. The radius in your logs would just sit
  at `initial_radius` forever.
- Even if you fixed that, the inner training loop
  (`forward_solve_step`, called up to `n_inner_iterations` times per outer
  iteration) called `.backward()` repeatedly on loss terms built from
  `sample_interior()`'s subdomain-0 points and `sample_hole_boundary()`,
  both of which *were* differentiable w.r.t. `r` (`rho0 = sqrt(r**2 + ...)`,
  `x_hole = r*cos(theta)`). Reusing that same graph across more than one
  `.backward()` call without `retain_graph=True` raises `RuntimeError:
  Trying to backward through the graph a second time` — this would have
  crashed on the second inner iteration.

**Fix:** `sample_interior()`'s subdomain-0 points and `sample_hole_boundary()`
now `.detach()` before being used in the forward-solve inner loop (numerically
identical, but no live graph back to `r`, so no crash and no wasted/ignored
gradient accumulation). A new `sample_hole_boundary_differentiable()` keeps
the live graph and is used *only* by the outer loop. `InverseDesignOptimizer`
now computes a `hole_region_sensitivity_objective()` — elastic strain-energy
density on a ring just outside the hole, evaluated with the frozen near-hole
network — as the thing it differentiates w.r.t. `r`. Read the docstring on
that method: it's a pragmatic proxy that makes the optimizer loop actually
run and move `r` in a sensible direction, **not** a rigorously derived
continuum shape derivative of structural compliance. If this needs to be
paper-accurate, that's a real derivation (Hadamard/Céa adjoint method), not a
one-line fix.

## 2. `r_mid`/`r_far` were fixed constants incompatible with your own radius bounds (sampling.py)

`PlateWithHoleSampler` hardcoded `r_mid = 0.6`, `r_far = 1.2`, but
`InverseDesignConfig.max_radius` defaults to `0.8`. The moment the
optimizer pushed `r` above `0.6`:

- `rho0 = sqrt(r**2 + xi0*(r_mid**2 - r**2))` takes `sqrt` of a **negative**
  number (`r_mid**2 - r**2 < 0`) → `NaN` propagating through the whole
  near-hole subdomain, silently poisoning training.
- The far-field rejection-sampling loop in `sample_interior()`
  (`while len(x2_list) < n: ...`) could also degrade badly or spin if
  `r_far` got close to the domain diagonal.

**Fix:** `r_mid`/`r_far` are now recomputed every call from the *current*
`r` plus fixed margins (`r + 0.3`, `r + 0.6`), via `_update_subdomain_radii()`,
called at the top of `sample_interior/_boundary/_interface`. Safe across the
whole configured `[0.1, 0.8]` range for `W=H=2.0`. `visualization.py` and
`validation.py` read `sampler.r_mid`/`r_far` as plain attributes, so they
automatically see the up-to-date values.

## 3. `ReduceLROnPlateau(..., verbose=True)` (loss_functions.py)

`verbose` was removed from `ReduceLROnPlateau` in newer PyTorch (2.4+) and
raises `TypeError` there. Removed the kwarg; `train()` now prints LR-drop
messages itself by comparing `param_groups[0]['lr']` before/after
`scheduler.step()`.

## 4. Matplotlib figures were never closed (visualization.py)

Every plotting function called `plt.savefig()` / `plt.show()` but never
`plt.close(fig)`. Fine for a single CLI run; leaks memory (and will
eventually OOM) if you call these in a loop — which the web UI does, once
per job. Added `plt.close(fig)` after each save/show.

## 5. `forward_solve_step` crashed on the very first call — `XPINNLoss` had no `compute_total_loss` (loss_functions.py, inverse_design.py)

Found by actually running the fixed inverse-design loop (see #1) rather
than by static review — `InverseDesignOptimizer.forward_solve_step()` calls
`loss_fn.compute_total_loss(...)`, but `loss_fn` there is an `XPINNLoss`
instance (that's what `atlas_main.run_inverse_design_on_plate()` constructs
and passes in), and `compute_total_loss` only ever existed on
`XPINNTrainer`. This raised `AttributeError: 'XPINNLoss' object has no
attribute 'compute_total_loss'` on the very first inner iteration of
`optimize()` — inverse design could never have run at all, independent of
the gradient-path bug in #1.

**Fix:** moved `compute_total_loss` onto `XPINNLoss` itself (now takes
`model` as an explicit argument instead of reading `self.model`), since
that's the object both `XPINNTrainer` and `InverseDesignOptimizer` actually
hold. `XPINNTrainer.compute_total_loss()` now just delegates to
`self.loss_fn.compute_total_loss(self.model, ...)` so there's one
implementation instead of two that can silently drift apart. Verified with
a live torch install that the full outer/inner loop now runs and
`sampler.r` moves with a real, non-`None` gradient each step.

## 6. Two-stage Adam → L-BFGS training (loss_functions.py, visualization.py, atlas_main.py)

The paper's methodology (Section VI.B) states: *"The optimizer for training
is Adam, and for fine-tuning, L-BFGS."* The code only ever had Adam.
`XPINNTrainer.train()` now runs Adam for `epochs` steps as before, then
(if `use_lbfgs=True`, the default) runs a second stage with
`torch.optim.LBFGS` via `train_lbfgs()`. Loss-component history from both
stages lands in the same `trainer.history` dict, and
`trainer.adam_epoch_count` records where the handoff happened so
`plot_training_convergence(..., adam_epoch_count=trainer.adam_epoch_count)`
can draw a marker on the convergence plot showing exactly where L-BFGS took
over. Gradient clipping (used during the Adam stage to tame occasional
large PDE-residual gradients) is deliberately *not* applied during L-BFGS —
see the docstring on `train_lbfgs()` for why: it would corrupt the
curvature history L-BFGS is building without adding real stability, since
L-BFGS already has its own safeguard (strong-Wolfe line search). Verified
end-to-end on all three benchmarks with a live torch install.

## 7. Loss numerically frozen due to raw SI units (atlas_main.py, webapp/backend/app.py)

Found by actually watching a training run (the flat convergence plot you
showed), then reproduced and confirmed with a live test: total loss was
**bit-for-bit identical** from epoch 1 to epoch 100 (ratio == 1.0 exactly)
when training with real steel properties (`E=210e9`).

Root cause: the PDE-residual loss term involves squared stress, which with
`E=210e9` lands around `(2e11)^2 ~ 4e22`. Two consequences: (1) this term
completely swamps the O(1) Dirichlet/interface terms in the combined
weighted loss, and (2) `clip_grad_norm_(max_norm=1.0)` - entirely
reasonable for a normal-scale loss - caps each Adam step to a change so
small relative to `1e22` that it's invisible at float32 precision. Training
runs, but produces no visible progress for thousands of epochs. (L-BFGS
isn't gradient-clipped and has its own line search, which is why the loss
visibly drops the instant L-BFGS takes over - a strong, specific tell that
confirmed this diagnosis before touching any code.)

**Fix:** non-dimensionalization, not a workaround. `train_forward_solver()`
and the web app's `_run_job()` now divide `E` and the applied load/traction
by a shared `stress_scale=1e11` before constructing the `physics` object
and boundary conditions used for the *loss* (`physics_train`). This is
mathematically exact, not an approximation: linear elasticity's governing
equations (equilibrium PDE + Neumann BC) are homogeneous of degree 1 in the
pair `(E, load)` - dividing both by the same constant is algebraically
equivalent to dividing the whole equation by that constant, which doesn't
change its solution. The network still learns the exact real-world
displacement field with no correction needed on its output - only
stress-*derived* quantities (von Mises plots, compliance) need real,
unscaled `E` to convert back to physical units, which is why both entry
points now return/use a separate `physics_real` (unscaled) specifically for
validation and visualization, while training internally uses
`physics_train` (scaled). `run_inverse_design_on_plate()` needed the same
treatment for the same reason - it continues training the same network, so
it has to use the identically-scaled physics or reintroduce the same
frozen-loss problem.

Verified with live runs at every level: raw loss computation (100 epochs,
frozen ratio 1.0 -> converging ratio 0.0006), the full `atlas_main.py`
entry point (ratio ~5.6e-6 through Adam+L-BFGS, `physics_real.E` confirmed
still real/unscaled), the inverse design path (real non-zero gradients,
radius actually moving, objective in a sane O(1e-3) range instead of
O(1e23)), all three benchmarks end-to-end, and the actual FastAPI web
endpoint via a real HTTP request.

## 8. Deformed-shape plot used a stale hardcoded scale (visualization.py, atlas_main.py, webapp/backend/app.py)

Found by looking at an actual screenshot: `plot_deformed_shape(..., scale=1e6)`
was tuned back when training was frozen (bug #7) and displacements never
moved from near-zero. With that fixed, real displacements land around
`1e-3` for the cantilever - `1e6 * 1e-3 = 1e3`, i.e. the "deformed" mesh
was drawn ~300x the beam's own length (4), producing an exploded,
unrecognizable trumpet shape instead of a legible bent beam.

**Fix:** `plot_deformed_shape()` now defaults `scale=None`, which
auto-computes a scale from the actual predicted displacement magnitude so
the deformed overlay always renders at roughly `target_fraction` (default
20%) of the domain's largest dimension, regardless of how large or small
the true displacements are for a given benchmark/load/material. An explicit
float still overrides this if you want a fixed scale across a side-by-side
comparison of multiple runs. Both `atlas_main.py` and the web app's
`_run_job()` were passing the stale `scale=1e6` explicitly and now just
call it with no scale argument. Verified with a live run: auto-computed
scale landed at 103 for a converged cantilever run, producing a plot that
visibly bends downward under the downward tip load at a legible size -
cross-checked against the classical Euler-Bernoulli tip deflection formula
(`PL^3/3EI = 1.219mm` predicted vs. `~1.20mm` from the trained network, ~2%
agreement) and the expected peak-stress location (fixed wall, extreme
fiber) to confirm the underlying physics is correct, not just the plot.

## 9. No actual FEM-vs-ATLAS comparison plot existed (visualization.py, atlas_main.py, webapp/backend/app.py)

`validation.py` computed relative L2 error correctly, but only ever
*printed* the number - nothing plotted what was actually being compared.
The paper's Visualization Module section explicitly calls for "comparison
of ATLAS and FEM results side by side," which didn't exist as an image
anywhere in the pipeline.

**Added:** `plot_fem_comparison()` in visualization.py - three panels
(ATLAS |displacement|, FEM |displacement|, pointwise |ATLAS-FEM| error),
using unstructured `tricontourf` since FEM node coordinates aren't a
regular grid. Deliberately reuses `FEMValidator`'s own
load/generate/evaluate methods (not a reimplementation) so the plot is
guaranteed to show the exact data the printed L2 number is computed from.
Also reuses the geometry-consistency check from bug-fix #... (the
stale-FEM-reference detector) and puts the warning directly in the plot
title in red if triggered, or if mock data was used - a wrong/fake
comparison should be obvious from a glance at the image itself, not require
reading a separate log line. Wired into both `atlas_main.py` (saves to
`results/{benchmark}_fem_comparison.png` when validation runs) and the web
app (added to `job.result_files` alongside the other plots). Verified with
a live run: correctly renders the true ATLAS displacement field, correctly
flags mock-data comparisons in the title (the mock reference is a
placeholder near-zero field, so its "error" against a real result is
enormous and *should* look alarming - that's the plot working, not a bug).

## 10. Added L-bracket re-entrant-corner fillet inverse design (sampling.py, inverse_design.py, atlas_main.py, webapp/backend/app.py, frontend)

Previously only `plate_hole` had inverse design, because it was the only
sampler with a geometry parameter (`r`) stored as a differentiable
`torch.tensor`. Cantilever's `L`/`H` and L-bracket's `size` were plain
Python floats - nothing for autograd to compute a gradient with respect
to. Your paper's own methodology section names a fillet radius as a second
target design variable, so this adds it for the L-bracket's re-entrant
corner (the same corner already tracked as a stress-concentration site in
the FEM scripts and validation notes).

**Geometry** (worked out and numerically verified before writing any code -
a reentrant-corner fillet is easy to get backwards): the fillet is a small
quarter-disk of NEW material, tangent to both edges meeting at the sharp
corner (s,s), centered at C=(s-fillet, s+fillet). Verified algebraically
and by test: the original sharp corner point stays outside the fillet disk
(distance fillet*sqrt(2) > fillet), while material just past the tangent
points falls inside it - i.e. the fillet rounds the point without eating
into sub0/sub1/sub2's existing material. The quarter-disk was assigned to
subdomain 0 (an arbitrary but consistent choice - it's tangent to both
sub0 and sub2 equally). Rendered and visually confirmed the geometry looks
like an actual rounded corner, not something malformed.

**Opt-in by design, specifically so no FEM reference needs regenerating**:
`LBracketSampler(fillet_init=None)` (the default) produces byte-identical
sampling to before this feature existed - verified live (identical point
counts, identical boundary segment count, fillet math never touches the
sampling path at all when disabled). Only setting `fillet_init` to a
number activates the extra quarter-disk interior points and the new
traction-free arc boundary condition. `train_forward_solver()`'s new
`l_bracket_fillet_init` parameter defaults to `None` for the same reason;
the CLI (`atlas_main.py`'s `main()`) only sets it when you actually ask for
inverse design on the L-bracket, so a plain "just train l_bracket" run is
completely unaffected. `sampler_geometry_dict()`'s l_bracket branch only
includes a `'fillet'` key when fillet is actually enabled, so the
geometry-consistency check (bug-fix note above) stays silent for the
common/default case and only starts flagging mismatches against your
existing sharp-corner FEM reference once you deliberately turn the fillet
on - verified both directions live.

**`InverseDesignOptimizer` generalized** rather than duplicated: it now
takes `design_param` (the trainable tensor), `resample_fn`, and
`sensitivity_fn` as arguments instead of hardcoding `sampler.r` and
plate-specific method names, so the exact same class drives both the
plate's hole radius and the L-bracket's fillet.
`fillet_region_sensitivity_objective()` mirrors
`hole_region_sensitivity_objective()`'s approach and carries the same
caveat: it's a pragmatic near-boundary strain-energy proxy, not a
rigorously-derived shape derivative.

Verified with live runs at every level: default (no-fillet) L-bracket
training unaffected (loss ratio ~1.7e-4, same as before this feature
existed), fillet-enabled training + inverse design with real non-zero
gradients and the fillet radius actually moving across outer iterations,
the pre-existing plate_hole inverse design still working correctly after
the `InverseDesignOptimizer` refactor (regression-tested, not just assumed
compatible), and all of the above again through the actual FastAPI web
endpoint via real HTTP requests (4 full end-to-end job runs: cantilever,
l_bracket without inverse design, plate_hole with inverse design,
l_bracket with inverse design).

## 11. Added von Mises stress FEM validation — closes a real paper-vs-code gap (validation.py, visualization.py, fenics_scripts/*.py)

A full re-audit against the paper (not just the code) found this: the
Methodology's Validation Module and the Testing Plan's Metric 2 both
explicitly call for a von Mises stress L2 error AND "maximum pointwise
error at stress concentration locations" - exactly at the L-bracket corner
and plate hole boundary the rest of this project cares most about.
`validation.py` only ever computed displacement (u, v) error; stress
wasn't validated against FEM anywhere.

**Added:**
- `fenics_scripts/*.py` now also export `{benchmark}_vonmises.npy`,
  projected onto the SAME P2 scalar space as u/v so it lines up node-for-
  node with the saved coordinates - no interpolation needed for a direct
  comparison.
- `FEMValidator` gained `evaluate_xpinn_vonmises_on_points()` and
  `compute_max_pointwise_error()`. `validate()` now reports
  `relative_L2_von_mises` and `max_pointwise_von_mises_error` (+ its (x,y)
  location) whenever the loaded reference has a `vonmises` field -
  gracefully degrades to `None` for older references or mock data rather
  than crashing, verified both ways with a live test.
- `plot_fem_comparison()` grows a second row (ATLAS stress / FEM stress /
  pointwise error) automatically when stress data is available, with the
  single worst point marked directly on the plot with an ✕ - a stress-
  concentration failure is exactly what an averaged L2 number can hide, so
  it's now visible at a glance, not just as a printed number.
- Found and fixed a real, previously-unverified bug while re-checking the
  L-bracket routing this required: the fillet-patch region (bug-fix #10
  above) was NEVER actually routed correctly in `visualization.py` - it
  fell through all three subdomain masks and would have silently rendered
  as u=v=0 in any fillet-enabled plot, even though the network had learned
  real values there. Confirmed with a live test: a point inside the fillet
  patch returned exactly `(0.0, 0.0)` before this fix, and real nonzero
  values after. Also refactored the routing logic (previously duplicated
  once between `_evaluate_model_on_grid` and `plot_von_mises_stress`) into
  a single shared `_route_and_forward()` - this bug is exactly the kind of
  thing that duplicated logic invites, so consolidating it removes the
  whole class of "fixed it in one copy, forgot the other" risk going
  forward.
- `run_validation()` now accepts `data_dir` (previously hardcoded to
  `'fem_reference'`, silently ignoring any other path you passed at the
  `FEMValidator` level).

Verified live at every step: stress error computed correctly against a
synthetic reference with `vonmises` data, correctly reports `None` without
crashing against one without it, the 6-panel comparison plot renders
correctly with the max-error point marked, the fillet-patch routing fix
confirmed with a direct point query, and a full three-benchmark + web-app
regression re-run afterward to confirm nothing else broke.

## 12. Added explicit PASS/FAIL against the paper's stated 5% target (validation.py, visualization.py, webapp frontend)

The code computed a relative L2 error number but never actually compared
it to anything - you had to eyeball whether e.g. `4.2e-2` was under the
paper's stated 5% target yourself. `FEMValidator.validate()` and
`plot_fem_comparison()` now both report `displacement_within_5pct_target`
(`True`/`False`/`None`), and the web UI shows it as a headline PASS/FAIL
badge rather than something buried in a JSON blob.

`None` (not a silent `False`) specifically when the comparison isn't
meaningful - mock data, or a real reference that failed the geometry-
consistency check (bug-fix #2/#9 above) - a pass/fail claim against
nonsense data would be actively misleading, worse than no claim. Verified
live: a close-to-correct synthetic reference reports `True`, mock data
reports `None` with the reason ("mock data" vs "geometry mismatch")
stated explicitly, and the mock case is correctly `None` (not `True` or
`False`) all the way through an actual HTTP round-trip via the web API.

No target is claimed for the von Mises stress metric (bug-fix #11 above),
since - checked directly against the paper's text for this - no numeric
target is actually stated there for stress specifically, only for
displacement (5%) and compliance (10%, for the inverse-design comparison
that still doesn't exist - see the open items list). Worth knowing
separately: the paper states these two targets (5% displacement, 5
repeatability runs) without a cited derivation or literature comparison
justifying the specific numbers - that's fine for a proposal, but if a
defense panel asks "why 5%," the honest current answer is "that's the
target the proposal set," not something derived from a formula or a cited
precedent.

## 13. The cantilever/L-bracket amplitude collapse — root cause, warm-start fix, and the honest limits (loss_functions.py, networks.py, atlas_main.py, visualization.py, fenics_scripts/*)

This is the big one. After items #1-#12 the pipeline *ran* end to end, but
the forward solves were quietly wrong: on the cantilever and L-bracket the
network converged to the **trivially-zero displacement field** (or a small
fraction of the true deflection), which scores a deceptively low loss.

### Root cause: the soft pointwise-residual loss is amplitude-degenerate

For a homogeneous, linear-elastic problem the governing PDE is linear in
the displacement field: if `u` is a solution then so is `α·u` for any
scalar `α`, *including* the empty field `α = 0`. A loss built from
**squared pointwise equilibrium residuals** (`||∇·σ||²`) is therefore
quadratic in the field amplitude: shrinking `α → 0` drives the PDE term to
zero. The only terms that can resist are the boundary terms — the
Dirichlet clamp penalty (which the empty field *also* satisfies: `0 = 0`)
and the Neumann traction mismatch (the one term the empty field genuinely
misses). With the interior Dirichlet field dominating and the load term
scaled small, ordinary Adam/L-BFGS descent happily slides into the empty
basin.

This was confirmed rather than assumed:

- A *perfect* Euler-Bernoulli/Timoshenko prefit (tip deflection −1.220 mm,
  regression loss 1.19e−11) followed by physics training **slides back to
  ≈ zero** (tip +0.009 mm) under the canonical weights.
- The EB prefit field scores `pde=2.55, neumann=239, total=2441` under the
  physical loss — i.e. the optimizer *correctly* rejects it (EB kinematics
  force `σ_yy = Eν/(1−ν²)·ε_xx ≠ 0` on the free edges), but every local
  descent path it takes from there lands in a worse (quieter) deception.
- Tested and insufficient on their own: raising the PDE weight (1→20),
  raising the Neumann weight, adding global-equilibrium **resultants** on
  the clamp section (holds the tip for ~5 L-BFGS steps, then slides),
  Adam fine-tuning, dense collocation (6000 pts still collapses; 20000 pts
  OOMs the 4 GB GPU), and a single shared "monolith" network.

### What actually works: warm-started monolith + sparse physics anchor

The shipped fix is `train_forward_solver(..., warm_start_eb=True)` for the
cantilever (the only benchmark with a closed-form approximate field):

1. The two subnetworks are replaced by a single shared network
   (`monolith=True` in `networks.py`; `get_subdomain_network`/
   `forward_subdomain` return `subnets[0]`, and the interface terms become
   identically zero). `visualization._route_and_forward` was updated to
   route through `get_subdomain_network` as well, so validation/mock/real
   FEM comparisons work for a monolith instead of crashing on
   `subnets[1]`.
2. The net is **prefit** to the EB/Timoshenko field (pure regression).
3. Training then runs with (a) the clamp-section **resultants** at weight
   100, (b) a high-weight sparse **displacement anchor** — 300 interior
   points pinned to the EB field at weight 3000 — and (c) `pde=10,
   neumann=10, dirichlet=100, interface=0`. The anchor and the resultants
   pin the load-response *amplitude*; `pde`/`neumann` then refine the shape
   while the optimizer is held at the correct deflection. Result:
   **tip −1.219 mm, stable over thousands of steps** (previously it either
   collapsed to ≈0 or overshot to −4.5 mm).

The new `energy` term (below) and the `anchor_data` hook are both opt-in
and default to `None`, so every non-warm-started path is byte-for-byte
unchanged.

### Total potential energy: principled, but numerically treacherous

Because the empty field is the *global minimiser of the residual norm* but
the *unique minimiser of the potential energy* `Π = ½∫σ:ε dA − ∫t·u dΓ`
(quadratic in the field minus linear in the load), an `energy_loss`
component was implemented in `loss_functions.py` (signed scalar, added
directly to the total rather than squared). On the cantilever it still
stalled in a near-zero configuration (the network's local landscape again,
not the functional), and on the **L-bracket it diverged** — with a
point-mean quadrature the network can drive the linear work term down
faster than the discretised quadratic strain energy grows, so Π is
effectively unbounded and Adam runs away (displacement L2 error ~1e9).
The energy objective is therefore **not enabled** on any shipped path; it
is documented here as a dead end in this discretisation.

### FEM references regenerated (fenics_scripts/*, fem_reference/)

The FEniCS scripts still applied the *old* uniform traction, so the
references disagreed with `sampling.py`'s parabolic load. All three scripts
were checked, and `cantilever_fenics.py` and `l_bracket_fenics.py` now
apply the exact parabolic profiles used by the sampler (`plate_hole`
already used the matching uniform `σ_∞`). The references were regenerated
with FEniCS 2019.1.0 and copied into `atlas_main/fem_reference/`:

- cantilever: 26001 nodes, max |v| = 1.2723e−3 m (2-D tip, vs EB −1.219e−3),
  peak von Mises 4.23e7 Pa
- L-bracket: 29808 nodes, max |u| = 1.443e−4 m, P2 peak von Mises
  2.66e7 Pa

(The FEniCS env needed `pkg-config` on `PATH` — it was in the conda cache
but not installed in the `Atlas` env; install it and run the scripts with
the env's `bin` on `PATH`.)

### Honest validation status against the regenerated references

Warm-started cantilever (2000 Adam epochs): **v error 5.0% L2**, |disp|
6.8%, u 162%, von Mises 118%. The dominant (v) displacement is at the
target; the |disp| slippage is the small secondary `u` field (whose
*relative* L2 is hypersensitive because `u ≪ v`), and the von Mises number
is dominated by the **clamped-corner stress singularity**, which no smooth
network can reproduce (the analytic EB field itself violates free-edge
traction there). The L-BFGS stage makes all of this *worse*, so the
warm-start path should be used Adam-only.

- **cantilever**: converges to v≈5% with the warm start; von Mises and the
  secondary u-component remain above target. Documented limitation.
- **L-bracket**: collapses to the zero field under the standard loss
  (|disp| L2 ≈ 99%, i.e. predictions ≈ 0) and has no closed-form field to
  anchor, so the warm-start trick does not apply. Unresolved.
- **plate with hole**: does *not* collapse (von Mises 30% L2) but does not
  reach 5%: u 35%, v 69%, |disp| 36%. The large `v` error is the current
  suspect and is not yet explained. Unresolved.

Two remaining mathematical levers were considered and are **not** in the
shipped code: (a) a genuine variational/FEM-discretisation energy with
proper quadrature (would fix the energy runaway, but is a substantial
rewrite), and (b) higher-order/adapted architectures for the corner
singularity. Both are method work, not bug fixes.


## 14. Inverse design radius "frozen" — same raw-SI scaling bug, second occurrence (inverse_design.py)

`sensitivity_fn` printed a gradient (`~1e-12`) and the geometry optimizer
ran, yet the hole radius never moved. Root cause is bug-fix #7 one layer
deeper: the network solves in the *scaled* space (`E_train ~ O(1)`,
outputs ~ `u_s ~ 1e-5`), so the strain-energy-density proxy `½σ:ε` is
`~1e-10` and its derivative w.r.t. the radius is `~1e-12` — below the
`lr_geometry = 1e-3` Adam step, so `r` was numerically inert.

Fixed by normalizing the sensitivity objectives to O(1):
`return mean(energy_density) / (detached-mean |energy_density| + 1e-12)`.
Dividing by a detached strictly-positive scale rescales the objective
without changing its descent direction, so `d r` becomes ~ `lr_geometry`
per step and the radius actually moves (verified: 0.3000 → 0.2970 over
three outer iterations, gradient ~ O(1)). Applied to both
`hole_region_sensitivity_objective` and
`fillet_region_sensitivity_objective`.

Caveat is unchanged and still true: this strain-energy proxy is a
*heuristic* descent direction, not a rigorous Hadamard/Céa shape
derivative, and the smaller-radius direction it favors (down toward the
min-radius bound) needs a finite-difference sanity check before being
reported as a paper-quality optimized radius.

## 15. Targets reached: deep-energy (Ritz) forward solver, and an L-bracket evaluation bug (energy_solver.py, atlas_main.py, visualization.py)

**Result** (seed 42, `python scripts/run_benchmark.py --benchmark X`, validated
through the unchanged `validation.py` path against the FEniCS references; no
FEM data is used in training):

| benchmark | displacement L2 (<5%) | von Mises L2, excluded region (<10%) |
|---|---|---|
| cantilever | 0.17% (CPU run: 0.19%) | 2.17% (CPU run: 2.29%) |
| plate_hole | 0.19% | 2.58% |
| l_bracket  | 0.19% | 4.20% |

(Numbers above are the GPU float64 runs. A CPU run of the same seed differs
slightly through floating-point ordering; the files currently in
`results/atlas_energy/forward/cantilever_*` are from the CPU run.)

Five-seed repeatability (`python scripts/run_repeatability_study.py --benchmark X
--formulation energy --epochs 2000`, seeds 42-46, five distinct trainings each;
results in `results/atlas_energy/repeatability/<benchmark>/`). All 15 runs pass both targets:

| benchmark | displacement L2, mean +/- std (range) | von Mises L2 excl., mean +/- std (range) |
|---|---|---|
| cantilever | 0.156% +/- 0.018% (0.130-0.170%) | 2.01% +/- 0.09% (1.94-2.17%) |
| plate_hole | 0.187% +/- 0.004% (0.183-0.193%) | 2.58% +/- 0.01% (2.57-2.59%) |
| l_bracket  | 0.185% +/- 0.001% (0.184-0.187%) | 4.18% +/- 0.02% (4.16-4.20%) |

(Note: `train_forward_solver` re-seeds itself with `seed=42` on entry, so a
caller that sets its own seed must pass `seed=` through; the repeatability
script does.)

**Why the residual PINN plateaued.** Random-point strong-form residuals plus
soft BC/interface penalties leave the loss badly conditioned and the field
amplitude weakly pinned; hyperparameter sweeps could not fix that. Section 13's
"energy dead end" failed only because of Monte-Carlo quadrature with soft BCs.

**What changed (`atlas/energy_solver.py`, `formulation='energy'`, now the
default of `train_forward_solver`; `'residual'` keeps the old path):**
- Minimise the total potential energy `1/2 int sigma:eps - int t.u` (convex,
  first derivatives only, load amplitude pinned).
- Essential BCs are exact: `u = mask(x) * net(x)` (clamp `x/L`, symmetry
  rollers `x`, `y`), so there are no boundary penalty weights.
- Deterministic graded Gauss-Legendre quadrature (refined toward the clamp and
  the re-entrant corner; O-grid polar blocks around the hole), so L-BFGS sees
  a fixed smooth objective. Training is float64 on the GPU; the result is cast
  to float32 for the existing validation/plot code.
- L-bracket: fixed Williams-eigenfunction enrichment (`r^lam cos/sin(k phi)`,
  lam = 0.5445, 0.9085 for the 270 deg corner) added linearly to the network
  output; the angle is gradient-safe at the corner.

**Evaluation bug that inflated every earlier L-bracket number**
(`visualization._route_and_forward`). Nodes on the inner edge `x = s, y > s`
matched none of the three subdomain masks (`x > s` was strict), so they were
scored as `u = v = 0` (82 of 29808 FEM nodes). The same checkpoint scored
6.2% displacement error before the fix and 0.19% after. Fixed with `x >= s`.
Old L-bracket figures in `results/archive_phases` are affected by this.

**Plot fixes.** Field/comparison plots now leave the notch/hole blank
(including the FEM comparison, whose Delaunay triangulation used to bridge
the notch), share colour scales between ATLAS and FEM, and size colorbars to
their axes. The convergence plot shows the energy gap and U/W on an explicit
iteration axis. `plt.show()` is skipped on headless backends.

**Caveats worth knowing**
- The FEM von Mises reference is an L2 projection onto P2, which overshoots
  and oscillates on boundary nodes with steep gradients. On the plate's hole
  edge the reference reads 2.2e6, 0.5e6, 5.6e6, 3.2e6 Pa on consecutive nodes
  while ATLAS decays smoothly from 3.30e6 (Kirsch: ~3.0e6); from ~0.03 off the
  edge the two agree within 1-2%. Part of the ~2.6% plate stress error is
  reference noise, not solver error.
- `SINGULAR_LOCATION_CONFIG['plate_hole']` excludes the point `(r, 0)`, which
  is the minimum-stress point; the stress peak is at `(0, r)`. Left unchanged
  so the reported metric stays comparable.
- Inverse design, plate hole radius: an energy-trained model only carries the
  hole geometry in its quadrature, so `run_inverse_design_on_plate` detects it
  and replaces the residual inner loop with an energy re-solve at the current
  radius (`make_plate_energy_inner_solver`: rebuild the quadrature, warm-start
  L-BFGS; checked U/(W/2) = 0.9998 at the moved radius). Residual-trained
  models keep the original inner loop unchanged. The L-bracket fillet still
  needs the residual formulation, so `l_bracket_fillet_init` falls back to
  `formulation='residual'`. `tests/validate_inverse.py` exercises both plate
  paths.
- The strain-energy-density sensitivity objective (section 14) is a heuristic
  and its sign is NOT physics-robust: in the same 3-iteration check the radius
  moves 0.300 -> 0.303 with the accurate energy field but 0.300 -> 0.297 with
  the residual field. With a frozen network the objective mostly measures how
  the offset ring moves through a decaying field. Do not report an optimised
  radius from it without an independent check (e.g. finite differences of a
  real objective, now cheap because the energy re-solve is ~seconds).
- `train_forward_solver`'s default is now the energy solver, so callers written
  for residual options pin it explicitly (`tests/validate_all_benchmarks.py`,
  `tests/validate_warm.py`, `--formulation` in `run_repeatability_study.py`).
  The web console's forward/inverse jobs use the energy solver with its
  mandatory L-BFGS stage (Adam alone leaves a ~3% energy gap).

## 16. ATLAS-XPINN was worse than the single network — causes and the improved recipe (energy_xpinn.py, energy_xpinn_design.py, energy_design.py, visualization.py)

**Causes found (measured, not assumed).**
1. *Pure-penalty interface coupling is inconsistent.* It lets neighbouring subnetworks drift apart like a
   soft spring: relative interface displacement jumps of 0.5–3 % on the saved models, every decomposed
   compliance too high (+0.8 / +0.5 / +2.2 %), and on the notched plate the decomposed energy fell below the
   single-network optimum. This matched the size of the extra displacement error (0.6–1.4 %).
2. *Slow start.* Each subnetwork starts from random weights and the interfaces have to be negotiated from
   scratch, which cost the decomposed solver many iterations (notched plate: 50 % plateau for ~5000 iterations).
3. *(notched plate only)* the root-patch layout left the slot — the near-jump in displacement that drives the
   plate's bending — inside the global subnetwork, so decomposing did not help with the hard part.

**Fixes (all within the paper's XPINN definition: 2–6 non-overlapping subdomains, one tanh network each,
displacement + traction continuity at interfaces, Adam then L-BFGS).**
- Augmented Lagrangian (method of multipliers) for displacement continuity: `DecomposedEnergyProblem.update_multipliers`,
  `train_decomposed(al_stages=3)` — the L-BFGS budget is split into 4 parts with a multiplier update between them.
  Enforces continuity exactly with the same moderate penalty (no ill-conditioning).
- Warm start: the first 1000 of the 2000 Adam steps train one network on the whole domain; it is copied into
  every subnetwork (`warm_start=1000`). Same total budget.
- Both are the new defaults (also in inverse design: `XPINNDesign.fit` / `resolve`). `al_stages=0, warm_start=0`
  reproduces the old solver. Old results: `results/archive_phases/atlas_xpinn_energy_penalty_v1/`.

**Tried and rejected** (L-bracket, seed 42, `results/comparison/xpinn_fix_study/`): per-subdomain input
normalisation (0.45 % vs 0.31 % with AL alone), a 4th subdomain around the re-entrant corner (0.52 %, 2.4x slower),
traction weight 1.0 instead of 0.1 (no change).

**Results, 5 seeds (displacement / von Mises excl.), same 2000 Adam + 2000 L-BFGS budget:**

| benchmark | ATLAS energy (single) | ATLAS-XPINN old | ATLAS-XPINN new |
|---|---|---|---|
| cantilever | 0.156 % / 2.01 % | 0.630 % / 3.04 % | 0.253 % / 2.14 % |
| plate_hole | 0.187 % / 2.58 % | 0.935 % / 3.44 % | 0.405 % / 2.89 % (seeds spread 0.21–0.75 %) |
| l_bracket | 0.185 % / 4.18 % | 1.305 % / 5.29 % | 0.254 % / 4.54 % |

Training is also slightly faster (seed 42: 312 / 455 / 875 s vs 336 / 495 / 974 s). Inverse designs (FEM-checked):
cantilever +0.03 % from target, plate −0.05 %, L-bracket −3.73 % (was +5.19 %; ATLAS peak stress vs FEM at the
design −1.5 %, was −5.3 %). The L-bracket fillet radius differs +11 % from the FEM reference because peak stress
is nearly flat in the fillet radius there.

**Honest summary.** The improved ATLAS-XPINN is 2–5x more accurate than before and now close to the single
network, but the single network is still slightly more accurate on all three benchmarks. On the notched plate
the warm start *hurts* (split layout: 0.89 % without it, 14.6 % with it — the copied network is still on the
"slot closed" plateau); the best decomposed result there (split layout + AL) is 0.89 % vs 0.08 % single network.

## 17. Multi-material support, and interface weights scaled by the stiffest material (energy_solver.py, energy_xpinn.py)

**Change 1 - per-block material.** `EnergyProblem` and `DecomposedEnergyProblem` blocks may now carry their
own material as a last element (`(pts, w, physics)` / `(sub, pts, w, physics)`), and an interface may name the
material on each side (`phys_a`, `phys_b`) for the traction-continuity term. Blocks without one behave exactly as
before: the total loss, strain energy and traction term were compared bit-for-bit on the cantilever, plate with
hole and L-bracket before and after the change. (`EnergyProblem.interior` now holds triples; the one test that
unpacked pairs, `tests/test_energy_solver.py`, was updated.)

**Bug - interface weights scaled by the base material.** The penalty weight gamma_u E / L, the traction weight
gamma_t L / E and the multiplier step rho = 2 gamma_u E / L used `physics.E`, the problem's base material. With a
single material that is the only modulus. On the fibre composite the base material was the SOFT matrix (3.5 GPa
against 72 GPa fibres): measured at the warm-start state, the traction term was 60x the strain energy (0.011x on
the soft-inclusion benchmark, whose base material is the stiff steel), and the optimiser removed the deformation
(displacement error rose from 75.7 % to 98 % the moment the subnetworks separated).

**Fix.** Scale all interface terms by `E_ref`, the stiffest material present in the problem (blocks and
interfaces). It is chosen from the problem definition, not from any comparison with FEM, and is identical to the
old behaviour whenever the base material is the stiffest - checked bit-for-bit on all six benchmarks that existed
(cantilever, plate with hole, L-bracket, soft inclusion at contrast 15 and 3, sandwich panel), so no earlier result
changes.

**What it did and did not fix.** On the composite the fibre-side stress error dropped from 90 % to 21 %, but the
displacement-continuity weight is now 20x stiff for the soft matrix and the matrix subnetwork barely deforms
(92 % displacement error): one weight set cannot serve both sides of a 20:1 contrast around eight interfaces.
Both composite runs are reported (results/multi_material/failed_candidates/fibre_composite/). The benchmarks
that pass (results/multi_material/SUMMARY.md) have one closed interface with the stiffer material either inside
(steel insert) or outside (inclusion, filled slot). The soft inclusion was run before the fix, which does
not change it (its base material is the stiffest); the steel insert and the filled slot were run after it.

## Not changed, but worth knowing about

- ~~The cantilever/L-bracket tip loads are applied as a **uniform** traction
  over the loaded edge (`P/H`), not the parabolic shear distribution a
  Timoshenko/Euler-Bernoulli reference solution assumes.~~ **Now fixed** -
  see item #13 above. The loaded-edge tractions are the corner-vanishing
  parabolic profiles (`t_y = 6(P/H)(y/H)(1-y/H)` for the cantilever,
  `t_x = 6P·u(1-u)` for the L-bracket), and the FEniCS reference scripts
  now apply the *same* profile, so the reference and the PINN solve the
  same BVP.
- `validation.py`'s mock FEM reference is a placeholder analytic field, not
  physically meaningful — fine for exercising the pipeline, meaningless
  for actually judging XPINN accuracy. You already knew that (it's
  commented), just flagging it stays true after these fixes.
