# FEniCS reference solvers

One standalone legacy-FEniCS (dolfin + mshr) script per benchmark. Each solves the same plane-stress
problem as its ATLAS benchmark (same domain, loads, supports and materials) and writes its
reference into `fem/reference/` (resolved from the script's own location). The references are used
only to score the solvers, never in training.

| script | benchmark | writes |
|---|---|---|
| `cantilever_fenics.py` | cantilever beam | `cantilever_*` |
| `l_bracket_fenics.py` | L-shaped bracket | `l_bracket_*` |
| `plate_hole_fenics.py` | plate with a circular hole | `plate_hole_*` |
| `notched_plate_fenics.py` | U-notched plate | `notched_plate_*`, `notched_plate_roi_*` |
| `multicutout_fenics.py` | multi-cutout bracket (also writes its training quadrature) | `multicutout_*`, `atlas/data/multicutout_quadrature.npz` |
| `bimaterial_fenics.py --contrast 15` | plate with a soft circular inclusion | `bimaterial_c15_*` (also `--contrast 3`) |
| `steel_insert_fenics.py` | steel insert in an aluminium plate | `steel_insert_*` |
| `filled_slot_fenics.py` | epoxy-filled elliptical slot | `filled_slot_*` |
| `sandwich_fenics.py` | sandwich panel | `sandwich_*` |
| `fibre_composite_fenics.py` | glass-fibre / epoxy composite | `fibre_composite_*` |
| `transition_joint_fenics.py` | steel-aluminium joint, wavy bond line | `transition_joint_*` |

The multi-material scripts use meshes that conform to every material interface, take stresses
element-wise (DG1, exact for P2 so the stress jump at an interface is kept), solve two mesh levels
and print the change between them, and write a `<benchmark>_meta.json` with the compliance and the
convergence figures.

## Run

From the project root, with the FEniCS env's `bin` on `PATH` (dolfin needs `pkg-config` at
import):

```bash
PATH=$CONDA_PREFIX/bin:$PATH python fem/fenics_scripts/steel_insert_fenics.py
```

Memory: WSL has about 3 GB of RAM. A P2 solve near 470k cells used it all; keep the finest level
near or below 120k cells (the multi-material scripts do).

## How these tie back to the paper

Section VI.C ("Testing and Evaluation Plan") states two accuracy targets:
displacement L2 error **< 5%** and (implicitly, same treatment) von Mises
stress L2 error, both against FEniCS. These scripts are that FEniCS side.
A few things worth knowing before you report those numbers:

- **Mesh resolution matters more than it looks.** `MESH_RESOLUTION` in each
  script (80–100) is a reasonable starting point, not a validated choice.
  Before trusting a 5% comparison, do a quick mesh-refinement check: double
  the resolution, see if peak von Mises / max displacement changes by more
  than a percent or two. If it does, your "FEM reference" isn't converged
  enough to be a fair yardstick for the XPINN yet.
- **The L-bracket's re-entrant corner is a genuine mathematical
  singularity** (stress theoretically unbounded there in the linear-elastic
  idealization) — the FEM peak-stress-at-the-corner number will keep
  climbing as you refine the mesh near that corner, forever. Report peak
  stress at a fixed small distance from the corner (not literally at the
  corner) for both XPINN and FEM, or the comparison isn't well-posed either
  way — this is a known modeling gotcha, not a bug in either solver.
- **The plate-with-hole script's docstring flags a physics question** worth
  resolving before you write it up: with `H=2.0, r` up to `0.8`, you're not
  in the "infinite plate" regime the classical Kirsch `Kt=3` stress
  concentration factor assumes. The script prints an implied `Kt` from the
  FEM solve so you can see how far off 3.0 it lands — if your paper wants
  to cite the Kirsch factor as a sanity check, either grow `H` well past
  `r` (e.g. `H ≈ 6r`) or treat "3.0" as a rough ballpark rather than the
  target to hit.
- **Regenerate `plate_hole` per radius you care about.** These aren't
  parametric — for the inverse-design comparison in Objective 5
  (ATLAS-optimized hole radius vs. FEM-parametric-optimized radius), you'll
  want a FEM solve at the ATLAS-optimized radius specifically (edit `r` at
  the top of `plate_hole_fenics.py` and rerun) to compute the compliance
  numbers you're comparing.

## Loaded-edge traction profiles (must match `sampling.py`)

The cantilever and L-bracket load faces use the **corner-vanishing
parabolic** traction that `sampling.py` applies, not a uniform load:

- cantilever right edge: `t_y(y) = 6*(P/H)*(y/H)*(1 - y/H)`
- L-bracket upper right face: `t_x(y) = 6*P*u*(1 - u)`, `u = (y-s)/s`

A uniform edge traction has a traction discontinuity at the corners and
creates spurious stress concentrations there; the parabolic profile
integrates to the same total load and vanishes at both corners. If you
change the profile in `sampling.py`, change it here too, or the reference
and the PINN will be solving different BVPs. (The `t_x`/`t_y` values are
applied with `df.UserExpression`, so FEniCS evaluates them at quadrature
points.)

## Requirements / running

On the FEniCS 2019.1.0 install used to generate the committed references,
`dolfin` needs `pkg-config` on `PATH` at import time. It was present in the
conda cache but not installed into the env; installing it and putting the
env's `bin` on `PATH` fixes the `OSError: pkg-config probably not
installed` import failure:

```bash
conda install -c conda-forge pkg-config   # into the FEniCS env
PATH=/path/to/env/bin:$PATH python cantilever_fenics.py
```

## Verified end-to-end

These have now been run end-to-end against a live legacy FEniCS 2019.1.0 +
mshr install (the earlier "syntax-checked but not run" caveat is resolved).
Regenerated references (parabolic loads) live in `fem/reference/` for
`validation.py` to consume:

- `cantilever`: max |v| = 1.2723e-3 m (2-D tip; the beam-theory EB estimate
  is −1.219e-3 m, ~4% stiffer in 2-D), peak von Mises 4.23e7 Pa
- `l_bracket`: max |u| = 1.443e-4 m, P2 peak von Mises 2.66e7 Pa
  (DG0 corner value 1.77e7 Pa — see the singularity note above)
