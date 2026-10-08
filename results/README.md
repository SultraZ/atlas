# Results index

All numbers for the six benchmarks in one file: [`ALL_RESULTS.md`](ALL_RESULTS.md).

Every number here is scored against a FEniCS reference in `fem/reference/` that no solver sees in
training. Targets: displacement error below 5 %, von Mises error below 10 %.

## The three paper benchmarks (cantilever, plate with hole, L-bracket)

| folder | contents |
|---|---|
| `atlas_xpinn_energy/` | **ATLAS-XPINN**: `forward/` (models, validation, plots), `repeatability/` (5 seeds), `inverse_design/` (designs + independent FEM checks), `logs/` |
| `atlas_energy/` | ATLAS energy, the single-network solver: same structure |
| `comparison/deepxde_vs_atlas_xpinn/` | DeepXDE PINN (defaults and configured) vs ATLAS-XPINN vs FEM: `COMPARISON.md` + figures |
| `comparison/atlas_vs_atlas_xpinn/` | single network vs ATLAS-XPINN |
| `comparison/all_solvers/` | in-house PINN, DeepXDE, ATLAS energy, ATLAS-XPINN |
| `comparison/convergence_timing/` | error vs iteration and wall-clock, processing time: `CONVERGENCE_TIMING.md` |
| `comparison/xpinn_fix_study/` | the experiments behind the ATLAS-XPINN improvements |
| `comparison/data/`, `comparison/logs/` | baseline fields/metrics the reports read; baseline logs |

## Multi-material benchmarks: where decomposition pays off

Start with [`multi_material/SUMMARY.md`](multi_material/SUMMARY.md). Each folder has a
`PROTOCOL.md` written before its first run, a report, per-solver `seed*.json` / `.npz` / model
files, figures and logs.

| folder | benchmark | outcome |
|---|---|---|
| `multi_material/steel_insert/` | steel insert in an aluminium plate | pass, 5 seeds (von Mises 10x lower) |
| `multi_material/bimaterial/` | plate with a soft circular inclusion (contrast 15 and 3) | pass, 5 seeds (von Mises 2x lower) |
| `multi_material/filled_slot/` | epoxy-filled elliptical slot in a steel plate | pass (displacement; von Mises a tie) |
| `multi_material/failed_candidates/transition_joint/` | steel-aluminium joint, wavy bond line | fail |
| `multi_material/failed_candidates/sandwich/` | sandwich panel | fail |
| `multi_material/failed_candidates/fibre_composite/` | glass-fibre / epoxy composite | fail (2 attempts) |

## Hard single-material benchmarks (set aside)

| folder | benchmark |
|---|---|
| `hard_single_material/notched_plate/` | U-notched plate: `NOTCHED_PLATE.md` (its headline ATLAS-XPINN rows predate the augmented-Lagrangian fix) |
| `hard_single_material/multicutout/` | L-bracket with three holes and a slot: `MULTICUTOUT.md` |
