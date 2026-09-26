# ATLAS — Adaptive Multi-Scale Framework for Structural Design

Deep-energy (Ritz) forward solver, domain-decomposed as **ATLAS-XPINN**, with inverse shape design
and FEM (FEniCS) validation. The paper/proposal and the investigation log live under `docs/`.

## Layout

```
atlas_main/
├── atlas/                   Python package (all solver code)
│   ├── energy_solver.py     ATLAS energy: single-network deep-energy solver; cantilever,
│   │                        plate-with-hole and L-bracket problems
│   ├── energy_xpinn.py      ATLAS-XPINN: subdomain networks + interface continuity
│   │                        (augmented Lagrangian, warm start)
│   ├── energy_design.py, energy_xpinn_design.py   inverse design on the energy solvers
│   ├── benchmarks/          the extra benchmarks (multi-material and hard single-material)
│   ├── physics.py, validation.py, visualization.py, paths.py, ...
│   └── atlas_main.py        interactive CLI (train / validate / inverse design)
├── scripts/
│   ├── run_benchmark.py, run_repeatability_study.py, compare_solvers.py,
│   │   run_convergence_timing.py, deepxde_baseline.py, verify_inverse_design.py,
│   │   make_comparison_report.py, make_convergence_timing_report.py   main study
│   ├── multi_material/      run_<benchmark>.py, run_protocol.sh, make_report.py
│   ├── hard_benchmarks/     notched plate and multi-cutout bracket runs + reports
│   ├── checks/              environment / FEniCS / FEM-reference checkers
│   └── experiments/         shell drivers that record how batches of runs were launched
├── fem/
│   ├── fenics_scripts/      FEniCS reference solvers, one per benchmark
│   └── reference/           FEM references (.npy / .json) the scripts score against
├── tests/                   test scripts (run each with python) + validation drivers
├── webapp/                  FastAPI console (results viewer + live solver)
├── results/                 all study outputs; index in results/README.md
└── docs/                    Atlas.md (write-up), ATLAS_spec.md, BUGFIXES.md, archive/
```

## Run

All commands from the project root with the Atlas conda env's python
(`/home/sultra/miniconda3/envs/Atlas/bin/python`). Run one GPU job at a time.

```bash
# the three paper benchmarks
python scripts/run_benchmark.py --benchmark cantilever          # or l_bracket / plate_hole
python scripts/run_repeatability_study.py --benchmark cantilever

# multi-material benchmarks: the full protocol (FEM reference, seed 42, control, 5 seeds)
bash scripts/multi_material/run_protocol.sh steel_insert vm_l2
python scripts/multi_material/run_steel_insert.py --solver xpinn --seed 42   # a single run
python scripts/multi_material/make_report.py --bench steel_insert

# tests
python tests/test_smoke.py
python tests/test_energy_solver.py

# web console
cd webapp/backend && python app.py        # then open http://localhost:8000/
```

FEniCS references need the legacy FEniCS 2019.1.0 env on `PATH`
(`PATH=$CONDA_PREFIX/bin:$PATH python fem/fenics_scripts/<benchmark>_fenics.py`); see
`fem/fenics_scripts/README.md`. WSL has about 3 GB of RAM: keep FEM meshes near or below
120k P2 cells.

## Results in one paragraph

On the three paper benchmarks both ATLAS solvers meet the targets by a wide margin (ATLAS-XPINN,
5 seeds: cantilever 0.25% / 2.14%, plate with hole 0.41% / 2.89%, L-bracket 0.25% / 4.54%
displacement / von Mises error), while DeepXDE PINNs, with default or tuned settings, stay at
40-100%. Domain decomposition pays off when a material interface is closed and inside the part:
ATLAS-XPINN beats the single network (including a parameter-matched one) on a steel insert
(10x lower von Mises error), a soft inclusion (2x) and an epoxy-filled slot (displacement only),
and loses on single-material parts and on three other multi-material candidates, all reported:
`results/multi_material/SUMMARY.md`.

## Notes

- All directory paths route through `atlas/paths.py`.
- `docs/BUGFIXES.md` records every fix with its evidence (section 16: how ATLAS-XPINN was
  improved; section 17: multi-material support and interface scaling).
- The display name is "ATLAS-XPINN"; the folder `results/atlas_xpinn_energy/` and the variant key
  `xpinn_energy` keep their old names so existing paths keep working.
