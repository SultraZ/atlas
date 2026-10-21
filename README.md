# ATLAS — Adaptive Multi-Scale Framework for Structural Design

Deep-energy (Ritz) forward solver, domain-decomposed as **ATLAS-XPINN**, with inverse shape design
and dual selectable finite-element validation (**genuine FEniCS 2019.1.0** + **Built-in Cross-Platform `P2` FEM**).

## Download & Install (GitHub Releases)

Grab the latest standalone **ATLAS Desktop Studio (`v1.1.5` — Published by `ATLAS`)** directly from **[GitHub Releases](https://github.com/SultraZ/atlas/releases/latest)**:

- **Full Complete Package (`.zip` — Includes Native `ATLAS-Setup.exe` & `ATLAS.exe`)**: **[Download `ATLAS-v1.1.5.zip`](https://github.com/SultraZ/atlas/archive/refs/tags/v1.1.5.zip)**
  - Extract the `.zip` archive and double-click **`ATLAS-Setup.exe`** *(Native Win32 GUI Installer — Publisher/Author: `ATLAS`)*.
- **Windows Native GUI Installer (`ATLAS-Setup.exe`)**: **[Download `ATLAS-Setup.exe`](https://github.com/SultraZ/atlas/raw/v1.1.5/ATLAS-Setup.exe)**
  - Real Windows x64 GUI Installer (`Publisher: ATLAS`) — lets you choose **Standard Cross-Platform Edition *(No WSL needed — GPU/CPU + Built-in P2 FEM)*** or **Full FEniCS Edition *(WSL2 + FEniCS 2019.1.0)***, installs `ATLAS.exe`, registers in *Windows Settings → Installed Apps*, and adds **Desktop & Start Menu shortcuts**.
- **macOS & Linux (1-Line Guided Installer)**:
  ```bash
  curl -fsSL https://github.com/SultraZ/atlas/raw/v1.1.5/setup_atlas.sh | bash
  ```
- **Rebuild Windows Executables / Publish a New Release Anytime**:
  ```bash
  python scripts/build_windows_exes.py
  python scripts/create_release.py --version v1.1.5 --publish
  ```

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
│   ├── live.py              parameterised ATLAS / ATLAS-XPINN problems for the live solver
│   ├── native_fem.py        built-in cross-platform P2 FEM solver (pure Python + SciPy)
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
│   ├── fem_live.py          on-demand FEniCS reference for any live-solver settings
│   └── reference/           FEM references (.npy / .json) the scripts score against
├── tests/                   test scripts (run each with python) + validation drivers
├── webapp/                  Desktop app & FastAPI console (results viewer + live solver)
│   └── desktop.py           Standalone desktop launcher (native window + local backend)
├── launch_atlas.sh          Linux / WSL2 / macOS desktop launcher
├── setup_atlas.sh           macOS / Linux guided installer
├── ATLAS-Setup.exe          Native Win32 GUI Setup Wizard (Publisher: ATLAS)
├── ATLAS.exe                Native Win32 Desktop Studio application launcher (Publisher: ATLAS)
└── results/                 all study outputs; index in results/README.md
```

## Run

All commands from the project root with the Atlas conda env's python
(`/home/sultra/miniconda3/envs/Atlas/bin/python`). Run one GPU job at a time.

```bash
# the three paper benchmarks
python scripts/run_benchmark.py --benchmark cantilever          # or l_bracket / plate_hole
python scripts/run_repeatability_study.py --benchmark cantilever --formulation energy_xpinn --epochs 2000
python scripts/run_repeatability_study.py --benchmark cantilever --formulation energy --epochs 2000   # single network

# multi-material benchmarks: the full protocol (FEM reference, seed 42, control, 5 seeds)
bash scripts/multi_material/run_protocol.sh steel_insert vm_l2
python scripts/multi_material/run_steel_insert.py --solver xpinn --seed 42   # a single run
python scripts/multi_material/make_report.py --bench steel_insert

# tests
python tests/test_smoke.py
python tests/test_energy_solver.py
python tests/test_desktop_app.py

# standalone desktop application (opens a dedicated native desktop window)
./launch_atlas.sh                              # Linux / WSL2 / macOS
python webapp/desktop.py --device auto         # or --device cuda / --device cpu --threads 8
# on Windows (double-click ATLAS.bat, or build a native icon-stamped ATLAS.exe in ~1 second):
build_exe.bat                                  # builds ATLAS.exe (uses built-in Windows compiler, 0 downloads)
build_exe.bat --standalone                     # optional: bundles Python + PyTorch into dist\ATLAS\ATLAS.exe
ATLAS.exe                                      # or double-click ATLAS.bat directly

# local browser / headless server mode
python webapp/desktop.py --no-window --port 8000   # or: cd webapp/backend && python app.py
```

FEniCS references use either the legacy FEniCS 2019.1.0 env on `PATH`
(`PATH=$CONDA_PREFIX/bin:$PATH python fem/fenics_scripts/<benchmark>_fenics.py`; see
`fem/fenics_scripts/README.md`) **or** the built-in cross-platform `P2` FEM engine
(`atlas/native_fem.py`, pure Python + SciPy — selectable in the desktop **Settings** view on
machines without WSL2/FEniCS). WSL has about 3 GB of RAM: keep FEM meshes near or below
120k P2 cells.

## Results in one paragraph

On the three paper benchmarks both ATLAS solvers meet the targets by a wide margin (ATLAS-XPINN,
5 seeds: cantilever 0.25% / 2.14%, plate with hole 0.40% / 2.89%, L-bracket 0.25% / 4.54%
displacement / von Mises error), while DeepXDE PINNs, with default or tuned settings, stay at
30-100%. Domain decomposition pays off when a material interface is closed and inside the part:
ATLAS-XPINN beats the single network (including a parameter-matched one) on a steel insert
(10x lower von Mises error), a soft inclusion (2x) and an epoxy-filled slot (displacement only),
and loses on single-material parts and on three other multi-material candidates, all reported:
`results/multi_material/SUMMARY.md`. Every number for the six benchmarks is collected in `results/ALL_RESULTS.md`.

## Notes

- All directory paths route through `atlas/paths.py`.
- The display name is "ATLAS-XPINN"; the folder `results/atlas_xpinn_energy/` and the variant key
  `xpinn_energy` keep their old names so existing paths keep working.
