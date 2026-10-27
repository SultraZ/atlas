# ATLAS — Structural Design Studio

Deep-energy (Ritz) structural solver and domain-decomposed **ATLAS-XPINN** suite with forward simulation,
inverse shape design on all six benchmarks, and dual finite-element validation
(**FEniCS 2019.1.0** + **built-in cross-platform `P2` FEM**).

## Download & Install (`v1.0.0`)

Get **ATLAS Desktop Studio v1.0.0** from **[GitHub Releases](https://github.com/SultraZ/atlas/releases/latest)**:

- **Windows (recommended)**: download **[`ATLAS-Desktop-Studio.zip`](https://github.com/SultraZ/atlas/releases/latest/download/ATLAS-Desktop-Studio.zip)**,
  extract it and double-click **`ATLAS-Setup.exe`**. The wizard installs Python and GPU/CPU PyTorch, lets you choose the
  **Standard edition** (no WSL, built-in P2 FEM) or the **Full FEniCS edition** (WSL2 + FEniCS 2019.1.0), and adds Desktop
  and Start Menu shortcuts. Run it again later and choose **Update** to update in place.
- **Windows installer only**: **[`ATLAS-Setup.exe`](https://github.com/SultraZ/atlas/releases/latest/download/ATLAS-Setup.exe)**
  (downloads the rest of the package itself).
- **macOS & Linux**:
  ```bash
  curl -fsSL https://github.com/SultraZ/atlas/releases/latest/download/setup_atlas.sh | bash
  ```

## What it does

- **Forward solve**: ATLAS-XPINN (one network per subdomain, augmented-Lagrangian interface coupling) on six 2D
  plane-stress benchmarks: cantilever beam, L-shaped bracket, plate with a hole, steel insert, soft inclusion and
  epoxy-filled slot. Materials, loads and geometry can be changed; every run is validated against FEM.
- **Inverse shape design**: one geometric parameter per benchmark is tuned to hit a target, with shape gradients from
  automatic differentiation, and the design is re-checked with FEM.
- **Field explorer**: after a forward run, hover anywhere on the part to read coordinates, displacements, the ATLAS
  stress tensor, von Mises (ATLAS and FEM) and the local error; drag a divider to compare ATLAS and FEM on one colour
  scale. Points near stress singularities or material interfaces are flagged.
- **Calculation report**: every finished run has a print-ready report (materials, loads, boundary conditions, solver
  settings, validation against FEM, convergence and figures) that saves to PDF from the print dialog.
- **Session Runs**: each run is saved in its own folder (`webapp/static/plots/<date>_<time>_<benchmark>_<kind>_<id>/`)
  with its figures, field data, log and report, and stays listed after ATLAS restarts.
- **Results view**: the stored study (5-seed repeatability, DeepXDE baselines, interface band errors, inverse designs).

### Stored inverse-design study (ATLAS-XPINN, checked with FEniCS)

| Benchmark | Parameter | Goal | ATLAS | FEM optimum | FEM at the ATLAS design |
|---|---|---|---|---|---|
| Cantilever | beam height | compliance × 0.60 | 1.193 | 1.193 | +0.03 % from target |
| Plate with a hole | hole radius | compliance × 1.10 | 0.481 | 0.482 | −0.05 % |
| L-bracket | fillet radius | peak von Mises × 0.75 | 0.239 | 0.215 | −3.7 % |
| Steel insert | insert radius | compliance × 0.97 | 0.527 | 0.527 | −0.003 % |
| Soft inclusion | inclusion radius | compliance × 1.10 | 0.516 | 0.515 | +0.05 % |
| Epoxy-filled slot | slot semi-axis along the load | peak von Mises × 0.75 | 0.333 | 0.315 | −3.4 % |

On the multi-material benchmarks the material interface moves with the design parameter and each material keeps its
own subnetwork. Reproduce the multi-material designs with `python scripts/run_multi_inverse_design.py`.

## Project Layout

```
atlas/
├── atlas/                   Core physics, deep-energy & ATLAS-XPINN solvers, inverse design
│   ├── energy_solver.py     Single-network deep-energy solver
│   ├── energy_xpinn.py      Domain-decomposed ATLAS-XPINN solver + augmented-Lagrangian interface continuity
│   ├── energy_design.py     Single-network inverse shape optimization
│   ├── energy_xpinn_design.py  ATLAS-XPINN inverse shape optimization (all six benchmarks)
│   ├── benchmarks/          Multi-material benchmark models (steel_insert, bimaterial, filled_slot)
│   ├── live.py              Parametric live problem builder (6 benchmarks)
│   ├── native_fem.py        Built-in cross-platform quadratic P2 FEM validator (SciPy sparse)
│   └── physics.py, networks.py, sampling.py, validation.py, visualization.py, design_plots.py
├── fem/
│   ├── fem_live.py          On-demand FEniCS 2019.1.0 reference solver (dolfin + mshr)
│   ├── fem_parametric.py    FEniCS check for inverse shape designs
│   └── reference/           Reference datasets for the 6 benchmarks
├── installer/               Native Win32 setup wizard, launcher, uninstaller & runtime checks
├── scripts/
│   ├── build_windows_exes.py       Cross-compiles the Windows binaries (ATLAS-Setup.exe, ATLAS.exe, uninstaller)
│   ├── create_release.py           Builds and publishes GitHub Releases
│   ├── run_multi_inverse_design.py Multi-material inverse-design study with FEniCS checks
│   └── run_inverse_seed_study.py   Inverse design on all six benchmarks over seeds 42–46, with FEniCS checks
├── webapp/
│   ├── desktop.py           Desktop window launcher (Edge WebView2 / pywebview)
│   ├── backend/app.py       FastAPI backend & live job runner
│   ├── backend/report.py    Printable engineering calculation report
│   └── frontend/            Desktop web interface (index.html, style.css, app.js)
├── results/                 Stored study (JSON summaries & figures) for the Results view
├── tests/                   Physics, solver, inverse design and desktop tests
├── ATLAS-Setup.exe          Windows x64 installer & updater
├── ATLAS.exe                Windows x64 application launcher
├── setup_atlas.sh           macOS / Linux installer
└── launch_atlas.sh          macOS / Linux / WSL2 launcher
```

## Quick Commands

```bash
# Launch Desktop Studio
./launch_atlas.sh                              # macOS / Linux / WSL2
python webapp/desktop.py --device auto         # desktop window (or --no-window --port 8000)

# Run the test suite
pytest -v

# Rebuild the Windows executables and publish a release
python scripts/build_windows_exes.py
python scripts/create_release.py --version v1.0.0 --publish
```
