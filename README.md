# ATLAS — Structural Design Studio

Deep-energy (Ritz) structural solver & domain-decomposed **ATLAS-XPINN** suite with forward simulation,
inverse shape design, and dual finite-element validation (**FEniCS 2019.1.0** + **Built-in Cross-Platform `P2` FEM**).

## Download & Install (`v1.1.7`)

Grab the latest standalone **ATLAS Desktop Studio (`v1.1.7` — Published by `ATLAS`)** directly from **[GitHub Releases](https://github.com/SultraZ/atlas/releases/latest)**:

- **Full Complete Package (`.zip` — Compact ~8 MB Archive with Native `ATLAS-Setup.exe` & `ATLAS.exe`)**: **[Download `ATLAS-v1.1.7.zip`](https://github.com/SultraZ/atlas/archive/refs/tags/v1.1.7.zip)**
  - Extract the `.zip` archive and double-click **`ATLAS-Setup.exe`** *(Native Win32 GUI Installer & Fast Updater — Publisher/Author: `ATLAS`)*.
- **Windows Native GUI Installer & Updater (`ATLAS-Setup.exe`)**: **[Download `ATLAS-Setup.exe`](https://github.com/SultraZ/atlas/raw/v1.1.7/ATLAS-Setup.exe)**
  - Native Windows x64 GUI Installer (`Publisher: ATLAS`) with dedicated **Update** & **Install ATLAS** actions — updates an existing installation in seconds without re-downloading Python/CUDA wheels, or installs **Standard Cross-Platform Edition *(No WSL needed — GPU/CPU + Built-in P2 FEM)*** / **Full FEniCS Edition *(WSL2 + FEniCS 2019.1.0)***.
- **macOS & Linux (1-Line Guided Installer)**:
  ```bash
  curl -fsSL https://github.com/SultraZ/atlas/raw/v1.1.7/setup_atlas.sh | bash
  ```

## Project Layout

```
atlas/
├── atlas/                   Core physics, deep-energy & ATLAS-XPINN solvers, inverse design,
│   ├── energy_solver.py     Single-network deep-energy solver
│   ├── energy_xpinn.py      Domain-decomposed ATLAS-XPINN solver + augmented-Lagrangian interface continuity
│   ├── energy_design.py     Single-network inverse shape optimization
│   ├── energy_xpinn_design.py  Domain-decomposed XPINN inverse shape optimization
│   ├── benchmarks/          Multi-material benchmark models (steel_insert, bimaterial, filled_slot)
│   ├── live.py              Parametric live problem builder (6 benchmarks)
│   ├── native_fem.py        Built-in cross-platform quadratic P2 FEM validator (SciPy sparse)
│   └── physics.py, networks.py, sampling.py, validation.py, visualization.py, design_plots.py
├── fem/
│   ├── fem_live.py          On-demand FEniCS 2019.1.0 reference solver (dolfin + mshr)
│   ├── fem_parametric.py    On-demand FEniCS parametric check for inverse shape design
│   └── reference/           Core reference datasets for the 6 benchmarks
├── installer/               Native Win32 C GUI Setup Wizard, launcher, uninstaller & runtime checks
├── scripts/
│   ├── build_windows_exes.py   Cross-compiles native x64 Win32 GUI binaries (ATLAS-Setup.exe, ATLAS.exe)
│   └── create_release.py       Builds and publishes versioned GitHub Releases
├── webapp/
│   ├── desktop.py           Standalone native desktop window launcher (Edge WebView2 / pywebview)
│   ├── backend/app.py       FastAPI backend server & live job runner
│   └── frontend/            Clean light/dark desktop web interface (index.html, style.css, app.js)
├── results/                 Compact benchmark summary JSONs & reference figures for the Results view
├── tests/                   Automated physics, solver, inverse design, UI & desktop test suite
├── ATLAS-Setup.exe          Native Windows x64 GUI Installer & Updater (Publisher: ATLAS)
├── ATLAS.exe                Native Windows x64 Desktop Application Launcher (Publisher: ATLAS)
├── setup_atlas.sh           macOS / Linux guided installer
└── launch_atlas.sh          macOS / Linux / WSL2 application launcher
```

## Quick Commands

```bash
# Launch Desktop Studio
./launch_atlas.sh                              # macOS / Linux / WSL2
python webapp/desktop.py --device auto         # Native desktop window (or --no-window --port 8000)

# Run automated test suite
pytest -v

# Rebuild native Windows executables & package release
python scripts/build_windows_exes.py
python scripts/create_release.py --version v1.1.7 --publish
```
