# ATLAS Desktop Application & Web Console

Standalone local application wrapping the full **ATLAS / ATLAS-XPINN** deep-energy solver alongside
**dual selectable FEM validation engines** — genuine **FEniCS 2019.1.0** (`dolfin` + `mshr` in `fem/`,
untouched) for WSL2/Linux AND a **Built-in Cross-Platform P2 FEM solver** (`atlas/native_fem.py`,
pure Python + SciPy, working natively on Windows/macOS/Linux without WSL) — inside a modern left-rail
desktop studio interface (`pywebview`, Windows Edge/Chrome app-mode from WSL2, or local browser).

Four workspace views in the collapsible left sidebar, with **Light**, **Dark**, and **Atlas** themes,
**Compact / Default / Large** interface scaling, and **Simple / Expert** mode:

- **Results**: reads the stored study in `results/` via `/api/results` (all three single-material
  benchmarks with ATLAS-XPINN, single-network ATLAS, and two DeepXDE PINN baselines against FEniCS,
  5-seed repeatability, inverse design, plus all three multi-material benchmarks with interface-band
  errors). Works completely offline with zero GPU retraining needed.
- **Live Solver**: trains **ATLAS-XPINN** or **ATLAS (single network)** locally on **GPU (`cuda`)**,
  **CPU (`cpu`)**, or **Auto**, validates against your chosen FEM backend (stored FEniCS reference at
  study inputs, or live on-demand solve via **FEniCS** OR **Built-in P2 FEM** for custom sliders and
  inverse shape checks).
- **Folder & Files**: browse generated figures (`webapp/backend/plots`), completed run JSON records,
  study figures (`results/`), and precomputed FEniCS references (`fem/reference/`), and open any local
  folder directly in Windows Explorer / macOS Finder / Linux file manager.
- **Settings**: dedicated studio page for **Theme (`Light` / `Dark` / `Atlas`)**, **UI Scale**,
  **Simple / Expert** detail mode, **Compute Device (`Auto` / `GPU` / `CPU` + threads)**, and
  **FEM Engine (`Auto` / `FEniCS WSL2·Linux` / `Built-in Cross-Platform P2 FEM`)** with a 1-click
  cantilever verification test.

## Adjustable Runtime & Solver Settings

- **Compute & Hardware (`Settings` drawer & left rail)**:
  - Switch compute device at any time between **Auto**, **GPU (`cuda`)**, and **CPU (`cpu`)**.
  - Live hardware telemetry (GPU model, VRAM, CUDA version, FEniCS availability, CPU logical cores,
    PyTorch version, plot storage size).
  - Adjustable **PyTorch CPU thread count**, **figure export resolution** (`100 / 150 / 220 / 300 DPI`),
    **live polling cadence** (`0.8 s / 1.5 s / 3.0 s`), **CUDA memory cache release**, **plot folder
    opener**, and **run history / plot storage cleanup** (persisted to `webapp/.app_settings.json`).
- **Solver & Design Controls (`Live solver` rail)**:
  - **Solver architecture**: switch between domain-decomposed `ATLAS-XPINN` and single-network `ATLAS`.
  - **Network size & presets**: `3×32 Fast`, `4×64 Study`, `4×91 Wide`, `5×128 Deep`, or custom hidden
    layers (`2–8`) and width (`16–256`), plus floating-point precision (`float64` study / `float32` faster).
  - **Optimizer & coupling**: Adam epochs (`50–40000`), L-BFGS steps (`0–8000`), learning rate, seed
    (`42–46` quick study seeds or custom), XPINN warm-start steps, augmented-Lagrangian stages, and
    interface penalties (`γ_u`, `γ_t`).
  - **Inverse design**: selectable objective (`compliance`, `peak_vm`, `volume`), target ratio vs
    baseline (`0.30×–1.60×` slider or numeric `0.10×–5.00×`), max outer design iterations, and inner
    forward L-BFGS steps.
  - **Run control**: cooperative **Stop** button on active runs and one-click **JSON metrics export**
    on completed runs.

## Launch

### 1. One-Click Desktop Launch
- **Windows** (supports WSL2 `Atlas` conda env seamlessly via `\\wsl.localhost\...` or native Python):
  ```bat
  :: Option A — run directly:
  ATLAS.bat

  :: Option B — generate a real native ATLAS.exe (with custom icon, Taskbar support, no console popup):
  build_exe.bat
  ATLAS.exe

  :: Option C — build a full PyInstaller bundle (dist\ATLAS\ATLAS.exe) for PCs without Python:
  build_exe.bat --standalone
  ```
- **Linux / WSLg / macOS**:
  ```bash
  ./launch_atlas.sh
  ```

### 2. CLI Options (`webapp/desktop.py`)
```bash
# Auto-pick native window shell (pywebview -> WSL2 Windows app window -> OS app window -> browser)
python webapp/desktop.py

# Pin CPU or GPU mode and CPU thread count at startup
python webapp/desktop.py --device cpu --threads 8
python webapp/desktop.py --device cuda

# Open directly on the Live Solver tab
python webapp/desktop.py --live

# Headless / browser server mode
python webapp/desktop.py --no-window --host 0.0.0.0 --port 8000
```

## API

| method | path | description |
|---|---|---|
| GET | `/api/results` | Stored study results, 5-seed statistics, inverse designs, and `/results/*` figure URLs |
| GET | `/api/benchmarks` | Benchmark metadata, geometry/material/load sliders, defaults, and inverse design specs |
| GET | `/api/health` | Active compute device, GPU/CPU hardware info, FEniCS availability, storage stats, settings |
| GET | `/api/settings` | Current persisted runtime settings (`device`, `cpu_threads`, `plot_dpi`, `poll_ms`, `auto_open_log`) |
| POST | `/api/settings` | Update and persist runtime settings (applies CPU thread count and default device immediately) |
| POST | `/api/cuda/clear` | Call `torch.cuda.empty_cache()` and return refreshed hardware telemetry |
| POST | `/api/open-plots` | Open `webapp/static/plots/` in the desktop file manager (Windows Explorer in WSL2, Finder, `xdg-open`) |
| POST | `/api/forward` | Queue a live forward solve + FEniCS validation (`device`, `formulation`, network/AL/optimizer params) |
| POST | `/api/inverse` | Queue an inverse shape design + FEniCS verification (`device`, `metric`, `target_ratio`, `n_outer`, etc.) |
| GET | `/api/jobs` | Recent jobs summary |
| GET | `/api/jobs/{id}` | Job status, stage, live log, energy descent history, solver config, validation metrics, plot URLs |
| POST | `/api/jobs/{id}/cancel` | Cooperatively stop a queued or running job |
| GET | `/api/jobs/{id}/export` | Download JSON report of a job's configuration, runtime hardware, and FEniCS validation metrics |
| POST | `/api/jobs/clear` | Clear finished job history and remove generated plot files in `webapp/static/plots/` |

## Layout

    webapp/
      desktop.py         Standalone desktop window & backend process launcher
      backend/app.py     FastAPI backend, settings manager, and background solver runner
      frontend/          index.html, style.css, app.js, icon.svg (no build step required)
      static/plots/      Figures written by live jobs (gitignored)
