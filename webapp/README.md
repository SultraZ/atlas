# ATLAS Web Console

Two views in one page, with a light, dark or Atlas theme (the choice is kept in the browser):

- **Results**: the stored study in `results/`, read through `/api/results`. The three paper
  benchmarks (ATLAS-XPINN, the single-network ATLAS solver and two DeepXDE PINN baselines against
  FEniCS, five-seed repeatability, inverse design) and the three multi-material benchmarks
  (ATLAS-XPINN against a single network and a parameter-matched one, interface-band errors, five
  seeds). Nothing is retrained in the browser.
- **Live solver**: runs ATLAS-XPINN on the server for any of the six benchmarks and validates it
  against FEniCS, or runs an inverse design on a paper benchmark. Each benchmark's materials, load
  and key dimension can be changed with sliders (with a live geometry preview). At the study's
  settings a run is scored against the stored reference; at any other settings the server first
  solves a FEniCS reference for exactly those inputs (`fem/fem_live.py`, a few seconds). The solver
  log, the pipeline steps and a live energy-descent chart update while the job runs; jobs run one at
  a time and queue.

## Run

```bash
cd webapp/backend
/home/sultra/miniconda3/envs/Atlas/bin/python app.py
```

Open http://localhost:8000/. No build step: `frontend/` is plain HTML, CSS and JavaScript.

## API

| method | path | |
|---|---|---|
| GET | `/api/results` | stored study: per benchmark, solver errors, params and train times, seed statistics, inverse design, figure URLs under `/results/` |
| GET | `/api/benchmarks` | benchmark metadata for the live solver: adjustable parameters (label, unit, study value, range), subdomains, network, inverse-design parameter; the L-BFGS budget and seed |
| GET | `/api/health` | device, torch version, active jobs |
| POST | `/api/forward` | `{benchmark, formulation: "energy_xpinn" \| "energy", epochs, params}`; `params` overrides the study settings (see `/api/benchmarks`); the UI always sends `energy_xpinn` |
| POST | `/api/inverse` | `{benchmark}` for cantilever, l_bracket or plate_hole (ATLAS-XPINN design loop) |
| GET | `/api/jobs` | recent jobs (without logs) |
| GET | `/api/jobs/{id}` | status, `stage` (setup, train, predict, validate, plots), streaming log, `progress` (energy per evaluation), `custom` flag, metrics, plot URLs |

A live forward run uses the study's settings (seed 42, the given Adam epochs, then 2000 L-BFGS
iterations), so with the default 2000 epochs it uses exactly the configuration of the stored seed-42
run (GPU arithmetic can still differ in the last digits).

## Layout

    webapp/
      backend/app.py     FastAPI app and job runner (parameterised problems: atlas/live.py)
      frontend/          index.html, style.css, app.js
      static/plots/      figures written by live jobs (not versioned)
