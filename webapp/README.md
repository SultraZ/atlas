# ATLAS Web Console

Two views in one page, with a light, dark or Atlas theme. A Simple / Expert switch sets the level of detail:
Simple (the default) explains every result in plain words, with "?" tips for each term and a material picker;
Expert shows the full metrics and solver settings. Both choices are kept in the browser.

- **Results**: the stored study in `results/`, read through `/api/results`. The three paper
  benchmarks (ATLAS-XPINN, the single-network ATLAS solver and two DeepXDE PINN baselines against
  FEniCS, five-seed repeatability, inverse design) and the three multi-material benchmarks
  (ATLAS-XPINN against a single network and a parameter-matched one, interface-band errors, five
  seeds). Nothing is retrained in the browser.
- **Live solver**: runs ATLAS-XPINN on the server for any of the six benchmarks and validates it
  against FEniCS, or runs an inverse design on a paper benchmark. Each benchmark's materials, load
  and key dimension can be changed with sliders (with a live geometry preview). At the study's
  settings a run is scored against the stored reference; at any other settings the server first
  solves a FEniCS reference for exactly those inputs (`fem/fem_live.py`, a few seconds). A forward
  run shows the displacement and stress against FEM, the deformed shape and the training curve. An
  inverse design starts from the chosen settings (for the cantilever and the plate, the height or
  radius on the sliders is the starting shape) and FEniCS checks the design it finds. The solver
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
| POST | `/api/forward` | `{benchmark, formulation: "energy_xpinn" \| "energy", epochs, lbfgs, params}`; `lbfgs` is at most 8000 (use multiples of 400); `params` overrides the study settings (see `/api/benchmarks`); the UI always sends `energy_xpinn` |
| POST | `/api/inverse` | `{benchmark, params}` for cantilever, l_bracket or plate_hole: the ATLAS-XPINN design loop starting from the given settings (study settings when `params` is omitted), then a FEniCS check of the design (`fem/fem_parametric.py --check`) |
| GET | `/api/jobs` | recent jobs (without logs) |
| GET | `/api/jobs/{id}` | status, `stage` (setup, train, predict, validate, plots), streaming log, `progress` (energy per evaluation), `custom` flag, metrics, plot URLs |

A live forward run uses seed 42 and the given Adam epochs and L-BFGS iterations. The default
(2000 + 2000) is exactly the configuration of the stored seed-42 run (GPU arithmetic can still differ
in the last digits). The Simple view's run-length slider picks 20, 40, 60, 80 or 100 % of that budget;
the Expert view also accepts longer L-BFGS budgets, up to 8000.
L-BFGS runs in four augmented-Lagrangian stages of whole 100-step chunks, so budgets are multiples of
400; others are rounded up per stage. Shorter runs are faster and less accurate: at 20 % (400 + 400)
the plate with a hole took 96 s and gave 5.9 % displacement and 10.0 % von Mises error, against
0.75 % and 3.4 % for the full run.

## Layout

    webapp/
      backend/app.py     FastAPI app and job runner (parameterised problems: atlas/live.py)
      frontend/          index.html, style.css, app.js
      static/plots/      figures written by live jobs (not versioned)
