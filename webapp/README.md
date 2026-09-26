# ATLAS Web Console

Browser UI for the ATLAS XPINN framework: forward solves, FEM validation and
inverse design with live plot output. The backend is a thin FastAPI wrapper
around the existing CLI pipeline (`atlas_main.py`, `validation.py`,
`visualization.py`) - no business logic is reimplemented here.

## Run

```bash
# from the atlas_main root
cd webapp/backend
/home/sultra/miniconda3/envs/Atlas/bin/python app.py
# or: uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/

Dependencies (already present in the Atlas conda env):
`fastapi`, `uvicorn`, `jinja2`, `pydantic`, plus the project stack
(torch, numpy, matplotlib).

## What it does

- **Forward + Validate** (`POST /api/forward`): trains the XPINN for the
  selected benchmark, runs FEM validation against `fem/reference/`,
  and renders five plots (training convergence, displacement field,
  von Mises stress, deformed shape, ATLAS-vs-FEM side-by-side) plus the
  relative L2 metrics with PASS/FAIL against the paper's <5% target.
- **Inverse Design** (`POST /api/inverse`): plate-with-hole -> hole radius,
  L-bracket -> re-entrant corner fillet. Reports the radius history and the
  inverse-design history plot. Not available for the cantilever.

Jobs are serialized (one GPU consumer at a time); the frontend polls the job
endpoint while training runs.

## Layout

    webapp/
      backend/app.py     FastAPI app + job runner (single-file)
      frontend/          index.html, app.js, style.css (no build step)
      static/plots/      generated PNGs, served under /static/plots/

## API

- `GET  /`                     frontend
- `GET  /api/health`           device + torch info
- `POST /api/forward`          `{benchmark, epochs, n_interior, use_lbfgs,
                                warm_start_eb, prefit_epochs, use_mock}`
- `POST /api/inverse`          `{benchmark, forward_epochs, n_outer, n_inner}`
- `GET  /api/jobs`             recent jobs
- `GET  /api/jobs/{id}`        job status / metrics / plots / console log

Known limits match the CLI (documented in BUGFIXES.md section 13): the
cantilever converges to ~5-6.8% L2 (clamp-corner stress singularity), the
L-bracket and plate-hole remain above the 5% target with the current soft
residual loss.