"""
webapp/backend/app.py - FastAPI backend for the ATLAS web console.

Thin wrapper around the existing CLI pipeline (atlas_main.py, validation.py,
visualization.py): forward solves, FEM validation and inverse design run in
background threads, and their plots are dropped into webapp/static/plots/ and
served back to the frontend. No business logic is reimplemented here - the
web layer only frames the same functions the CLI already drives.

Run (from anywhere):
    uvicorn app:app --host 0.0.0.0 --port 8000
(or:  python app.py   which launches uvicorn with uvicorn.run)
Then open http://localhost:8000/
"""
import os
import sys
import io
import uuid
import threading
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]          # /atlas_main
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")  # headless server; must be set before pyplot

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from atlas.paths import FEM_REFERENCE_DIR

PLOTS_DIR = ROOT / "webapp" / "static" / "plots"
FRONTEND_DIR = ROOT / "webapp" / "frontend"
FEM_REF_DIR = FEM_REFERENCE_DIR
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STORED_RESULTS = ROOT / "results"

app = FastAPI(title="ATLAS Web Console", version="1.0.0")
app.mount("/static/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/results", StaticFiles(directory=str(STORED_RESULTS)), name="results")

# ---------------------------------------------------------------------------
# Job store (in-memory): one-job-at-a-time execution with a queue.
# ---------------------------------------------------------------------------
JOBS: dict = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()          # serializes GPU-heavy training jobs

BENCHMARKS = {"cantilever", "l_bracket", "plate_hole"}


def auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def cap_log(log: str, n: int = 4000) -> str:
    if len(log) <= n:
        return log
    return "..." + log[-(n - 3):]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ForwardRequest(BaseModel):
    benchmark: str = Field(..., pattern="^(cantilever|l_bracket|plate_hole)$")
    epochs: int = Field(2000, ge=50, le=40000)
    n_interior: int = Field(800, ge=200, le=5000)
    use_lbfgs: bool = False
    warm_start_eb: bool = True          # cantilever only; ignored elsewhere
    prefit_epochs: int = Field(4000, ge=0, le=40000)
    use_mock: bool = False


class InverseRequest(BaseModel):
    """Inverse design: find the shape parameter (cantilever beam height / plate hole
    radius / L-bracket fillet radius) whose metric meets a target. `target_ratio`
    is the target as a multiple of the baseline design's metric (compliance for
    cantilever and plate, peak von Mises for the L-bracket unless `metric` says
    otherwise); leave it unset for the per-benchmark default."""
    benchmark: str = Field(..., pattern="^(cantilever|l_bracket|plate_hole)$")
    metric: Optional[str] = Field(None, pattern="^(compliance|peak_vm|volume)$")
    target_ratio: Optional[float] = Field(None, gt=0.0, le=10.0)
    forward_epochs: int = Field(2000, ge=100, le=10000)
    forward_lbfgs: int = Field(1500, ge=0, le=6000)
    n_outer: int = Field(12, ge=1, le=100)      # max solver iterations
    n_inner: int = Field(40, ge=5, le=2000)     # accepted for older clients; unused
    use_mock: bool = False


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------
def submit(kind: str, payload: dict) -> str:
    job_id = uuid.uuid4().hex[:8]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "log": "",
            "metrics": {},
            "plots": {},
            "created": datetime.now().isoformat(timespec="seconds"),
            "started": None,
            "finished": None,
            "error": None,
        }
    t = threading.Thread(target=_run_job, args=(job_id, kind, payload), daemon=True)
    t.start()
    return job_id


def _run_job(job_id: str, kind: str, payload: dict) -> None:
    buf = io.StringIO()
    with RUN_LOCK:  # single GPU consumer at a time
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "running"
            JOBS[job_id]["started"] = datetime.now().isoformat(timespec="seconds")
        try:
            with contextlib.redirect_stdout(buf):
                if kind == "forward":
                    _run_forward(job_id, payload)
                else:
                    _run_inverse(job_id, payload)
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "done"
                JOBS[job_id]["finished"] = datetime.now().isoformat(timespec="seconds")
        except Exception as exc:  # surface backend errors to the UI, not just logs
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["finished"] = datetime.now().isoformat(timespec="seconds")
                JOBS[job_id]["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with JOBS_LOCK:
                JOBS[job_id]["log"] = buf.getvalue()
                JOBS[job_id]["log"] = cap_log(JOBS[job_id]["log"])


def _run_forward(job_id: str, payload: dict) -> None:
    from atlas.atlas_main import train_forward_solver
    from atlas.validation import run_validation
    from atlas.visualization import (plot_fem_comparison, plot_displacement_field,
                                     plot_von_mises_stress, plot_deformed_shape,
                                     plot_training_convergence)

    bench = payload["benchmark"]
    device = auto_device()
    use_mock = payload.get("use_mock", False)
    warm = bench == "cantilever" and payload.get("warm_start_eb", True)

    model, trainer, sampler, physics = train_forward_solver(
        benchmark=bench,
        device=device,
        epochs=payload["epochs"],
        n_interior=payload["n_interior"],
        n_boundary=max(150, payload["n_interior"] // 5),
        n_interface=max(80, payload["n_interior"] // 10),
        # Default formulation is the deep-energy solver: `epochs` Adam steps
        # then a mandatory L-BFGS stage (Adam alone leaves a ~3% energy gap,
        # i.e. a visibly wrong field). n_interior / warm_start_eb /
        # prefit_epochs / use_lbfgs only apply to the residual formulation.
        use_lbfgs=True, lbfgs_steps=15, lbfgs_max_iter_per_step=100,
        warm_start_eb=warm,
        prefit_epochs=payload.get("prefit_epochs", 4000),
    )

    metrics = run_validation(
        model=model, sampler=sampler, physics=physics,
        benchmark=bench, device=device,
        use_mock=use_mock, data_dir=str(FEM_REF_DIR),
        r_excl=payload.get("r_excl"),  # None -> auto (factor x local point spacing)
    )
    if isinstance(metrics, dict):
        metrics["benchmark"] = bench
    else:  # run_validation may return JSON-serializable wrapper; guard anyway
        metrics = {"raw": str(metrics)}

    plot_fem_comparison(model, sampler, physics, benchmark=bench,
                        data_dir=str(FEM_REF_DIR), use_mock=use_mock,
                        r_excl=payload.get("r_excl"),
                        save_path=str(PLOTS_DIR / f"{job_id}_fem_comparison.png"))
    plot_displacement_field(model, sampler, physics, n_grid=90,
                            save_path=str(PLOTS_DIR / f"{job_id}_displacement.png"))
    plot_von_mises_stress(model, sampler, physics, n_grid=90,
                          save_path=str(PLOTS_DIR / f"{job_id}_vonmises.png"))
    plot_deformed_shape(model, sampler, n_grid=50,
                        save_path=str(PLOTS_DIR / f"{job_id}_deformed.png"))
    plot_training_convergence(
        trainer.history,
        save_path=str(PLOTS_DIR / f"{job_id}_convergence.png"),
        adam_epoch_count=trainer.adam_epoch_count,
    )

    plots = {
        "fem_comparison": f"/static/plots/{job_id}_fem_comparison.png",
        "displacement": f"/static/plots/{job_id}_displacement.png",
        "vonmises": f"/static/plots/{job_id}_vonmises.png",
        "deformed": f"/static/plots/{job_id}_deformed.png",
        "convergence": f"/static/plots/{job_id}_convergence.png",
    }
    with JOBS_LOCK:
        JOBS[job_id]["metrics"] = metrics
        JOBS[job_id]["plots"] = plots


def _run_inverse(job_id: str, payload: dict) -> None:
    from atlas.atlas_main import run_inverse_design_all, design_result_to_json

    bench = payload["benchmark"]
    device = auto_device()
    out_dir = PLOTS_DIR / f"{job_id}_inv"
    res = run_inverse_design_all(
        bench, device, metric=payload.get("metric"),
        target_ratio=payload.get("target_ratio"), out_dir=out_dir,
        forward_epochs=payload["forward_epochs"], forward_lbfgs=payload["forward_lbfgs"],
        max_iter=max(payload["n_outer"], 8), verbose=True,
    )
    data = design_result_to_json(res)
    hist = [r for r in data["history"] if r["tag"] not in ("probe", "return")]
    metric = data["metric"]
    metrics = dict(
        data,
        # keys the current frontend already reads
        param=data["param_name"],
        initial=data["param_baseline"],
        final=data["p_star"],
        radius_history=[r["param"] for r in hist],
        compliance_history=[r[metric] for r in hist],
        n_outer=len(hist),
    )
    base = f"/static/plots/{job_id}_inv/{bench}"
    plots = {
        "inverse_history": f"{base}_history.png",
        "design": f"{base}_design.png",
    }
    with JOBS_LOCK:
        JOBS[job_id]["metrics"] = metrics
        JOBS[job_id]["plots"] = plots


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


BENCH_TITLES = {"cantilever": "Cantilever beam", "l_bracket": "L-shaped bracket",
                "plate_hole": "Plate with circular hole"}


def _load(*parts):
    """Read one stored JSON result; None when that file was never produced."""
    import json
    p = STORED_RESULTS.joinpath(*parts)
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _img(*parts):
    p = STORED_RESULTS.joinpath(*parts)
    return "/results/" + "/".join(parts) if p.exists() else None


@app.get("/api/results")
def stored_results():
    """Everything the Results view shows: the finished study in results/ (no training)."""
    out = {"benchmarks": {}, "figures": {}, "targets": {"displacement": 0.05, "von_mises": 0.10}}
    for b in BENCH_TITLES:
        xf = _load("atlas_xpinn_energy", "forward", f"{b}_xpinn_energy.json")
        af = _load("atlas_energy", "forward", f"{b}_validation.json")
        dd = _load("comparison", "data", f"{b}_deepxde.json")
        dc = _load("comparison", "data", f"{b}_deepxde_configured.json")
        xs = _load("atlas_xpinn_energy", "repeatability", b, "summary.json")
        as_ = _load("atlas_energy", "repeatability", b, "summary.json")
        inv = _load("atlas_xpinn_energy", "inverse_design", f"{b}_fem_check.json")
        seeds = lambda d: None if not d else {
            "disp_mean": d["summary"]["relative_L2_displacement_mean"],
            "disp_std": d["summary"]["relative_L2_displacement_std"],
            "vm_mean": d["summary"]["von_mises_l2_excl_mean"],
            "vm_std": d["summary"]["von_mises_l2_excl_std"],
            "n": len(d["per_seed"])}
        solver = lambda d, disp="relative_L2_displacement", vm="von_mises_l2_excl": None if not d else {
            "disp": d.get(disp), "vm": d.get(vm), "seconds": d.get("train_seconds"),
            "params": d.get("params")}
        out["benchmarks"][b] = {
            "title": BENCH_TITLES[b],
            "solvers": {
                "atlas_xpinn": solver(xf), "atlas": solver(af), "deepxde": solver(dd),
                "deepxde_configured": solver(dc)},
            "seeds": {"atlas_xpinn": seeds(xs), "atlas": seeds(as_)},
            "inverse": None if not inv else {
                "param": inv.get("metric"), "p_star": inv.get("p_atlas"),
                "error_vs_target": inv.get("metric_error_vs_target"),
                "within_10pct": inv.get("within_10pct")},
            "plots": {k: v for k, v in {
                "ATLAS-XPINN vs FEM": _img("atlas_xpinn_energy", "forward", f"{b}_fem_comparison.png"),
                "Displacement field": _img("atlas_xpinn_energy", "forward", f"{b}_displacement.png"),
                "von Mises stress": _img("atlas_xpinn_energy", "forward", f"{b}_stress.png"),
                "Deformed shape": _img("atlas_xpinn_energy", "forward", f"{b}_deformed.png"),
                "Training convergence": _img("atlas_xpinn_energy", "forward", f"{b}_convergence.png"),
                "Five-seed study": _img("atlas_xpinn_energy", "repeatability", b, "summary.png"),
                "Inverse design": _img("atlas_xpinn_energy", "inverse_design", f"{b}_design.png"),
                "FEM check of the design": _img("atlas_xpinn_energy", "inverse_design", f"{b}_fem_check.png"),
                "All solvers vs FEM": _img("comparison", "deepxde_vs_atlas_xpinn", f"{b}_comparison.png"),
            }.items() if v},
        }
    out["figures"] = {k: v for k, v in {
        "Error summary (all benchmarks)": _img("comparison", "deepxde_vs_atlas_xpinn", "comparison_summary.png"),
        "Error vs training time": _img("comparison", "convergence_timing", "convergence_vs_time.png"),
        "Error vs iteration": _img("comparison", "convergence_timing", "convergence_vs_iteration.png"),
        "Processing time": _img("comparison", "convergence_timing", "processing_time.png"),
    }.items() if v}
    return out


@app.get("/api/health")
def health():
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        torch_ver = torch.__version__
    except Exception as e:
        dev, torch_ver = "unknown", str(e)
    return {
        "status": "ok",
        "device": dev,
        "torch": torch_ver,
        "active_jobs": sum(1 for j in JOBS.values() if j["status"] in ("queued", "running")),
        "total_jobs": len(JOBS),
    }


@app.post("/api/forward")
def forward(req: ForwardRequest):
    job_id = submit("forward", req.dict())
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/inverse")
def inverse(req: InverseRequest):
    job_id = submit("inverse", req.dict())
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs")
def list_jobs(limit: int = 25):
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)[:max(1, limit)]
        return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return job


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)