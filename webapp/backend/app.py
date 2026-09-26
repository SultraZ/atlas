"""Run from the project root:
    cd webapp/backend && python app.py
then open http://localhost:8000/
"""

import io
import json
import sys
import threading
import time
import traceback
import uuid
from contextlib import redirect_stdout
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from atlas.paths import FEM_REFERENCE_DIR

PLOTS_DIR = ROOT / "webapp" / "static" / "plots"
FRONTEND_DIR = ROOT / "webapp" / "frontend"
STORED_RESULTS = ROOT / "results"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ATLAS Web Console", version="2.0.0")
app.mount("/static/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/results", StaticFiles(directory=str(STORED_RESULTS)), name="results")

JOBS: dict = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()
LOG_CAP = 20000
HIDDEN_LAYERS, NEURONS, SEED, LBFGS_ITERS = 4, 64, 42, 2000

PAPER = ["cantilever", "l_bracket", "plate_hole"]
MULTI = ["steel_insert", "bimaterial", "filled_slot"]
META = {
    "cantilever": dict(title="Cantilever beam", geometry="4 × 1 beam"),
    "l_bracket": dict(title="L-shaped bracket", geometry="2 × 2 L-shape"),
    "plate_hole": dict(title="Plate with circular hole", geometry="2 × 2 quarter plate"),
    "steel_insert": dict(title="Steel insert", geometry="aluminium plate, steel insert",
                         inside="Insert side", outside="Plate side"),
    "bimaterial": dict(title="Soft inclusion", geometry="steel plate, 15:1 inclusion",
                       inside="Inclusion side", outside="Plate side"),
    "filled_slot": dict(title="Epoxy-filled slot", geometry="steel plate, elliptical filler",
                        inside="Filler side", outside="Plate side"),
}
MULTI_DIR = {"steel_insert": ("multi_material", "steel_insert"),
             "bimaterial": ("multi_material", "bimaterial", "c15"),
             "filled_slot": ("multi_material", "filled_slot")}
MULTI_REF = {"steel_insert": "steel_insert", "bimaterial": "bimaterial_c15", "filled_slot": "filled_slot"}
BIMATERIAL_CONTRAST = 15.0
INVERSE_OBJECTIVE = {"compliance": "Compliance target", "peak_vm": "Peak von Mises target", "volume": "Volume target"}


def auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load(*parts):
    try:
        return json.loads(STORED_RESULTS.joinpath(*parts).read_text())
    except Exception:
        return None


def _img(*parts):
    return "/results/" + "/".join(parts) if STORED_RESULTS.joinpath(*parts).exists() else None


def _multi_module(bench):
    import importlib
    return importlib.import_module(f"atlas.benchmarks.{bench}")


def _multi_build(bench, formulation, device):
    mod = _multi_module(bench)
    args = (BIMATERIAL_CONTRAST,) if bench == "bimaterial" else ()
    if formulation == "energy_xpinn":
        return mod.build_decomposed(*args, hidden_layers=HIDDEN_LAYERS, neurons=NEURONS, device=device)
    return mod.build_single(*args, hidden_layers=HIDDEN_LAYERS, neurons=NEURONS, device=device)


@lru_cache(maxsize=None)
def subdomain_count(bench: str) -> int:
    if bench in MULTI:
        return len(_multi_build(bench, "energy_xpinn", "cpu")[0].subnets)
    from atlas.energy_xpinn import E_MOD, NU, STRESS_SCALE, build_xpinn_energy
    from atlas.physics import PlaneStressElasticity
    phys = PlaneStressElasticity(E=E_MOD / STRESS_SCALE, nu=NU)
    return int(build_xpinn_energy(bench, phys, device="cpu")[2]["n_sub"])


@lru_cache(maxsize=None)
def materials(bench: str):
    if bench in PAPER:
        from atlas.energy_xpinn import E_MOD, NU
        return [dict(name="Steel", E=E_MOD, nu=NU)]
    mod = _multi_module(bench)
    if bench == "steel_insert":
        return [dict(name="Aluminium plate", E=mod.E_AL, nu=mod.NU_AL), dict(name="Steel insert", E=mod.E_ST, nu=mod.NU_ST)]
    if bench == "bimaterial":
        return [dict(name="Steel plate", E=mod.E_MOD, nu=mod.NU),
                dict(name="Soft inclusion", E=mod.e_inclusion(BIMATERIAL_CONTRAST), nu=mod.NU)]
    return [dict(name="Steel plate", E=mod.E_ST, nu=mod.NU_ST), dict(name="Epoxy filler", E=mod.E_EP, nu=mod.NU_EP)]


@lru_cache(maxsize=None)
def saved_param_count(path: str) -> Optional[int]:
    try:
        import torch
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.state_dict() if hasattr(sd, "state_dict") else sd
        return int(sum(v.numel() for k, v in sd.items() if hasattr(v, "numel") and not k.split(".")[-1].startswith("_")))
    except Exception:
        return None


def bench_meta(bench: str) -> dict:
    m = dict(META[bench], key=bench, group="paper" if bench in PAPER else "multi",
             subdomains=subdomain_count(bench), materials=materials(bench),
             network=dict(hidden_layers=HIDDEN_LAYERS, neurons=NEURONS, activation="tanh"), inverse=None)
    if bench in PAPER:
        inv = _load("atlas_xpinn_energy", "inverse_design", f"{bench}_inverse.json") or {}
        m["inverse"] = dict(param=inv.get("param_name"), objective=INVERSE_OBJECTIVE.get(inv.get("metric"), inv.get("metric")))
    return m


class JobLog(io.TextIOBase):
    def __init__(self, job_id: str):
        self.job_id = job_id

    def writable(self):
        return True

    def write(self, s):
        with JOBS_LOCK:
            job = JOBS[self.job_id]
            log = job["log"] + s
            job["log"] = log if len(log) <= LOG_CAP else "…" + log[-(LOG_CAP - 1):]
        return len(s)


def set_stage(job_id: str, stage: str) -> None:
    with JOBS_LOCK:
        JOBS[job_id]["stage"] = stage


class ForwardRequest(BaseModel):
    benchmark: str = Field(..., pattern="^(cantilever|l_bracket|plate_hole|steel_insert|bimaterial|filled_slot)$")
    formulation: str = Field("energy_xpinn", pattern="^(energy|energy_xpinn)$")
    epochs: int = Field(2000, ge=50, le=40000)
    n_interior: int = Field(800, ge=200, le=5000)
    warm_start_eb: bool = True
    prefit_epochs: int = Field(4000, ge=0, le=40000)
    use_mock: bool = False


class InverseRequest(BaseModel):
    benchmark: str = Field(..., pattern="^(cantilever|l_bracket|plate_hole)$")
    metric: Optional[str] = Field(None, pattern="^(compliance|peak_vm|volume)$")
    target_ratio: Optional[float] = Field(None, gt=0.0, le=10.0)
    forward_epochs: int = Field(2000, ge=100, le=10000)
    forward_lbfgs: int = Field(1500, ge=0, le=6000)
    n_outer: int = Field(12, ge=1, le=100)
    n_inner: int = Field(40, ge=5, le=2000)
    use_mock: bool = False


def submit(kind: str, payload: dict) -> str:
    job_id = uuid.uuid4().hex[:8]
    with JOBS_LOCK:
        JOBS[job_id] = dict(id=job_id, kind=kind, benchmark=payload["benchmark"],
                            formulation=payload.get("formulation", "energy_xpinn"), status="queued", stage="queued",
                            log="", metrics={}, plots={}, created=datetime.now().isoformat(timespec="seconds"),
                            started=None, finished=None, error=None)
    threading.Thread(target=_run_job, args=(job_id, kind, payload), daemon=True).start()
    return job_id


def _run_job(job_id: str, kind: str, payload: dict) -> None:
    with RUN_LOCK:
        with JOBS_LOCK:
            JOBS[job_id].update(status="running", stage="setup", started=datetime.now().isoformat(timespec="seconds"))
        log = JobLog(job_id)
        try:
            with redirect_stdout(log):
                if kind == "inverse":
                    _run_inverse(job_id, payload)
                elif payload["benchmark"] in MULTI:
                    _run_multi(job_id, payload)
                else:
                    _run_forward(job_id, payload)
            status, error = "done", None
        except Exception as exc:
            log.write("\n" + traceback.format_exc())
            status, error = "error", f"{type(exc).__name__}: {exc}"
        with JOBS_LOCK:
            JOBS[job_id].update(status=status, error=error, stage="finished" if status == "done" else JOBS[job_id]["stage"],
                                finished=datetime.now().isoformat(timespec="seconds"))


def _param_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _run_forward(job_id: str, payload: dict) -> None:
    from atlas.atlas_main import train_forward_solver
    from atlas.validation import run_validation
    from atlas.visualization import plot_fem_comparison, plot_training_convergence

    bench, device = payload["benchmark"], auto_device()
    set_stage(job_id, "train")
    t0 = time.perf_counter()
    model, trainer, sampler, physics = train_forward_solver(
        benchmark=bench, device=device, epochs=payload["epochs"], formulation=payload["formulation"],
        n_interior=payload["n_interior"], n_boundary=max(150, payload["n_interior"] // 5),
        n_interface=max(80, payload["n_interior"] // 10), use_lbfgs=True,
        lbfgs_steps=LBFGS_ITERS // 100, lbfgs_max_iter_per_step=100,
        warm_start_eb=bench == "cantilever" and payload.get("warm_start_eb", True),
        prefit_epochs=payload.get("prefit_epochs", 4000), seed=SEED)
    seconds = time.perf_counter() - t0
    set_stage(job_id, "predict")
    params = _param_count(model)
    set_stage(job_id, "validate")
    metrics = run_validation(model=model, sampler=sampler, physics=physics, benchmark=bench, device=device,
                             use_mock=payload.get("use_mock", False), data_dir=str(FEM_REFERENCE_DIR))
    metrics = dict(metrics, benchmark=bench, formulation=payload["formulation"], params=params,
                   train_seconds=seconds, seed=SEED,
                   subdomains=subdomain_count(bench) if payload["formulation"] == "energy_xpinn" else 1)
    set_stage(job_id, "plots")
    plot_fem_comparison(model, sampler, physics, benchmark=bench, data_dir=str(FEM_REFERENCE_DIR),
                        use_mock=payload.get("use_mock", False),
                        save_path=str(PLOTS_DIR / f"{job_id}_fem_comparison.png"))
    plot_training_convergence(trainer.history, save_path=str(PLOTS_DIR / f"{job_id}_convergence.png"),
                              adam_epoch_count=trainer.adam_epoch_count)
    with JOBS_LOCK:
        JOBS[job_id]["metrics"] = metrics
        JOBS[job_id]["plots"] = {"fem_comparison": f"/static/plots/{job_id}_fem_comparison.png",
                                 "convergence": f"/static/plots/{job_id}_convergence.png"}


def _rel(a, b):
    return float(np.sqrt(np.sum((a - b) ** 2) / np.sum(b ** 2)))


def _run_multi(job_id: str, payload: dict) -> None:
    import torch
    from atlas.energy_solver import history_for_plot, train_energy
    from atlas.energy_xpinn import train_decomposed
    from atlas.visualization import plot_training_convergence

    bench, device, form = payload["benchmark"], auto_device(), payload["formulation"]
    mod, ref = _multi_module(bench), MULTI_REF[bench]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model, prob = _multi_build(bench, form, device)
    print(f"{META[bench]['title']}: {_param_count(model)} parameters, {prob.n_points} quadrature points, {device}")
    set_stage(job_id, "train")
    t0 = time.perf_counter()
    if form == "energy_xpinn":
        hist = train_decomposed(model, prob, epochs=payload["epochs"], lbfgs_per_stage=LBFGS_ITERS,
                                gammas=(30.0,), verbose=True)
    else:
        hist = train_energy(model, prob, epochs=payload["epochs"], lbfgs_iters=LBFGS_ITERS, lbfgs_chunk=100)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    set_stage(job_id, "predict")
    R = FEM_REFERENCE_DIR
    load = lambda tag, f: np.load(R / f"{tag}_{f}.npy").ravel() if f != "coords" else np.load(R / f"{tag}_coords.npy")
    coords, u_f, v_f, vm_f = (load(ref, f) for f in ("coords", "u", "v", "vonmises"))
    roi, vmr_f = load(ref + "_roi", "coords"), load(ref + "_roi", "vonmises")
    extra = (BIMATERIAL_CONTRAST,) if bench == "bimaterial" else ()
    up, vp, vm = mod.predict(model, coords, *extra, device)
    _, _, vmr = mod.predict(model, roi, *extra, device)

    set_stage(job_id, "validate")
    inside_fn = getattr(mod, "in_filler", None) or mod.in_inclusion
    inside = inside_fn(roi[:, 0], roi[:, 1])
    fem_meta = json.loads((R / f"{ref}_meta.json").read_text())
    C = mod.compliance(model, device)
    metrics = dict(
        benchmark=bench, formulation=form, params=_param_count(model), train_seconds=seconds, seed=SEED,
        subdomains=len(model.subnets) if hasattr(model, "subnets") else 1,
        relative_L2_displacement=float(np.sqrt(np.sum((up - u_f) ** 2 + (vp - v_f) ** 2) / np.sum(u_f ** 2 + v_f ** 2))),
        relative_L2_u=_rel(up, u_f), relative_L2_v=_rel(vp, v_f), von_mises_l2=_rel(vm, vm_f),
        band_vm_l2_inside=_rel(vmr[inside], vmr_f[inside]), band_vm_l2_outside=_rel(vmr[~inside], vmr_f[~inside]),
        compliance_error=float((C - fem_meta["compliance"]) / fem_meta["compliance"]),
        interface_jump_sq=prob.last.get("jump_u") if form == "energy_xpinn" else None)
    print(f"displacement L2 {metrics['relative_L2_displacement']:.3%}   von Mises L2 {metrics['von_mises_l2']:.3%}   "
          f"interface band {metrics['band_vm_l2_inside']:.3%} / {metrics['band_vm_l2_outside']:.3%}")

    set_stage(job_id, "plots")
    _plot_multi(bench, coords, vm, vm_f, inside_fn, PLOTS_DIR / f"{job_id}_fem_comparison.png")
    plot_training_convergence(history_for_plot(hist), save_path=str(PLOTS_DIR / f"{job_id}_convergence.png"),
                              adam_epoch_count=payload["epochs"])
    with JOBS_LOCK:
        JOBS[job_id]["metrics"] = metrics
        JOBS[job_id]["plots"] = {"fem_comparison": f"/static/plots/{job_id}_fem_comparison.png",
                                 "convergence": f"/static/plots/{job_id}_convergence.png"}


def _plot_multi(bench, coords, vm, vm_f, inside_fn, path):
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    tri = Triangulation(coords[:, 0], coords[:, 1])
    cap = np.percentile(vm_f, 99.8)
    ind = inside_fn(coords[:, 0], coords[:, 1]).astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    panels = [(vm_f / 1e6, "FEM von Mises [MPa]", "inferno", cap / 1e6),
              (vm / 1e6, "ATLAS-XPINN von Mises [MPa]", "inferno", cap / 1e6),
              (np.abs(vm - vm_f) / cap * 100, "|error| / FEM 99.8th pct [%]", "magma", 10.0)]
    for ax, (z, title, cmap, hi) in zip(axes, panels):
        cs = ax.tricontourf(tri, np.clip(z, 0, hi), levels=np.linspace(0, hi, 30), cmap=cmap)
        ax.tricontour(tri, ind, levels=[0.5], colors="w", linewidths=0.8, linestyles="--")
        fig.colorbar(cs, ax=ax, shrink=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{META[bench]['title']}: ATLAS-XPINN vs FEM", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _run_inverse(job_id: str, payload: dict) -> None:
    from atlas.atlas_main import design_result_to_json, run_inverse_design_all

    bench, device = payload["benchmark"], auto_device()
    set_stage(job_id, "train")
    t0 = time.perf_counter()
    res = run_inverse_design_all(bench, device, metric=payload.get("metric"), target_ratio=payload.get("target_ratio"),
                                 out_dir=PLOTS_DIR / f"{job_id}_inv", forward_epochs=payload["forward_epochs"],
                                 forward_lbfgs=payload["forward_lbfgs"], max_iter=max(payload["n_outer"], 8),
                                 verbose=True, decomposed=True)
    set_stage(job_id, "plots")
    data = design_result_to_json(res)
    hist = [r for r in data["history"] if r["tag"] not in ("probe", "return")]
    metrics = dict(data, param=data["param_name"], initial=data["param_baseline"], final=data["p_star"],
                   objective=INVERSE_OBJECTIVE.get(data["metric"], data["metric"]), n_outer=len(hist),
                   train_seconds=time.perf_counter() - t0, formulation="energy_xpinn")
    base = f"/static/plots/{job_id}_inv/{bench}"
    with JOBS_LOCK:
        JOBS[job_id]["metrics"] = metrics
        JOBS[job_id]["plots"] = {"design": f"{base}_design.png", "inverse_history": f"{base}_history.png"}


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api/benchmarks")
def benchmarks():
    return {"benchmarks": [bench_meta(b) for b in PAPER + MULTI], "lbfgs_iters": LBFGS_ITERS, "seed": SEED}


def _paper_results(b: str) -> dict:
    xf = _load("atlas_xpinn_energy", "forward", f"{b}_xpinn_energy.json")
    af = _load("atlas_energy", "forward", f"{b}_validation.json")
    dd = _load("comparison", "data", f"{b}_deepxde.json")
    dc = _load("comparison", "data", f"{b}_deepxde_configured.json")
    inv = _load("atlas_xpinn_energy", "inverse_design", f"{b}_fem_check.json")

    def solver(d, params=None):
        if not d:
            return None
        cfg = d.get("config") if isinstance(d.get("config"), dict) else {}
        return dict(disp=d.get("relative_L2_displacement"), vm=d.get("von_mises_l2_excl"),
                    seconds=d.get("train_seconds"), params=d.get("params") or params, library=cfg.get("library"))

    def seeds(d):
        if not d:
            return None
        s, runs = d["summary"], d["per_seed"]
        return dict(disp_mean=s["relative_L2_displacement_mean"], disp_std=s["relative_L2_displacement_std"],
                    vm_mean=s["von_mises_l2_excl_mean"], vm_std=s["von_mises_l2_excl_std"],
                    seeds=[r["seed"] for r in runs], n=len(runs),
                    n_pass_both=sum(1 for r in runs if r["displacement_within_5pct_target"] and r["von_mises_l2_excl_within_10pct"]))

    return dict(
        solvers=dict(atlas_xpinn=solver(xf),
                     atlas=solver(af, saved_param_count(str(STORED_RESULTS / "atlas_energy" / "forward" / f"{b}_model.pt"))),
                     deepxde=solver(dd), deepxde_configured=solver(dc)),
        seeds=dict(atlas_xpinn=seeds(_load("atlas_xpinn_energy", "repeatability", b, "summary.json")),
                   atlas=seeds(_load("atlas_energy", "repeatability", b, "summary.json"))),
        inverse=None if not inv else dict(metric=inv.get("metric"), value=inv.get("p_atlas"),
                                          error_vs_target=inv.get("metric_error_vs_target"),
                                          within_10pct=inv.get("within_10pct"), target_ratio=inv.get("target_ratio")),
        exclusion=dict(factor=(af or {}).get("von_mises_exclusion_factor"),
                       radius=(af or {}).get("von_mises_exclusion_radius")),
        plots={k: v for k, v in dict(
            fem_comparison=_img("atlas_xpinn_energy", "forward", f"{b}_fem_comparison.png"),
            displacement=_img("atlas_xpinn_energy", "forward", f"{b}_displacement.png"),
            stress=_img("atlas_xpinn_energy", "forward", f"{b}_stress.png"),
            convergence=_img("atlas_xpinn_energy", "forward", f"{b}_convergence.png"),
            deformed=_img("atlas_xpinn_energy", "forward", f"{b}_deformed.png"),
            all_solvers=_img("comparison", "deepxde_vs_atlas_xpinn", f"{b}_comparison.png"),
        ).items() if v})


def _multi_results(b: str) -> dict:
    parts = MULTI_DIR[b]
    root = STORED_RESULTS.joinpath(*parts)

    def run(solver, seed=42):
        d = _load(*parts, solver, f"seed{seed}.json")
        if not d:
            return None
        f = d["final"]
        return dict(disp=f["disp_vec_l2"], vm=f["vm_l2"], band_inside=f["band_vm_l2_inclusion"],
                    band_outside=f["band_vm_l2_matrix"], compliance_error=f["compliance_err"],
                    seconds=d["train_seconds"], params=d["params"])

    def finals(solver):
        return {json.loads(p.read_text())["seed"]: json.loads(p.read_text())["final"]
                for p in sorted((root / solver).glob("seed*.json"))}

    def seeds(solver):
        runs = finals(solver)
        if len(runs) < 2:
            return None
        d = np.array([r["disp_vec_l2"] for r in runs.values()])
        v = np.array([r["vm_l2"] for r in runs.values()])
        return dict(disp_mean=float(d.mean()), disp_std=float(d.std()), vm_mean=float(v.mean()), vm_std=float(v.std()),
                    seeds=sorted(runs), n=len(runs), n_pass_both=int(np.sum((d < 0.05) & (v < 0.10))))

    x, w = finals("xpinn"), finals("atlas_wide")
    common = sorted(set(x) & set(w))
    return dict(
        solvers=dict(atlas_xpinn=run("xpinn"), atlas=run("atlas"), atlas_wide=run("atlas_wide")),
        seeds=dict(atlas_xpinn=seeds("xpinn"), atlas_wide=seeds("atlas_wide")),
        paired=dict(n=len(common),
                    xpinn_lower_both=sum(1 for s in common if x[s]["disp_vec_l2"] < w[s]["disp_vec_l2"] and x[s]["vm_l2"] < w[s]["vm_l2"]),
                    xpinn_lower_disp=sum(1 for s in common if x[s]["disp_vec_l2"] < w[s]["disp_vec_l2"]),
                    xpinn_lower_vm=sum(1 for s in common if x[s]["vm_l2"] < w[s]["vm_l2"])),
        plots={k: v for k, v in dict(fields=_img(*parts, "fields.png"), profiles=_img(*parts, "profiles.png"),
                                     convergence=_img(*parts, "convergence.png")).items() if v})


@app.get("/api/results")
def stored_results():
    out = {"benchmarks": {}, "order": PAPER + MULTI, "targets": {"displacement": 0.05, "von_mises": 0.10}}
    for b in PAPER + MULTI:
        meta = bench_meta(b)
        res = _paper_results(b) if b in PAPER else _multi_results(b)
        if res.get("inverse") and meta["inverse"]:
            res["inverse"].update(meta["inverse"])
        out["benchmarks"][b] = dict(meta, **res)
    return out


@app.get("/api/health")
def health():
    try:
        import torch
        dev, torch_ver = ("cuda" if torch.cuda.is_available() else "cpu"), torch.__version__
    except Exception as e:
        dev, torch_ver = "unknown", str(e)
    with JOBS_LOCK:
        active = sum(1 for j in JOBS.values() if j["status"] in ("queued", "running"))
        total = len(JOBS)
    return {"status": "ok", "device": dev, "torch": torch_ver, "active_jobs": active, "total_jobs": total}


@app.post("/api/forward")
def forward(req: ForwardRequest):
    return {"job_id": submit("forward", req.dict()), "status": "queued"}


@app.post("/api/inverse")
def inverse(req: InverseRequest):
    return {"job_id": submit("inverse", req.dict()), "status": "queued"}


@app.get("/api/jobs")
def list_jobs(limit: int = 25):
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)[:max(1, limit)]
        return {"jobs": [{k: v for k, v in j.items() if k != "log"} for j in jobs]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        return dict(job)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
