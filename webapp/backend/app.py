"""Run from the project root:
    cd webapp/backend && python app.py
then open http://localhost:8000/
"""

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from contextlib import contextmanager, redirect_stdout
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

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

from atlas import live
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
MAX_LBFGS = 4 * LBFGS_ITERS

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
             network=dict(hidden_layers=HIDDEN_LAYERS, neurons=NEURONS, activation="tanh"), inverse=None,
             params=live.PARAMS[bench])
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
    lbfgs: int = Field(LBFGS_ITERS, ge=100, le=MAX_LBFGS)
    n_interior: int = Field(800, ge=200, le=5000)
    warm_start_eb: bool = True
    prefit_epochs: int = Field(4000, ge=0, le=40000)
    use_mock: bool = False
    params: Optional[Dict[str, float]] = None


class InverseRequest(BaseModel):
    benchmark: str = Field(..., pattern="^(cantilever|l_bracket|plate_hole)$")
    metric: Optional[str] = Field(None, pattern="^(compliance|peak_vm|volume)$")
    target_ratio: Optional[float] = Field(None, gt=0.0, le=10.0)
    forward_epochs: int = Field(2000, ge=100, le=10000)
    forward_lbfgs: int = Field(1500, ge=0, le=6000)
    n_outer: int = Field(12, ge=1, le=100)
    n_inner: int = Field(40, ge=5, le=2000)
    use_mock: bool = False
    params: Optional[Dict[str, float]] = None


def submit(kind: str, payload: dict) -> str:
    job_id = uuid.uuid4().hex[:8]
    bench = payload["benchmark"]
    payload["custom"] = not live.is_default(bench, payload.get("params"))
    with JOBS_LOCK:
        JOBS[job_id] = dict(id=job_id, kind=kind, benchmark=bench, custom=payload["custom"],
                            settings=live.values(bench, payload.get("params")),
                            formulation=payload.get("formulation", "energy_xpinn"),
                            budget=dict(epochs=payload["epochs"], lbfgs=payload["lbfgs"]) if kind == "forward" else None,
                            status="queued", stage="queued",
                            log="", metrics={}, plots={}, progress=None,
                            created=datetime.now().isoformat(timespec="seconds"),
                            started=None, finished=None, error=None)
    threading.Thread(target=_run_job, args=(job_id, kind, payload), daemon=True).start()
    return job_id


def _run_job(job_id: str, kind: str, payload: dict) -> None:
    with RUN_LOCK:
        with JOBS_LOCK:
            JOBS[job_id].update(status="running", stage="setup", started=datetime.now().isoformat(timespec="seconds"))
        log = JobLog(job_id)
        try:
            with redirect_stdout(log), record_energy(job_id):
                if kind == "inverse":
                    _run_inverse(job_id, payload)
                elif payload["custom"]:
                    _run_custom(job_id, payload)
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


@contextmanager
def record_energy(job_id: str):
    from atlas.energy_solver import EnergyProblem
    from atlas.energy_xpinn import DecomposedEnergyProblem
    originals = {cls: cls.__call__ for cls in (EnergyProblem, DecomposedEnergyProblem)}
    points, count = [], [0]

    def publish():
        step = max(1, len(points) // 300)
        sample = points[::step] + (points[-1:] if points and (len(points) - 1) % step else [])
        with JOBS_LOCK:
            JOBS[job_id]["progress"] = dict(evaluations=count[0], points=sample)

    def wrap(orig):
        def call(self, net):
            out = orig(self, net)
            count[0] += 1
            points.append([count[0], float(out[0].detach())])
            if count[0] % 25 == 0:
                publish()
            return out
        return call

    for cls, orig in originals.items():
        cls.__call__ = wrap(orig)
    try:
        yield
    finally:
        for cls, orig in originals.items():
            cls.__call__ = orig
        publish()


def _param_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _run_forward(job_id: str, payload: dict) -> None:
    from atlas.atlas_main import train_forward_solver
    from atlas.validation import run_validation
    from atlas.visualization import plot_deformed_shape, plot_fem_comparison, plot_training_convergence

    bench, device = payload["benchmark"], auto_device()
    set_stage(job_id, "train")
    t0 = time.perf_counter()
    model, trainer, sampler, physics = train_forward_solver(
        benchmark=bench, device=device, epochs=payload["epochs"], formulation=payload["formulation"],
        n_interior=payload["n_interior"], n_boundary=max(150, payload["n_interior"] // 5),
        n_interface=max(80, payload["n_interior"] // 10), use_lbfgs=True,
        lbfgs_steps=payload["lbfgs"] // 100, lbfgs_max_iter_per_step=100,
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
    plot_deformed_shape(model, sampler, n_grid=50 if bench == "cantilever" else 70,
                        save_path=str(PLOTS_DIR / f"{job_id}_deformed.png"))
    with JOBS_LOCK:
        JOBS[job_id]["metrics"] = metrics
        JOBS[job_id]["plots"] = {"fem_comparison": f"/static/plots/{job_id}_fem_comparison.png",
                                 "deformed": f"/static/plots/{job_id}_deformed.png",
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
        hist = train_decomposed(model, prob, epochs=payload["epochs"], lbfgs_per_stage=payload["lbfgs"],
                                gammas=(30.0,), verbose=True)
    else:
        hist = train_energy(model, prob, epochs=payload["epochs"], lbfgs_iters=payload["lbfgs"], lbfgs_chunk=100)
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
    _plot_live(bench, None, coords, up, vp, vm, u_f, v_f, vm_f, PLOTS_DIR / f"{job_id}_fem_comparison.png")
    _plot_deformed(bench, None, coords, up, vp, vm, PLOTS_DIR / f"{job_id}_deformed.png")
    plot_training_convergence(history_for_plot(hist), save_path=str(PLOTS_DIR / f"{job_id}_convergence.png"),
                              adam_epoch_count=payload["epochs"])
    with JOBS_LOCK:
        JOBS[job_id]["metrics"] = metrics
        JOBS[job_id]["plots"] = {"fem_comparison": f"/static/plots/{job_id}_fem_comparison.png",
                                 "deformed": f"/static/plots/{job_id}_deformed.png",
                                 "convergence": f"/static/plots/{job_id}_convergence.png"}


def _fem_reference(bench: str, params) -> dict:
    spec = live.fem_spec(bench, params)
    with tempfile.TemporaryDirectory() as tmp:
        spec_path, out_path = os.path.join(tmp, "spec.json"), os.path.join(tmp, "ref.npz")
        json.dump(spec, open(spec_path, "w"))
        env = dict(os.environ, PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
        run = subprocess.run([sys.executable, str(ROOT / "fem" / "fem_live.py"), spec_path, out_path],
                             capture_output=True, text=True, env=env, timeout=900, cwd=str(ROOT))
        for line in run.stdout.splitlines():
            if "JIT" not in line:
                print("  " + line)
        if run.returncode != 0:
            raise RuntimeError("FEniCS reference failed: " + (run.stderr.strip().splitlines() or ["no output"])[-1])
        with np.load(out_path) as z:
            return {k: z[k] for k in z.files}


def _run_custom(job_id: str, payload: dict) -> None:
    import torch
    from atlas.energy_solver import history_for_plot
    from atlas.energy_xpinn import train_decomposed
    from atlas.validation import default_exclusion_radius
    from atlas.visualization import plot_training_convergence

    bench, device, params = payload["benchmark"], auto_device(), payload.get("params") or {}
    shown = live.values(bench, params)
    units = {s["key"]: s["unit"] for s in live.PARAMS[bench]}
    print(f"{META[bench]['title']}, custom settings: " + ", ".join(f"{k}={v:g}{(' ' + units[k]) if units[k] else ''}"
                                                              for k, v in shown.items()))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model, prob = live.build(bench, params, device, HIDDEN_LAYERS, NEURONS)
    print(f"ATLAS-XPINN: {len(model.subnets)} subnetworks, {_param_count(model)} parameters, "
          f"{prob.n_points} quadrature points, {device}")
    set_stage(job_id, "train")
    t0 = time.perf_counter()
    hist = train_decomposed(model, prob, epochs=payload["epochs"], lbfgs_per_stage=payload["lbfgs"], gammas=(30.0,), verbose=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    set_stage(job_id, "predict")
    C = live.compliance(model, bench, params, device)
    set_stage(job_id, "validate")
    print("Solving the FEniCS reference for these settings ...")
    ref = _fem_reference(bench, params)
    coords, u_f, v_f, vm_f = ref["coords"], ref["u"], ref["v"], ref["vm"]
    up, vp, vm = live.predict(model, coords, bench, params, device)
    metrics = dict(
        benchmark=bench, formulation="energy_xpinn", custom=True, settings=shown, params=_param_count(model),
        train_seconds=seconds, seed=SEED, subdomains=len(model.subnets), fem_cells=int(ref["cells"]),
        relative_L2_displacement=float(np.sqrt(np.sum((up - u_f) ** 2 + (vp - v_f) ** 2) / np.sum(u_f ** 2 + v_f ** 2))),
        relative_L2_u=_rel(up, u_f), relative_L2_v=_rel(vp, v_f), von_mises_l2_global=_rel(vm, vm_f),
        compliance_error=float((C - float(ref["compliance"])) / float(ref["compliance"])),
        interface_jump_sq=prob.last.get("jump_u"))
    singular = live.singular_points(bench, params)
    if singular:
        r = default_exclusion_radius(coords, singular)
        keep = np.min([np.hypot(coords[:, 0] - x, coords[:, 1] - y) for x, y in singular], axis=0) > r
        metrics.update(von_mises_l2_excl=_rel(vm[keep], vm_f[keep]), von_mises_exclusion_radius=float(r),
                       von_mises_excluded_points=int((~keep).sum()), von_mises_total_points=int(len(keep)))
    else:
        keep = np.ones(len(coords), dtype=bool)
        metrics["von_mises_l2"] = metrics["von_mises_l2_global"]
    err = np.abs(vm - vm_f)
    worst = int(np.argmax(np.where(keep, err, -1.0)))
    metrics.update(max_pointwise_von_mises_error=float(err[worst]),
                   max_pointwise_von_mises_error_location=[float(coords[worst, 0]), float(coords[worst, 1])])
    if bench in live.INCLUSION:
        band = live.interface_distance(bench, params, coords[:, 0], coords[:, 1]) < 0.05
        ins = live.inside(bench, params, coords[:, 0], coords[:, 1])
        metrics.update(band_vm_l2_inside=_rel(vm[band & ins], vm_f[band & ins]),
                       band_vm_l2_outside=_rel(vm[band & ~ins], vm_f[band & ~ins]))
    vm_key = "von_mises_l2_excl" if singular else "von_mises_l2"
    print(f"displacement L2 {metrics['relative_L2_displacement']:.3%}   von Mises L2 {metrics[vm_key]:.3%}   "
          f"compliance {metrics['compliance_error']:+.3%} vs FEM")

    set_stage(job_id, "plots")
    _plot_live(bench, params, coords, up, vp, vm, u_f, v_f, vm_f, PLOTS_DIR / f"{job_id}_fem_comparison.png")
    _plot_deformed(bench, params, coords, up, vp, vm, PLOTS_DIR / f"{job_id}_deformed.png")
    plot_training_convergence(history_for_plot(hist), save_path=str(PLOTS_DIR / f"{job_id}_convergence.png"),
                              adam_epoch_count=payload["epochs"])
    with JOBS_LOCK:
        JOBS[job_id]["metrics"] = metrics
        JOBS[job_id]["plots"] = {"fem_comparison": f"/static/plots/{job_id}_fem_comparison.png",
                                 "deformed": f"/static/plots/{job_id}_deformed.png",
                                 "convergence": f"/static/plots/{job_id}_convergence.png"}


def _live_triangulation(bench, params, coords, dx=None, dy=None):
    from matplotlib.tri import Triangulation
    base = Triangulation(coords[:, 0], coords[:, 1])
    cx, cy = coords[base.triangles].mean(axis=1).T
    p = live.si(bench, params)
    mask = None
    if bench == "plate_hole":
        mask = np.hypot(cx, cy) < p["r"]
    elif bench == "l_bracket":
        mask = (cx < live.L_BRACKET_S) & (cy > live.L_BRACKET_S)
    tri = base if dx is None else Triangulation(coords[:, 0] + dx, coords[:, 1] + dy, triangles=base.triangles)
    if mask is not None:
        tri.set_mask(mask)
    return tri


def _colorbar(fig, ax, cs, label=None):
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.08)
    cb = fig.colorbar(cs, cax=cax, label=label)
    lo, hi = cs.levels[0], cs.levels[-1]
    cb.set_ticks([t for t in MaxNLocator(5).tick_values(lo, hi) if lo <= t <= hi * (1 + 1e-9)])
    cb.ax.tick_params(labelsize=7)


def _plot_live(bench, params, coords, up, vp, vm, u_f, v_f, vm_f, path):
    import matplotlib.pyplot as plt
    tri = _live_triangulation(bench, params, coords)
    ind = live.inside(bench, params, coords[:, 0], coords[:, 1]).astype(float) if bench in live.INCLUSION else None
    aspect = np.ptp(coords[:, 1]) / np.ptp(coords[:, 0])
    wide = aspect < 1 / 1.6
    fields = [("|u|", "mm", 1e3 * np.hypot(u_f, v_f), 1e3 * np.hypot(up, vp), "viridis"),
              ("von Mises", "MPa", vm_f / 1e6, vm / 1e6, "inferno")]
    if wide:
        fig, axes = plt.subplots(3, 2, figsize=(12, min(9.0, 3 * (5.0 * aspect + 0.75))))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4))
    for fi, (name, unit, ref, pred, cmap) in enumerate(fields):
        cap = np.percentile(ref, 99.8)
        panels = [(ref, f"FEM {name} [{unit}]", cmap, cap), (pred, f"ATLAS-XPINN {name} [{unit}]", cmap, cap),
                  (np.abs(pred - ref) / cap * 100, f"{name} error / FEM 99.8th pct [%]", "magma", 10.0)]
        for pi, (z, title, cm, hi) in enumerate(panels):
            ax = axes[pi, fi] if wide else axes[fi, pi]
            cs = ax.tricontourf(tri, z, levels=np.linspace(0, hi, 30), cmap=cm, extend="max")
            if ind is not None:
                ax.tricontour(tri, ind, levels=[0.5], colors="w", linewidths=0.8, linestyles="--")
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=7)
            _colorbar(fig, ax, cs)
    fig.suptitle(f"{META[bench]['title']}: ATLAS-XPINN vs FEM", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_deformed(bench, params, coords, up, vp, vm, path):
    import matplotlib.pyplot as plt
    extent = max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))
    peak = float(np.max(np.hypot(up, vp))) or 1.0
    want = 0.08 * extent / peak
    scale = float(10 ** np.floor(np.log10(want)))
    scale *= 5 if want >= 5 * scale else 2 if want >= 2 * scale else 1
    base = _live_triangulation(bench, params, coords)
    moved = _live_triangulation(bench, params, coords, scale * up, scale * vp)
    xs = np.concatenate([coords[:, 0], coords[:, 0] + scale * up])
    ys = np.concatenate([coords[:, 1], coords[:, 1] + scale * vp])
    aspect = np.ptp(ys) / np.ptp(xs)
    fig, ax = plt.subplots(figsize=(11, min(8.5, max(3.2, 9.5 * aspect + 1.2))) if aspect < 0.7 else (7.6, 6.6))
    ax.tricontourf(base, np.zeros(len(coords)), levels=[-1, 1], colors=["#d9d9d9"], alpha=0.6)
    cap = np.percentile(vm, 99.8) / 1e6
    cs = ax.tricontourf(moved, vm / 1e6, levels=np.linspace(0, cap, 30), cmap="inferno", extend="max")
    if bench in live.INCLUSION:
        ax.tricontour(moved, live.inside(bench, params, coords[:, 0], coords[:, 1]).astype(float), levels=[0.5],
                      colors="w", linewidths=0.8, linestyles="--")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.set_title(f"{META[bench]['title']}: deformed shape (displacement \u00d7 {scale:g}), grey = unloaded shape", fontsize=10)
    _colorbar(fig, ax, cs, "von Mises [MPa]")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


INVERSE_FEM_KW = {
    "cantilever": lambda p: dict(L=p["L"], P=-p["load"], E=p["E"], nu=p["nu"]),
    "plate_hole": lambda p: dict(sigma_inf=p["sigma"], E=p["E"], nu=p["nu"]),
    "l_bracket": lambda p: dict(P=p["load"], E=p["E"], nu=p["nu"]),
}
INVERSE_CASE_KW = {
    "cantilever": lambda p: dict(E=p["E"], nu=p["nu"], L=p["L"], load=p["load"], p0=p["H"]),
    "plate_hole": lambda p: dict(E=p["E"], nu=p["nu"], sigma=p["sigma"], p0=p["r"]),
    "l_bracket": lambda p: dict(E=p["E"], nu=p["nu"], load=p["load"]),
}


def _fem_design_check(bench, params, points):
    spec = dict(benchmark=bench, points=[float(x) for x in points], kwargs=INVERSE_FEM_KW[bench](live.si(bench, params)))
    with tempfile.TemporaryDirectory() as tmp:
        spec_path, out_path = os.path.join(tmp, "spec.json"), os.path.join(tmp, "out.json")
        json.dump(spec, open(spec_path, "w"))
        env = dict(os.environ, PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
        run = subprocess.run([sys.executable, str(ROOT / "fem" / "fem_parametric.py"), "--check", spec_path, out_path],
                             capture_output=True, text=True, env=env, timeout=1800, cwd=str(ROOT))
        if run.returncode != 0:
            raise RuntimeError("FEniCS design check failed: " + (run.stderr.strip().splitlines() or ["no output"])[-1])
        return json.load(open(out_path))


def _run_inverse(job_id: str, payload: dict) -> None:
    from atlas.atlas_main import design_result_to_json, run_inverse_design_all

    bench, device, params = payload["benchmark"], auto_device(), payload.get("params") or {}
    if payload["custom"]:
        units = {s["key"]: s["unit"] for s in live.PARAMS[bench]}
        print(f"{META[bench]['title']}, custom settings: " + ", ".join(f"{k}={v:g}{(' ' + units[k]) if units[k] else ''}"
                                                              for k, v in live.values(bench, params).items()))
    set_stage(job_id, "train")
    t0 = time.perf_counter()
    res = run_inverse_design_all(bench, device, metric=payload.get("metric"), target_ratio=payload.get("target_ratio"),
                                 out_dir=PLOTS_DIR / f"{job_id}_inv", forward_epochs=payload["forward_epochs"],
                                 forward_lbfgs=payload["forward_lbfgs"], max_iter=max(payload["n_outer"], 8),
                                 verbose=True, decomposed=True,
                                 case_kw=INVERSE_CASE_KW[bench](live.si(bench, params)) if payload["custom"] else None)
    seconds = time.perf_counter() - t0
    data = design_result_to_json(res)
    metric = data["metric"]
    ratio = data["target"] / data["baseline"][metric]

    set_stage(job_id, "validate")
    print("Checking the design with FEniCS ...", flush=True)
    start, found = _fem_design_check(bench, params, [data["param_baseline"], data["p_star"]])
    fem_target = ratio * start[metric]
    check = dict(target_ratio=ratio, fem_baseline=start[metric], fem_at_design=found[metric], fem_target=fem_target,
                 error_vs_target=found[metric] / fem_target - 1.0,
                 atlas_vs_fem=data["metrics"][metric] / found[metric] - 1.0)
    check["within_10pct"] = bool(abs(check["error_vs_target"]) < 0.10)
    print(f"FEM check: {metric} at the design is {check['error_vs_target']:+.2%} from the target; "
          f"ATLAS-XPINN vs FEM there {check['atlas_vs_fem']:+.2%}", flush=True)

    set_stage(job_id, "plots")
    hist = [r for r in data["history"] if r["tag"] not in ("probe", "return")]
    metrics = dict(data, param=data["param_name"], initial=data["param_baseline"], final=data["p_star"],
                   objective=INVERSE_OBJECTIVE.get(metric, metric), n_outer=len(hist), train_seconds=seconds,
                   formulation="energy_xpinn", custom=payload["custom"], settings=live.values(bench, params),
                   fem_check=check)
    base = f"/static/plots/{job_id}_inv/{bench}"
    with JOBS_LOCK:
        JOBS[job_id]["metrics"] = metrics
        JOBS[job_id]["plots"] = {"design": f"{base}_design.png", "inverse_history": f"{base}_history.png"}


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api/benchmarks")
def benchmarks():
    return {"benchmarks": [bench_meta(b) for b in PAPER + MULTI], "lbfgs_iters": LBFGS_ITERS, "max_lbfgs": MAX_LBFGS, "seed": SEED}


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
                                          within_10pct=inv.get("within_10pct"), target_ratio=inv.get("target_ratio"),
                                          baseline=(_load("atlas_xpinn_energy", "inverse_design", f"{b}_inverse.json")
                                                    or {}).get("param_baseline")),
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
        return {"jobs": [{k: v for k, v in j.items() if k not in ("log", "progress")} for j in jobs]}


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
