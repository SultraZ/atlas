"""Run from the project root:
    python webapp/desktop.py          # standalone desktop application window
    cd webapp/backend && python app.py  # local console server on http://localhost:8000/
"""

import gc
import io
import json
import os
import platform
import shutil
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
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from atlas import live
from atlas.paths import FEM_REFERENCE_DIR

PLOTS_DIR = ROOT / "webapp" / "static" / "plots"
FRONTEND_DIR = ROOT / "webapp" / "frontend"
STORED_RESULTS = ROOT / "results"
SETTINGS_FILE = ROOT / "webapp" / ".app_settings.json"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ATLAS Application Console", version="2.1.0")
app.mount("/static/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/results", StaticFiles(directory=str(STORED_RESULTS)), name="results")

JOBS: dict = {}
JOBS_LOCK = threading.Lock()
RUN_LOCK = threading.Lock()
SETTINGS_LOCK = threading.Lock()
LOG_CAP = 24000
HIDDEN_LAYERS, NEURONS, SEED, LBFGS_ITERS = 4, 64, 42, 2000
MAX_LBFGS = 4 * LBFGS_ITERS
DEFAULT_LR = 2e-3
DEFAULT_WARM_START = 1000
DEFAULT_AL_STAGES = 3
DEFAULT_GAMMA_U = 30.0
DEFAULT_GAMMA_T = 0.1
CPU_COUNT = max(1, os.cpu_count() or 4)

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
INVERSE_DEFAULTS = {
    "cantilever": dict(param="beam height", metric="compliance", ratio=0.60, bounds=[0.5, 2.0], p0=1.0),
    "plate_hole": dict(param="hole radius", metric="compliance", ratio=1.10, bounds=[0.1, 0.8], p0=0.3),
    "l_bracket": dict(param="fillet radius", metric="peak_vm", ratio=0.75, bounds=[0.05, 0.4], p0=0.1),
}

DEFAULT_SETTINGS = {
    "device": os.environ.get("ATLAS_DEVICE", "auto") if os.environ.get("ATLAS_DEVICE") in ("auto", "cuda", "cpu") else "auto",
    "fem_backend": os.environ.get("ATLAS_FEM_BACKEND", "auto") if os.environ.get("ATLAS_FEM_BACKEND") in ("auto", "fenics", "builtin") else "auto",
    "cpu_threads": min(CPU_COUNT, max(1, int(os.environ.get("ATLAS_CPU_THREADS", "0") or CPU_COUNT))),
    "plot_dpi": 140,
    "poll_ms": 1500,
    "auto_open_log": True,
}


def _model_dump(model: BaseModel) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _load_app_settings() -> dict:
    cfg = dict(DEFAULT_SETTINGS)
    try:
        if SETTINGS_FILE.exists():
            saved = json.loads(SETTINGS_FILE.read_text())
            if isinstance(saved, dict):
                if saved.get("device") in ("auto", "cuda", "cpu") and not os.environ.get("ATLAS_DEVICE"):
                    cfg["device"] = saved["device"]
                if saved.get("fem_backend") in ("auto", "fenics", "builtin") and not os.environ.get("ATLAS_FEM_BACKEND"):
                    cfg["fem_backend"] = saved["fem_backend"]
                if isinstance(saved.get("cpu_threads"), int) and not os.environ.get("ATLAS_CPU_THREADS"):
                    cfg["cpu_threads"] = min(CPU_COUNT, max(1, saved["cpu_threads"]))
                if isinstance(saved.get("plot_dpi"), int):
                    cfg["plot_dpi"] = min(400, max(72, saved["plot_dpi"]))
                if isinstance(saved.get("poll_ms"), int):
                    cfg["poll_ms"] = min(10000, max(500, saved["poll_ms"]))
                if isinstance(saved.get("auto_open_log"), bool):
                    cfg["auto_open_log"] = saved["auto_open_log"]
    except Exception:
        pass
    return cfg


SETTINGS = _load_app_settings()


def _apply_cpu_threads(threads: int) -> None:
    try:
        import torch
        torch.set_num_threads(max(1, min(CPU_COUNT, int(threads))))
    except Exception:
        pass


_apply_cpu_threads(SETTINGS["cpu_threads"])


def _save_app_settings() -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(SETTINGS, indent=2))
    except Exception:
        pass


def cuda_details() -> Tuple[bool, Optional[str], Optional[str], Optional[dict]]:
    try:
        import torch
        if not torch.cuda.is_available():
            return False, None, getattr(torch.version, "cuda", None), None
        name = torch.cuda.get_device_name(0)
        mem = None
        try:
            free_b, total_b = torch.cuda.mem_get_info(0)
            mem = dict(
                free_mb=round(free_b / (1024 ** 2)),
                total_mb=round(total_b / (1024 ** 2)),
                allocated_mb=round(torch.cuda.memory_allocated(0) / (1024 ** 2)),
                reserved_mb=round(torch.cuda.memory_reserved(0) / (1024 ** 2)),
            )
        except Exception:
            pass
        return True, name, getattr(torch.version, "cuda", None), mem
    except Exception:
        return False, None, None, None


def _win_to_wsl_path(win_path: str, distro: Optional[str] = None) -> str:
    p = str(win_path)
    norm = p.replace("/", "\\")
    for prefix in ("\\\\wsl.localhost\\", "\\\\wsl$\\"):
        if norm.lower().startswith(prefix):
            rest = norm[len(prefix):]
            parts = rest.split("\\", 1)
            return "/" + (parts[1].replace("\\", "/") if len(parts) > 1 else "")
    cmd = ["wsl.exe"] + (["-d", distro] if distro else []) + ["wslpath", "-a", p]
    return subprocess.check_output(cmd, text=True, timeout=8).strip()


@lru_cache(maxsize=1)
def _find_fenics_runner() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return `(mode, executable_or_distro, version_label)` where `mode` is `'local'` or `'wsl'`."""
    candidates = []
    if os.environ.get("ATLAS_FENICS_PYTHON"):
        candidates.append(os.environ["ATLAS_FENICS_PYTHON"])
    candidates.append(sys.executable)
    home = Path.home()
    for rel in ("miniconda3/envs/Atlas/bin/python", "anaconda3/envs/Atlas/bin/python",
                "miniforge3/envs/Atlas/bin/python", "mambaforge/envs/Atlas/bin/python"):
        p = str(home / rel)
        if p not in candidates:
            candidates.append(p)
    for py in candidates:
        if not py or not os.path.isfile(py):
            continue
        try:
            env = dict(os.environ, PATH=os.path.dirname(py) + os.pathsep + os.environ.get("PATH", ""))
            run = subprocess.run(
                [py, "-c", "import dolfin, mshr; print(dolfin.__version__)"],
                capture_output=True, text=True, env=env, timeout=12, cwd=str(ROOT),
            )
            if run.returncode == 0:
                ver = (run.stdout.strip().splitlines() or ["installed"])[-1].strip()
                return "local", py, ver
        except Exception:
            pass

    if sys.platform == "win32" and shutil.which("wsl.exe"):
        distros: list[Optional[str]] = [None]
        try:
            raw = subprocess.check_output(["wsl.exe", "-l", "-q"], timeout=6)
            names = [line.strip() for line in raw.decode("utf-16le", errors="ignore").splitlines() if line.strip()]
            distros = [d for d in names if not d.lower().startswith("docker-desktop")] or distros
        except Exception:
            pass
        probe = (
            "for p in \"$HOME/miniconda3/envs/Atlas/bin/python\" "
            "\"$HOME/anaconda3/envs/Atlas/bin/python\" "
            "\"$HOME/miniforge3/envs/Atlas/bin/python\" "
            "\"$HOME/mambaforge/envs/Atlas/bin/python\" "
            "\"$(command -v python3 2>/dev/null)\"; do "
            "if [ -x \"$p\" ] && PATH=\"$(dirname \"$p\"):$PATH\" \"$p\" -c 'import dolfin, mshr; print(dolfin.__version__)' >/dev/null 2>&1; then "
            "ver=$(PATH=\"$(dirname \"$p\"):$PATH\" \"$p\" -c 'import dolfin; print(dolfin.__version__)'); "
            "echo \"$p|$ver\"; exit 0; fi; done; exit 1"
        )
        for d in distros:
            try:
                cmd = ["wsl.exe"] + (["-d", d] if d else []) + ["bash", "-lc", probe]
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=15).strip()
                if "|" in out:
                    wsl_py, ver = out.splitlines()[-1].split("|", 1)
                    tag = f"{ver.strip()} (via WSL2{f' {d}' if d else ''})"
                    return "wsl", f"{d or ''}::{wsl_py.strip()}", tag
            except Exception:
                pass

    return None, None, None


def fenics_status() -> Tuple[bool, Optional[str]]:
    mode, _, ver = _find_fenics_runner()
    return (mode is not None), ver


@lru_cache(maxsize=1)
def builtin_fem_status() -> Tuple[bool, Optional[str]]:
    try:
        import scipy
        from atlas import native_fem  # noqa: F401
        return True, f"SciPy {scipy.__version__} P2 FEM"
    except Exception:
        return False, None


def resolve_fem_backend(requested: Optional[str] = None, strict: bool = True) -> Tuple[str, str]:
    pref = (requested or SETTINGS.get("fem_backend") or "auto").lower()
    has_fenics, fenics_ver = fenics_status()
    has_builtin, builtin_ver = builtin_fem_status()
    fenics_label = f"FEniCS {fenics_ver}" if fenics_ver else "FEniCS"
    builtin_label = builtin_ver or "Built-in P2 FEM"
    if pref == "fenics":
        if not has_fenics:
            if strict:
                raise HTTPException(
                    status_code=400,
                    detail="FEniCS (dolfin + mshr) is not installed on this machine (WSL2 / Conda). "
                           "Switch FEM Engine to Auto or Built-in P2 FEM, or install WSL2 FEniCS.",
                )
            return ("builtin", builtin_label) if has_builtin else ("none", "Unavailable")
        return "fenics", fenics_label
    if pref == "builtin":
        if not has_builtin:
            if strict:
                raise HTTPException(status_code=400, detail="SciPy is required for the Built-in P2 FEM solver.")
            return ("fenics", fenics_label) if has_fenics else ("none", "Unavailable")
        return "builtin", builtin_label
    if has_fenics:
        return "fenics", fenics_label
    if has_builtin:
        return "builtin", builtin_label
    if strict:
        raise HTTPException(status_code=400, detail="No FEM backend available (install SciPy or FEniCS).")
    return "none", "Unavailable"


def _run_fenics_script(script_rel: str, path_args: list[str], extra_flags: list[str] = (), timeout: int = 900):
    mode, target, _ = _find_fenics_runner()
    if mode == "wsl" and target:
        distro, wsl_py = target.split("::", 1)
        distro_opt = distro or None
        wsl_root = _win_to_wsl_path(str(ROOT), distro_opt)
        wsl_script = f"{wsl_root.rstrip('/')}/{script_rel.lstrip('/')}"
        wsl_paths = [_win_to_wsl_path(p, distro_opt) for p in path_args]
        py_dir = os.path.dirname(wsl_py)
        quoted = " ".join([f'"{x}"' for x in [wsl_py, wsl_script, *extra_flags, *wsl_paths]])
        bash_cmd = f'cd "{wsl_root}" && PATH="{py_dir}:$PATH" {quoted}'
        cmd = ["wsl.exe"] + (["-d", distro_opt] if distro_opt else []) + ["bash", "-lc", bash_cmd]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    py = target if (mode == "local" and target) else sys.executable
    env = dict(os.environ, PATH=os.path.dirname(py) + os.pathsep + os.environ.get("PATH", ""))
    cmd = [py, str(ROOT / script_rel), *extra_flags, *path_args]
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout, cwd=str(ROOT))


def plots_storage() -> Tuple[int, int]:
    count, total = 0, 0
    try:
        for p in PLOTS_DIR.rglob("*"):
            if p.is_file():
                count += 1
                total += p.stat().st_size
    except Exception:
        pass
    return count, total


def resolve_device(requested: Optional[str] = None, strict_cuda: bool = True) -> Tuple[str, str]:
    pref = (requested or SETTINGS.get("device") or "auto").lower()
    has_cuda, gpu_name, _, _ = cuda_details()
    threads = SETTINGS.get("cpu_threads", CPU_COUNT)
    if pref == "cuda":
        if not has_cuda:
            if strict_cuda:
                raise HTTPException(
                    status_code=400,
                    detail="CUDA GPU is not available in this Python environment. Switch Compute Device to Auto or CPU, or launch ATLAS with your CUDA conda environment.",
                )
            return "cpu", f"CPU ({threads} threads)"
        return "cuda", gpu_name or "CUDA GPU"
    if pref == "cpu":
        return "cpu", f"CPU ({threads} threads)"
    if has_cuda:
        return "cuda", gpu_name or "CUDA GPU"
    return "cpu", f"CPU ({threads} threads)"


def auto_device() -> str:
    return resolve_device(None, strict_cuda=False)[0]


def _dtype_of(precision: str):
    import torch
    return torch.float32 if precision == "float32" else torch.float64


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


def _multi_build(bench, formulation, device, hidden_layers=HIDDEN_LAYERS, neurons=NEURONS,
                 gamma_u=DEFAULT_GAMMA_U, gamma_t=DEFAULT_GAMMA_T, dtype=None):
    import torch
    dt = dtype or torch.float64
    mod = _multi_module(bench)
    args = (BIMATERIAL_CONTRAST,) if bench == "bimaterial" else ()
    if formulation == "energy_xpinn":
        return mod.build_decomposed(*args, hidden_layers=hidden_layers, neurons=neurons,
                                    gamma_u=gamma_u, gamma_t=gamma_t, device=device, dtype=dt)
    return mod.build_single(*args, hidden_layers=hidden_layers, neurons=neurons, device=device, dtype=dt)


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
        d = INVERSE_DEFAULTS[bench]
        metric = inv.get("metric") or d["metric"]
        m["inverse"] = dict(
            param=inv.get("param_name") or d["param"],
            metric=metric,
            metrics=["compliance", "peak_vm", "volume"],
            objective=INVERSE_OBJECTIVE.get(metric, metric),
            default_ratio=d["ratio"],
            bounds=inv.get("bounds") or d["bounds"],
            baseline=inv.get("param_baseline", d["p0"]),
        )
    return m


class JobCancelledError(Exception):
    pass


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


def _check_cancel(job_id: str) -> None:
    with JOBS_LOCK:
        if JOBS.get(job_id, {}).get("cancel_requested"):
            raise JobCancelledError("Run cancelled by user")


def set_stage(job_id: str, stage: str) -> None:
    _check_cancel(job_id)
    with JOBS_LOCK:
        JOBS[job_id]["stage"] = stage


@contextmanager
def _with_plot_dpi(dpi: int):
    from matplotlib.figure import Figure
    orig = Figure.savefig

    def savefig_with_dpi(self, *args, **kwargs):
        kwargs["dpi"] = int(dpi)
        return orig(self, *args, **kwargs)

    Figure.savefig = savefig_with_dpi
    try:
        yield
    finally:
        Figure.savefig = orig


class SettingsUpdate(BaseModel):
    device: Optional[str] = Field(None, pattern="^(auto|cuda|cpu)$")
    fem_backend: Optional[str] = Field(None, pattern="^(auto|fenics|builtin)$")
    cpu_threads: Optional[int] = Field(None, ge=1, le=256)
    plot_dpi: Optional[int] = Field(None, ge=72, le=400)
    poll_ms: Optional[int] = Field(None, ge=500, le=10000)
    auto_open_log: Optional[bool] = None


class ForwardRequest(BaseModel):
    benchmark: str = Field(..., pattern="^(cantilever|l_bracket|plate_hole|steel_insert|bimaterial|filled_slot)$")
    formulation: str = Field("energy_xpinn", pattern="^(energy|energy_xpinn)$")
    device: Optional[str] = Field(None, pattern="^(auto|cuda|cpu)$")
    fem_backend: Optional[str] = Field(None, pattern="^(auto|fenics|builtin)$")
    epochs: int = Field(2000, ge=50, le=40000)
    lbfgs: int = Field(LBFGS_ITERS, ge=0, le=MAX_LBFGS)
    hidden_layers: int = Field(HIDDEN_LAYERS, ge=2, le=8)
    neurons: int = Field(NEURONS, ge=16, le=256)
    lr: float = Field(DEFAULT_LR, ge=1e-5, le=5e-2)
    precision: str = Field("float64", pattern="^(float64|float32)$")
    seed: int = Field(SEED, ge=0, le=999999)
    warm_start: int = Field(DEFAULT_WARM_START, ge=0, le=20000)
    al_stages: int = Field(DEFAULT_AL_STAGES, ge=0, le=8)
    gamma_u: float = Field(DEFAULT_GAMMA_U, ge=0.1, le=1000.0)
    gamma_t: float = Field(DEFAULT_GAMMA_T, ge=0.0, le=100.0)
    n_interior: int = Field(800, ge=200, le=5000)
    warm_start_eb: bool = True
    prefit_epochs: int = Field(4000, ge=0, le=40000)
    use_mock: bool = False
    params: Optional[Dict[str, float]] = None


class InverseRequest(BaseModel):
    benchmark: str = Field(..., pattern="^(cantilever|l_bracket|plate_hole)$")
    formulation: str = Field("energy_xpinn", pattern="^(energy|energy_xpinn)$")
    device: Optional[str] = Field(None, pattern="^(auto|cuda|cpu)$")
    fem_backend: Optional[str] = Field(None, pattern="^(auto|fenics|builtin)$")
    metric: Optional[str] = Field(None, pattern="^(compliance|peak_vm|volume)$")
    target_ratio: Optional[float] = Field(None, gt=0.0, le=10.0)
    forward_epochs: int = Field(2000, ge=100, le=20000)
    forward_lbfgs: int = Field(2000, ge=0, le=8000)
    n_outer: int = Field(12, ge=1, le=100)
    n_inner: int = Field(40, ge=5, le=2000)
    hidden_layers: int = Field(HIDDEN_LAYERS, ge=2, le=8)
    neurons: int = Field(NEURONS, ge=16, le=256)
    precision: str = Field("float64", pattern="^(float64|float32)$")
    seed: int = Field(SEED, ge=0, le=999999)
    warm_start: int = Field(DEFAULT_WARM_START, ge=0, le=20000)
    al_stages: int = Field(DEFAULT_AL_STAGES, ge=0, le=8)
    gamma_u: float = Field(DEFAULT_GAMMA_U, ge=0.1, le=1000.0)
    gamma_t: float = Field(DEFAULT_GAMMA_T, ge=0.0, le=100.0)
    use_mock: bool = False
    params: Optional[Dict[str, float]] = None


def submit(kind: str, payload: dict) -> str:
    dev, dev_name = resolve_device(payload.get("device"), strict_cuda=True)
    bench = payload["benchmark"]
    payload["custom"] = not live.is_default(bench, payload.get("params"))
    needs_live_fem = payload["custom"] or kind == "inverse"
    if needs_live_fem:
        fem_key, fem_name = resolve_fem_backend(payload.get("fem_backend"), strict=True)
    else:
        fem_key, fem_name = "stored_fenics", "Stored FEniCS 2019.1.0 reference"
    payload["_resolved_device"] = dev
    payload["_resolved_device_name"] = dev_name
    payload["_resolved_fem"] = fem_key
    payload["_resolved_fem_name"] = fem_name
    payload["_plot_dpi"] = int(SETTINGS.get("plot_dpi", 140))

    job_id = uuid.uuid4().hex[:8]
    solver_cfg = dict(
        formulation=payload.get("formulation", "energy_xpinn"),
        hidden_layers=payload.get("hidden_layers", HIDDEN_LAYERS),
        neurons=payload.get("neurons", NEURONS),
        precision=payload.get("precision", "float64"),
        seed=payload.get("seed", SEED),
        warm_start=payload.get("warm_start", DEFAULT_WARM_START),
        al_stages=payload.get("al_stages", DEFAULT_AL_STAGES),
        gamma_u=payload.get("gamma_u", DEFAULT_GAMMA_U),
        gamma_t=payload.get("gamma_t", DEFAULT_GAMMA_T),
    )
    if kind == "forward":
        solver_cfg["lr"] = payload.get("lr", DEFAULT_LR)
    else:
        solver_cfg.update(
            metric=payload.get("metric"),
            target_ratio=payload.get("target_ratio"),
            forward_epochs=payload.get("forward_epochs", 2000),
            forward_lbfgs=payload.get("forward_lbfgs", 2000),
            n_outer=payload.get("n_outer", 12),
        )
    with JOBS_LOCK:
        JOBS[job_id] = dict(
            id=job_id, kind=kind, benchmark=bench, custom=payload["custom"],
            device_preference=payload.get("device") or SETTINGS.get("device", "auto"),
            device=dev, device_name=dev_name,
            fem_backend=fem_key, fem_engine=fem_name,
            settings=live.values(bench, payload.get("params")),
            formulation=payload.get("formulation", "energy_xpinn"),
            solver_config=solver_cfg,
            budget=dict(epochs=payload["epochs"], lbfgs=payload["lbfgs"]) if kind == "forward"
            else dict(epochs=payload["forward_epochs"], lbfgs=payload["forward_lbfgs"], n_outer=payload["n_outer"]),
            status="queued", stage="queued", cancel_requested=False,
            log="", metrics={}, plots={}, progress=None,
            created=datetime.now().isoformat(timespec="seconds"),
            started=None, finished=None, error=None,
        )
    threading.Thread(target=_run_job, args=(job_id, kind, payload), daemon=True).start()
    return job_id


def _run_job(job_id: str, kind: str, payload: dict) -> None:
    with RUN_LOCK:
        with JOBS_LOCK:
            if JOBS[job_id].get("cancel_requested"):
                JOBS[job_id].update(status="cancelled", stage="cancelled",
                                    finished=datetime.now().isoformat(timespec="seconds"))
                return
            JOBS[job_id].update(status="running", stage="setup",
                                started=datetime.now().isoformat(timespec="seconds"))
        log = JobLog(job_id)
        try:
            _apply_cpu_threads(SETTINGS.get("cpu_threads", CPU_COUNT))
            dpi = int(payload.get("_plot_dpi") or SETTINGS.get("plot_dpi", 140))
            with redirect_stdout(log), record_energy(job_id), _with_plot_dpi(dpi):
                if kind == "inverse":
                    _run_inverse(job_id, payload)
                elif payload["custom"]:
                    _run_custom(job_id, payload)
                elif payload["benchmark"] in MULTI:
                    _run_multi(job_id, payload)
                else:
                    _run_forward(job_id, payload)
            status, error = "done", None
        except JobCancelledError:
            log.write("\n[Stopped by user]\n")
            status, error = "cancelled", None
        except Exception as exc:
            log.write("\n" + traceback.format_exc())
            status, error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            if payload.get("_resolved_device", "").startswith("cuda"):
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        with JOBS_LOCK:
            next_stage = "finished" if status == "done" else ("cancelled" if status == "cancelled" else JOBS[job_id]["stage"])
            JOBS[job_id].update(status=status, error=error, stage=next_stage,
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
            if JOBS.get(job_id, {}).get("cancel_requested"):
                raise JobCancelledError("Run cancelled by user")
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
    import torch
    from atlas.energy_solver import EnergyTrainer, history_for_plot, train_energy_forward_solver
    from atlas.energy_xpinn import E_MOD, NU, STRESS_SCALE, build_xpinn_energy, train_decomposed
    from atlas.physics import PlaneStressElasticity
    from atlas.sampling import CantileverBeamSampler, LBracketSampler, PlateWithHoleSampler
    from atlas.validation import run_validation
    from atlas.visualization import plot_deformed_shape, plot_fem_comparison, plot_training_convergence

    bench = payload["benchmark"]
    device = payload["_resolved_device"]
    device_name = payload["_resolved_device_name"]
    form = payload["formulation"]
    dtype = _dtype_of(payload.get("precision", "float64"))
    seed = int(payload.get("seed", SEED))
    hidden_layers = int(payload.get("hidden_layers", HIDDEN_LAYERS))
    neurons = int(payload.get("neurons", NEURONS))
    lr = float(payload.get("lr", DEFAULT_LR))

    print(f"Compute device: {device_name} ({device}, {payload.get('precision', 'float64')}, seed {seed})")
    set_stage(job_id, "train")
    t0 = time.perf_counter()
    if form == "energy_xpinn":
        torch.manual_seed(seed)
        np.random.seed(seed)
        model, problem, _ = build_xpinn_energy(
            bench, PlaneStressElasticity(E=E_MOD / STRESS_SCALE, nu=NU),
            hidden_layers=hidden_layers, neurons=neurons,
            gamma_u=float(payload.get("gamma_u", DEFAULT_GAMMA_U)),
            gamma_t=float(payload.get("gamma_t", DEFAULT_GAMMA_T)),
            device=device, dtype=dtype,
        )
        print(f"ATLAS-XPINN: {len(model.subnets)} subnetworks, {_param_count(model)} parameters, {problem.n_points} quadrature points")
        hist = train_decomposed(
            model, problem, epochs=payload["epochs"], lr=lr, lbfgs_per_stage=payload["lbfgs"],
            gammas=(float(payload.get("gamma_u", DEFAULT_GAMMA_U)),), verbose=True,
            al_stages=int(payload.get("al_stages", DEFAULT_AL_STAGES)),
            warm_start=int(payload.get("warm_start", DEFAULT_WARM_START)),
        )
        model = model.float().to(device)
        sampler = {"cantilever": CantileverBeamSampler, "plate_hole": PlateWithHoleSampler,
                   "l_bracket": LBracketSampler}[bench](device=device)
        trainer = EnergyTrainer()
        trainer.history = history_for_plot(hist)
        trainer.adam_epoch_count = payload["epochs"] if payload["lbfgs"] > 0 else None
        physics = PlaneStressElasticity(E=E_MOD, nu=NU)
    else:
        model, trainer, sampler, physics = train_energy_forward_solver(
            bench, device=device, epochs=payload["epochs"], lr=lr,
            lbfgs_iters=payload["lbfgs"], lbfgs_chunk=100,
            hidden_layers=hidden_layers, neurons=neurons, dtype=dtype, seed=seed,
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    set_stage(job_id, "predict")
    params = _param_count(model)
    set_stage(job_id, "validate")
    metrics = run_validation(model=model, sampler=sampler, physics=physics, benchmark=bench, device=device,
                             use_mock=payload.get("use_mock", False), data_dir=str(FEM_REFERENCE_DIR))
    metrics = dict(metrics, benchmark=bench, formulation=form, params=params,
                   train_seconds=seconds, seed=seed, device=device, device_name=device_name,
                   fem_engine=payload.get("_resolved_fem_name", "Stored FEniCS 2019.1.0 reference"),
                   precision=payload.get("precision", "float64"),
                   hidden_layers=hidden_layers, neurons=neurons,
                   subdomains=subdomain_count(bench) if form == "energy_xpinn" else 1)
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

    bench, form = payload["benchmark"], payload["formulation"]
    device = payload["_resolved_device"]
    device_name = payload["_resolved_device_name"]
    dtype = _dtype_of(payload.get("precision", "float64"))
    seed = int(payload.get("seed", SEED))
    hidden_layers = int(payload.get("hidden_layers", HIDDEN_LAYERS))
    neurons = int(payload.get("neurons", NEURONS))
    lr = float(payload.get("lr", DEFAULT_LR))
    gamma_u = float(payload.get("gamma_u", DEFAULT_GAMMA_U))
    gamma_t = float(payload.get("gamma_t", DEFAULT_GAMMA_T))

    mod, ref = _multi_module(bench), MULTI_REF[bench]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, prob = _multi_build(bench, form, device, hidden_layers=hidden_layers, neurons=neurons,
                               gamma_u=gamma_u, gamma_t=gamma_t, dtype=dtype)
    print(f"Compute device: {device_name} ({device}, {payload.get('precision', 'float64')}, seed {seed})")
    print(f"{META[bench]['title']}: {_param_count(model)} parameters, {prob.n_points} quadrature points, {form}")
    set_stage(job_id, "train")
    t0 = time.perf_counter()
    if form == "energy_xpinn":
        hist = train_decomposed(
            model, prob, epochs=payload["epochs"], lr=lr, lbfgs_per_stage=payload["lbfgs"],
            gammas=(gamma_u,), verbose=True,
            al_stages=int(payload.get("al_stages", DEFAULT_AL_STAGES)),
            warm_start=int(payload.get("warm_start", DEFAULT_WARM_START)),
        )
    else:
        hist = train_energy(model, prob, epochs=payload["epochs"], lr=lr,
                            lbfgs_iters=payload["lbfgs"], lbfgs_chunk=100)
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
        benchmark=bench, formulation=form, params=_param_count(model), train_seconds=seconds, seed=seed,
        device=device, device_name=device_name,
        fem_engine=payload.get("_resolved_fem_name", "Stored FEniCS 2019.1.0 reference"),
        precision=payload.get("precision", "float64"),
        hidden_layers=hidden_layers, neurons=neurons,
        subdomains=len(model.subnets) if hasattr(model, "subnets") else 1,
        relative_L2_displacement=float(np.sqrt(np.sum((up - u_f) ** 2 + (vp - v_f) ** 2) / np.sum(u_f ** 2 + v_f ** 2))),
        relative_L2_u=_rel(up, u_f), relative_L2_v=_rel(vp, v_f), von_mises_l2=_rel(vm, vm_f),
        band_vm_l2_inside=_rel(vmr[inside], vmr_f[inside]), band_vm_l2_outside=_rel(vmr[~inside], vmr_f[~inside]),
        compliance_error=float((C - fem_meta["compliance"]) / fem_meta["compliance"]),
        interface_jump_sq=prob.last.get("jump_u") if form == "energy_xpinn" else None)
    print(f"displacement L2 {metrics['relative_L2_displacement']:.3%}   von Mises L2 {metrics['von_mises_l2']:.3%}   "
          f"interface band {metrics['band_vm_l2_inside']:.3%} / {metrics['band_vm_l2_outside']:.3%}")

    set_stage(job_id, "plots")
    _plot_live(bench, None, coords, up, vp, vm, u_f, v_f, vm_f, PLOTS_DIR / f"{job_id}_fem_comparison.png", form=form)
    _plot_deformed(bench, None, coords, up, vp, vm, PLOTS_DIR / f"{job_id}_deformed.png", form=form)
    plot_training_convergence(history_for_plot(hist), save_path=str(PLOTS_DIR / f"{job_id}_convergence.png"),
                              adam_epoch_count=payload["epochs"] if payload["lbfgs"] > 0 else None)
    with JOBS_LOCK:
        JOBS[job_id]["metrics"] = metrics
        JOBS[job_id]["plots"] = {"fem_comparison": f"/static/plots/{job_id}_fem_comparison.png",
                                 "deformed": f"/static/plots/{job_id}_deformed.png",
                                 "convergence": f"/static/plots/{job_id}_convergence.png"}


def _fem_error_message(prefix: str, stderr: str) -> str:
    if "No module named 'dolfin'" in stderr or "No module named 'mshr'" in stderr:
        return (f"{prefix}: FEniCS (dolfin + mshr) is not installed in the active Python environment. "
                "Launch ATLAS from your FEniCS conda environment (e.g. Atlas in WSL2) for custom geometry/material "
                "FEniCS validation, or press Reset on Properties to validate against the stored FEniCS references.")
    tail = (stderr.strip().splitlines() or ["no output"])[-1]
    return f"{prefix}: {tail}"


def _fem_reference(bench: str, params, fem_pref: Optional[str] = None) -> dict:
    spec = live.fem_spec(bench, params)
    engine, _ = resolve_fem_backend(fem_pref, strict=True)
    if engine == "builtin":
        from atlas import native_fem
        return native_fem.solve_live_spec(spec)
    with tempfile.TemporaryDirectory() as tmp:
        spec_path, out_path = os.path.join(tmp, "spec.json"), os.path.join(tmp, "ref.npz")
        with open(spec_path, "w") as fh:
            json.dump(spec, fh)
        run = _run_fenics_script("fem/fem_live.py", [spec_path, out_path], timeout=900)
        for line in run.stdout.splitlines():
            if "JIT" not in line:
                print("  " + line)
        if run.returncode != 0:
            raise RuntimeError(_fem_error_message("FEniCS reference failed", run.stderr))
        with np.load(out_path) as z:
            return {k: z[k] for k in z.files}


def _run_custom(job_id: str, payload: dict) -> None:
    import torch
    from atlas.energy_solver import history_for_plot, train_energy
    from atlas.energy_xpinn import train_decomposed
    from atlas.validation import default_exclusion_radius
    from atlas.visualization import plot_training_convergence

    bench, params = payload["benchmark"], payload.get("params") or {}
    form = payload.get("formulation", "energy_xpinn")
    device = payload["_resolved_device"]
    device_name = payload["_resolved_device_name"]
    dtype = _dtype_of(payload.get("precision", "float64"))
    seed = int(payload.get("seed", SEED))
    hidden_layers = int(payload.get("hidden_layers", HIDDEN_LAYERS))
    neurons = int(payload.get("neurons", NEURONS))
    lr = float(payload.get("lr", DEFAULT_LR))
    gamma_u = float(payload.get("gamma_u", DEFAULT_GAMMA_U))
    gamma_t = float(payload.get("gamma_t", DEFAULT_GAMMA_T))

    shown = live.values(bench, params)
    units = {s["key"]: s["unit"] for s in live.PARAMS[bench]}
    print(f"Compute device: {device_name} ({device}, {payload.get('precision', 'float64')}, seed {seed})")
    print(f"{META[bench]['title']}, custom settings: " + ", ".join(f"{k}={v:g}{(' ' + units[k]) if units[k] else ''}"
                                                              for k, v in shown.items()))
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, prob = live.build(bench, params, device, hidden_layers, neurons, dtype=dtype,
                             formulation=form, gamma_u=gamma_u, gamma_t=gamma_t)
    n_sub = len(model.subnets) if hasattr(model, "subnets") else 1
    solver_label = "ATLAS-XPINN" if form == "energy_xpinn" else "ATLAS (single network)"
    print(f"{solver_label}: {n_sub} {'subnetworks' if n_sub > 1 else 'network'}, {_param_count(model)} parameters, "
          f"{prob.n_points} quadrature points")
    set_stage(job_id, "train")
    t0 = time.perf_counter()
    if form == "energy_xpinn":
        hist = train_decomposed(
            model, prob, epochs=payload["epochs"], lr=lr, lbfgs_per_stage=payload["lbfgs"],
            gammas=(gamma_u,), verbose=True,
            al_stages=int(payload.get("al_stages", DEFAULT_AL_STAGES)),
            warm_start=int(payload.get("warm_start", DEFAULT_WARM_START)),
        )
    else:
        hist = train_energy(model, prob, epochs=payload["epochs"], lr=lr,
                            lbfgs_iters=payload["lbfgs"], lbfgs_chunk=100)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    set_stage(job_id, "predict")
    C = live.compliance(model, bench, params, device)
    set_stage(job_id, "validate")
    fem_label = payload.get("_resolved_fem_name", "FEM reference")
    print(f"Solving the {fem_label} reference for these settings ...")
    ref = _fem_reference(bench, params, fem_pref=payload.get("_resolved_fem"))
    coords, u_f, v_f, vm_f = ref["coords"], ref["u"], ref["v"], ref["vm"]
    up, vp, vm = live.predict(model, coords, bench, params, device)
    metrics = dict(
        benchmark=bench, formulation=form, custom=True, settings=shown, params=_param_count(model),
        train_seconds=seconds, seed=seed, device=device, device_name=device_name,
        fem_engine=fem_label,
        precision=payload.get("precision", "float64"),
        hidden_layers=hidden_layers, neurons=neurons,
        subdomains=n_sub, fem_cells=int(ref["cells"]),
        relative_L2_displacement=float(np.sqrt(np.sum((up - u_f) ** 2 + (vp - v_f) ** 2) / np.sum(u_f ** 2 + v_f ** 2))),
        relative_L2_u=_rel(up, u_f), relative_L2_v=_rel(vp, v_f), von_mises_l2_global=_rel(vm, vm_f),
        compliance_error=float((C - float(ref["compliance"])) / float(ref["compliance"])),
        interface_jump_sq=prob.last.get("jump_u") if form == "energy_xpinn" else None)
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
    _plot_live(bench, params, coords, up, vp, vm, u_f, v_f, vm_f, PLOTS_DIR / f"{job_id}_fem_comparison.png", form=form)
    _plot_deformed(bench, params, coords, up, vp, vm, PLOTS_DIR / f"{job_id}_deformed.png", form=form)
    plot_training_convergence(history_for_plot(hist), save_path=str(PLOTS_DIR / f"{job_id}_convergence.png"),
                              adam_epoch_count=payload["epochs"] if payload["lbfgs"] > 0 else None)
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


def _plot_live(bench, params, coords, up, vp, vm, u_f, v_f, vm_f, path, form="energy_xpinn"):
    import matplotlib.pyplot as plt
    solver_tag = "ATLAS-XPINN" if form == "energy_xpinn" else "ATLAS"
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
        panels = [(ref, f"FEM {name} [{unit}]", cmap, cap), (pred, f"{solver_tag} {name} [{unit}]", cmap, cap),
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
    fig.suptitle(f"{META[bench]['title']}: {solver_tag} vs FEM", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_deformed(bench, params, coords, up, vp, vm, path, form="energy_xpinn"):
    import matplotlib.pyplot as plt
    solver_tag = "ATLAS-XPINN" if form == "energy_xpinn" else "ATLAS"
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
    ax.set_title(f"{META[bench]['title']} ({solver_tag}): deformed shape (displacement \u00d7 {scale:g}), grey = unloaded shape", fontsize=10)
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


def _fem_design_check(bench, params, points, fem_pref: Optional[str] = None):
    spec = dict(benchmark=bench, points=[float(x) for x in points], kwargs=INVERSE_FEM_KW[bench](live.si(bench, params)))
    engine, _ = resolve_fem_backend(fem_pref, strict=True)
    if engine == "builtin":
        from atlas import native_fem
        return native_fem.solve_parametric_check(spec)
    with tempfile.TemporaryDirectory() as tmp:
        spec_path, out_path = os.path.join(tmp, "spec.json"), os.path.join(tmp, "out.json")
        with open(spec_path, "w") as fh:
            json.dump(spec, fh)
        run = _run_fenics_script("fem/fem_parametric.py", [spec_path, out_path], extra_flags=["--check"], timeout=1800)
        if run.returncode != 0:
            raise RuntimeError(_fem_error_message("FEniCS design check failed", run.stderr))
        with open(out_path) as fh:
            return json.load(fh)


def _run_inverse(job_id: str, payload: dict) -> None:
    from atlas.atlas_main import design_result_to_json, run_inverse_design_all

    bench, params = payload["benchmark"], payload.get("params") or {}
    form = payload.get("formulation", "energy_xpinn")
    decomposed = form == "energy_xpinn"
    device = payload["_resolved_device"]
    device_name = payload["_resolved_device_name"]
    dtype = _dtype_of(payload.get("precision", "float64"))
    seed = int(payload.get("seed", SEED))
    hidden_layers = int(payload.get("hidden_layers", HIDDEN_LAYERS))
    neurons = int(payload.get("neurons", NEURONS))

    print(f"Compute device: {device_name} ({device}, {payload.get('precision', 'float64')}, seed {seed})")
    if payload["custom"]:
        units = {s["key"]: s["unit"] for s in live.PARAMS[bench]}
        print(f"{META[bench]['title']}, custom settings: " + ", ".join(f"{k}={v:g}{(' ' + units[k]) if units[k] else ''}"
                                                              for k, v in live.values(bench, params).items()))
    case_kw = dict(hidden_layers=hidden_layers, neurons=neurons, dtype=dtype)
    if decomposed:
        case_kw.update(
            gamma_u=float(payload.get("gamma_u", DEFAULT_GAMMA_U)),
            gamma_t=float(payload.get("gamma_t", DEFAULT_GAMMA_T)),
            al_stages=int(payload.get("al_stages", DEFAULT_AL_STAGES)),
            warm_start=int(payload.get("warm_start", DEFAULT_WARM_START)),
        )
    if payload["custom"]:
        case_kw.update(INVERSE_CASE_KW[bench](live.si(bench, params)))

    set_stage(job_id, "train")
    t0 = time.perf_counter()
    res = run_inverse_design_all(
        bench, device, metric=payload.get("metric"), target_ratio=payload.get("target_ratio"),
        out_dir=PLOTS_DIR / f"{job_id}_inv", forward_epochs=payload["forward_epochs"],
        forward_lbfgs=payload["forward_lbfgs"], max_iter=max(payload["n_outer"], 1),
        seed=seed, verbose=True, decomposed=decomposed, case_kw=case_kw,
    )
    seconds = time.perf_counter() - t0
    data = design_result_to_json(res)
    metric = data["metric"]
    ratio = data["target"] / data["baseline"][metric]

    set_stage(job_id, "validate")
    fem_label = payload.get("_resolved_fem_name", "FEM validator")
    print(f"Checking the design with {fem_label} ...", flush=True)
    start, found = _fem_design_check(bench, params, [data["param_baseline"], data["p_star"]],
                                     fem_pref=payload.get("_resolved_fem"))
    fem_target = ratio * start[metric]
    check = dict(target_ratio=ratio, fem_baseline=start[metric], fem_at_design=found[metric], fem_target=fem_target,
                 error_vs_target=found[metric] / fem_target - 1.0,
                 atlas_vs_fem=data["metrics"][metric] / found[metric] - 1.0)
    check["within_10pct"] = bool(abs(check["error_vs_target"]) < 0.10)
    print(f"FEM check: {metric} at the design is {check['error_vs_target']:+.2%} from the target; "
          f"ATLAS vs FEM there {check['atlas_vs_fem']:+.2%}", flush=True)

    set_stage(job_id, "plots")
    hist = [r for r in data["history"] if r["tag"] not in ("probe", "return")]
    metrics = dict(data, param=data["param_name"], initial=data["param_baseline"], final=data["p_star"],
                   objective=INVERSE_OBJECTIVE.get(metric, metric), n_outer=len(hist), train_seconds=seconds,
                   formulation=form, custom=payload["custom"], settings=live.values(bench, params),
                   device=device, device_name=device_name, fem_engine=fem_label,
                   precision=payload.get("precision", "float64"),
                   hidden_layers=hidden_layers, neurons=neurons, seed=seed,
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
    return {
        "benchmarks": [bench_meta(b) for b in PAPER + MULTI],
        "lbfgs_iters": LBFGS_ITERS,
        "max_lbfgs": MAX_LBFGS,
        "seed": SEED,
        "defaults": {
            "formulation": "energy_xpinn",
            "epochs": 2000,
            "lbfgs": LBFGS_ITERS,
            "hidden_layers": HIDDEN_LAYERS,
            "neurons": NEURONS,
            "lr": DEFAULT_LR,
            "precision": "float64",
            "seed": SEED,
            "warm_start": DEFAULT_WARM_START,
            "al_stages": DEFAULT_AL_STAGES,
            "gamma_u": DEFAULT_GAMMA_U,
            "gamma_t": DEFAULT_GAMMA_T,
            "forward_epochs": 2000,
            "forward_lbfgs": 2000,
            "n_outer": 12,
        },
    }


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
            res["inverse"].update({k: v for k, v in meta["inverse"].items() if k != "baseline" or res["inverse"].get("baseline") is None})
        out["benchmarks"][b] = dict(meta, **res)
    return out


def _system_snapshot() -> dict:
    try:
        import torch
        torch_ver = torch.__version__
    except Exception as e:
        torch_ver = str(e)
    has_cuda, gpu_name, cuda_ver, gpu_mem = cuda_details()
    eff_dev, eff_name = resolve_device(None, strict_cuda=False)
    fenics_ok, fenics_ver = fenics_status()
    builtin_ok, builtin_ver = builtin_fem_status()
    eff_fem, eff_fem_name = resolve_fem_backend(None, strict=False)
    p_count, p_bytes = plots_storage()
    with JOBS_LOCK:
        active = sum(1 for j in JOBS.values() if j["status"] in ("queued", "running"))
        running_job = next((j for j in JOBS.values() if j["status"] == "running"), None)
        total = len(JOBS)
    return {
        "status": "ok",
        "device": eff_dev,
        "device_name": eff_name,
        "selected_device": SETTINGS.get("device", "auto"),
        "cuda_available": has_cuda,
        "gpu_name": gpu_name,
        "cuda_version": cuda_ver,
        "gpu_memory": gpu_mem,
        "cpu_count": CPU_COUNT,
        "cpu_threads": SETTINGS.get("cpu_threads", CPU_COUNT),
        "torch": torch_ver,
        "fem_backend": eff_fem,
        "fem_backend_name": eff_fem_name,
        "selected_fem_backend": SETTINGS.get("fem_backend", "auto"),
        "fenics_available": fenics_ok,
        "fenics_version": fenics_ver,
        "builtin_fem_available": builtin_ok,
        "builtin_fem_version": builtin_ver,
        "app_mode": os.environ.get("ATLAS_APP_MODE", "browser"),
        "platform": platform.system(),
        "plots_count": p_count,
        "plots_bytes": p_bytes,
        "active_jobs": active,
        "running_device": running_job.get("device") if running_job else None,
        "total_jobs": total,
        "settings": dict(SETTINGS),
    }


@app.get("/api/health")
def health():
    return _system_snapshot()


@app.get("/api/settings")
def get_settings():
    return _system_snapshot()


@app.post("/api/settings")
def update_settings(req: SettingsUpdate):
    with SETTINGS_LOCK:
        if req.device is not None:
            if req.device == "cuda" and not cuda_details()[0]:
                raise HTTPException(
                    status_code=400,
                    detail="CUDA GPU is not available in this Python environment.",
                )
            SETTINGS["device"] = req.device
        if req.fem_backend is not None:
            resolve_fem_backend(req.fem_backend, strict=True)
            SETTINGS["fem_backend"] = req.fem_backend
        if req.cpu_threads is not None:
            SETTINGS["cpu_threads"] = min(CPU_COUNT, max(1, int(req.cpu_threads)))
            _apply_cpu_threads(SETTINGS["cpu_threads"])
        if req.plot_dpi is not None:
            SETTINGS["plot_dpi"] = min(400, max(72, int(req.plot_dpi)))
        if req.poll_ms is not None:
            SETTINGS["poll_ms"] = min(10000, max(500, int(req.poll_ms)))
        if req.auto_open_log is not None:
            SETTINGS["auto_open_log"] = bool(req.auto_open_log)
        _save_app_settings()
    return _system_snapshot()


class FemTestRequest(BaseModel):
    fem_backend: Optional[str] = Field(None, pattern="^(auto|fenics|builtin)$")


@app.post("/api/fem/test")
def test_fem_backend(req: FemTestRequest = FemTestRequest()):
    engine, engine_name = resolve_fem_backend(req.fem_backend, strict=True)
    t0 = time.perf_counter()
    res = _fem_design_check("cantilever", {}, [1.0], fem_pref=engine)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    comp = float(res[0]["compliance"])
    return {
        "status": "ok",
        "engine": engine,
        "engine_name": engine_name,
        "benchmark": "cantilever",
        "compliance": comp,
        "elapsed_ms": elapsed_ms,
    }


@app.get("/api/workspace")
def workspace_files():
    plot_items = []
    for p in sorted(PLOTS_DIR.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_file():
            st = p.stat()
            plot_items.append({
                "name": p.name,
                "url": f"/static/plots/{p.name}",
                "bytes": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "is_image": p.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg", ".webp"),
            })

    result_items = []
    res_bytes = 0
    if STORED_RESULTS.exists():
        for p in sorted(STORED_RESULTS.rglob("*")):
            if not p.is_file():
                continue
            st = p.stat()
            res_bytes += st.st_size
            if p.suffix.lower() in (".png", ".json", ".md") and len(result_items) < 120:
                rel = p.relative_to(STORED_RESULTS).as_posix()
                result_items.append({
                    "path": rel,
                    "group": rel.split("/")[0],
                    "url": f"/results/{rel}",
                    "bytes": st.st_size,
                    "is_image": p.suffix.lower() == ".png",
                    "ext": p.suffix.lower().lstrip("."),
                })

    ref_groups: Dict[str, dict] = {}
    ref_bytes = 0
    if FEM_REFERENCE_DIR.exists():
        for p in sorted(FEM_REFERENCE_DIR.iterdir()):
            if not p.is_file():
                continue
            st = p.stat()
            ref_bytes += st.st_size
            stem = p.name
            for suf in ("_coords.npy", "_u.npy", "_v.npy", "_vonmises.npy", "_meta.json"):
                if stem.endswith(suf):
                    stem = stem[: -len(suf)]
                    break
            g = ref_groups.setdefault(stem, {"key": stem, "files": 0, "bytes": 0, "points": None, "compliance": None})
            g["files"] += 1
            g["bytes"] += st.st_size
            if p.name.endswith("_meta.json"):
                try:
                    meta = json.loads(p.read_text())
                    g["compliance"] = meta.get("compliance")
                    g["points"] = meta.get("n_points") or meta.get("points")
                except Exception:
                    pass
            elif p.name.endswith("_u.npy") and g["points"] is None:
                try:
                    arr = np.load(p, mmap_mode="r")
                    g["points"] = int(arr.size)
                except Exception:
                    pass

    return {
        "plots": {
            "path": str(PLOTS_DIR),
            "count": len(plot_items),
            "bytes": sum(i["bytes"] for i in plot_items),
            "items": plot_items,
        },
        "results": {
            "path": str(STORED_RESULTS),
            "count": len(result_items),
            "bytes": res_bytes,
            "items": result_items,
        },
        "references": {
            "path": str(FEM_REFERENCE_DIR),
            "count": len(ref_groups),
            "bytes": ref_bytes,
            "items": list(ref_groups.values()),
        },
    }


@app.post("/api/cuda/clear")
def clear_cuda_cache():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return _system_snapshot()


@app.post("/api/forward")
def forward(req: ForwardRequest):
    return {"job_id": submit("forward", _model_dump(req)), "status": "queued"}


@app.post("/api/inverse")
def inverse(req: InverseRequest):
    return {"job_id": submit("inverse", _model_dump(req)), "status": "queued"}


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


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        if job["status"] == "queued":
            job["cancel_requested"] = True
            job["status"] = "cancelled"
            job["stage"] = "cancelled"
            job["finished"] = datetime.now().isoformat(timespec="seconds")
        elif job["status"] == "running":
            job["cancel_requested"] = True
        return dict(job)


@app.get("/api/jobs/{job_id}/export")
def export_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        payload = {k: v for k, v in job.items() if k != "cancel_requested"}
    filename = f"atlas_{job['benchmark']}_{job_id}.json"
    return JSONResponse(
        content=payload,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/jobs/clear")
def clear_jobs(remove_plots: bool = True):
    with JOBS_LOCK:
        keep = {k: v for k, v in JOBS.items() if v["status"] in ("queued", "running")}
        removed = len(JOBS) - len(keep)
        active_prefixes = tuple(keep.keys())
        JOBS.clear()
        JOBS.update(keep)
    deleted_files = 0
    if remove_plots:
        for item in PLOTS_DIR.iterdir():
            if active_prefixes and any(item.name.startswith(p) for p in active_prefixes):
                continue
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                deleted_files += 1
            except Exception:
                pass
    snap = _system_snapshot()
    snap.update(removed_jobs=removed, deleted_files=deleted_files)
    return snap


class FolderOpenRequest(BaseModel):
    target: str = Field("plots", pattern="^(plots|results|references)$")


def _open_local_dir(request: Request, target_dir: Path) -> dict:
    client_host = request.client.host if request.client else ""
    if client_host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        return {"opened": False, "path": str(target_dir), "detail": "Remote preview mode — path shown below."}
    target_dir.mkdir(parents=True, exist_ok=True)
    rel = platform.release().lower()
    is_wsl = bool(os.environ.get("WSL_DISTRO_NAME")) or "microsoft" in rel
    try:
        if is_wsl and shutil.which("wslpath") and shutil.which("explorer.exe"):
            win_path = subprocess.check_output(["wslpath", "-w", str(target_dir)], text=True).strip()
            subprocess.Popen(["explorer.exe", win_path])
        elif sys.platform == "win32":
            os.startfile(str(target_dir))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target_dir)])
        elif shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", str(target_dir)])
        else:
            return {"opened": False, "path": str(target_dir)}
        return {"opened": True, "path": str(target_dir)}
    except Exception as exc:
        return {"opened": False, "path": str(target_dir), "detail": str(exc)}


@app.post("/api/open-plots")
def open_plots_folder(request: Request):
    return _open_local_dir(request, PLOTS_DIR)


@app.post("/api/open-folder")
def open_workspace_folder(request: Request, req: FolderOpenRequest = FolderOpenRequest()):
    mapping = {
        "plots": PLOTS_DIR,
        "results": STORED_RESULTS,
        "references": FEM_REFERENCE_DIR,
    }
    return _open_local_dir(request, mapping[req.target])


@app.get("/icon.svg")
def app_icon_svg():
    return FileResponse(FRONTEND_DIR / "icon.svg", media_type="image/svg+xml")


@app.get("/favicon.ico")
def app_favicon():
    ico = FRONTEND_DIR / "icon.ico"
    if ico.exists():
        return FileResponse(ico, media_type="image/x-icon")
    return FileResponse(FRONTEND_DIR / "icon.svg", media_type="image/svg+xml")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("ATLAS_PORT", "8000")))
