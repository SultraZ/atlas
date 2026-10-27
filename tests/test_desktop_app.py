"""End-to-end verification of the ATLAS Desktop Application backend, GPU/CPU
switching, adjustable runtime/solver settings, live CPU forward execution
against genuine stored FEniCS references, export, cleanup, and desktop launcher.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from webapp import desktop
from webapp.backend import app as backend_app


def _wait_job(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200
        last = r.json()
        if last["status"] in ("done", "error", "cancelled"):
            return last
        time.sleep(0.25)
    raise TimeoutError(f"Job {job_id} timed out: {last.get('status')} / {last.get('stage')}")


def test_desktop_launcher_helpers(tmp_path):
    import zipfile
    from scripts import build_windows_exes, create_release

    port = desktop.find_free_port("127.0.0.1")
    assert isinstance(port, int) and port > 0
    assert (ROOT / "launch_atlas.sh").exists()
    assert (ROOT / "setup_atlas.sh").exists()
    assert (ROOT / "ATLAS-Setup.exe").exists()
    assert (ROOT / "ATLAS.exe").exists()
    assert (ROOT / "installer" / "Uninstall-ATLAS.exe").exists()
    assert (ROOT / "scripts" / "create_release.py").exists()
    assert (ROOT / "webapp" / "frontend" / "icon.ico").exists()
    assert os.access(ROOT / "launch_atlas.sh", os.X_OK)
    assert os.access(ROOT / "setup_atlas.sh", os.X_OK)
    assert backend_app._win_to_wsl_path(r"\\wsl.localhost\Ubuntu\home\sultra\atlas") == "/home/sultra/atlas"
    assert backend_app._win_to_wsl_path(r"\\wsl$\Ubuntu-22.04\home\sultra\atlas\fem") == "/home/sultra/atlas/fem"

    # Verify PE headers, Windows GUI subsystem, and Publisher/Author = 'ATLAS' metadata on all 3 EXEs
    m_setup = build_windows_exes.verify_exe_metadata(ROOT / "ATLAS-Setup.exe", "ATLAS Setup Wizard")
    m_app = build_windows_exes.verify_exe_metadata(ROOT / "ATLAS.exe", "ATLAS Desktop Studio")
    m_uninst = build_windows_exes.verify_exe_metadata(ROOT / "installer" / "Uninstall-ATLAS.exe", "ATLAS Uninstaller")
    for meta in (m_setup, m_app, m_uninst):
        assert meta["CompanyName"] == "ATLAS"
        assert meta["Publisher"] == "ATLAS"
        assert meta["Author"] == "ATLAS"
        assert meta["FileVersion"] == "1.0.0"
        assert meta["ProductVersion"] == "1.0.0"

    wizard_c = (ROOT / "installer" / "setup_wizard.c").read_text(encoding="utf-8")
    assert "ID_BTN_UPDATE" in wizard_c
    assert 'L"Update"' in wizard_c
    assert "v1.0.0" in wizard_c

    # Execute standalone installer verification scripts AND guard against PowerShell traps
    import subprocess
    ps1_text = (ROOT / "installer" / "setup_worker.ps1").read_text(encoding="utf-8")
    assert '\\"' not in ps1_text, "Backslash-quote detected in PowerShell script!"
    assert "Invoke-Native" not in ps1_text, "Pipeline-capturing function detected in PowerShell script!"
    assert " -c " not in ps1_text.split("# Provisioning Miniconda")[0].split("$wslCmd")[0], (
        "Inline python -c string detected in PowerShell section!"
    )
    assert "ATLAS_ACTION" in ps1_text and "v1.0.0" in ps1_text
    assert "please close and re-open ATLAS-Setup.exe" not in ps1_text
    assert "Install-OfficialPython312" in ps1_text
    assert "PythonCore" in ps1_text
    assert "python-3.12.10-amd64.exe" in ps1_text
    assert "pip uninstall -y torch" in ps1_text, "Must uninstall CPU PyTorch before installing CUDA wheel!"
    for tag in ("cu130", "cu128", "cu126", "cu124"):
        assert tag in ps1_text, f"Missing CUDA index tag {tag} in setup_worker.ps1!"

    subprocess.run([sys.executable, str(ROOT / "installer" / "check_python.py")], check=True)
    cuda_probe = subprocess.run([sys.executable, str(ROOT / "installer" / "check_cuda.py")])
    assert cuda_probe.returncode in (0, 2, 3)
    subprocess.run([sys.executable, str(ROOT / "installer" / "verify_install.py")], check=True)

    out_dir, assets, notes_file = create_release.build_release_bundle("v1.0.0", out_root=tmp_path)
    assert out_dir.is_dir()
    assert notes_file.is_file()
    names = {a.name for a in assets}
    assert {
        "ATLAS-Setup.exe",
        "setup_atlas.sh",
        "ATLAS-Desktop-Studio-v1.0.0.zip",
        "ATLAS-Desktop-Studio.zip",
        "SHA256SUMS.txt",
    } == names
    with zipfile.ZipFile(out_dir / "ATLAS-Desktop-Studio.zip") as zf:
        znames = set(zf.namelist())
        assert "ATLAS-Desktop-Studio-v1.0.0/ATLAS-Setup.exe" in znames
        assert "ATLAS-Desktop-Studio-v1.0.0/ATLAS.exe" in znames
        assert "ATLAS-Desktop-Studio-v1.0.0/installer/Uninstall-ATLAS.exe" in znames
        assert "ATLAS-Desktop-Studio-v1.0.0/installer/check_python.py" in znames
        assert "ATLAS-Desktop-Studio-v1.0.0/installer/check_cuda.py" in znames
        assert "ATLAS-Desktop-Studio-v1.0.0/installer/verify_install.py" in znames
        assert "ATLAS-Desktop-Studio-v1.0.0/atlas/native_fem.py" in znames
        assert "ATLAS-Desktop-Studio-v1.0.0/webapp/desktop.py" in znames
        assert not any(n.endswith((".npz", ".pt", ".log", ".out")) for n in znames)
        assert not any("/.venv/" in n or "/.git/" in n for n in znames)
        assert not any(n.split("/")[1] in ("tests", "scripts") or "/inverse_design/seeds/" in n for n in znames)


def test_health_and_adjustable_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_app, "SETTINGS_FILE", tmp_path / ".app_settings.json")
    client = TestClient(backend_app.app)

    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    assert h["device"] in ("cpu", "cuda")
    assert "fenics_available" in h
    assert "cpu_count" in h and h["cpu_count"] >= 1
    assert "settings" in h

    # Switch explicitly to CPU, adjust threads, plot DPI, polling cadence, and auto_open_log
    r = client.post("/api/settings", json={
        "device": "cpu",
        "cpu_threads": 2,
        "plot_dpi": 180,
        "poll_ms": 750,
        "auto_open_log": False,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["device"] == "cpu"
    assert body["selected_device"] == "cpu"
    assert body["cpu_threads"] == 2
    assert body["settings"]["device"] == "cpu"
    assert body["settings"]["plot_dpi"] == 180
    assert body["settings"]["poll_ms"] == 750
    assert body["settings"]["auto_open_log"] is False
    assert (tmp_path / ".app_settings.json").exists()

    # If CUDA is unavailable, explicitly requesting 'cuda' raises a clear 400 error
    if not h["cuda_available"]:
        bad = client.post("/api/settings", json={"device": "cuda"})
        assert bad.status_code == 400
        assert "CUDA" in bad.json()["detail"]

    # Switch back to auto cleanly
    ok = client.post("/api/settings", json={"device": "auto", "auto_open_log": True})
    assert ok.status_code == 200
    assert ok.json()["settings"]["device"] == "auto"

    # CUDA cache clear & runtime setup endpoints always respond cleanly
    cc = client.post("/api/cuda/clear")
    assert cc.status_code == 200
    assert cc.json()["status"] == "ok"
    assert "system_gpu_name" in cc.json()
    cs = client.post("/api/cuda/setup")
    assert cs.status_code == 200
    assert cs.json()["status"] in ("already_active", "manual", "launching_installer")
    up = client.post("/api/app/update")
    assert up.status_code == 200
    assert up.json()["status"] in ("manual", "launching_updater")


def test_static_frontend_and_study_results():
    client = TestClient(backend_app.app)
    idx = client.get("/")
    assert idx.status_code == 200
    assert 'id="sidebar"' in idx.text
    assert 'data-view="results"' in idx.text
    assert 'data-view="live"' in idx.text
    assert 'data-view="files"' in idx.text
    assert 'data-view="settings"' in idx.text
    assert 'id="computeBox"' in idx.text
    assert 'id="solverCfg"' in idx.text
    assert 'data-theme-pick="atlas"' not in idx.text
    assert 'id="flow"' not in idx.text
    assert 'class="contours' not in idx.text
    assert "Inter" in idx.text

    css_text = (ROOT / "webapp" / "frontend" / "style.css").read_text(encoding="utf-8")
    js_text = (ROOT / "webapp" / "frontend" / "app.js").read_text(encoding="utf-8")
    assert 'data-theme="atlas"' not in css_text
    assert "view-transition-new(root)" not in css_text
    assert "--mx" not in css_text and "--mx" not in js_text
    assert "startViewTransition" not in js_text

    assert client.get("/static/icon.svg").status_code == 200
    assert client.get("/icon.svg").status_code == 200
    assert client.get("/favicon.ico").status_code == 200

    ws = client.get("/api/workspace")
    assert ws.status_code == 200
    ws_data = ws.json()
    assert set(ws_data.keys()) == {"plots"}
    assert "results" not in ws_data
    assert "references" not in ws_data

    allowed_prefixes = ("cantilever_", "l_bracket_", "plate_hole_", "steel_insert_", "bimaterial_c15_", "filled_slot_")
    for ref_file in (ROOT / "fem" / "reference").iterdir():
        if ref_file.is_file():
            assert ref_file.name.startswith(allowed_prefixes), f"Unexpected FEniCS reference file: {ref_file.name}"

    fem_chk = client.post("/api/fem/test", json={"fem_backend": "builtin"})
    assert fem_chk.status_code == 200
    assert fem_chk.json()["engine"] == "builtin"
    assert abs(fem_chk.json()["compliance"] - 1272.36) < 2.0

    res = client.get("/api/results")
    assert res.status_code == 200
    data = res.json()
    assert {"cantilever", "plate_hole", "l_bracket", "steel_insert"}.issubset(set(data["order"]))
    for bkey in ("cantilever", "plate_hole", "l_bracket"):
        assert data["benchmarks"][bkey]["solvers"]["atlas"]["params"] > 10000
        assert len(data["benchmarks"][bkey]["plots"]) >= 5

    benches = client.get("/api/benchmarks")
    assert benches.status_code == 200
    assert len(benches.json()["benchmarks"]) == 6


def test_live_cpu_forward_xpinn_and_single_network_with_custom_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_app, "SETTINGS_FILE", tmp_path / ".app_settings.json")
    client = TestClient(backend_app.app)

    # 1. Fast forward run on CPU using single-network ATLAS ('energy') with custom 3x24 network
    sub = client.post("/api/forward", json={
        "benchmark": "cantilever",
        "device": "cpu",
        "formulation": "energy",
        "epochs": 50,
        "lbfgs": 0,
        "hidden_layers": 3,
        "neurons": 24,
        "lr": 0.003,
        "precision": "float64",
        "seed": 43,
    })
    assert sub.status_code == 200
    jid = sub.json()["job_id"]

    done = _wait_job(client, jid, timeout=90.0)
    assert done["status"] == "done", f"Unexpected job error: {done.get('error')}\n{done.get('log')}"
    assert done["device"] == "cpu"
    assert done["formulation"] == "energy"
    mx = done["metrics"]
    assert mx["device"] == "cpu"
    assert mx["hidden_layers"] == 3
    assert mx["neurons"] == 24
    assert mx["seed"] == 43
    assert mx["subdomains"] == 1
    assert mx["relative_L2_displacement"] > 0.0
    assert set(done["plots"].keys()) >= {"fem_comparison", "deformed", "convergence"}

    # Verify JSON export works and contains metrics
    exp = client.get(f"/api/jobs/{jid}/export")
    assert exp.status_code == 200
    assert exp.json()["id"] == jid
    assert exp.json()["metrics"]["seed"] == 43

    # 2. Fast forward run on CPU using ATLAS-XPINN ('energy_xpinn') on multi-material steel_insert
    sub2 = client.post("/api/forward", json={
        "benchmark": "steel_insert",
        "device": "cpu",
        "formulation": "energy_xpinn",
        "epochs": 50,
        "lbfgs": 0,
        "warm_start": 0,
        "al_stages": 1,
        "hidden_layers": 2,
        "neurons": 16,
        "precision": "float32",
        "seed": 42,
    })
    assert sub2.status_code == 200
    jid2 = sub2.json()["job_id"]
    done2 = _wait_job(client, jid2, timeout=90.0)
    assert done2["status"] == "done", f"Unexpected XPINN job error: {done2.get('error')}\n{done2.get('log')}"
    assert done2["device"] == "cpu"
    assert done2["formulation"] == "energy_xpinn"
    assert done2["metrics"]["precision"] == "float32"
    assert done2["metrics"]["subdomains"] == 2

    # 3. Custom-geometry live forward solve on CPU using Built-in P2 FEM (zero WSL/FEniCS required!)
    sub3 = client.post("/api/forward", json={
        "benchmark": "cantilever",
        "device": "cpu",
        "fem_backend": "builtin",
        "formulation": "energy",
        "epochs": 50,
        "lbfgs": 0,
        "hidden_layers": 2,
        "neurons": 16,
        "seed": 42,
        "params": {"H": 1.3},
    })
    assert sub3.status_code == 200
    jid3 = sub3.json()["job_id"]
    done3 = _wait_job(client, jid3, timeout=90.0)
    assert done3["status"] == "done", f"Unexpected custom job error: {done3.get('error')}\n{done3.get('log')}"
    assert done3["custom"] is True
    assert done3["metrics"]["fem_cells"] > 1000
    assert "SciPy" in done3["metrics"]["fem_engine"]

    # 4. Clear job history and generated plots cleanly
    clr = client.post("/api/jobs/clear")
    assert clr.status_code == 200
    assert clr.json()["removed_jobs"] >= 3


def test_sidebar_cleanup_and_consistent_simple_expert_ui(tmp_path, monkeypatch):
    import re
    import subprocess

    monkeypatch.setattr(backend_app, "SETTINGS_FILE", tmp_path / ".app_settings.json")
    client = TestClient(backend_app.app)

    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "no-cache" in root_resp.headers.get("cache-control", "")

    html = (ROOT / "webapp" / "frontend" / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "webapp" / "frontend" / "style.css").read_text(encoding="utf-8")
    js = (ROOT / "webapp" / "frontend" / "app.js").read_text(encoding="utf-8")

    # 1. Sidebar has NO Appearance quick section, while keeping GPU runtime button (#health) in .side-bottom
    assert "side-quick" not in html
    assert "side-seg" not in html
    assert "side-quick" not in css
    assert "data-theme-pick" not in html
    assert "data-level-pick" not in html
    side_match = re.search(r'<aside class="sidebar"[^>]*>(.*?)</aside>', html, flags=re.S)
    assert side_match is not None, "Missing sidebar in index.html!"
    assert 'data-view="settings"' in side_match.group(1)
    side_bottom_match = re.search(r'<div class="side-bottom">(.*?)</div>\s*</aside>', html, flags=re.S)
    assert side_bottom_match is not None, "Missing .side-bottom in sidebar!"
    assert 'id="health"' in side_bottom_match.group(1)

    # 2. Appearance & Detail level controls exist inside Settings view only
    settings_fn = js.split("function renderSettingsDrawer()")[1].split("async function pollHealth()")[0]
    assert '["light", "Light"' in settings_fn and '["dark", "Dark"' in settings_fn
    assert 'data-theme-pick="${id}"' in settings_fn
    assert 'data-level-pick="simple"' in settings_fn
    assert 'data-level-pick="expert"' in settings_fn
    assert js.count('data-theme-pick="') == 1
    assert js.count('data-level-pick="') == 2

    # 3. No word-swapping tables or S(...) rewrites remain in app.js
    assert re.search(r"\bS\(", js) is None, "Legacy S(...) word-rewriting call left in app.js!"
    for banned in ("SOLVER_NAMES_SIMPLE", "FIGS_SIMPLE", "STEPS_SIMPLE", "SIMPLE_LABEL", "SIMPLE_SUB"):
        assert banned not in js, f"Legacy word-swapping table {banned} still in app.js!"
    assert "const extra =" in js

    # 4. Session Runs view contains ONLY session runs & saved plots (no Study Figures/JSON, FEniCS References, or Study Files)
    for removed_ui in (
        "Study Figures",
        "FEniCS References",
        "Study Files",
        "Open Study Results",
        "Open FEniCS Refs",
        "data-files-tab",
        "data-files-filter",
    ):
        assert removed_ui not in js and removed_ui not in html, f"Removed UI element still found: {removed_ui}"

    # 5. Live Solver Run length has NO time estimation & no duplicate cross-view buttons remain
    assert "runEstimate" not in js
    assert "ADAM_SHARE" not in js
    for dup_btn in (
        "Browse in Folder View",
        "Saved Runs &amp; Plots",
        "Open folder",
        "quickTestFem",
        "openPlotsBtn",
        "data-try-bench",
        "data-view-jump",
    ):
        assert dup_btn not in js and dup_btn not in html, f"Duplicate navigation/action button still present: {dup_btn}"

    # 6. Repository reclean check: zero .npz/.pt/.log/.out bloat in results/ or discarded study scripts
    assert not list((ROOT / "results").rglob("*.npz"))
    assert not list((ROOT / "results").rglob("*.pt"))
    assert not list((ROOT / "results").rglob("*.log"))
    assert not (ROOT / "atlas" / "data").exists()
    assert not (ROOT / "fem" / "fenics_scripts").exists()
    assert {p.name for p in (ROOT / "scripts").glob("*.py")} == {"build_windows_exes.py", "create_release.py", "run_inverse_seed_study.py", "run_multi_inverse_design.py"}

    subprocess.run(["node", "--check", str(ROOT / "webapp" / "frontend" / "app.js")], check=True)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
