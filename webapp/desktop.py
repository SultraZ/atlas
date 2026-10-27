#!/usr/bin/env python3
"""Standalone Desktop Application launcher for ATLAS.

Runs the untouched ATLAS Python + CUDA/CPU + FEniCS backend locally and opens
a dedicated desktop application window (native pywebview or chromeless app window).

Usage (from project root):
    python webapp/desktop.py
    python webapp/desktop.py --device cpu --threads 4
    python webapp/desktop.py --device cuda --live
    python webapp/desktop.py --no-window --port 8000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import List, Optional

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    ROOT = Path(getattr(sys, "_MEIPASS"))
else:
    ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

APP_TITLE = "ATLAS — Structural Design Console"
DEFAULT_WIDTH = 1440
DEFAULT_HEIGHT = 920
MIN_WIDTH = 960
MIN_HEIGHT = 640
APP_ID = "ATLAS.DesktopStudio"
ICON_ICO = ROOT / "webapp" / "frontend" / "icon.ico"


def _set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass


def _apply_window_icon(title: str) -> None:
    if sys.platform != "win32" or not ICON_ICO.exists():
        return
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.FindWindowW.restype = wintypes.HWND
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT, ctypes.c_int, ctypes.c_int, wintypes.UINT]
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        hwnd = None
        for _ in range(50):
            hwnd = user32.FindWindowW(None, title)
            if hwnd:
                break
            time.sleep(0.1)
        if not hwnd:
            return
        for which, metric in ((0, 49), (1, 11)):
            size = user32.GetSystemMetrics(metric)
            hicon = user32.LoadImageW(None, str(ICON_ICO), 1, size, size, 0x10)
            if hicon:
                user32.SendMessageW(hwnd, 0x80, which, hicon)
    except Exception:
        pass


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    return "microsoft" in platform.release().lower()


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_for_backend(url: str, timeout: float = 30.0) -> Optional[dict]:
    deadline = time.monotonic() + timeout
    health_url = f"{url.rstrip('/')}/api/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.5) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception:
            time.sleep(0.15)
    return None


class DesktopBridge:
    """JS bridge exposed to `window.pywebview.api` when running inside pywebview."""

    def __init__(self, server_url: str):
        self.server_url = server_url
        self._window = None

    def attach(self, window) -> None:
        self._window = window

    def info(self) -> dict:
        return {"shell": "pywebview", "server_url": self.server_url}


def _try_pywebview(url: str) -> bool:
    if is_wsl() and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        import webview  # type: ignore

        bridge = DesktopBridge(url)
        window = webview.create_window(
            APP_TITLE,
            url,
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            min_size=(MIN_WIDTH, MIN_HEIGHT),
            text_select=True,
            js_api=bridge,
        )
        bridge.attach(window)
        window.events.shown += lambda: _apply_window_icon(APP_TITLE)
        webview.start(debug=False, icon=str(ICON_ICO) if ICON_ICO.exists() else None)
        return True
    except Exception as exc:
        print(f"[ATLAS] Native webview unavailable ({exc}); opening standalone app window instead.")
        return False


def _wsl_windows_profile_dir() -> tuple[Optional[str], Optional[Path]]:
    try:
        out = subprocess.check_output(
            ["cmd.exe", "/c", "echo %TEMP%"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        if out and "%" not in out:
            win_dir = out + rf"\ATLAS-Window-{os.getpid()}"
            wsl_dir = Path(subprocess.check_output(["wslpath", "-u", win_dir], text=True, timeout=5).strip())
            wsl_dir.mkdir(parents=True, exist_ok=True)
            return win_dir, wsl_dir
    except Exception:
        pass
    return None, None


def _candidate_app_browsers() -> List[tuple[str, bool]]:
    """Return `(executable, is_windows_binary_under_wsl)` candidates supporting `--app=<url>`."""
    found: List[tuple[str, bool]] = []
    if is_wsl():
        win_bins = [
            "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
            "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe",
            "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
            "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        ]
        for b in win_bins:
            if os.path.isfile(b):
                found.append((b, True))
    if sys.platform == "win32":
        local_app = os.environ.get("LOCALAPPDATA", "")
        win_native = [
            r"C:\Program Files (x86)\Microsoft\Edge/Application/msedge.exe".replace("/", "\\"),
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.join(local_app, r"Microsoft\Edge\Application\msedge.exe"),
            os.path.join(local_app, r"Google\Chrome\Application\chrome.exe"),
        ]
        for b in win_native:
            if b and os.path.isfile(b):
                found.append((b, False))
    elif sys.platform == "darwin":
        mac_bins = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
        for b in mac_bins:
            if os.path.isfile(b):
                found.append((b, False))
    for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge", "brave-browser"):
        path = shutil.which(name)
        if path:
            found.append((path, False))
    return found


def _try_app_window(url: str, server_thread: threading.Thread) -> bool:
    for binary, win_under_wsl in _candidate_app_browsers():
        tmp_ctx = None
        wsl_profile_cleanup: Optional[Path] = None
        try:
            if win_under_wsl:
                profile_dir, wsl_profile_cleanup = _wsl_windows_profile_dir()
            else:
                tmp_ctx = tempfile.TemporaryDirectory(prefix="atlas_app_")
                profile_dir = tmp_ctx.name
            cmd = [
                binary,
                f"--app={url}",
                f"--window-size={DEFAULT_WIDTH},{DEFAULT_HEIGHT}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-mode",
            ]
            if profile_dir:
                cmd.append(f"--user-data-dir={profile_dir}")
            t_launch = time.monotonic()
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc.wait()
            if time.monotonic() - t_launch < 3.0:
                print(f"[ATLAS] App window handed off to active browser process; keeping backend active at {url}. Press Ctrl+C to stop.", flush=True)
                while server_thread.is_alive():
                    time.sleep(0.5)
            return True
        except Exception:
            pass
        finally:
            if tmp_ctx is not None:
                try:
                    tmp_ctx.cleanup()
                except Exception:
                    pass
            if wsl_profile_cleanup is not None:
                shutil.rmtree(wsl_profile_cleanup, ignore_errors=True)
    return False


def main() -> int:
    _set_windows_app_id()
    log_path_env = os.environ.get("ATLAS_LOG_FILE")
    if log_path_env or sys.stdout is None or sys.stderr is None:
        log_file = Path(log_path_env) if log_path_env else (ROOT / "webapp" / "atlas_desktop.log")
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = log_file.open("w", encoding="utf-8", buffering=1, errors="replace")
            sys.stdout = fh
            sys.stderr = fh
        except Exception:
            pass
    else:
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    pass

    parser = argparse.ArgumentParser(description="Launch the ATLAS standalone desktop application.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=0, help="TCP port (default: 0 = pick a free local port)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=None,
                        help="Initial compute device override (auto, cuda, or cpu)")
    parser.add_argument("--threads", type=int, default=None, help="CPU worker thread limit for PyTorch")
    parser.add_argument("--shell", choices=("auto", "webview", "app-window"), default="auto",
                        help="Desktop window backend (default: auto)")
    parser.add_argument("--live", action="store_true", help="Open directly on the Live Solver tab")
    parser.add_argument("--no-window", action="store_true", help="Start local backend without opening a GUI window")
    args = parser.parse_args()

    os.environ["ATLAS_APP_MODE"] = "desktop"
    if args.device:
        os.environ["ATLAS_DEVICE"] = args.device
    if args.threads and args.threads > 0:
        os.environ["ATLAS_CPU_THREADS"] = str(args.threads)

    # Ensure conda bin path is on PATH so FEniCS (`dolfin` / `pkg-config`) resolves automatically
    py_bin = os.path.dirname(sys.executable)
    if py_bin and py_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = py_bin + os.pathsep + os.environ.get("PATH", "")

    port = args.port if args.port > 0 else find_free_port(args.host)

    import uvicorn
    from webapp.backend.app import app as fastapi_app

    config = uvicorn.Config(fastapi_app, host=args.host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, name="atlas-uvicorn", daemon=True)
    server_thread.start()

    connect_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    base_url = f"http://{connect_host}:{port}"
    target_url = f"{base_url}/#live" if args.live else f"{base_url}/"

    info = wait_for_backend(base_url, timeout=60.0)
    if info is None:
        print("[ATLAS] Backend failed to start within 60 seconds.", file=sys.stderr)
        server.should_exit = True
        return 1

    fem_str = f"{info.get('fem_backend_name', 'Built-in P2 FEM (SciPy)')} [{info.get('selected_fem_backend', 'auto')} -> {info.get('fem_backend', 'builtin')}]"
    print("=" * 68)
    print(f"  {APP_TITLE}")
    print(f"  Local endpoint : {base_url}")
    print(f"  Compute device : {info.get('device_name')} [{info.get('selected_device')} -> {info.get('device')}]")
    print(f"  FEM validation : {fem_str}")
    print("=" * 68, flush=True)

    try:
        if args.no_window:
            print("Running in local server mode (--no-window). Press Ctrl+C to stop.", flush=True)
            while server_thread.is_alive():
                time.sleep(0.5)
            return 0

        opened = False
        if args.shell in ("auto", "webview"):
            opened = _try_pywebview(target_url)
        if not opened and args.shell in ("auto", "app-window"):
            opened = _try_app_window(target_url, server_thread)
        if not opened:
            print(f"[ATLAS] Opening {target_url} in system browser. Press Ctrl+C to stop.", flush=True)
            webbrowser.open(target_url)
            while server_thread.is_alive():
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[ATLAS] Shutting down...", flush=True)
    finally:
        server.should_exit = True
        server_thread.join(timeout=4.0)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
