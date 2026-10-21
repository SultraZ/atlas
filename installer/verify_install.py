#!/usr/bin/env python3
"""
Post-install verification script run automatically at 93% by ATLAS-Setup.exe
and setup_atlas.sh. Verifies all Python dependencies, Built-in P2 FEM physics
accuracy, and FastAPI desktop application imports cleanly without inline quotes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import fastapi  # noqa: F401
    import matplotlib  # noqa: F401
    import numpy as np
    import pydantic  # noqa: F401
    import scipy
    import torch
    import uvicorn  # noqa: F401

    from atlas.native_fem import solve_forward
    from webapp.backend import app as backend_app  # noqa: F401

    res = solve_forward("cantilever")
    comp = float(res["compliance"])
    if not (1200.0 <= comp <= 1350.0):
        raise RuntimeError(f"Unexpected cantilever compliance during self-test: {comp:.4f}")

    dev = "CUDA GPU" if torch.cuda.is_available() else "CPU"
    print(
        f"Verified ATLAS runtime (PyTorch {torch.__version__} [{dev}], "
        f"NumPy {np.__version__}, SciPy {scipy.__version__}, "
        f"Built-in P2 FEM compliance = {comp:.2f} N*m)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
