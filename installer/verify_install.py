#!/usr/bin/env python3
"""Standalone post-install self-test invoked by the ATLAS Setup Wizard at 93%."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import numpy
    import scipy
    import matplotlib
    import fastapi
    import uvicorn
    import pydantic
    import torch

    from atlas import native_fem
    from webapp import desktop

    res = native_fem.solve_forward("cantilever")
    comp = float(res["compliance"])
    if not (comp > 0.0):
        raise RuntimeError(f"Unexpected non-positive cantilever compliance: {comp}")

    port = desktop.find_free_port("127.0.0.1")
    gpu_info = (
        f"CUDA GPU ({torch.cuda.get_device_name(0)})"
        if torch.cuda.is_available()
        else f"CPU ({torch.__version__})"
    )
    print(
        f"ATLAS self-test OK | Python {sys.version.split()[0]} "
        f"| PyTorch {torch.__version__} [{gpu_info}] | SciPy {scipy.__version__} "
        f"| Cantilever compliance = {comp:.2f} N*m | Local port check = {port}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
