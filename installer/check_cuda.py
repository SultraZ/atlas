#!/usr/bin/env python3
"""Standalone CUDA capability probe for the ATLAS Windows/Linux installer.

Exit codes:
  0 — PyTorch is installed AND torch.cuda.is_available() is True (GPU ready!)
  1 — PyTorch is not installed in this environment
  2 — PyTorch CUDA build is installed, but torch.cuda.is_available() is False
      (e.g. wheel CUDA toolkit version is newer than the installed NVIDIA driver)
  3 — CPU-only PyTorch build is installed (torch.version.cuda is None)
  4 — DLL load error while importing PyTorch (e.g. missing VC++ Redistributable)
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        print("PyTorch status: not installed")
        return 1
    except OSError as exc:
        print(f"PyTorch DLL load error: {exc}")
        return 4
    except Exception as exc:
        print(f"PyTorch import error: {exc}")
        return 1

    cuda_build = getattr(torch.version, "cuda", None)
    if not cuda_build:
        print(f"PyTorch {torch.__version__} installed (CPU-only build; no CUDA runtime bundled)")
        return 3

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(
            f"CUDA GPU Verified: {gpu_name} ({vram_gb:.1f} GB VRAM) "
            f"| PyTorch {torch.__version__} (CUDA {cuda_build})"
        )
        return 0

    print(
        f"PyTorch {torch.__version__} includes CUDA {cuda_build}, "
        "but torch.cuda.is_available() returned False (driver may require an older CUDA wheel)."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
