#!/usr/bin/env python3
"""Verify that this interpreter is standard GIL-enabled Python 3.10 - 3.14."""
from __future__ import annotations

import sys


def main() -> int:
    gil_ok = (not hasattr(sys, "_is_gil_enabled")) or bool(sys._is_gil_enabled())
    if (3, 10) <= sys.version_info < (3, 15) and gil_ok:
        print(sys.executable)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
