#!/usr/bin/env python3
"""Verify that this interpreter is 64-bit, GIL-enabled Python 3.10 - 3.14."""
from __future__ import annotations

import struct
import sys


def main() -> int:
    gil_ok = (not hasattr(sys, "_is_gil_enabled")) or bool(sys._is_gil_enabled())
    is_64bit = struct.calcsize("P") == 8
    if (3, 10) <= sys.version_info < (3, 15) and gil_ok and is_64bit:
        print(sys.executable)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
