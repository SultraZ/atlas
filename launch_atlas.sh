#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN=""
for candidate in \
  "${ATLAS_PYTHON:-}" \
  "$HOME/miniconda3/envs/Atlas/bin/python" \
  "$HOME/anaconda3/envs/Atlas/bin/python" \
  "$HOME/miniforge3/envs/Atlas/bin/python" \
  "$HOME/mambaforge/envs/Atlas/bin/python" \
  "$ROOT/.venv/bin/python" \
  "$(command -v python3 || true)" \
  "$(command -v python || true)"
do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    if "$candidate" -c "import torch, fastapi, uvicorn" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Error: Could not find a Python environment with torch, fastapi, and uvicorn installed." >&2
  echo "Activate your Atlas conda environment or set ATLAS_PYTHON=/path/to/python." >&2
  exit 1
fi

export PATH="$(dirname "$PYTHON_BIN"):$PATH"
exec "$PYTHON_BIN" "$ROOT/webapp/desktop.py" "$@"
