#!/usr/bin/env bash
# ==============================================================================
# ATLAS Desktop Studio — Guided Cross-Platform Setup Wizard (macOS & Linux)
# Works standalone (downloads release archive automatically) OR inside a clone!
# ==============================================================================
set -euo pipefail

REPO_SLUG="SultraZ/atlas"
APP_VERSION="v1.1.5"
ZIP_URL_PRI="https://github.com/${REPO_SLUG}/releases/latest/download/ATLAS-Desktop-Studio.zip"
ZIP_URL_ALT="https://github.com/${REPO_SLUG}/archive/refs/tags/${APP_VERSION}.zip"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/webapp/desktop.py" ]; then
    DEFAULT_DIR="${SCRIPT_DIR}"
else
    DEFAULT_DIR="${HOME}/.local/share/atlas-studio"
fi

AUTO_MODE=0
WITH_FENICS=0
NO_LAUNCH=0
DEVICE_PREF="auto"
TARGET_DIR="${DEFAULT_DIR}"

while [ $# -gt 0 ]; do
    case "$1" in
        --auto|--unattended|-y) AUTO_MODE=1 ;;
        --with-fenics)          WITH_FENICS=1 ;;
        --no-launch)            NO_LAUNCH=1 ;;
        --device)               DEVICE_PREF="${2:-auto}"; shift ;;
        --dir)                  TARGET_DIR="${2:-$DEFAULT_DIR}"; shift ;;
    esac
    shift
done

echo "======================================================================"
echo "  ATLAS Desktop Studio — Guided Setup (${APP_VERSION})"
echo "======================================================================"

OS_NAME="$(uname -s)"
HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    HAS_NVIDIA=1
    GPU_LABEL="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
else
    GPU_LABEL="CPU / Apple Silicon (no NVIDIA CUDA GPU detected)"
fi
echo "  Platform     : ${OS_NAME}"
echo "  Hardware     : ${GPU_LABEL}"
echo "  Install path : ${TARGET_DIR}"

if [ "${AUTO_MODE}" -eq 0 ] && [ "${OS_NAME}" = "Linux" ]; then
    echo ""
    echo "  Choose FEM Validation Edition:"
    echo "    [1] Standard Cross-Platform Edition (Recommended — Built-in SciPy P2 FEM)"
    echo "    [2] Full FEniCS Edition (Also creates Miniconda 'Atlas' FEniCS 2019.1.0 env)"
    printf "  Select [1/2] (default: 1): "
    read -r ans || ans="1"
    if [ "${ans}" = "2" ]; then
        WITH_FENICS=1
    fi
fi

mkdir -p "${TARGET_DIR}"
if [ ! -f "${TARGET_DIR}/webapp/desktop.py" ]; then
    echo ""
    echo "[1/5] Downloading ATLAS Desktop Studio release package..."
    TMP_ZIP="$(mktemp /tmp/atlas_release_XXXXXX.zip)"
    TMP_DIR="$(mktemp -d /tmp/atlas_extract_XXXXXX)"
    if ! curl -fsSL "${ZIP_URL_PRI}" -o "${TMP_ZIP}"; then
        curl -fsSL "${ZIP_URL_ALT}" -o "${TMP_ZIP}"
    fi
    unzip -q "${TMP_ZIP}" -d "${TMP_DIR}"
    INNER="$(find "${TMP_DIR}" -mindepth 1 -maxdepth 1 -type d | head -n1)"
    if [ -n "${INNER}" ] && [ -f "${INNER}/webapp/desktop.py" ]; then
        cp -R "${INNER}/." "${TARGET_DIR}/"
    else
        cp -R "${TMP_DIR}/." "${TARGET_DIR}/"
    fi
    rm -rf "${TMP_ZIP}" "${TMP_DIR}"
fi

chmod +x "${TARGET_DIR}/launch_atlas.sh" "${TARGET_DIR}/webapp/desktop.py" 2>/dev/null || true

PY_BIN=""
for cand in python3.11 python3.12 python3.10 python3; do
    if command -v "${cand}" >/dev/null 2>&1; then
        if "${cand}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
            PY_BIN="$(command -v "${cand}")"
            break
        fi
    fi
done

if [ -z "${PY_BIN}" ]; then
    echo "Error: Python 3.10+ is required. Install Python 3.11 (e.g. brew install python@3.11 or sudo apt install python3 python3-venv) and re-run."
    exit 1
fi

echo "[2/5] Setting up isolated Python environment (${PY_BIN})..."
if [ ! -x "${TARGET_DIR}/.venv/bin/python" ]; then
    "${PY_BIN}" -m venv "${TARGET_DIR}/.venv"
fi
"${TARGET_DIR}/.venv/bin/python" -m pip install --upgrade pip wheel -q

echo "[3/5] Installing PyTorch & ATLAS dependencies..."
if [ "${OS_NAME}" = "Linux" ] && [ "${HAS_NVIDIA}" -eq 1 ] && [ "${DEVICE_PREF}" != "cpu" ]; then
    if ! "${TARGET_DIR}/.venv/bin/python" "${TARGET_DIR}/installer/check_cuda.py" >/dev/null 2>&1; then
        "${TARGET_DIR}/.venv/bin/python" -m pip uninstall -y torch >/dev/null 2>&1 || true
        "${TARGET_DIR}/.venv/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || \
            "${TARGET_DIR}/.venv/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cu130 || \
            "${TARGET_DIR}/.venv/bin/python" -m pip install torch
    fi
elif [ "${OS_NAME}" = "Linux" ] && [ "${DEVICE_PREF}" != "cuda" ]; then
    "${TARGET_DIR}/.venv/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || \
        "${TARGET_DIR}/.venv/bin/python" -m pip install torch
else
    "${TARGET_DIR}/.venv/bin/python" -m pip install torch
fi
"${TARGET_DIR}/.venv/bin/python" -m pip install --prefer-binary -r "${TARGET_DIR}/requirements.txt"
"${TARGET_DIR}/.venv/bin/python" -m pip install --prefer-binary "pywebview>=5.0" || true

if [ "${WITH_FENICS}" -eq 1 ] && [ "${OS_NAME}" = "Linux" ]; then
    echo "[4/5] Configuring Miniconda 'Atlas' environment with genuine FEniCS 2019.1.0..."
    if [ ! -x "${HOME}/miniconda3/bin/conda" ]; then
        curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p "${HOME}/miniconda3"
        rm -f /tmp/miniconda.sh
    fi
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    if [ ! -x "${HOME}/miniconda3/envs/Atlas/bin/python" ]; then
        conda create -y -n Atlas -c conda-forge python=3.9 fenics=2019.1.0 mshr numpy scipy matplotlib
    fi
else
    echo "[4/5] Verifying Built-in Cross-Platform P2 FEM engine..."
fi

"${TARGET_DIR}/.venv/bin/python" "${TARGET_DIR}/installer/verify_install.py"

if [ "${OS_NAME}" = "Linux" ]; then
    APPS_DIR="${HOME}/.local/share/applications"
    mkdir -p "${APPS_DIR}"
    cat > "${APPS_DIR}/atlas-studio.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=ATLAS Desktop Studio
Comment=Adaptive Multi-Scale Deep-Energy & Finite-Element Studio
Exec="${TARGET_DIR}/launch_atlas.sh"
Path=${TARGET_DIR}
Icon=${TARGET_DIR}/webapp/frontend/icon.svg
Terminal=false
Categories=Science;Engineering;Education;
EOF
    chmod +x "${APPS_DIR}/atlas-studio.desktop"
fi
mkdir -p "${HOME}/.local/bin"
ln -sf "${TARGET_DIR}/launch_atlas.sh" "${HOME}/.local/bin/atlas-studio" 2>/dev/null || true

echo "[5/5] Setup complete! Launch anytime via: ${TARGET_DIR}/launch_atlas.sh"
if [ "${NO_LAUNCH}" -eq 0 ]; then
    exec "${TARGET_DIR}/launch_atlas.sh"
fi
