#!/usr/bin/env bash
# Run inside WSL (the Windows installer calls this automatically for the Full FEniCS edition):
#     bash installer/setup_fenics_wsl.sh
#
# Example:
#     wsl.exe -d Ubuntu --exec bash /mnt/c/Users/me/AppData/Local/Programs/ATLAS/installer/setup_fenics_wsl.sh
set -euo pipefail

find_conda() {
    for base in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/mambaforge" "$HOME/anaconda3"; do
        if [ -x "$base/bin/conda" ]; then
            echo "$base"
            return 0
        fi
    done
    return 1
}

fetch() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$1" -o "$2"
    else
        wget -q "$1" -O "$2"
    fi
}

if ! BASE="$(find_conda)"; then
    echo "Installing Miniforge (conda-forge) into $HOME/miniforge3 ..."
    fetch "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
    rm -f /tmp/miniforge.sh
    BASE="$HOME/miniforge3"
fi

source "$BASE/etc/profile.d/conda.sh"
ENV_PY="$BASE/envs/Atlas/bin/python"

has_fenics() {
    PATH="$(dirname "$1"):$PATH" "$1" -c "import dolfin, mshr" >/dev/null 2>&1
}

if [ -d "$BASE/envs/Atlas" ]; then
    if ! has_fenics "$ENV_PY"; then
        echo "An 'Atlas' conda environment already exists at $BASE/envs/Atlas but has no FEniCS." >&2
        echo "It was left untouched. Install FEniCS into it, or rename it, then run setup again." >&2
        exit 2
    fi
else
    echo "Creating the 'Atlas' environment with FEniCS 2019.1.0 (this can take 10-20 minutes) ..."
    conda create -y -n Atlas --override-channels -c conda-forge \
        python=3.10 fenics=2019.1.0 mshr numpy scipy matplotlib
fi

PATH="$(dirname "$ENV_PY"):$PATH" "$ENV_PY" -c "import dolfin, mshr; print('WSL2 FEniCS ready:', dolfin.__version__)"
