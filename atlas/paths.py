"""Central path resolution for the ATLAS project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOCS_DIR = ROOT / "docs"
FEM_DIR = ROOT / "fem"
FENICS_SCRIPTS_DIR = FEM_DIR / "fenics_scripts"
FEM_REFERENCE_DIR = FEM_DIR / "reference"
RESULTS_DIR = ROOT / "results"
SCRIPTS_DIR = ROOT / "scripts"
TESTS_DIR = ROOT / "tests"
WEBAPP_DIR = ROOT / "webapp"
WEBAPP_PLOTS_DIR = WEBAPP_DIR / "static" / "plots"