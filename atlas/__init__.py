"""ATLAS - Adaptive Multi-Scale Framework for Structural Design.

Typical use (from the project root):
    python -m atlas.atlas_main            # interactive CLI
    from atlas.atlas_main import train_forward_solver, run_inverse_design_on_plate
    from atlas.paths import FEM_REFERENCE_DIR
"""
from atlas.paths import (
    ROOT,
    DOCS_DIR,
    FEM_DIR,
    FENICS_SCRIPTS_DIR,
    FEM_REFERENCE_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
    WEBAPP_DIR,
    WEBAPP_PLOTS_DIR,
)

__version__ = "0.1.0"

__all__ = [
    "ROOT",
    "DOCS_DIR",
    "FEM_DIR",
    "FENICS_SCRIPTS_DIR",
    "FEM_REFERENCE_DIR",
    "RESULTS_DIR",
    "SCRIPTS_DIR",
    "TESTS_DIR",
    "WEBAPP_DIR",
    "WEBAPP_PLOTS_DIR",
    "__version__",
]