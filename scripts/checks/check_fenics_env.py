"""Run from the project root with the FEniCS env on PATH:
    PATH=$CONDA_PREFIX/bin:$PATH python scripts/checks/check_fenics_env.py
"""

try:
    import dolfin
    print("dolfin", dolfin.__version__)
except ImportError as e:
    print("NO dolfin:", e)
try:
    import mshr
    print("mshr ok")
except ImportError as e:
    print("NO mshr:", e)