"""Run from the project root:
    python scripts/checks/check_env.py
"""

import sys
print("python:", sys.version)
for mod in ["torch", "numpy", "matplotlib"]:
    try:
        m = __import__(mod)
        print(mod, getattr(m, "__version__", "?"))
    except Exception as e:
        print(mod, "ERROR:", e)
try:
    import torch
    print("cuda:", torch.cuda.is_available())
except Exception as e:
    print("cuda check skipped:", e)