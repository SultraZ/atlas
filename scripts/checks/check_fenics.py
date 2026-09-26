import importlib.util, subprocess, shutil
for m in ["dolfin", "dolfinx", "fenics", "mshr"]:
    print(m, "->", importlib.util.find_spec(m))
print("docker:", shutil.which("docker"))
print("fenicsproject:", shutil.which("fenicsproject"))