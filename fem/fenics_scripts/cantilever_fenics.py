"""Run from the project root with the FEniCS env on PATH:
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fenics_scripts/cantilever_fenics.py
"""

import dolfin as df
import numpy as np
import os

L, H = 4.0, 1.0
E, nu = 210e9, 0.3
P = -1e6
NX, NY = 160, 40

OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference"))
BENCHMARK = "cantilever"

mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / (1.0 - nu**2)


def sigma(u):
    eps = df.sym(df.grad(u))
    return lmbda * df.tr(eps) * df.Identity(2) + 2 * mu * eps


def main():
    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(L, H), NX, NY)
    V = df.VectorFunctionSpace(mesh, "Lagrange", degree=2)

    def left(x, on_boundary):
        return on_boundary and df.near(x[0], 0.0)

    bc = df.DirichletBC(V, df.Constant((0.0, 0.0)), left)

    class ShearTraction(df.UserExpression):
        def eval(self, values, x):
            values[0] = 0.0
            values[1] = 6.0 * (P / H) * (x[1] / H) * (1.0 - x[1] / H)
        def value_shape(self):
            return (2,)

    traction = ShearTraction(degree=2)

    class RightBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[0], L)

    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    RightBoundary().mark(boundaries, 1)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    a = df.inner(sigma(u), df.sym(df.grad(v))) * df.dx
    Lform = df.dot(traction, v) * ds(1)

    uh = df.Function(V, name="displacement")
    df.solve(a == Lform, uh, bc)

    os.makedirs(OUT_DIR, exist_ok=True)
    coords = V.tabulate_dof_coordinates()
    Vx, Vy = V.sub(0).collapse(), V.sub(1).collapse()
    ux = df.project(uh.sub(0), Vx)
    uy = df.project(uh.sub(1), Vy)

    coords_x = Vx.tabulate_dof_coordinates()
    u_vals = ux.vector().get_local().reshape(-1, 1)
    v_vals = uy.vector().get_local().reshape(-1, 1)

    sig = sigma(uh)
    vm_expr = df.sqrt(sig[0, 0]**2 - sig[0, 0]*sig[1, 1] + sig[1, 1]**2 + 3*sig[0, 1]**2)
    vm_field = df.project(vm_expr, Vx)
    vm_vals = vm_field.vector().get_local().reshape(-1, 1)

    np.save(os.path.join(OUT_DIR, f"{BENCHMARK}_coords.npy"), coords_x)
    np.save(os.path.join(OUT_DIR, f"{BENCHMARK}_u.npy"), u_vals)
    np.save(os.path.join(OUT_DIR, f"{BENCHMARK}_v.npy"), v_vals)
    np.save(os.path.join(OUT_DIR, f"{BENCHMARK}_vonmises.npy"), vm_vals)

    import json
    with open(os.path.join(OUT_DIR, f"{BENCHMARK}_meta.json"), "w") as f:
        json.dump({"L": L, "H": H, "E": E, "nu": nu, "P": P}, f, indent=2)

    print(f"Saved {coords_x.shape[0]} nodes to {OUT_DIR}/{BENCHMARK}_{{coords,u,v,vonmises}}.npy")
    print(f"Max |u| = {np.abs(u_vals).max():.6e}, Max |v| = {np.abs(v_vals).max():.6e}")
    print(f"Peak von Mises stress = {vm_vals.max():.6e} Pa")


if __name__ == "__main__":
    main()
