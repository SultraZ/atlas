"""Run from the project root with the FEniCS env on PATH:
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fenics_scripts/l_bracket_fenics.py
"""

import dolfin as df
import mshr
import numpy as np
import os

s = 1.0
E, nu = 210e9, 0.3
P = 1e6
MESH_RESOLUTION = 80

OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference"))
BENCHMARK = "l_bracket"

mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / (1.0 - nu**2)


def sigma(u):
    eps = df.sym(df.grad(u))
    return lmbda * df.tr(eps) * df.Identity(2) + 2 * mu * eps


def main():
    outer = mshr.Rectangle(df.Point(0, 0), df.Point(2 * s, 2 * s))
    notch = mshr.Rectangle(df.Point(0, s), df.Point(s, 2 * s))
    domain = outer - notch
    mesh = mshr.generate_mesh(domain, MESH_RESOLUTION)

    V = df.VectorFunctionSpace(mesh, "Lagrange", degree=2)

    def fixed(x, on_boundary):
        return on_boundary and df.near(x[0], 0.0)

    bc = df.DirichletBC(V, df.Constant((0.0, 0.0)), fixed)

    class LoadBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[0], 2 * s) and x[1] >= s - 1e-8

    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    LoadBoundary().mark(boundaries, 1)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    class ParabolicTraction(df.UserExpression):
        def eval(self, values, x):
            u_load = (x[1] - s) / s
            values[0] = 6.0 * P * u_load * (1.0 - u_load)
            values[1] = 0.0
        def value_shape(self):
            return (2,)

    traction = ParabolicTraction(degree=2)

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    a = df.inner(sigma(u), df.sym(df.grad(v))) * df.dx
    Lform = df.dot(traction, v) * ds(1)

    uh = df.Function(V, name="displacement")
    df.solve(a == Lform, uh, bc)

    os.makedirs(OUT_DIR, exist_ok=True)
    Vx, Vy = V.sub(0).collapse(), V.sub(1).collapse()
    ux = df.project(uh.sub(0), Vx)
    uy = df.project(uh.sub(1), Vy)

    coords_x = Vx.tabulate_dof_coordinates()
    u_vals = ux.vector().get_local().reshape(-1, 1)
    v_vals = uy.vector().get_local().reshape(-1, 1)

    eps = df.sym(df.grad(uh))
    sig = lmbda * df.tr(eps) * df.Identity(2) + 2 * mu * eps
    vm_expr = df.sqrt(sig[0, 0]**2 - sig[0, 0]*sig[1, 1] + sig[1, 1]**2 + 3*sig[0, 1]**2)
    vm_field = df.project(vm_expr, Vx)
    vm_vals = vm_field.vector().get_local().reshape(-1, 1)

    np.save(os.path.join(OUT_DIR, f"{BENCHMARK}_coords.npy"), coords_x)
    np.save(os.path.join(OUT_DIR, f"{BENCHMARK}_u.npy"), u_vals)
    np.save(os.path.join(OUT_DIR, f"{BENCHMARK}_v.npy"), v_vals)
    np.save(os.path.join(OUT_DIR, f"{BENCHMARK}_vonmises.npy"), vm_vals)

    import json
    with open(os.path.join(OUT_DIR, f"{BENCHMARK}_meta.json"), "w") as f:
        json.dump({"size": s, "E": E, "nu": nu, "P": P}, f, indent=2)

    print(f"Saved {coords_x.shape[0]} nodes to {OUT_DIR}/{BENCHMARK}_{{coords,u,v,vonmises}}.npy")
    print(f"Max |u| = {np.abs(u_vals).max():.6e}, Max |v| = {np.abs(v_vals).max():.6e}")
    print(f"Peak von Mises stress (P2 nodal, matches saved reference): {vm_vals.max():.6e} Pa")

    vm_dg0 = df.project(vm_expr, df.FunctionSpace(mesh, "DG", 0))
    print(f"Peak von Mises stress (DG0, mesh-refinement sanity check): "
          f"{vm_dg0.vector().get_local().max():.6e} Pa")


if __name__ == "__main__":
    main()
