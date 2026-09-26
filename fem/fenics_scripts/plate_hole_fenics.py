"""Run from the project root with the FEniCS env on PATH:
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fenics_scripts/plate_hole_fenics.py
"""

import dolfin as df
import mshr
import numpy as np
import os

W, H = 2.0, 2.0
r = 0.3
E, nu = 210e9, 0.3
sigma_inf = 1e6
MESH_RESOLUTION = 100
CIRCLE_SEGMENTS = 64

OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference"))
BENCHMARK = "plate_hole"

mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / (1.0 - nu**2)


def sigma(u):
    eps = df.sym(df.grad(u))
    return lmbda * df.tr(eps) * df.Identity(2) + 2 * mu * eps


def main():
    outer = mshr.Rectangle(df.Point(0, 0), df.Point(W, H))
    hole = mshr.Circle(df.Point(0, 0), r, CIRCLE_SEGMENTS)
    domain = outer - hole
    mesh = mshr.generate_mesh(domain, MESH_RESOLUTION)

    V = df.VectorFunctionSpace(mesh, "Lagrange", degree=2)

    tol = 1e-8

    def left_symmetry(x, on_boundary):
        return on_boundary and df.near(x[0], 0.0, tol) and x[1] > r + tol

    def bottom_symmetry(x, on_boundary):
        return on_boundary and df.near(x[1], 0.0, tol) and x[0] > r + tol

    bcs = [
        df.DirichletBC(V.sub(0), df.Constant(0.0), left_symmetry),
        df.DirichletBC(V.sub(1), df.Constant(0.0), bottom_symmetry),
    ]

    class RightBoundary(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[0], W)

    boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    RightBoundary().mark(boundaries, 1)
    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)

    traction = df.Constant((sigma_inf, 0.0))

    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    a = df.inner(sigma(u), df.sym(df.grad(v))) * df.dx
    Lform = df.dot(traction, v) * ds(1)

    uh = df.Function(V, name="displacement")
    df.solve(a == Lform, uh, bcs)

    os.makedirs(OUT_DIR, exist_ok=True)
    Vx, Vy = V.sub(0).collapse(), V.sub(1).collapse()
    ux = df.project(uh.sub(0), Vx)
    uy = df.project(uh.sub(1), Vy)

    coords_x = Vx.tabulate_dof_coordinates()
    u_vals = ux.vector().get_local().reshape(-1, 1)
    v_vals = uy.vector().get_local().reshape(-1, 1)

    np.save(os.path.join(OUT_DIR, f"{BENCHMARK}_coords.npy"), coords_x)
    np.save(os.path.join(OUT_DIR, f"{BENCHMARK}_u.npy"), u_vals)
    np.save(os.path.join(OUT_DIR, f"{BENCHMARK}_v.npy"), v_vals)

    eps = df.sym(df.grad(uh))
    sig = lmbda * df.tr(eps) * df.Identity(2) + 2 * mu * eps
    vm_expr = df.sqrt(sig[0, 0]**2 - sig[0, 0]*sig[1, 1] + sig[1, 1]**2 + 3*sig[0, 1]**2)
    vm_field = df.project(vm_expr, Vx)
    vm_vals = vm_field.vector().get_local().reshape(-1, 1)
    np.save(os.path.join(OUT_DIR, f"{BENCHMARK}_vonmises.npy"), vm_vals)

    import json
    with open(os.path.join(OUT_DIR, f"{BENCHMARK}_meta.json"), "w") as f:
        json.dump({"W": W, "H": H, "r": r, "E": E, "nu": nu, "sigma_inf": sigma_inf}, f, indent=2)

    print(f"Saved {coords_x.shape[0]} nodes to {OUT_DIR}/{BENCHMARK}_{{coords,u,v,vonmises}}.npy "
          f"(r={r})")
    print(f"Max |u| = {np.abs(u_vals).max():.6e}, Max |v| = {np.abs(v_vals).max():.6e}")
    print(f"Peak von Mises stress (P2 nodal, matches saved reference): {vm_vals.max():.6e} Pa")

    vm_dg0 = df.project(vm_expr, df.FunctionSpace(mesh, "DG", 0))
    peak_vm = vm_dg0.vector().get_local().max()
    print(f"Peak von Mises stress (DG0, mesh-refinement sanity check): {peak_vm:.6e} Pa")
    print(f"  -> implied Kt = peak_vm / sigma_inf = {peak_vm / sigma_inf:.3f} "
          f"(infinite-plate Kirsch Kt = 3.0; expect somewhat above 3 here "
          f"since H/r = {H/r:.2f} isn't 'infinite', and the top edge is free "
          f"rather than a matched far-field condition)")


if __name__ == "__main__":
    main()
