"""plate_hole_fenics.py - Legacy FEniCS (dolfin + mshr) plane-stress reference
solve for the ATLAS plate-with-hole benchmark.

    Geometry: rectangle [0,W] x [0,H] minus a quarter-disc of radius r
              centered at the origin (the sampler only ever samples
              theta in [0, pi/2]).

    BCs:
      - hole boundary (arc):        traction-free (natural, no term needed)
      - x=W, y in [0,H]:             traction t_x = sigma_inf, t_y = 0
      - y=H, x in [0,W]:             traction-free (natural)
      - x=0, y in (r,H]:             symmetry roller -> u = 0 (v free)
      - y=0, x in (r,W]:             symmetry roller -> v = 0 (u free)

    Material: plane stress, E, nu (same constants as physics.py)

    NOTE on the physics: this reproduces exactly what sampling.py encodes
    (right edge loaded, top edge free), which is the standard doubly-
    symmetric quarter model of an infinite plate with a hole ONLY if you
    also intend the top edge to be a symmetry-adjacent free surface (i.e.
    a plate that is genuinely finite in y, not an approximation of the
    Kirsch infinite-plate solution). If your paper's intent was the
    classical Kirsch stress-concentration factor (Kt=3 at the hole edge for
    an infinite plate), you'd want sigma_inf applied as a far-field
    condition rather than a finite-edge traction, and H should be several
    multiples of r so edge effects are negligible. With H=2.0, r up to 0.8,
    that ratio is not huge (H/r as low as 2.5) - worth checking against the
    analytical Kirsch formula (sigma_theta = sigma_inf*(1 - 2cos(2theta))
    at the hole boundary) before trusting this as a Kt=3 validation case.

Run:  python plate_hole_fenics.py
Needs legacy FEniCS + mshr.
"""

import dolfin as df
import mshr
import numpy as np
import os

# ---- must match sampling.py / physics.py exactly ---------------------------
W, H = 2.0, 2.0
r = 0.3              # PlateWithHoleSampler(r_init=...) -- set to whatever
E, nu = 210e9, 0.3
sigma_inf = 1e6       # same as train_forward_solver's sigma_inf for plate_hole
MESH_RESOLUTION = 100
CIRCLE_SEGMENTS = 64

OUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference"))
BENCHMARK = "plate_hole"

mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / (1.0 - nu**2)  # plane-stress effective lambda


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
        # x=0 edge, but only the part outside the hole (y > r)
        return on_boundary and df.near(x[0], 0.0, tol) and x[1] > r + tol

    def bottom_symmetry(x, on_boundary):
        # y=0 edge, but only the part outside the hole (x > r)
        return on_boundary and df.near(x[1], 0.0, tol) and x[0] > r + tol

    bcs = [
        df.DirichletBC(V.sub(0), df.Constant(0.0), left_symmetry),   # u = 0
        df.DirichletBC(V.sub(1), df.Constant(0.0), bottom_symmetry),  # v = 0
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
