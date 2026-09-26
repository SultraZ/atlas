"""
fem_parametric.py - independent FEniCS (legacy dolfin 2019.1 + mshr) solves of the
three ATLAS design problems at an arbitrary shape parameter. Used ONLY to verify
inverse-design results (docs/ATLAS_spec.md: "compliance difference between the
ATLAS-optimised design and the FEM parametric reference, target within 10%").

Nothing here is read by training, and nothing is written to fem/reference/.

    cantilever : p = beam height H
    plate_hole : p = hole radius r
    l_bracket  : p = fillet radius f  (region between the re-entrant corner and
                 the arc of radius f is added; smooth rounded corner)

Returns physical compliance C = int t.u ds  [N m per unit thickness] (= 2 x strain
energy), the peak von Mises stress along the same curves ATLAS samples, and the
solid area. Stresses are L2-projected onto DISCONTINUOUS P1: the strain of a P2
displacement field is exactly piecewise-P1, so that projection is exact and free of
the boundary overshoot a continuous P2 projection of von Mises shows.

Run from a shell where the Atlas conda env's bin is first on PATH (pkg-config).
"""

import math
import os
import sys
from typing import Dict, Optional

import numpy as np

E, NU = 210e9, 0.3
MU = E / (2.0 * (1.0 + NU))
LMBDA = E * NU / (1.0 - NU ** 2)


def _df():
    import dolfin as df
    df.set_log_level(40)
    return df


def _sigma(df, u):
    eps = df.sym(df.grad(u))
    return LMBDA * df.tr(eps) * df.Identity(2) + 2 * MU * eps


def _vm_from_dg(df, uh, mesh):
    """Return a callable (x, y) -> von Mises [Pa] evaluated from the exact DG1 strain."""
    W = df.FunctionSpace(mesh, "DG", 1)
    s = _sigma(df, uh)
    fxx, fyy, fxy = (df.project(s[0, 0], W), df.project(s[1, 1], W), df.project(s[0, 1], W))

    def vm(x, y):
        pt = df.Point(float(x), float(y))
        a, b, c = fxx(pt), fyy(pt), fxy(pt)
        return math.sqrt(max(a * a - a * b + b * b + 3.0 * c * c, 0.0))

    return vm


def _peak(vm, pts) -> float:
    best = 0.0
    for x, y in pts:
        try:
            best = max(best, vm(x, y))
        except RuntimeError:      # point outside the (polygonal) mesh
            continue
    return best


def solve_cantilever(H: float, L: float = 4.0, P: float = -1e6, nx: int = 160,
                     ny_per_unit: int = 40) -> Dict[str, float]:
    df = _df()
    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(L, H), nx, max(8, int(round(ny_per_unit * H))))
    V = df.VectorFunctionSpace(mesh, "Lagrange", 2)
    bc = df.DirichletBC(V, df.Constant((0.0, 0.0)),
                        lambda x, on: on and df.near(x[0], 0.0))

    class Trac(df.UserExpression):
        def eval(self, values, x):
            values[0] = 0.0
            values[1] = 6.0 * (P / H) * (x[1] / H) * (1.0 - x[1] / H)

        def value_shape(self):
            return (2,)

    class Right(df.SubDomain):
        def inside(self, x, on):
            return on and df.near(x[0], L)

    mf = df.MeshFunction("size_t", mesh, 1, 0)
    Right().mark(mf, 1)
    ds = df.Measure("ds", domain=mesh, subdomain_data=mf)
    t = Trac(degree=2)
    u, v = df.TrialFunction(V), df.TestFunction(V)
    uh = df.Function(V)
    df.solve(df.inner(_sigma(df, u), df.sym(df.grad(v))) * df.dx == df.dot(t, v) * ds(1), uh, bc)
    return dict(compliance=float(df.assemble(df.dot(t, uh) * ds(1))), volume=L * H)


def solve_plate_hole(r: float, W: float = 2.0, H: float = 2.0, sigma_inf: float = 1e6,
                     resolution: int = 100, segments: int = 128) -> Dict[str, float]:
    import mshr
    df = _df()
    dom = mshr.Rectangle(df.Point(0, 0), df.Point(W, H)) - mshr.Circle(df.Point(0, 0), r, segments)
    mesh = mshr.generate_mesh(dom, resolution)
    # refine twice around the hole edge, where the stress concentration lives
    for _ in range(2):
        cf = df.MeshFunction("bool", mesh, mesh.topology().dim(), False)
        for c in df.cells(mesh):
            m = c.midpoint()
            if math.hypot(m.x(), m.y()) < r + 0.25:
                cf[c] = True
        mesh = df.refine(mesh, cf)
    V = df.VectorFunctionSpace(mesh, "Lagrange", 2)
    tol = 1e-8
    bcs = [df.DirichletBC(V.sub(0), df.Constant(0.0),
                          lambda x, on: on and df.near(x[0], 0.0, tol) and x[1] > r + tol),
           df.DirichletBC(V.sub(1), df.Constant(0.0),
                          lambda x, on: on and df.near(x[1], 0.0, tol) and x[0] > r + tol)]

    class Right(df.SubDomain):
        def inside(self, x, on):
            return on and df.near(x[0], W)

    mf = df.MeshFunction("size_t", mesh, 1, 0)
    Right().mark(mf, 1)
    ds = df.Measure("ds", domain=mesh, subdomain_data=mf)
    t = df.Constant((sigma_inf, 0.0))
    u, v = df.TrialFunction(V), df.TestFunction(V)
    uh = df.Function(V)
    df.solve(df.inner(_sigma(df, u), df.sym(df.grad(v))) * df.dx == df.dot(t, v) * ds(1), uh, bcs)
    vm = _vm_from_dg(df, uh, mesh)
    th = np.linspace(0.0, 0.5 * math.pi, 181)
    arc = [((r * (1 + 3e-4)) * math.cos(a), (r * (1 + 3e-4)) * math.sin(a)) for a in th]
    return dict(compliance=float(df.assemble(df.dot(t, uh) * ds(1))),
                peak_vm=_peak(vm, arc), volume=W * H - math.pi * r * r / 4.0)


def solve_l_bracket_fillet(f: float, s: float = 1.0, P: float = 1e6, resolution: int = 50,
                           refinements: int = 2, arc_segments: int = 96) -> Dict[str, float]:
    import mshr
    df = _df()
    cx, cy = s - f, s + f
    th = np.linspace(0.0, -0.5 * math.pi, arc_segments + 1)       # (s, s+f) -> (s-f, s)
    arc = [df.Point(cx + f * math.cos(a), cy + f * math.sin(a)) for a in th]
    poly = [df.Point(0, 0), df.Point(2 * s, 0), df.Point(2 * s, 2 * s), df.Point(s, 2 * s)] \
        + arc + [df.Point(0, s)]
    mesh = mshr.generate_mesh(mshr.Polygon(poly), resolution)
    for _ in range(refinements):
        cf = df.MeshFunction("bool", mesh, mesh.topology().dim(), False)
        for c in df.cells(mesh):
            m = c.midpoint()
            if math.hypot(m.x() - s, m.y() - s) < 0.3 + 2 * f:
                cf[c] = True
        mesh = df.refine(mesh, cf)
    print(f"  [fem] fillet mesh: {mesh.num_cells()} cells", file=sys.stderr)
    V = df.VectorFunctionSpace(mesh, "Lagrange", 2)
    bc = df.DirichletBC(V, df.Constant((0.0, 0.0)), lambda x, on: on and df.near(x[0], 0.0))

    class Trac(df.UserExpression):
        def eval(self, values, x):
            u_ = (x[1] - s) / s
            values[0] = 6.0 * P * u_ * (1.0 - u_)
            values[1] = 0.0

        def value_shape(self):
            return (2,)

    class Load(df.SubDomain):
        def inside(self, x, on):
            return on and df.near(x[0], 2 * s) and x[1] >= s - 1e-8

    mf = df.MeshFunction("size_t", mesh, 1, 0)
    Load().mark(mf, 1)
    ds = df.Measure("ds", domain=mesh, subdomain_data=mf)
    t = Trac(degree=2)
    u, v = df.TrialFunction(V), df.TestFunction(V)
    uh = df.Function(V)
    df.solve(df.inner(_sigma(df, u), df.sym(df.grad(v))) * df.dx == df.dot(t, v) * ds(1), uh, bc)
    vm = _vm_from_dg(df, uh, mesh)
    a = np.linspace(-0.5 * math.pi, 0.0, 181)
    k = 1.0 + 3e-4
    pts = [(cx + f * k * math.cos(x), cy + f * k * math.sin(x)) for x in a]
    pts += [(x, s + 1e-5) for x in np.linspace(s - 0.5, s - f, 60)]
    pts += [(s + 1e-5, y) for y in np.linspace(s + f, s + 0.5, 60)]
    return dict(compliance=float(df.assemble(df.dot(t, uh) * ds(1))), peak_vm=_peak(vm, pts),
                volume=3.0 * s * s + f * f * (1.0 - math.pi / 4.0))


SOLVERS = {'cantilever': solve_cantilever, 'plate_hole': solve_plate_hole,
           'l_bracket': solve_l_bracket_fillet}


def fem_metrics(benchmark: str, p: float, **kw) -> Dict[str, float]:
    return SOLVERS[benchmark](p, **kw)


if __name__ == "__main__":
    b, p = sys.argv[1], float(sys.argv[2])
    print(b, p, fem_metrics(b, p))
