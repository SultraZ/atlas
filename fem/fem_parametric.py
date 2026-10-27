"""Run from the project root with the FEniCS env on PATH:
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fem_parametric.py BENCHMARK PARAMETER
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fem_parametric.py --check SPEC.json OUT.json

Example:
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fem_parametric.py plate_hole 0.3
"""

import json
import math
import sys
from typing import Dict

import numpy as np

E, NU = 210e9, 0.3
MU = E / (2.0 * (1.0 + NU))
LMBDA = E * NU / (1.0 - NU ** 2)


def _df():
    import dolfin as df
    df.set_log_level(40)
    return df


def _lame(E_: float, nu_: float):
    return E_ / (2.0 * (1.0 + nu_)), E_ * nu_ / (1.0 - nu_ ** 2)


def _sigma(df, u, mu=MU, lmbda=LMBDA):
    eps = df.sym(df.grad(u))
    return lmbda * df.tr(eps) * df.Identity(2) + 2 * mu * eps


def _vm_from_dg(df, uh, mesh, mu=MU, lmbda=LMBDA):
    W = df.FunctionSpace(mesh, "DG", 1)
    s = _sigma(df, uh, mu, lmbda)
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
        except RuntimeError:
            continue
    return best


def solve_cantilever(H: float, L: float = 4.0, P: float = -1e6, nx: int = 160,
                     ny_per_unit: int = 40, E: float = E, nu: float = NU) -> Dict[str, float]:
    df = _df()
    mu, lmbda = _lame(E, nu)
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
    df.solve(df.inner(_sigma(df, u, mu, lmbda), df.sym(df.grad(v))) * df.dx == df.dot(t, v) * ds(1), uh, bc)
    return dict(compliance=float(df.assemble(df.dot(t, uh) * ds(1))), volume=L * H)


def solve_plate_hole(r: float, W: float = 2.0, H: float = 2.0, sigma_inf: float = 1e6,
                     resolution: int = 100, segments: int = 128, E: float = E, nu: float = NU) -> Dict[str, float]:
    import mshr
    df = _df()
    mu, lmbda = _lame(E, nu)
    dom = mshr.Rectangle(df.Point(0, 0), df.Point(W, H)) - mshr.Circle(df.Point(0, 0), r, segments)
    mesh = mshr.generate_mesh(dom, resolution)
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
    df.solve(df.inner(_sigma(df, u, mu, lmbda), df.sym(df.grad(v))) * df.dx == df.dot(t, v) * ds(1), uh, bcs)
    vm = _vm_from_dg(df, uh, mesh, mu, lmbda)
    th = np.linspace(0.0, 0.5 * math.pi, 181)
    arc = [((r * (1 + 3e-4)) * math.cos(a), (r * (1 + 3e-4)) * math.sin(a)) for a in th]
    return dict(compliance=float(df.assemble(df.dot(t, uh) * ds(1))),
                peak_vm=_peak(vm, arc), volume=W * H - math.pi * r * r / 4.0)


def solve_l_bracket_fillet(f: float, s: float = 1.0, P: float = 1e6, resolution: int = 50,
                           refinements: int = 2, arc_segments: int = 96, E: float = E,
                           nu: float = NU) -> Dict[str, float]:
    import mshr
    df = _df()
    mu, lmbda = _lame(E, nu)
    cx, cy = s - f, s + f
    th = np.linspace(0.0, -0.5 * math.pi, arc_segments + 1)
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
    df.solve(df.inner(_sigma(df, u, mu, lmbda), df.sym(df.grad(v))) * df.dx == df.dot(t, v) * ds(1), uh, bc)
    vm = _vm_from_dg(df, uh, mesh, mu, lmbda)
    a = np.linspace(-0.5 * math.pi, 0.0, 181)
    k = 1.0 + 3e-4
    pts = [(cx + f * k * math.cos(x), cy + f * k * math.sin(x)) for x in a]
    pts += [(x, s + 1e-5) for x in np.linspace(s - 0.5, s - f, 60)]
    pts += [(s + 1e-5, y) for y in np.linspace(s + f, s + 0.5, 60)]
    return dict(compliance=float(df.assemble(df.dot(t, uh) * ds(1))), peak_vm=_peak(vm, pts),
                volume=3.0 * s * s + f * f * (1.0 - math.pi / 4.0))


INCLUSION_DEFAULTS = {
    'steel_insert': dict(E1=70e9, nu1=0.33, E2=210e9, nu2=0.30, sigma=1e6),
    'bimaterial': dict(E1=210e9, nu1=0.30, E2=14e9, nu2=0.30, sigma=1e6),
    'filled_slot': dict(E1=210e9, nu1=0.30, E2=3.5e9, nu2=0.35, sigma=1e6, b=0.4),
}


def solve_inclusion(bench: str, p: float, W: float = 2.0, H: float = 2.0, **kw) -> Dict[str, float]:
    df = _df()
    import mshr
    d = dict(INCLUSION_DEFAULTS[bench], **kw)
    a, b = (p, float(d['b'])) if bench == 'filled_slot' else (p, p)
    r_e = lambda th: a * b / np.sqrt((b * np.cos(th)) ** 2 + (a * np.sin(th)) ** 2)
    dom = mshr.Rectangle(df.Point(0, 0), df.Point(W, H))
    dom.set_subdomain(1, mshr.Ellipse(df.Point(0, 0), a, b, 512))
    mesh = mshr.generate_mesh(dom, 48)
    for rad in (0.2, 0.1, 0.05):
        mid = np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
        mark = df.MeshFunction('bool', mesh, 2, False)
        mark.array()[:] = np.abs(np.hypot(mid[:, 0], mid[:, 1]) - r_e(np.arctan2(mid[:, 1], mid[:, 0]))) < rad
        mesh = df.refine(mesh, mark)
    mid = np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
    ins = (mid[:, 0] / a) ** 2 + (mid[:, 1] / b) ** 2 < 1.0
    Q = df.FunctionSpace(mesh, 'DG', 0)
    Ef, nuf = df.Function(Q), df.Function(Q)
    Ef.vector()[:] = np.where(ins, d['E2'], d['E1'])
    nuf.vector()[:] = np.where(ins, d['nu2'], d['nu1'])
    mu, lmbda = Ef / (2 * (1 + nuf)), Ef * nuf / (1 - nuf ** 2)
    V = df.VectorFunctionSpace(mesh, 'Lagrange', 2)
    bcs = [df.DirichletBC(V.sub(0), df.Constant(0.0), lambda x, on: on and df.near(x[0], 0.0, 1e-9)),
           df.DirichletBC(V.sub(1), df.Constant(0.0), lambda x, on: on and df.near(x[1], 0.0, 1e-9))]
    mf = df.MeshFunction('size_t', mesh, 1, 0)
    df.CompiledSubDomain('near(x[0], W) && on_boundary', W=W).mark(mf, 1)
    ds = df.Measure('ds', domain=mesh, subdomain_data=mf)
    t = df.Constant((float(d['sigma']), 0.0))
    u, v = df.TrialFunction(V), df.TestFunction(V)
    uh = df.Function(V)
    df.solve(df.inner(_sigma(df, u, mu, lmbda), df.sym(df.grad(v))) * df.dx == df.dot(t, v) * ds(1), uh, bcs,
             solver_parameters={'linear_solver': 'mumps'})
    vm = _vm_from_dg(df, uh, mesh, mu, lmbda)
    th = np.linspace(0.0, 0.5 * math.pi, 181)
    r = r_e(th) * (1.0 + 2e-3)
    return dict(compliance=float(df.assemble(df.dot(t, uh) * ds(1))),
                peak_vm=_peak(vm, zip(r * np.cos(th), r * np.sin(th))), volume=math.pi * a * b / 4.0)


SOLVERS = {'cantilever': solve_cantilever, 'plate_hole': solve_plate_hole,
           'l_bracket': solve_l_bracket_fillet,
           'steel_insert': lambda p, **kw: solve_inclusion('steel_insert', p, **kw),
           'bimaterial': lambda p, **kw: solve_inclusion('bimaterial', p, **kw),
           'filled_slot': lambda p, **kw: solve_inclusion('filled_slot', p, **kw)}


def fem_metrics(benchmark: str, p: float, **kw) -> Dict[str, float]:
    return SOLVERS[benchmark](p, **kw)


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        spec = json.load(open(sys.argv[2]))
        out = [dict(p=p, **fem_metrics(spec["benchmark"], p, **spec.get("kwargs", {}))) for p in spec["points"]]
        json.dump(out, open(sys.argv[3], "w"))
        print(json.dumps(out), flush=True)
    else:
        b, p = sys.argv[1], float(sys.argv[2])
        print(b, p, fem_metrics(b, p))
