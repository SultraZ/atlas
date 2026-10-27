"""Run from the project root with the FEniCS env on PATH:
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fem_live.py SPEC.json OUT.npz

Example:
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fem_live.py spec.json reference.npz
"""
import json
import math
import sys

import numpy as np
import dolfin as df
import mshr

df.set_log_level(40)
MAX_CELLS = 40000
N_EVAL = 10000


def _lame(E, nu):
    return E * nu / (1 - nu ** 2), E / (2 * (1 + nu))


def _refine(mesh, near, radii):
    for rad in radii:
        mid = np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
        mark = df.MeshFunction("bool", mesh, mesh.topology().dim(), False)
        mark.array()[:] = near(mid) < rad
        mesh = df.refine(mesh, mark)
    return mesh


def _solve(mesh, lam, mu, bcs_of, load_marker, traction):
    V = df.VectorFunctionSpace(mesh, "Lagrange", 2)
    sig = lambda u: lam * df.tr(df.sym(df.grad(u))) * df.Identity(2) + 2 * mu * df.sym(df.grad(u))
    bnd = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    load_marker.mark(bnd, 1)
    ds = df.Measure("ds", domain=mesh, subdomain_data=bnd)
    u, v = df.TrialFunction(V), df.TestFunction(V)
    uh = df.Function(V)
    df.solve(df.inner(sig(u), df.sym(df.grad(v))) * df.dx == df.dot(traction, v) * ds(1), uh, bcs_of(V),
             solver_parameters={"linear_solver": "mumps"})
    S = df.FunctionSpace(mesh, "DG", 1)
    s = sig(uh)
    comps = [df.project(s[0, 0], S), df.project(s[1, 1], S), df.project(s[0, 1], S)]
    return uh, comps, float(df.assemble(df.dot(traction, uh) * ds(1)))


def _evaluate(uh, comps, pts, lo, hi):
    out = np.full((len(pts), 5), np.nan)
    for i, (x, y) in enumerate(pts):
        P = df.Point(min(max(x, lo[0] + 1e-10), hi[0] - 1e-10), min(max(y, lo[1] + 1e-10), hi[1] - 1e-10))
        try:
            u = uh(P)
            out[i] = [u[0], u[1], comps[0](P), comps[1](P), comps[2](P)]
        except RuntimeError:
            pass
    sx, sy, tau = out[:, 2], out[:, 3], out[:, 4]
    return out[:, 0], out[:, 1], np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)


def _grid(x0, x1, y0, y1, area):
    step = math.sqrt(area / N_EVAL)
    xs, ys = np.arange(x0, x1 + 1e-12, step), np.arange(y0, y1 + 1e-12, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def cantilever(L, H, P, E, nu):
    k = min(1.0, math.sqrt(MAX_CELLS / (6400.0 * L * H)))
    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(L, H), max(8, round(40 * L * k)), max(4, round(40 * H * k)), "crossed")
    mesh = _refine(mesh, lambda m: np.minimum(np.hypot(m[:, 0], m[:, 1]), np.hypot(m[:, 0], m[:, 1] - H)), (0.1 * H, 0.05 * H))
    lam, mu = _lame(E, nu)
    bcs = lambda V: [df.DirichletBC(V, df.Constant((0.0, 0.0)), lambda x, on: on and df.near(x[0], 0.0, 1e-9))]
    t = df.Expression(("0.0", "-6.0*P/H*(x[1]/H)*(1.0-x[1]/H)"), P=P, H=H, degree=2)
    uh, comps, C = _solve(mesh, lam, mu, bcs, df.CompiledSubDomain("near(x[0], L) && on_boundary", L=L), t)
    pts = _grid(0.0, L, 0.0, H, L * H)
    return mesh, uh, comps, C, pts, (0.0, 0.0), (L, H)


def l_bracket(s, P, E, nu):
    poly = [df.Point(0, 0), df.Point(2 * s, 0), df.Point(2 * s, 2 * s), df.Point(s, 2 * s), df.Point(s, s), df.Point(0, s)]
    mesh = mshr.generate_mesh(mshr.Polygon(poly), 50)
    mesh = _refine(mesh, lambda m: np.hypot(m[:, 0] - s, m[:, 1] - s), (0.3 * s, 0.15 * s, 0.075 * s, 0.04 * s, 0.02 * s))
    lam, mu = _lame(E, nu)
    bcs = lambda V: [df.DirichletBC(V, df.Constant((0.0, 0.0)), lambda x, on: on and df.near(x[0], 0.0, 1e-9))]
    t = df.Expression(("6.0*P*((x[1]-s)/s)*(1.0-(x[1]-s)/s)", "0.0"), P=P, s=s, degree=2)
    load = df.CompiledSubDomain("near(x[0], 2*s) && x[1] >= s - 1e-9 && on_boundary", s=s)
    uh, comps, C = _solve(mesh, lam, mu, bcs, load, t)
    pts = _grid(0.0, 2 * s, 0.0, 2 * s, 3 * s * s)
    pts = pts[~((pts[:, 0] < s - 1e-12) & (pts[:, 1] > s + 1e-12))]
    return mesh, uh, comps, C, pts, (0.0, 0.0), (2 * s, 2 * s)


def _sym_bcs(V):
    return [df.DirichletBC(V.sub(0), df.Constant(0.0), lambda x, on: on and df.near(x[0], 0.0, 1e-9)),
            df.DirichletBC(V.sub(1), df.Constant(0.0), lambda x, on: on and df.near(x[1], 0.0, 1e-9))]


def plate_hole(r, W, H, sigma, E, nu):
    dom = mshr.Rectangle(df.Point(0, 0), df.Point(W, H)) - mshr.Circle(df.Point(0, 0), r, 256)
    mesh = mshr.generate_mesh(dom, 60)
    mesh = _refine(mesh, lambda m: np.hypot(m[:, 0], m[:, 1]) - r, (0.3, 0.12))
    lam, mu = _lame(E, nu)
    uh, comps, C = _solve(mesh, lam, mu, _sym_bcs, df.CompiledSubDomain("near(x[0], W) && on_boundary", W=W),
                          df.Constant((sigma, 0.0)))
    pts = _grid(0.0, W, 0.0, H, W * H)
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) > r + 1e-3]
    return mesh, uh, comps, C, pts, (0.0, 0.0), (W, H)


def inclusion(a, b, W, H, sigma, E1, nu1, E2, nu2):
    dom = mshr.Rectangle(df.Point(0, 0), df.Point(W, H))
    dom.set_subdomain(1, mshr.Ellipse(df.Point(0, 0), a, b, 512))
    mesh = mshr.generate_mesh(dom, 48)
    dist = lambda m: np.abs(np.hypot(m[:, 0], m[:, 1]) - a * b / np.sqrt((b * np.cos(np.arctan2(m[:, 1], m[:, 0]))) ** 2
                                                                          + (a * np.sin(np.arctan2(m[:, 1], m[:, 0]))) ** 2))
    mesh = _refine(mesh, dist, (0.2, 0.1))
    mid = np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
    ins = (mid[:, 0] / a) ** 2 + (mid[:, 1] / b) ** 2 < 1.0
    Q = df.FunctionSpace(mesh, "DG", 0)
    Ef, nuf = df.Function(Q), df.Function(Q)
    Ef.vector()[:] = np.where(ins, E2, E1)
    nuf.vector()[:] = np.where(ins, nu2, nu1)
    lam, mu = Ef * nuf / (1 - nuf ** 2), Ef / (2 * (1 + nuf))
    uh, comps, C = _solve(mesh, lam, mu, _sym_bcs, df.CompiledSubDomain("near(x[0], W) && on_boundary", W=W),
                          df.Constant((sigma, 0.0)))
    pts = _grid(0.0, W, 0.0, H, W * H)
    pts = pts[dist(pts) > 1e-3]
    return mesh, uh, comps, C, pts, (0.0, 0.0), (W, H)


KINDS = {"cantilever": cantilever, "l_bracket": l_bracket, "plate_hole": plate_hole, "inclusion": inclusion}


def main():
    spec = json.load(open(sys.argv[1]))
    kind = spec.pop("kind")
    mesh, uh, comps, C, pts, lo, hi = KINDS[kind](**spec)
    print(f"FEniCS reference ({kind}): {mesh.num_cells()} cells, compliance {C:.6e}", flush=True)
    u, v, vm = _evaluate(uh, comps, pts, lo, hi)
    ok = np.isfinite(vm)
    np.savez_compressed(sys.argv[2], coords=pts[ok], u=u[ok], v=v[ok], vm=vm[ok], compliance=C, cells=mesh.num_cells())
    print(f"evaluated at {int(ok.sum())} points", flush=True)


if __name__ == "__main__":
    main()
