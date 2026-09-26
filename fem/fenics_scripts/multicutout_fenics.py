"""Run from the project root with the FEniCS env on PATH:
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fenics_scripts/multicutout_fenics.py
"""

import json
import os
import sys

import numpy as np
import dolfin as df
import mshr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from atlas.benchmarks import multicutout as mc   # noqa: E402

df.set_log_level(40)
SEG = 96
REF = os.path.join(ROOT, "fem", "reference")

_A, _WA = 0.445948490915965, 0.223381589678011
_B, _WB = 0.091576213509771, 0.109951743655322
BARY = np.array([[_A, _A, 1 - 2 * _A], [_A, 1 - 2 * _A, _A], [1 - 2 * _A, _A, _A],
                 [_B, _B, 1 - 2 * _B], [_B, 1 - 2 * _B, _B], [1 - 2 * _B, _B, _B]])
BW = np.array([_WA] * 3 + [_WB] * 3)
BARY3 = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
BW3 = np.full(3, 1 / 3)
_a1, _b1, _w1 = 0.059715871789770, 0.470142064105115, 0.132394152788506
_a2, _b2, _w2 = 0.797426985353087, 0.101286507323456, 0.125939180544827
BARY7 = np.array([[1 / 3, 1 / 3, 1 / 3], [_a1, _b1, _b1], [_b1, _a1, _b1], [_b1, _b1, _a1],
                  [_a2, _b2, _b2], [_b2, _a2, _b2], [_b2, _b2, _a2]])
BW7 = np.array([0.225, _w1, _w1, _w1, _w2, _w2, _w2])


def geometry():
    P = df.Point
    body = mshr.Polygon([P(0, 0), P(mc.LEG, 0), P(mc.LEG, mc.ARM), P(mc.ARM, mc.ARM),
                         P(mc.ARM, mc.LEG), P(0, mc.LEG)])
    for cx, cy, r in mc.HOLES:
        body = body - mshr.Circle(P(cx, cy), r, SEG)
    (x0, y0), (x1, y1), (x2, y2) = mc.SLOT_PATH
    h = mc.SLOT_HALF
    slot = (mshr.Rectangle(P(x0 - h, y1), P(x0 + h, y0)) + mshr.Rectangle(P(x1, y1 - h), P(x2, y1 + h))
            + mshr.Circle(P(x0, y0), h, SEG) + mshr.Circle(P(x1, y1), h, SEG) + mshr.Circle(P(x2, y2), h, SEG))
    return body - slot


def feature_distance(pts):
    d = np.full(len(pts), np.inf)
    for cx, cy, r in mc.HOLES:
        d = np.minimum(d, np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r))
    ds = np.minimum(mc._seg_dist(pts[:, 0], pts[:, 1], mc.SLOT_PATH[0], mc.SLOT_PATH[1]),
                    mc._seg_dist(pts[:, 0], pts[:, 1], mc.SLOT_PATH[1], mc.SLOT_PATH[2]))
    d = np.minimum(d, np.abs(ds - mc.SLOT_HALF))
    dc = np.hypot(pts[:, 0] - mc.CORNER[0], pts[:, 1] - mc.CORNER[1])
    return d, dc


def refine(mesh, radii):
    for band, corner in radii:
        mid = np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
        d, dc = feature_distance(mid)
        mark = df.MeshFunction("bool", mesh, 2, False)
        mark.array()[:] = (d < band) | (dc < corner)
        mesh = df.refine(mesh, mark)
    return mesh


def quadrature(res=24, radii=((0.03, 0.08),)):
    P = df.Point
    geo = geometry()
    regions = [geo * mshr.Rectangle(P(-1, mc.Y_CUT), P(2, 2)),
               geo * mshr.Rectangle(P(-1, -1), P(mc.X_CUT, mc.Y_CUT)),
               geo * mshr.Rectangle(P(mc.X_CUT, -1), P(2, 2))]
    out, total = {}, 0.0
    for i, reg in enumerate(regions):
        mesh = refine(mshr.generate_mesh(reg, res), radii)
        X = mesh.coordinates()
        T = mesh.cells()
        tri = X[T]
        ar = 0.5 * np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                          - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
        pts = np.einsum("qk,nkd->nqd", BARY, tri).reshape(-1, 2)
        w = (ar[:, None] * BW[None, :]).reshape(-1)
        out[f"pts{i}"], out[f"w{i}"] = pts, w
        out[f"chk_pts{i}"] = np.einsum("qk,nkd->nqd", BARY7, tri).reshape(-1, 2)
        out[f"chk_w{i}"] = (ar[:, None] * BW7[None, :]).reshape(-1)
        total += w.sum()
        sub = mc.subdomain(pts[:, 0], pts[:, 1])
        print(f"subdomain {i}: {mesh.num_cells()} triangles, {len(w)} points, area {w.sum():.6f}, "
              f"routed correctly {bool((sub == i).all())}, in material {mc.material(*pts.T).mean():.4%}",
              flush=True)
    os.makedirs(os.path.dirname(mc.QUAD_FILE), exist_ok=True)
    np.savez(mc.QUAD_FILE, **out)
    print(f"total area {total:.6f}; saved {mc.QUAD_FILE}", flush=True)
    return total


def solve(mesh):
    lam = mc.E_MOD * mc.NU / (1 - mc.NU ** 2)
    mu = mc.E_MOD / (2 * (1 + mc.NU))
    sig = lambda u: lam * df.tr(df.sym(df.grad(u))) * df.Identity(2) + 2 * mu * df.sym(df.grad(u))
    V = df.VectorFunctionSpace(mesh, "Lagrange", 2)
    bc = df.DirichletBC(V, df.Constant((0.0, 0.0)), lambda x, on: on and df.near(x[0], mc.LEG, 1e-9))
    bnd = df.MeshFunction("size_t", mesh, 1, 0)
    df.CompiledSubDomain("near(x[1], L) && on_boundary", L=mc.LEG).mark(bnd, 1)
    ds = df.Measure("ds", domain=mesh, subdomain_data=bnd)
    u, v = df.TrialFunction(V), df.TestFunction(V)
    uh = df.Function(V)
    t = df.Constant((0.0, mc.SIGMA))
    df.solve(df.inner(sig(u), df.sym(df.grad(v))) * df.dx == df.dot(t, v) * ds(1), uh, bc,
             solver_parameters={"linear_solver": "mumps"})
    compliance = df.assemble(df.dot(t, uh) * ds(1))
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    s = sig(uh)
    comps = [df.project(s[0, 0], S1), df.project(s[1, 1], S1), df.project(s[0, 1], S1)]
    return uh, comps, compliance


def evaluate(uh, comps, pts):
    out = np.full((len(pts), 5), np.nan)
    for i, p in enumerate(pts):
        P = df.Point(*p)
        try:
            u = uh(P)
            out[i] = [u[0], u[1], comps[0](P), comps[1](P), comps[2](P)]
        except RuntimeError:
            pass
    sx, sy, tau = out[:, 2], out[:, 3], out[:, 4]
    return out[:, 0], out[:, 1], np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)


def ring_peaks(pts, vm):
    rows = {}
    for name, (cx, cy, r) in zip("ABC", mc.HOLES):
        m = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) < r + 0.03
        rows[f"hole {name}"] = float(np.nanmax(vm[m]))
    ds = np.minimum(mc._seg_dist(pts[:, 0], pts[:, 1], mc.SLOT_PATH[0], mc.SLOT_PATH[1]),
                    mc._seg_dist(pts[:, 0], pts[:, 1], mc.SLOT_PATH[1], mc.SLOT_PATH[2]))
    rows["slot"] = float(np.nanmax(vm[ds < mc.SLOT_HALF + 0.03]))
    far = np.hypot(pts[:, 0] - mc.CORNER[0], pts[:, 1] - mc.CORNER[1]) > 0.03
    for cx, cy in mc.CLAMP_CORNERS:
        far &= np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) > 0.03
    rows["bulk (away from singular points)"] = float(np.nanmax(vm[far]))
    rows["mean over the part"] = float(np.nanmean(vm))
    return rows


def main():
    area = quadrature()
    if "--quadrature-only" in sys.argv:
        return
    xs = np.arange(0.0, mc.LEG + 1e-9, 0.005)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)
    grid = grid[mc.material(grid[:, 0], grid[:, 1])]
    print(f"Monte-Carlo-free check: grid points in material {len(grid)} (area ~ {len(grid) * 0.005 ** 2:.4f}, "
          f"quadrature area {area:.4f})", flush=True)
    geo = geometry()
    results = {}
    for lvl, radii in (("coarse", ((0.04, 0.10), (0.02, 0.05))),
                       ("fine", ((0.04, 0.10), (0.02, 0.05), (0.01, 0.025)))):
        mesh = refine(mshr.generate_mesh(geo, 60), radii)
        uh, comps, C = solve(mesh)
        u, v, vm = evaluate(uh, comps, grid)
        peaks = ring_peaks(grid, vm)
        results[lvl] = dict(cells=mesh.num_cells(), compliance=C, peaks=peaks, u=u, v=v, vm=vm)
        print(f"[{lvl}] cells {mesh.num_cells()}, compliance {C:.6e}, max |u| {np.nanmax(np.hypot(u, v)):.4e}",
              flush=True)
        for k, val in peaks.items():
            print(f"    peak von Mises, {k:34s} {val:.4e} Pa", flush=True)
    c, f = results["coarse"], results["fine"]
    print(f"mesh convergence: compliance {abs(f['compliance'] / c['compliance'] - 1):.3%}, "
          + ", ".join(f"{k} {abs(f['peaks'][k] / c['peaks'][k] - 1):.2%}" for k in f["peaks"]), flush=True)
    ok = np.isfinite(f["vm"])
    os.makedirs(REF, exist_ok=True)
    np.save(os.path.join(REF, "multicutout_coords.npy"), grid[ok])
    np.save(os.path.join(REF, "multicutout_u.npy"), f["u"][ok].reshape(-1, 1))
    np.save(os.path.join(REF, "multicutout_v.npy"), f["v"][ok].reshape(-1, 1))
    np.save(os.path.join(REF, "multicutout_vonmises.npy"), f["vm"][ok].reshape(-1, 1))
    json.dump(dict(E=mc.E_MOD, nu=mc.NU, sigma=mc.SIGMA, cells=f["cells"], compliance=f["compliance"],
                   compliance_coarse=c["compliance"], peaks=f["peaks"], peaks_coarse=c["peaks"],
                   n_points=int(ok.sum())),
              open(os.path.join(REF, "multicutout_meta.json"), "w"), indent=2)
    print(f"saved {int(ok.sum())} reference points", flush=True)


if __name__ == "__main__":
    main()
