"""Run from the project root with the FEniCS env on PATH:
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fenics_scripts/sandwich_fenics.py
"""

import json
import os
import sys

import numpy as np
import dolfin as df

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from atlas.benchmarks.sandwich import (E_CORE, E_FACE, NU_CORE, NU_FACE, Q, H, L, T,   # noqa: E402
                            SINGULAR, characteristic_displacement, layer_of)

OUT_DIR = os.path.join(ROOT, "fem", "reference")
BENCH = "sandwich"
SKIP = 1e-3
df.set_log_level(40)


def build_mesh(level):
    n = 2 ** level
    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(L, H), 40 * n, 10 * n, "crossed")
    for rad in (0.1, 0.05, 0.025, 0.0125):
        mid = np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
        mark = df.MeshFunction("bool", mesh, mesh.topology().dim(), False)
        mark.array()[:] = np.min([np.hypot(mid[:, 0] - a, mid[:, 1] - b) for a, b in SINGULAR], axis=0) < rad
        mesh = df.refine(mesh, mark)
    return mesh


def solve(mesh):
    mid = np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
    core = layer_of(mid[:, 1]) == 1
    Q0 = df.FunctionSpace(mesh, "DG", 0)
    E, nu = df.Function(Q0), df.Function(Q0)
    E.vector()[:] = np.where(core, E_CORE, E_FACE)
    nu.vector()[:] = np.where(core, NU_CORE, NU_FACE)
    lam = E * nu / (1 - nu ** 2)
    mu = E / (2 * (1 + nu))
    sig = lambda u: lam * df.tr(df.sym(df.grad(u))) * df.Identity(2) + 2 * mu * df.sym(df.grad(u))
    V = df.VectorFunctionSpace(mesh, "Lagrange", 2)
    bc = df.DirichletBC(V, df.Constant((0.0, 0.0)), lambda x, on: on and df.near(x[0], 0.0, 1e-9))
    bnd = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    df.CompiledSubDomain("near(x[1], H) && on_boundary", H=H).mark(bnd, 1)
    ds = df.Measure("ds", domain=mesh, subdomain_data=bnd)
    u, v = df.TrialFunction(V), df.TestFunction(V)
    uh = df.Function(V)
    t = df.Constant((0.0, -Q))
    df.solve(df.inner(sig(u), df.sym(df.grad(v))) * df.dx == df.dot(t, v) * ds(1), uh, bc,
             solver_parameters={"linear_solver": "mumps"})
    S = df.FunctionSpace(mesh, "DG", 1)
    s = sig(uh)
    comps = [df.project(s[0, 0], S), df.project(s[1, 1], S), df.project(s[0, 1], S)]
    return uh, comps, float(df.assemble(df.dot(t, uh) * ds(1)))


def evaluate(uh, comps, pts):
    out = np.full((len(pts), 5), np.nan)
    for i, p in enumerate(pts):
        P = df.Point(min(max(p[0], 1e-10), L - 1e-10), min(max(p[1], 1e-10), H - 1e-10))
        try:
            u = uh(P)
            out[i] = [u[0], u[1], comps[0](P), comps[1](P), comps[2](P)]
        except RuntimeError:
            pass
    sx, sy, tau = out[:, 2], out[:, 3], out[:, 4]
    return out[:, 0], out[:, 1], np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)


def main():
    xs, ys = np.arange(0.0, L + 1e-12, 0.01), np.arange(0.0, H + 1e-12, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    pts = pts[(np.abs(pts[:, 1] - T) > SKIP) & (np.abs(pts[:, 1] - (H - T)) > SKIP)]
    far = np.min([np.hypot(pts[:, 0] - a, pts[:, 1] - b) for a, b in
                  [(0, 0), (0, H), (0, T), (0, H - T)]], axis=0) > 0.1
    res = {}
    for level in (2, 3):
        mesh = build_mesh(level)
        uh, comps, C = solve(mesh)
        u, v, vm = evaluate(uh, comps, pts)
        tip = uh(df.Point(L, 0.5))[1]
        res[level] = dict(C=C, u=u, v=v, vm=vm, tip=tip, cells=mesh.num_cells())
        print(f"level {level}: cells={mesh.num_cells()}  compliance {C:.10e}  tip deflection {tip:.8e}", flush=True)
    a, b = res[2], res[3]
    d_c = abs(a["C"] - b["C"]) / abs(b["C"])
    d_u = float(np.sqrt(np.nansum((a["u"] - b["u"]) ** 2 + (a["v"] - b["v"]) ** 2) / np.nansum(b["u"] ** 2 + b["v"] ** 2)))
    ok = far & np.isfinite(a["vm"]) & np.isfinite(b["vm"])
    d_vm = float(np.linalg.norm(a["vm"][ok] - b["vm"][ok]) / np.linalg.norm(b["vm"][ok]))
    print(f"change between levels: compliance {d_c:.2e}, displacement L2 {d_u:.2e}, "
          f"von Mises L2 (> 0.1 m from the clamp singularities) {d_vm:.2e}")

    ok = np.isfinite(b["vm"])
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, f"{BENCH}_coords.npy"), pts[ok])
    np.save(os.path.join(OUT_DIR, f"{BENCH}_u.npy"), b["u"][ok].reshape(-1, 1))
    np.save(os.path.join(OUT_DIR, f"{BENCH}_v.npy"), b["v"][ok].reshape(-1, 1))
    np.save(os.path.join(OUT_DIR, f"{BENCH}_vonmises.npy"), b["vm"][ok].reshape(-1, 1))
    json.dump({"L": L, "H": H, "T": T, "E_face": E_FACE, "nu_face": NU_FACE, "E_core": E_CORE,
               "nu_core": NU_CORE, "q": Q, "cells": b["cells"], "compliance": b["C"], "tip_deflection": b["tip"],
               "compliance_change_last_level": d_c, "disp_change_last_level": d_u,
               "vm_change_last_level_away_from_clamp": d_vm, "n_main": int(ok.sum())},
              open(os.path.join(OUT_DIR, f"{BENCH}_meta.json"), "w"), indent=2)
    print(f"saved {int(ok.sum())} points; tip deflection {b['tip']:.6e} m "
          f"(sandwich beam theory {-characteristic_displacement():.6e})")


if __name__ == "__main__":
    main()
