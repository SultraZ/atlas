"""Run from the project root with the FEniCS env on PATH:
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fenics_scripts/fibre_composite_fenics.py
"""

import json
import os
import sys

import numpy as np
import dolfin as df
import mshr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from atlas.benchmarks.fibre_composite import (CENTRES, E_FIB, E_MAT, NU_FIB, NU_MAT, RF, SIGMA, H, W,   # noqa: E402
                                   in_fibre)

OUT_DIR = os.path.join(ROOT, "fem", "reference")
BENCH = "fibre_composite"
SKIP = 1e-3
df.set_log_level(40)


def dist_to_interface(P):
    return np.min([np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - RF) for cx, cy in CENTRES], axis=0)


def build_mesh(levels, base=40, segments=128):
    dom = mshr.Rectangle(df.Point(0, 0), df.Point(W, H))
    for k, (cx, cy) in enumerate(CENTRES):
        dom.set_subdomain(k + 1, mshr.Circle(df.Point(cx, cy), RF, segments))
    mesh = mshr.generate_mesh(dom, base)
    for band in [0.03, 0.015][:levels]:
        mid = np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
        mark = df.MeshFunction("bool", mesh, mesh.topology().dim(), False)
        mark.array()[:] = dist_to_interface(mid) < band
        mesh = df.refine(mesh, mark)
    return mesh


def solve(mesh):
    mid = np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
    fib = in_fibre(mid[:, 0], mid[:, 1])
    Q0 = df.FunctionSpace(mesh, "DG", 0)
    E, nu = df.Function(Q0), df.Function(Q0)
    E.vector()[:] = np.where(fib, E_FIB, E_MAT)
    nu.vector()[:] = np.where(fib, NU_FIB, NU_MAT)
    lam = E * nu / (1 - nu ** 2)
    mu = E / (2 * (1 + nu))
    sig = lambda u: lam * df.tr(df.sym(df.grad(u))) * df.Identity(2) + 2 * mu * df.sym(df.grad(u))
    V = df.VectorFunctionSpace(mesh, "Lagrange", 2)
    bcs = [df.DirichletBC(V.sub(0), df.Constant(0.0), lambda x, on: on and df.near(x[0], 0.0, 1e-9)),
           df.DirichletBC(V.sub(1), df.Constant(0.0), lambda x, on: on and df.near(x[1], 0.0, 1e-9))]
    bnd = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    df.CompiledSubDomain("near(x[0], W) && on_boundary", W=W).mark(bnd, 1)
    ds = df.Measure("ds", domain=mesh, subdomain_data=bnd)
    u, v = df.TrialFunction(V), df.TestFunction(V)
    uh = df.Function(V)
    t = df.Constant((SIGMA, 0.0))
    df.solve(df.inner(sig(u), df.sym(df.grad(v))) * df.dx == df.dot(t, v) * ds(1), uh, bcs,
             solver_parameters={"linear_solver": "mumps"})
    S = df.FunctionSpace(mesh, "DG", 1)
    s = sig(uh)
    comps = [df.project(s[0, 0], S), df.project(s[1, 1], S), df.project(s[0, 1], S)]
    return uh, comps, float(df.assemble(df.dot(t, uh) * ds(1)))


def evaluate(uh, comps, pts):
    out = np.full((len(pts), 5), np.nan)
    for i, p in enumerate(pts):
        P = df.Point(min(max(p[0], 1e-10), W - 1e-10), min(max(p[1], 1e-10), H - 1e-10))
        try:
            u = uh(P)
            out[i] = [u[0], u[1], comps[0](P), comps[1](P), comps[2](P)]
        except RuntimeError:
            pass
    sx, sy, tau = out[:, 2], out[:, 3], out[:, 4]
    return out[:, 0], out[:, 1], np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)


def grid(step):
    xs, ys = np.arange(0.0, W + 1e-12, step), np.arange(0.0, H + 1e-12, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    return P[dist_to_interface(P) > SKIP]


def main():
    main_pts = grid(0.01)
    roi = grid(0.004)
    roi = roi[dist_to_interface(roi) < 0.03]
    res = {}
    for levels in (1, 2):
        mesh = build_mesh(levels)
        uh, comps, C = solve(mesh)
        _, _, vm_r = evaluate(uh, comps, roi)
        res[levels] = dict(C=C, vm_r=vm_r, cells=mesh.num_cells(), peak=float(np.nanmax(vm_r)))
        print(f"levels={levels}: cells={mesh.num_cells()}  compliance {C:.10e}  peak vM near interfaces "
              f"{res[levels]['peak']:.6e}", flush=True)
        keep = (mesh, uh, comps)
    a, b = res[1], res[2]
    ok = np.isfinite(a["vm_r"]) & np.isfinite(b["vm_r"])
    d_vm = float(np.linalg.norm(a["vm_r"][ok] - b["vm_r"][ok]) / np.linalg.norm(b["vm_r"][ok]))
    d_c = abs(a["C"] - b["C"]) / abs(b["C"])
    print(f"change between the last two levels: compliance {d_c:.2e}, interface-band von Mises L2 {d_vm:.2e}")

    mesh, uh, comps = keep
    u, v, vm = evaluate(uh, comps, main_pts)
    ok = np.isfinite(vm)
    u_r, v_r, vm_r = evaluate(uh, comps, roi)
    okr = np.isfinite(vm_r)
    os.makedirs(OUT_DIR, exist_ok=True)
    for t, P, x, y, z in ((BENCH, main_pts[ok], u[ok], v[ok], vm[ok]),
                          (BENCH + "_roi", roi[okr], u_r[okr], v_r[okr], vm_r[okr])):
        np.save(os.path.join(OUT_DIR, f"{t}_coords.npy"), P)
        np.save(os.path.join(OUT_DIR, f"{t}_u.npy"), x.reshape(-1, 1))
        np.save(os.path.join(OUT_DIR, f"{t}_v.npy"), y.reshape(-1, 1))
        np.save(os.path.join(OUT_DIR, f"{t}_vonmises.npy"), z.reshape(-1, 1))
    json.dump({"W": W, "H": H, "r": RF, "centres": CENTRES, "E_fibre": E_FIB, "nu_fibre": NU_FIB,
               "E_matrix": E_MAT, "nu_matrix": NU_MAT, "sigma": SIGMA, "cells": int(mesh.num_cells()),
               "compliance": b["C"], "compliance_change_last_level": d_c, "band_vm_change_last_level": d_vm,
               "peak_vm_band": b["peak"], "n_main": int(ok.sum()), "n_roi": int(okr.sum())},
              open(os.path.join(OUT_DIR, f"{BENCH}_meta.json"), "w"), indent=2)
    print(f"saved {int(ok.sum())} main points and {int(okr.sum())} interface-band points")


if __name__ == "__main__":
    main()
