"""Run from the project root with the FEniCS env on PATH:
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fenics_scripts/filled_slot_fenics.py
"""

import json
import os
import sys

import numpy as np
import dolfin as df
import mshr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from atlas.benchmarks.filled_slot import (AX, BY, E_EP, E_ST, NU_EP, NU_ST, SIGMA, H, W, in_filler,   # noqa: E402
                               radial_distance)

OUT_DIR = os.path.join(ROOT, "fem", "reference")
BENCH = "filled_slot"
SKIP = 1e-3
df.set_log_level(40)


def build_mesh(levels, base=48, segments=1024):
    dom = mshr.Rectangle(df.Point(0, 0), df.Point(W, H))
    dom.set_subdomain(1, mshr.Ellipse(df.Point(0, 0), AX, BY, segments))
    mesh = mshr.generate_mesh(dom, base)
    for band in [0.20, 0.10, 0.05, 0.025][:levels]:
        mid = np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
        mark = df.MeshFunction("bool", mesh, mesh.topology().dim(), False)
        mark.array()[:] = radial_distance(mid[:, 0], mid[:, 1]) < band
        mesh = df.refine(mesh, mark)
    return mesh


def solve(mesh):
    mid = np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
    fil = in_filler(mid[:, 0], mid[:, 1])
    Q = df.FunctionSpace(mesh, "DG", 0)
    E, nu = df.Function(Q), df.Function(Q)
    E.vector()[:] = np.where(fil, E_EP, E_ST)
    nu.vector()[:] = np.where(fil, NU_EP, NU_ST)
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
        P = df.Point(p[0] * (1 - 1e-9) + 5e-10, p[1] * (1 - 1e-9) + 5e-10)
        try:
            u = uh(P)
            out[i] = [u[0], u[1], comps[0](P), comps[1](P), comps[2](P)]
        except RuntimeError:
            pass
    sx, sy, tau = out[:, 2], out[:, 3], out[:, 4]
    return out[:, 0], out[:, 1], np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)


def grid(x0, x1, y0, y1, step):
    xs, ys = np.arange(x0, x1 + 1e-12, step), np.arange(y0, y1 + 1e-12, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    return P[radial_distance(P[:, 0], P[:, 1]) > SKIP]


def main():
    roi = grid(0.0, AX + 0.05, 0.0, BY + 0.05, 0.0025)
    roi = roi[radial_distance(roi[:, 0], roi[:, 1]) < 0.05]
    res = {}
    for levels in (2, 3):
        mesh = build_mesh(levels)
        uh, comps, C = solve(mesh)
        _, _, vm = evaluate(uh, comps, roi)
        res[levels] = dict(C=C, vm=vm, cells=mesh.num_cells(), peak=float(np.nanmax(vm)))
        print(f"levels={levels}: cells={mesh.num_cells()}  compliance {C:.10e}  peak vM near the interface "
              f"{res[levels]['peak']:.6e}", flush=True)
        keep = (mesh, uh, comps)
    a, b = res[2], res[3]
    ok = np.isfinite(a["vm"]) & np.isfinite(b["vm"])
    d_vm = float(np.linalg.norm(a["vm"][ok] - b["vm"][ok]) / np.linalg.norm(b["vm"][ok]))
    d_c = abs(a["C"] - b["C"]) / abs(b["C"])
    print(f"change between the last two levels: compliance {d_c:.2e}, interface-band von Mises L2 {d_vm:.2e}, "
          f"peak {abs(a['peak'] - b['peak']) / b['peak']:.2e}; Kt (peak / sigma) = {b['peak'] / SIGMA:.3f}")
    mesh, uh, comps = keep
    main_pts = grid(0.0, W, 0.0, H, 0.01)
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
    json.dump({"W": W, "H": H, "AX": AX, "BY": BY, "E_plate": E_ST, "nu_plate": NU_ST, "E_filler": E_EP,
               "nu_filler": NU_EP, "sigma": SIGMA, "cells": int(mesh.num_cells()), "compliance": b["C"],
               "compliance_change_last_level": d_c, "band_vm_change_last_level": d_vm, "peak_vm_band": b["peak"],
               "n_main": int(ok.sum()), "n_roi": int(okr.sum())},
              open(os.path.join(OUT_DIR, f"{BENCH}_meta.json"), "w"), indent=2)
    print(f"saved {int(ok.sum())} main points and {int(okr.sum())} interface-band points")


if __name__ == "__main__":
    main()
