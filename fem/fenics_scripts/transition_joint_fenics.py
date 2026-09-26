"""Run from the project root with the FEniCS env on PATH:
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fenics_scripts/transition_joint_fenics.py
"""

import json
import os
import sys

import numpy as np
import dolfin as dl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from atlas.benchmarks.transition_joint import (E_AL, E_ST, NU_AL, NU_ST, SIGMA, SINGULAR, H, W, X0, f,   # noqa: E402
                                    in_steel)

OUT_DIR = os.path.join(ROOT, "fem", "reference")
BENCH = "transition_joint"
SKIP = 2e-3
dl.set_log_level(40)


def build_mesh(level):
    n = 2 ** level
    mesh = dl.RectangleMesh(dl.Point(0, 0), dl.Point(W, H), 50 * n, 25 * n, "crossed")
    mid = np.array([c.midpoint().array()[:2] for c in dl.cells(mesh)])
    steel = mid[:, 0] < X0
    xy = mesh.coordinates()
    F = f(xy[:, 1])
    left = xy[:, 0] <= X0
    xy[:, 0] = np.where(left, xy[:, 0] * F / X0, F + (xy[:, 0] - X0) * (W - F) / (W - X0))
    marker = dl.MeshFunction("size_t", mesh, 2, 0)
    marker.array()[:] = steel.astype(np.uint64)
    for rad in (0.1, 0.05, 0.025):
        mid = np.array([c.midpoint().array()[:2] for c in dl.cells(mesh)])
        mark = dl.MeshFunction("bool", mesh, 2, False)
        mark.array()[:] = np.hypot(mid[:, 0] - SINGULAR[0][0], mid[:, 1] - SINGULAR[0][1]) < rad
        mesh = dl.refine(mesh, mark)
        marker = dl.adapt(marker, mesh)
    return mesh, marker


def solve(mesh, marker):
    steel = marker.array() == 1
    Q0 = dl.FunctionSpace(mesh, "DG", 0)
    E, nu = dl.Function(Q0), dl.Function(Q0)
    E.vector()[:] = np.where(steel, E_ST, E_AL)
    nu.vector()[:] = np.where(steel, NU_ST, NU_AL)
    lam = E * nu / (1 - nu ** 2)
    mu = E / (2 * (1 + nu))
    sig = lambda u: lam * dl.tr(dl.sym(dl.grad(u))) * dl.Identity(2) + 2 * mu * dl.sym(dl.grad(u))
    V = dl.VectorFunctionSpace(mesh, "Lagrange", 2)
    bcs = [dl.DirichletBC(V.sub(0), dl.Constant(0.0), lambda x, on: on and dl.near(x[0], 0.0, 1e-9)),
           dl.DirichletBC(V.sub(1), dl.Constant(0.0), lambda x, on: on and dl.near(x[1], 0.0, 1e-9))]
    bnd = dl.MeshFunction("size_t", mesh, 1, 0)
    dl.CompiledSubDomain("near(x[0], W) && on_boundary", W=W).mark(bnd, 1)
    ds = dl.Measure("ds", domain=mesh, subdomain_data=bnd)
    u, v = dl.TrialFunction(V), dl.TestFunction(V)
    uh = dl.Function(V)
    t = dl.Constant((SIGMA, 0.0))
    dl.solve(dl.inner(sig(u), dl.sym(dl.grad(v))) * dl.dx == dl.dot(t, v) * ds(1), uh, bcs,
             solver_parameters={"linear_solver": "mumps"})
    S = dl.FunctionSpace(mesh, "DG", 1)
    s = sig(uh)
    comps = [dl.project(s[0, 0], S), dl.project(s[1, 1], S), dl.project(s[0, 1], S)]
    return uh, comps, float(dl.assemble(dl.dot(t, uh) * ds(1)))


def evaluate(uh, comps, pts):
    out = np.full((len(pts), 5), np.nan)
    for i, p in enumerate(pts):
        P = dl.Point(min(max(p[0], 1e-10), W - 1e-10), min(max(p[1], 1e-10), H - 1e-10))
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
    return P[np.abs(P[:, 0] - f(P[:, 1])) > SKIP]


def main():
    main_pts = grid(0.01)
    roi = grid(0.004)
    roi = roi[np.abs(roi[:, 0] - f(roi[:, 1])) < 0.05]
    far = np.hypot(main_pts[:, 0] - SINGULAR[0][0], main_pts[:, 1] - SINGULAR[0][1]) > 0.1
    res = {}
    for level in (1, 2):
        mesh, marker = build_mesh(level)
        uh, comps, C = solve(mesh, marker)
        u, v, vm = evaluate(uh, comps, main_pts)
        res[level] = dict(C=C, u=u, v=v, vm=vm, cells=mesh.num_cells(), uh=uh, comps=comps)
        print(f"level {level}: cells={mesh.num_cells()}  compliance {C:.10e}", flush=True)
    a, b = res[1], res[2]
    d_c = abs(a["C"] - b["C"]) / abs(b["C"])
    d_u = float(np.sqrt(np.nansum((a["u"] - b["u"]) ** 2 + (a["v"] - b["v"]) ** 2) / np.nansum(b["u"] ** 2 + b["v"] ** 2)))
    ok = far & np.isfinite(a["vm"]) & np.isfinite(b["vm"])
    d_vm = float(np.linalg.norm(a["vm"][ok] - b["vm"][ok]) / np.linalg.norm(b["vm"][ok]))
    print(f"change between levels: compliance {d_c:.2e}, displacement L2 {d_u:.2e}, "
          f"von Mises L2 (> 0.1 m from the singular point) {d_vm:.2e}")
    in_ok = np.isfinite(b["vm"])
    u_r, v_r, vm_r = evaluate(b["uh"], b["comps"], roi)
    okr = np.isfinite(vm_r)
    os.makedirs(OUT_DIR, exist_ok=True)
    for t, P, x, y, z in ((BENCH, main_pts[in_ok], b["u"][in_ok], b["v"][in_ok], b["vm"][in_ok]),
                          (BENCH + "_roi", roi[okr], u_r[okr], v_r[okr], vm_r[okr])):
        np.save(os.path.join(OUT_DIR, f"{t}_coords.npy"), P)
        np.save(os.path.join(OUT_DIR, f"{t}_u.npy"), x.reshape(-1, 1))
        np.save(os.path.join(OUT_DIR, f"{t}_v.npy"), y.reshape(-1, 1))
        np.save(os.path.join(OUT_DIR, f"{t}_vonmises.npy"), z.reshape(-1, 1))
    json.dump({"W": W, "H": H, "E_steel": E_ST, "nu_steel": NU_ST, "E_al": E_AL, "nu_al": NU_AL,
               "sigma": SIGMA, "cells": b["cells"], "compliance": b["C"], "compliance_change_last_level": d_c,
               "disp_change_last_level": d_u, "vm_change_last_level_away_from_singularity": d_vm,
               "n_main": int(in_ok.sum()), "n_roi": int(okr.sum())},
              open(os.path.join(OUT_DIR, f"{BENCH}_meta.json"), "w"), indent=2)
    print(f"saved {int(in_ok.sum())} main points and {int(okr.sum())} bond-line band points")


if __name__ == "__main__":
    main()
