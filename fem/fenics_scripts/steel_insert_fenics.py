"""steel_insert_fenics.py - legacy FEniCS plane-stress reference for the steel insert in an
aluminium plate (atlas/benchmarks/steel_insert.py). Same interface-conforming mesh, refinement, evaluation grids
and mesh-convergence check as bimaterial_fenics.py; only the materials differ.

    -> fem/reference/steel_insert_{coords,u,v,vonmises}.npy, steel_insert_roi_*.npy, steel_insert_meta.json

Run from the repo root with the Atlas env's bin on PATH:
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fenics_scripts/steel_insert_fenics.py
"""
import json
import os
import sys

import numpy as np
import dolfin as df

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from atlas.benchmarks.steel_insert import E_AL, E_ST, NU_AL, NU_ST, R_INC, SIGMA, H, W   # noqa: E402
from bimaterial_fenics import OUT_DIR, build_mesh, evaluate, grid              # noqa: E402

BENCH = "steel_insert"


def solve(mesh):
    mid = np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
    ins = np.hypot(mid[:, 0], mid[:, 1]) < R_INC
    Q = df.FunctionSpace(mesh, "DG", 0)
    E, nu = df.Function(Q), df.Function(Q)
    E.vector()[:] = np.where(ins, E_ST, E_AL)
    nu.vector()[:] = np.where(ins, NU_ST, NU_AL)
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


def main():
    roi = grid(0.0, R_INC + 0.05, 0.0, R_INC + 0.05, 0.0025)
    roi = roi[np.abs(np.hypot(roi[:, 0], roi[:, 1]) - R_INC) < 0.05]
    res = {}
    for levels in (2, 3):
        mesh = build_mesh(levels)
        uh, comps, C = solve(mesh)
        _, _, vm, _, _ = evaluate(uh, comps, roi)
        res[levels] = dict(C=C, vm=vm, cells=mesh.num_cells(), peak=float(np.nanmax(vm)))
        print(f"levels={levels}: cells={mesh.num_cells()}  compliance {C:.10e}  peak vM near the interface "
              f"{res[levels]['peak']:.6e}", flush=True)
        keep = (mesh, uh, comps)
    a, b = res[2], res[3]
    ok = np.isfinite(a["vm"]) & np.isfinite(b["vm"])
    d_vm = float(np.linalg.norm(a["vm"][ok] - b["vm"][ok]) / np.linalg.norm(b["vm"][ok]))
    d_c = abs(a["C"] - b["C"]) / abs(b["C"])
    print(f"change between the last two levels: compliance {d_c:.2e}, interface-band von Mises L2 {d_vm:.2e}")
    mesh, uh, comps = keep
    _, _, _, sx0, sy0 = evaluate(uh, comps, np.array([[0.05, 0.05]]))
    print(f"insert stress at (0.05, 0.05): sxx/sigma = {sx0[0] / SIGMA:.4f}, syy/sigma = {sy0[0] / SIGMA:.4f}")
    main_pts = grid(0.0, W, 0.0, H, 0.01)
    u, v, vm, _, _ = evaluate(uh, comps, main_pts)
    ok = np.isfinite(vm)
    u_r, v_r, vm_r, _, _ = evaluate(uh, comps, roi)
    okr = np.isfinite(vm_r)
    for t, P, x, y, z in ((BENCH, main_pts[ok], u[ok], v[ok], vm[ok]),
                          (BENCH + "_roi", roi[okr], u_r[okr], v_r[okr], vm_r[okr])):
        np.save(os.path.join(OUT_DIR, f"{t}_coords.npy"), P)
        np.save(os.path.join(OUT_DIR, f"{t}_u.npy"), x.reshape(-1, 1))
        np.save(os.path.join(OUT_DIR, f"{t}_v.npy"), y.reshape(-1, 1))
        np.save(os.path.join(OUT_DIR, f"{t}_vonmises.npy"), z.reshape(-1, 1))
    json.dump({"W": W, "H": H, "r": R_INC, "E_plate": E_AL, "nu_plate": NU_AL, "E_insert": E_ST,
               "nu_insert": NU_ST, "sigma": SIGMA, "cells": int(mesh.num_cells()), "compliance": b["C"],
               "compliance_change_last_level": d_c, "band_vm_change_last_level": d_vm, "peak_vm_band": b["peak"],
               "insert_stress_over_sigma": [float(sx0[0] / SIGMA), float(sy0[0] / SIGMA)],
               "n_main": int(ok.sum()), "n_roi": int(okr.sum())},
              open(os.path.join(OUT_DIR, f"{BENCH}_meta.json"), "w"), indent=2)
    print(f"saved {int(ok.sum())} main points and {int(okr.sum())} interface-band points")


if __name__ == "__main__":
    main()
