"""notched_plate_fenics.py - legacy FEniCS (dolfin + mshr) plane-stress reference for the
notched-plate benchmark (atlas/benchmarks/notched_plate.py).

    plate [0,1] x [0,2], U-notch from the left edge at y = 1 (depth 0.25, root radius 0.025)
    y = 0  clamped (u = v = 0);  y = 2  uniform traction t_y = sigma;  all else free.

The mesh is refined in nested discs around the notch root; P2 displacements, stresses
recovered by L2 projection onto continuous P1 (so the root is not polluted by the
element-wise-constant DG0 jumps), evaluated on
  * a uniform grid (spacing 0.01) over the material  -> notched_plate_{coords,u,v,vonmises}.npy
  * a dense grid (spacing 0.0025) around the root   -> notched_plate_roi_*.npy
The FEM result is used ONLY to score the ATLAS solvers, never in training.

Run from the repo root with the Atlas env's bin on PATH:
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fenics_scripts/notched_plate_fenics.py
"""
import json
import os
import sys

import numpy as np
import dolfin as df
import mshr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from atlas.benchmarks.notched_plate import E_MOD, NU, SIGMA, NotchGeometry   # noqa: E402

OUT_DIR = os.path.join(ROOT, "fem", "reference")
BENCH = "notched_plate"
df.set_log_level(40)


def build_mesh(g, levels, base=16, segments=64):
    outer = mshr.Rectangle(df.Point(0, 0), df.Point(g.W, g.H))
    notch = (mshr.Rectangle(df.Point(-0.2, g.y0 - g.r), df.Point(g.cx, g.y0 + g.r))
             + mshr.Circle(df.Point(*g.C), g.r, segments))
    mesh = mshr.generate_mesh(outer - notch, base)
    radii = [0.40, 0.30, 0.20, 0.13, 0.08, 0.05, 0.03][:levels]
    for rad in radii:
        mid = np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
        mark = df.MeshFunction("bool", mesh, mesh.topology().dim(), False)
        mark.array()[:] = np.hypot(mid[:, 0] - g.cx, mid[:, 1] - g.y0) < rad
        mesh = df.refine(mesh, mark)
    return mesh


def solve(mesh, g):
    lam = E_MOD * NU / (1 - NU ** 2)
    mu = E_MOD / (2 * (1 + NU))
    sig = lambda u: lam * df.tr(df.sym(df.grad(u))) * df.Identity(2) + 2 * mu * df.sym(df.grad(u))
    V = df.VectorFunctionSpace(mesh, "Lagrange", 2)
    bc = df.DirichletBC(V, df.Constant((0.0, 0.0)), lambda x, on: on and df.near(x[1], 0.0, 1e-9))
    bnd = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    df.CompiledSubDomain("near(x[1], H) && on_boundary", H=g.H).mark(bnd, 1)
    ds = df.Measure("ds", domain=mesh, subdomain_data=bnd)
    u, v = df.TrialFunction(V), df.TestFunction(V)
    uh = df.Function(V)
    df.solve(df.inner(sig(u), df.sym(df.grad(v))) * df.dx == df.dot(df.Constant((0.0, SIGMA)), v) * ds(1),
             uh, bc)
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    s = sig(uh)
    comps = [df.project(s[0, 0], S1), df.project(s[1, 1], S1), df.project(s[0, 1], S1)]
    return uh, comps


def evaluate(uh, comps, pts):
    tree = uh.function_space().mesh().bounding_box_tree()
    out = np.full((len(pts), 5), np.nan)
    for i, p in enumerate(pts):
        P = df.Point(*p)
        if tree.compute_first_entity_collision(P) > 10 ** 9:       # just outside: nudge inward
            P = df.Point(p[0] * (1 - 1e-9) + 5e-10, p[1] * (1 - 1e-9) + 5e-10)
        try:
            u = uh(P)
            out[i] = [u[0], u[1], comps[0](P), comps[1](P), comps[2](P)]
        except RuntimeError:
            pass
    sx, sy, tau = out[:, 2], out[:, 3], out[:, 4]
    vm = np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)
    return out[:, 0], out[:, 1], vm, sy


def grid(g, x0, x1, y0, y1, step):
    xs = np.arange(x0, x1 + 1e-12, step)
    ys = np.arange(y0, y1 + 1e-12, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    keep = g.material(P[:, 0], P[:, 1])
    return P[keep]


def main():
    g = NotchGeometry()
    sigma_net = SIGMA * g.W / (g.W - g.h)
    roi = grid(g, g.cx - 0.1, g.cx + 0.1, g.y0 - 0.1, g.y0 + 0.1, 0.0025)
    peaks = {}
    for levels in (3, 4):
        mesh = build_mesh(g, levels)
        uh, comps = solve(mesh, g)
        u, v, vm, sy = evaluate(uh, comps, roi)
        peaks[levels] = (float(np.nanmax(vm)), float(np.nanmax(sy)), mesh.num_cells(), uh)
        print(f"levels={levels}: cells={mesh.num_cells()}  peak vM (ROI) {peaks[levels][0]:.6e}  "
              f"peak sigma_yy {peaks[levels][1]:.6e}  Kt_net = {peaks[levels][1] / sigma_net:.3f}",
              flush=True)
        keep = (mesh, uh, comps)
    d = abs(peaks[4][0] - peaks[3][0]) / peaks[4][0]
    print(f"peak von Mises change between the last two levels: {d:.2%}")

    mesh, uh, comps = keep
    main_pts = grid(g, 0.0, g.W, 0.0, g.H, 0.01)
    u, v, vm, _ = evaluate(uh, comps, main_pts)
    ok = np.isfinite(vm)
    main_pts, u, v, vm = main_pts[ok], u[ok], v[ok], vm[ok]
    u_r, v_r, vm_r, _ = evaluate(uh, comps, roi)
    ok = np.isfinite(vm_r)
    os.makedirs(OUT_DIR, exist_ok=True)
    for tag, P, a, b, c in ((BENCH, main_pts, u, v, vm), (BENCH + "_roi", roi[ok], u_r[ok], v_r[ok], vm_r[ok])):
        np.save(os.path.join(OUT_DIR, f"{tag}_coords.npy"), P)
        np.save(os.path.join(OUT_DIR, f"{tag}_u.npy"), a.reshape(-1, 1))
        np.save(os.path.join(OUT_DIR, f"{tag}_v.npy"), b.reshape(-1, 1))
        np.save(os.path.join(OUT_DIR, f"{tag}_vonmises.npy"), c.reshape(-1, 1))
    json.dump({"W": g.W, "H": g.H, "h": g.h, "r": g.r, "R": g.R, "E": E_MOD, "nu": NU,
               "sigma": SIGMA, "n_main": int(len(main_pts)), "n_roi": int(ok.sum()),
               "cells": int(mesh.num_cells()), "peak_vm_roi": peaks[4][0],
               "peak_sigma_yy": peaks[4][1], "Kt_net": peaks[4][1] / sigma_net,
               "mesh_change_last_level": d},
              open(os.path.join(OUT_DIR, f"{BENCH}_meta.json"), "w"), indent=2)
    print(f"saved {len(main_pts)} main points and {int(ok.sum())} ROI points; "
          f"max |u| = {np.abs(u).max():.4e}, max |v| = {np.abs(v).max():.4e}")


if __name__ == "__main__":
    main()
