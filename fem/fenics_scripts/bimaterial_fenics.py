"""bimaterial_fenics.py - legacy FEniCS (dolfin + mshr) plane-stress reference for the
two-material benchmark (atlas/benchmarks/bimaterial.py).

    quarter plate [0,2]^2, inclusion rho < 0.3 (E / contrast), rollers on x = 0 and y = 0,
    uniform tension on x = 2.

The mesh conforms to the interface (mshr subdomain, 1024-segment circle) and is refined in bands
around it; P2 displacements, stresses taken element-wise in DG1 (exact for P2, so the stress jump
at the interface is kept, not smoothed). Evaluated on
  * a uniform 0.01 grid over the plate         -> bimaterial_c<contrast>_{coords,u,v,vonmises}.npy
  * a 0.0025 grid within 0.05 of the interface -> bimaterial_c<contrast>_roi_*.npy
Points within 1e-3 of the interface are skipped (which side of the discrete interface they fall
on is ambiguous). The FEM result is used ONLY to score the ATLAS solvers, never in training.

Run from the repo root with the Atlas env's bin on PATH:
    PATH=$CONDA_PREFIX/bin:$PATH python fem/fenics_scripts/bimaterial_fenics.py --contrast 15
"""
import argparse
import json
import os
import sys

import numpy as np
import dolfin as df
import mshr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from atlas.benchmarks.bimaterial import E_MOD, NU, R_INC, SIGMA, H, W, e_inclusion   # noqa: E402

OUT_DIR = os.path.join(ROOT, "fem", "reference")
SKIP = 1e-3
df.set_log_level(40)


def build_mesh(levels, base=48, segments=1024):
    plate = mshr.Rectangle(df.Point(0, 0), df.Point(W, H))
    dom = plate
    dom.set_subdomain(1, mshr.Circle(df.Point(0, 0), R_INC, segments))   # mshr clips it to the plate
    mesh = mshr.generate_mesh(dom, base)
    for band in [0.20, 0.10, 0.05, 0.025][:levels]:
        mid = np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
        mark = df.MeshFunction("bool", mesh, mesh.topology().dim(), False)
        mark.array()[:] = np.abs(np.hypot(mid[:, 0], mid[:, 1]) - R_INC) < band
        mesh = df.refine(mesh, mark)
    return mesh


def solve(mesh, contrast):
    mid = np.array([c.midpoint().array()[:2] for c in df.cells(mesh)])
    Q = df.FunctionSpace(mesh, "DG", 0)
    E = df.Function(Q)
    E.vector()[:] = np.where(np.hypot(mid[:, 0], mid[:, 1]) < R_INC, e_inclusion(contrast), E_MOD)
    lam = E * NU / (1 - NU ** 2)
    mu = E / (2 * (1 + NU))
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
        P = df.Point(p[0] * (1 - 1e-9) + 5e-10, p[1] * (1 - 1e-9) + 5e-10)   # keep edge points inside
        try:
            u = uh(P)
            out[i] = [u[0], u[1], comps[0](P), comps[1](P), comps[2](P)]
        except RuntimeError:
            pass
    sx, sy, tau = out[:, 2], out[:, 3], out[:, 4]
    return out[:, 0], out[:, 1], np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2), sx, sy


def grid(x0, x1, y0, y1, step):
    xs, ys = np.arange(x0, x1 + 1e-12, step), np.arange(y0, y1 + 1e-12, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    return P[np.abs(np.hypot(P[:, 0], P[:, 1]) - R_INC) > SKIP]


def goodier(contrast):
    """Uniform stress inside a circular inclusion in an INFINITE plate under remote uniaxial
    tension (plane stress, equal nu): (sigma_xx, sigma_yy) / sigma. A cross-check only - the
    plate here is finite (W / r = 6.7)."""
    kap = (3 - NU) / (1 + NU)
    gam = 1.0 / contrast
    mean = 0.5 * gam * (kap + 1) / (2 * gam + kap - 1)
    dev = 0.5 * gam * (kap + 1) / (gam * kap + 1)
    return mean + dev, mean - dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", type=float, required=True)
    args = ap.parse_args()
    tag = f"bimaterial_c{args.contrast:g}"
    roi = grid(0.0, R_INC + 0.05, 0.0, R_INC + 0.05, 0.0025)
    roi = roi[np.abs(np.hypot(roi[:, 0], roi[:, 1]) - R_INC) < 0.05]
    res = {}
    for levels in (2, 3):
        mesh = build_mesh(levels)
        uh, comps, C = solve(mesh, args.contrast)
        _, _, vm, _, _ = evaluate(uh, comps, roi)
        res[levels] = dict(cells=mesh.num_cells(), compliance=C, peak=float(np.nanmax(vm)), vm=vm)
        print(f"levels={levels}: cells={mesh.num_cells()}  compliance {C:.10e}  "
              f"peak vM near the interface {res[levels]['peak']:.6e}", flush=True)
        keep = (mesh, uh, comps)
    a, b = res[2], res[3]
    ok = np.isfinite(a["vm"]) & np.isfinite(b["vm"])
    d_vm = float(np.linalg.norm(a["vm"][ok] - b["vm"][ok]) / np.linalg.norm(b["vm"][ok]))
    d_c = abs(a["compliance"] - b["compliance"]) / abs(b["compliance"])
    print(f"change between the last two levels: compliance {d_c:.2e}, interface-band von Mises L2 {d_vm:.2e}, "
          f"peak {abs(a['peak'] - b['peak']) / b['peak']:.2e}")

    mesh, uh, comps = keep
    _, _, _, sx0, sy0 = evaluate(uh, comps, np.array([[0.05, 0.05]]))
    gx, gy = goodier(args.contrast)
    print(f"inclusion stress at (0.05, 0.05): sxx/sigma = {sx0[0] / SIGMA:.4f}, syy/sigma = {sy0[0] / SIGMA:.4f}"
          f"   (infinite-plate Goodier solution {gx:.4f}, {gy:.4f})")

    main_pts = grid(0.0, W, 0.0, H, 0.01)
    u, v, vm, _, _ = evaluate(uh, comps, main_pts)
    ok = np.isfinite(vm)
    main_pts, u, v, vm = main_pts[ok], u[ok], v[ok], vm[ok]
    u_r, v_r, vm_r, _, _ = evaluate(uh, comps, roi)
    okr = np.isfinite(vm_r)
    os.makedirs(OUT_DIR, exist_ok=True)
    for t, P, x, y, z in ((tag, main_pts, u, v, vm), (tag + "_roi", roi[okr], u_r[okr], v_r[okr], vm_r[okr])):
        np.save(os.path.join(OUT_DIR, f"{t}_coords.npy"), P)
        np.save(os.path.join(OUT_DIR, f"{t}_u.npy"), x.reshape(-1, 1))
        np.save(os.path.join(OUT_DIR, f"{t}_v.npy"), y.reshape(-1, 1))
        np.save(os.path.join(OUT_DIR, f"{t}_vonmises.npy"), z.reshape(-1, 1))
    json.dump({"W": W, "H": H, "r": R_INC, "E": E_MOD, "E_inclusion": e_inclusion(args.contrast), "nu": NU,
               "contrast": args.contrast, "sigma": SIGMA, "cells": int(mesh.num_cells()),
               "compliance": b["compliance"], "compliance_change_last_level": d_c,
               "band_vm_change_last_level": d_vm, "peak_vm_band": b["peak"],
               "inclusion_stress_over_sigma": [float(sx0[0] / SIGMA), float(sy0[0] / SIGMA)],
               "goodier_infinite_plate": [gx, gy], "n_main": int(len(main_pts)), "n_roi": int(okr.sum())},
              open(os.path.join(OUT_DIR, f"{tag}_meta.json"), "w"), indent=2)
    print(f"saved {len(main_pts)} main points and {int(okr.sum())} interface-band points; "
          f"max |u| = {np.abs(u).max():.4e}, max |v| = {np.abs(v).max():.4e}")


if __name__ == "__main__":
    main()
