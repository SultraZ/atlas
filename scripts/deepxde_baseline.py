"""Run from the project root (all options: --help):
    python scripts/deepxde_baseline.py --benchmark {cantilever,plate_hole,l_bracket,notched_plate,multicutout} [--configured]

Example:
    python scripts/deepxde_baseline.py --benchmark cantilever --configured
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import deepxde as dde

from atlas.paths import FEM_REFERENCE_DIR, RESULTS_DIR
from atlas.validation import (compute_von_mises_metrics, default_exclusion_radius,
                              relative_l2, sampler_geometry_dict, singular_locations)

E_MOD, NU, SIGMA0 = 210e9, 0.3, 1e6
EH = E_MOD / SIGMA0
C11 = EH / (1 - NU ** 2)
G12 = EH / (2 * (1 + NU))
close = np.isclose


def stresses(x, y):
    ux = dde.grad.jacobian(y, x, i=0, j=0)
    uy = dde.grad.jacobian(y, x, i=0, j=1)
    vx = dde.grad.jacobian(y, x, i=1, j=0)
    vy = dde.grad.jacobian(y, x, i=1, j=1)
    return C11 * (ux + NU * vy), C11 * (vy + NU * ux), G12 * (uy + vx)


def pde(x, y):
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    u_xy = dde.grad.hessian(y, x, component=0, i=0, j=1)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    v_xy = dde.grad.hessian(y, x, component=1, i=0, j=1)
    return [C11 * (u_xx + NU * v_xy) + G12 * (u_yy + v_xy),
            G12 * (u_xy + v_xx) + C11 * (v_yy + NU * u_xy)]


def traction_bc(geom, on, normal, target, comp):
    def fn(inputs, outputs, X):
        sx, sy, tau = stresses(inputs, outputs)
        n = normal(inputs.detach())
        nx, ny = n[:, 0:1], n[:, 1:2]
        t = (sx * nx + tau * ny) if comp == 0 else (tau * nx + sy * ny)
        return t - target(inputs.detach())[:, None]
    return dde.icbc.OperatorBC(geom, fn, on)


def fixed_bc(geom, on, comp, u_s):
    def fn(inputs, outputs, X):
        return outputs[:, comp:comp + 1] / u_s
    return dde.icbc.OperatorBC(geom, fn, on)


def constant_normal(nx, ny):
    return lambda p: torch.tensor([nx, ny], dtype=p.dtype, device=p.device).expand(len(p), 2)


def zero(p):
    return torch.zeros(len(p), dtype=p.dtype, device=p.device)


def build_problem(b, n_domain, n_boundary, hard_clamp=False, anchors=None):
    n_fix = 2
    if b == 'cantilever':
        L, H, P = 4.0, 1.0, -1e6
        u_s = 1e6 * L ** 3 / (3.0 * E_MOD * (H ** 3 / 12.0))
        geom = dde.geometry.Rectangle([0, 0], [L, H])
        on = lambda f: (lambda X, ob: ob and f(X))
        bcs = [] if hard_clamp else [fixed_bc(geom, on(lambda X: close(X[0], 0)), 0, u_s),
                                     fixed_bc(geom, on(lambda X: close(X[0], 0)), 1, u_s)]
        n_fix = 0 if hard_clamp else 2
        edges = [(lambda X: close(X[0], L), constant_normal(1, 0),
                  lambda p: 6.0 * (P / SIGMA0 / H) * (p[:, 1] / H) * (1 - p[:, 1] / H), zero),
                 (lambda X: close(X[1], H), constant_normal(0, 1), zero, zero),
                 (lambda X: close(X[1], 0), constant_normal(0, -1), zero, zero)]
        for f, nrm, tx, ty in edges:
            bcs += [traction_bc(geom, on(f), nrm, tx, 0), traction_bc(geom, on(f), nrm, ty, 1)]
    elif b == 'l_bracket':
        s, P = 1.0, 1e6
        u_s = 1e6 * s ** 3 / (3.0 * E_MOD * (s ** 3 / 12.0))
        geom = dde.geometry.Polygon([[0, 0], [2 * s, 0], [2 * s, 2 * s], [s, 2 * s], [s, s], [0, s]])
        on = lambda f: (lambda X, ob: ob and f(X))
        bcs = [] if hard_clamp else [fixed_bc(geom, on(lambda X: close(X[0], 0)), 0, u_s),
                                     fixed_bc(geom, on(lambda X: close(X[0], 0)), 1, u_s)]
        n_fix = 0 if hard_clamp else 2
        edges = [(lambda X: close(X[0], 2 * s) and X[1] >= s - 1e-8, constant_normal(1, 0),
                  lambda p: 6.0 * (P / SIGMA0) * ((p[:, 1] - s) / s) * (1 - (p[:, 1] - s) / s), zero),
                 (lambda X: close(X[0], 2 * s) and X[1] <= s + 1e-8, constant_normal(1, 0), zero, zero),
                 (lambda X: close(X[1], 2 * s), constant_normal(0, 1), zero, zero),
                 (lambda X: close(X[0], s) and X[1] >= s - 1e-8, constant_normal(-1, 0), zero, zero),
                 (lambda X: close(X[1], s) and X[0] <= s + 1e-8, constant_normal(0, 1), zero, zero),
                 (lambda X: close(X[1], 0), constant_normal(0, -1), zero, zero)]
        for f, nrm, tx, ty in edges:
            bcs += [traction_bc(geom, on(f), nrm, tx, 0), traction_bc(geom, on(f), nrm, ty, 1)]
    elif b == 'notched_plate':
        from atlas.benchmarks.notched_plate import NotchGeometry
        g = NotchGeometry()
        W, H, r, cx, y0, sig = g.W, g.H, g.r, g.cx, g.y0, 1e6
        u_s = sig * H / E_MOD
        notch = dde.geometry.Rectangle([-0.2, y0 - r], [cx, y0 + r]) | dde.geometry.Disk([cx, y0], r)
        geom = dde.geometry.Rectangle([0, 0], [W, H]) - notch
        on = lambda f: (lambda X, ob: ob and f(X))
        tol = 1e-8
        bcs = [fixed_bc(geom, on(lambda X: close(X[1], 0)), 0, u_s),
               fixed_bc(geom, on(lambda X: close(X[1], 0)), 1, u_s)]
        root_normal = lambda p: torch.stack([cx - p[:, 0], y0 - p[:, 1]], dim=1) / \
            torch.hypot(cx - p[:, 0], y0 - p[:, 1]).clamp_min(1e-12)[:, None]
        edges = [(lambda X: close(X[1], H), constant_normal(0, 1),
                  zero, lambda p: torch.full((len(p),), sig / SIGMA0, dtype=p.dtype, device=p.device)),
                 (lambda X: close(X[0], W), constant_normal(1, 0), zero, zero),
                 (lambda X: close(X[0], 0) and abs(X[1] - y0) > r + tol, constant_normal(-1, 0), zero, zero),
                 (lambda X: close(X[1], y0 + r) and X[0] <= cx + tol, constant_normal(0, -1), zero, zero),
                 (lambda X: close(X[1], y0 - r) and X[0] <= cx + tol, constant_normal(0, 1), zero, zero),
                 (lambda X: close(np.hypot(X[0] - cx, X[1] - y0), r, atol=1e-6) and X[0] >= cx - tol,
                  root_normal, zero, zero)]
        for f, nrm, tx, ty in edges:
            bcs += [traction_bc(geom, on(f), nrm, tx, 0), traction_bc(geom, on(f), nrm, ty, 1)]
    elif b == 'multicutout':
        from atlas.benchmarks import multicutout as mc
        u_s = mc.characteristic_displacement()
        body = dde.geometry.Polygon([[0, 0], [mc.LEG, 0], [mc.LEG, mc.ARM], [mc.ARM, mc.ARM],
                                     [mc.ARM, mc.LEG], [0, mc.LEG]])
        (x0, y0), (x1, y1), (x2, y2) = mc.SLOT_PATH
        h = mc.SLOT_HALF
        slot = (dde.geometry.Rectangle([x0 - h, y1], [x0 + h, y0]) | dde.geometry.Rectangle([x1, y1 - h], [x2, y1 + h])
                | dde.geometry.Disk([x0, y0], h) | dde.geometry.Disk([x1, y1], h) | dde.geometry.Disk([x2, y2], h))
        geom = body - slot
        for cx, cy, r in mc.HOLES:
            geom = geom - dde.geometry.Disk([cx, cy], r)
        clamp = lambda X: close(X[0], mc.LEG)
        load = lambda X: close(X[1], mc.LEG) and X[0] <= mc.ARM + 1e-8
        on = lambda f: (lambda X, ob: ob and f(X))
        free = lambda X, ob: ob and not clamp(X) and not load(X)
        _ncache = {}

        def normal(p):
            key = (len(p), float(p[0, 0]), float(p[0, 1]), float(p[-1, 0]), float(p[-1, 1]))
            if key not in _ncache:
                _ncache[key] = torch.as_tensor(geom.boundary_normal(p.cpu().numpy()),
                                               dtype=p.dtype, device=p.device)
            return _ncache[key]
        sig1 = lambda p: torch.full((len(p),), mc.SIGMA / SIGMA0, dtype=p.dtype, device=p.device)
        bcs = [] if hard_clamp else [fixed_bc(geom, on(clamp), 0, u_s), fixed_bc(geom, on(clamp), 1, u_s)]
        n_fix = 0 if hard_clamp else 2
        bcs += [traction_bc(geom, on(load), constant_normal(0, 1), zero, 0),
                traction_bc(geom, on(load), constant_normal(0, 1), sig1, 1),
                traction_bc(geom, free, normal, zero, 0), traction_bc(geom, free, normal, zero, 1)]
    else:
        W, r, sig = 2.0, 0.3, 1e6
        u_s = 1e6 * W / E_MOD
        geom = dde.geometry.Rectangle([0, 0], [W, W]) - dde.geometry.Disk([0, 0], r)
        on = lambda f: (lambda X, ob: ob and f(X))
        hole = lambda X: close(np.hypot(X[0], X[1]), r, atol=1e-6)
        bcs = [] if hard_clamp else [fixed_bc(geom, on(lambda X: close(X[0], 0) and X[1] > r - 1e-8), 0, u_s),
                                     fixed_bc(geom, on(lambda X: close(X[1], 0) and X[0] > r - 1e-8), 1, u_s)]
        n_fix = 0 if hard_clamp else 2
        hole_normal = lambda p: -p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
        edges = [(lambda X: close(X[0], W), constant_normal(1, 0), lambda p: torch.full((len(p),), sig / SIGMA0, dtype=p.dtype, device=p.device), zero),
                 (lambda X: close(X[1], W), constant_normal(0, 1), zero, zero),
                 (hole, hole_normal, zero, zero)]
        for f, nrm, tx, ty in edges:
            bcs += [traction_bc(geom, on(f), nrm, tx, 0), traction_bc(geom, on(f), nrm, ty, 1)]
    data = dde.data.PDE(geom, pde, bcs, num_domain=n_domain, num_boundary=n_boundary,
                        train_distribution="pseudo", num_test=1000, anchors=anchors)
    return data, u_s, n_fix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["cantilever", "plate_hole", "l_bracket", "notched_plate",
                                                                 "multicutout"])
    ap.add_argument("--adam", type=int, default=10000)
    ap.add_argument("--lbfgs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n-domain", type=int, default=4000)
    ap.add_argument("--n-boundary", type=int, default=2000)
    ap.add_argument("--w-pde", type=float, default=1.0, help="loss weight of the two PDE residuals")
    ap.add_argument("--w-fix", type=float, default=1.0, help="loss weight of the displacement constraints")
    ap.add_argument("--w-trac", type=float, default=1.0, help="loss weight of the traction conditions")
    ap.add_argument("--float64", action="store_true", help="dde.config.set_default_float('float64')")
    ap.add_argument("--configured", action="store_true",
                    help="multicutout sensitivity run with settings fixed in advance from standard DeepXDE "
                         "practice (never chosen by FEM error): exact clamp via output transform, float64, "
                         "extra collocation points near the cutouts, loss weights balanced so every term "
                         "starts at 1 (from the initial training losses only)")
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="deepxde")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    b = args.benchmark
    dde.config.set_random_seed(args.seed)
    if args.configured:
        args.float64 = True
    if args.float64:
        dde.config.set_default_float("float64")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_default_device("cuda")

    anchors = None
    if args.configured:
        rng = np.random.default_rng(args.seed)
        if b == 'multicutout':
            from atlas.benchmarks import multicutout as mc
            cand = rng.uniform(0.0, mc.LEG, size=(200000, 2))
            cand = cand[mc.material(cand[:, 0], cand[:, 1])]
            near = np.zeros(len(cand), dtype=bool)
            for m in mc.cutout_zones(cand, band=0.03).values():
                near |= m
            near |= np.hypot(cand[:, 0] - mc.CORNER[0], cand[:, 1] - mc.CORNER[1]) < 0.05
        elif b == 'cantilever':
            cand = rng.uniform([0, 0], [4.0, 1.0], size=(200000, 2))
            near = cand[:, 0] < 0.4
        elif b == 'plate_hole':
            cand = rng.uniform(0.0, 2.0, size=(200000, 2))
            rad = np.hypot(cand[:, 0], cand[:, 1])
            cand, rad = cand[rad >= 0.3], rad[rad >= 0.3]
            near = rad < 0.6
        else:
            cand = rng.uniform(0.0, 2.0, size=(200000, 2))
            cand = cand[~((cand[:, 0] < 1.0) & (cand[:, 1] > 1.0))]
            near = np.hypot(cand[:, 0] - 1.0, cand[:, 1] - 1.0) < 0.35
        anchors = cand[near][:4000]
    data, u_s, n_fix = build_problem(b, args.n_domain, args.n_boundary, hard_clamp=args.configured,
                                     anchors=anchors)
    net = dde.nn.FNN([2] + [args.width] * args.depth + [2], "tanh", "Glorot uniform")
    if args.configured:
        if b == 'multicutout':
            from atlas.benchmarks import multicutout as mc
            tr = lambda x, y: u_s * (mc.LEG - x[:, 0:1]) * y
        elif b in ('cantilever', 'l_bracket'):
            tr = lambda x, y: u_s * x[:, 0:1] * y
        else:
            tr = lambda x, y: u_s * torch.cat([x[:, 0:1] * y[:, 0:1], x[:, 1:2] * y[:, 1:2]], dim=1)
        net.apply_output_transform(tr)
    else:
        net.apply_output_transform(lambda x, y: u_s * y)
    model = dde.Model(data, net)
    n_params = sum(p.numel() for p in net.parameters())

    R = FEM_REFERENCE_DIR
    coords = np.load(R / f"{b}_coords.npy")
    u_f = np.load(R / f"{b}_u.npy").ravel()
    v_f = np.load(R / f"{b}_v.npy").ravel()
    vm_f = np.load(R / f"{b}_vonmises.npy")
    if b == 'multicutout':
        from atlas.benchmarks import multicutout as mc
        centers = mc.singular_points()
    elif b == 'notched_plate':
        from atlas.benchmarks.notched_plate import NotchGeometry
        ng = NotchGeometry()
        centers = [ng.apex, (0.0, 0.0), (ng.W, 0.0)]
        roi_c = np.load(R / "notched_plate_roi_coords.npy")
        ur_f = np.load(R / "notched_plate_roi_u.npy").ravel()
        vr_f = np.load(R / "notched_plate_roi_v.npy").ravel()
        vmr_f = np.load(R / "notched_plate_roi_vonmises.npy").ravel()
        far = np.hypot(roi_c[:, 0] - ng.apex[0], roi_c[:, 1] - ng.apex[1]) > 0.005
    else:
        from atlas.sampling import CantileverBeamSampler, LBracketSampler, PlateWithHoleSampler
        samp = {'cantilever': CantileverBeamSampler, 'plate_hole': PlateWithHoleSampler,
                'l_bracket': LBracketSampler}[b](device="cpu")
        centers = [loc for _, loc in singular_locations(b, sampler_geometry_dict(samp, b))]
    r_excl = default_exclusion_radius(coords, centers)
    mag_f = np.sqrt(u_f ** 2 + v_f ** 2)

    def fields(pts=None):
        X = torch.tensor(coords if pts is None else pts, dtype=next(net.parameters()).dtype, requires_grad=True)
        out = net(X)
        g = lambda k, j: torch.autograd.grad(out[:, k].sum(), X, create_graph=False,
                                             retain_graph=True)[0][:, j:j + 1]
        ux, uy = g(0, 0), g(0, 1)
        vx, vy = g(1, 0), g(1, 1)
        c11, g12 = E_MOD / (1 - NU ** 2), E_MOD / (2 * (1 + NU))
        sx, sy, tau = c11 * (ux + NU * vy), c11 * (vy + NU * ux), g12 * (uy + vx)
        vm = torch.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * tau ** 2)
        return (out[:, 0].detach().cpu().numpy(), out[:, 1].detach().cpu().numpy(),
                vm.detach().cpu().numpy().ravel())

    def score():
        up, vp, vm = fields()
        vec = float(np.sqrt(np.sum((up - u_f) ** 2 + (vp - v_f) ** 2) / np.sum(u_f ** 2 + v_f ** 2)))
        mag = float(relative_l2(np.sqrt(up ** 2 + vp ** 2), mag_f))
        vm_l2 = compute_von_mises_metrics(vm.reshape(-1, 1), vm_f, coords, centers, r_excl)['von_mises_l2_excl']
        return vec, mag, None if vm_l2 is None else float(vm_l2), (up, vp, vm)

    def notch_metrics():
        if b != 'notched_plate':
            return {}
        ur, vr, vmr = fields(roi_c)
        return dict(notch_vm_l2=float(np.sqrt(np.sum((vmr[far] - vmr_f[far]) ** 2) / np.sum(vmr_f[far] ** 2))),
                    notch_disp_l2=float(np.sqrt(np.sum((ur - ur_f) ** 2 + (vr - vr_f) ** 2)
                                                / np.sum(ur_f ** 2 + vr_f ** 2))),
                    peak_vm_ratio=float(vmr[far].max() / vmr_f[far].max()))

    rows, state = [], {"probe_s": 0.0, "last": -10 ** 9}

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    class Probe(dde.callbacks.Callback):
        def on_epoch_end(self):
            step = self.model.train_state.step
            if step - state["last"] < args.probe_every:
                return
            state["last"] = step
            do_probe(step)

    def do_probe(step):
        sync()
        t = time.perf_counter()
        elapsed = t - t_start - state["probe_s"]
        vec, mag, vm, _ = score()
        rows.append(dict(iter=int(step), seconds=elapsed, disp_vec_l2=vec, disp_mag_l2=mag, vm_l2_excl=vm,
                         **notch_metrics()))
        sync()
        state["probe_s"] += time.perf_counter() - t
        print(f"  it {step:5d}  t={elapsed:7.1f}s  disp(vec)={vec:.3%}  disp(|u|)={mag:.3%}  "
              f"vm={'n/a' if vm is None else format(vm, '.3%')}", flush=True)

    print(f"{b} / DeepXDE {dde.__version__}: {n_params} parameters, device {device}", flush=True)
    sync()
    t_start = time.perf_counter()
    n_trac = len(data.bcs) - n_fix
    weights = [args.w_pde] * 2 + [args.w_fix] * n_fix + [args.w_trac] * n_trac
    if args.configured:
        model.compile("adam", lr=args.lr, loss_weights=[1.0] * len(weights))
        model.train(iterations=0, display_every=1000)
        l0 = np.asarray(model.train_state.loss_train, dtype=float)
        weights = [float(1.0 / max(v, 1e-12)) for v in l0]
        print("  balanced loss weights: " + ", ".join(f"{w:.3g}" for w in weights), flush=True)
    model.compile("adam", lr=args.lr, loss_weights=weights)
    model.train(iterations=args.adam, display_every=1000, callbacks=[Probe()])
    n_adam = model.train_state.step
    if args.lbfgs > 0:
        dde.optimizers.set_LBFGS_options(maxiter=args.lbfgs)
        dde.optimizers.config.LBFGS_options["iter_per_step"] = 100
        dde.optimizers.config.LBFGS_options["fun_per_step"] = 125
        model.compile("L-BFGS", loss_weights=weights)
        model.train(callbacks=[Probe()], display_every=500)
    sync()
    total_s = time.perf_counter() - t_start - state["probe_s"]
    vec, mag, vm, (up, vp, vmp) = score()
    do_probe(model.train_state.step)
    print(f"DONE {b}/deepxde: train {total_s:.1f}s (probing {state['probe_s']:.1f}s excluded), "
          f"final disp(vec) {vec:.3%}, |u| {mag:.3%}, vm {'n/a' if vm is None else format(vm, '.3%')}", flush=True)
    if args.no_save:
        return
    if b == 'multicutout':
        from atlas.benchmarks import multicutout as mc
        d = RESULTS_DIR / "hard_single_material" / "multicutout" / args.tag
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"seed{args.seed}.npz", coords=coords, u=up, v=vp, vm=vmp,
                            u_fem=u_f, v_fem=v_f, vm_fem=vm_f.ravel())
        f = vm_f.ravel()
        Z = mc.cutout_zones(coords)
        cut = {k: float(np.sqrt(np.sum((vmp[m] - f[m]) ** 2) / np.sum(f[m] ** 2))) for k, m in Z.items()}
        final = dict(disp_vec_l2=vec, disp_mag_l2=mag, vm_l2_excl=vm, cutout_vm_l2=cut,
                     cutout_peak_ratio={k: float(vmp[m].max() / f[m].max()) for k, m in Z.items()},
                     cutout_vm_l2_mean=float(np.mean(list(cut.values()))))
        json.dump(dict(benchmark=b, solver=args.tag, seed=args.seed, params=int(n_params),
                       train_seconds=total_s, probe_seconds=state["probe_s"], epochs=args.adam,
                       lbfgs=args.lbfgs, final=final, probes=rows, configured=args.configured,
                       loss_weights=weights,
                       config=dict(library="deepxde " + dde.__version__, n_domain=args.n_domain,
                                   n_boundary=args.n_boundary, width=args.width, depth=args.depth,
                                   float64=args.float64)),
                  open(d / f"seed{args.seed}.json", "w"))
        return
    if b == 'notched_plate':
        d = RESULTS_DIR / "hard_single_material" / "notched_plate" / args.tag
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / f"seed{args.seed}.npz", coords=coords, u=up, v=vp, vm=vmp,
                            u_fem=u_f, v_fem=v_f, vm_fem=vm_f.ravel())
        final = dict(disp_vec_l2=vec, disp_mag_l2=mag, vm_l2_excl=vm, **notch_metrics())
        json.dump(dict(benchmark=b, solver=args.tag, seed=args.seed, params=int(n_params),
                       train_seconds=total_s, probe_seconds=state["probe_s"], epochs=args.adam,
                       lbfgs=args.lbfgs, final=final, probes=rows,
                       config=dict(library="deepxde " + dde.__version__, n_domain=args.n_domain,
                                   n_boundary=args.n_boundary, width=args.width, depth=args.depth)),
                  open(d / f"seed{args.seed}.json", "w"))
        return
    cmp_dir = RESULTS_DIR / "comparison" / "data"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cmp_dir / f"{b}_{args.tag}.npz", coords=coords, u=up, v=vp, vm=vmp,
                        u_fem=u_f, v_fem=v_f, vm_fem=vm_f.ravel())
    vm_glob = float(relative_l2(vmp.reshape(-1, 1), vm_f))
    json.dump(dict(relative_L2_displacement=mag, relative_L2_vector=vec, von_mises_l2_excl=vm,
                   von_mises_l2_global=vm_glob, benchmark=b, variant=args.tag, seed=args.seed,
                   params=int(n_params), train_seconds=round(total_s, 1),
                   pred_max_abs_disp=float(np.hypot(up, vp).max()), fem_max_abs_disp=float(np.hypot(u_f, v_f).max()),
                   config=dict(library="deepxde " + dde.__version__, adam=args.adam, lbfgs=args.lbfgs,
                               lr=args.lr, width=args.width, depth=args.depth, n_domain=args.n_domain,
                               n_boundary=args.n_boundary, loss_weights=dict(pde=args.w_pde, fix=args.w_fix, trac=args.w_trac), float64=args.float64)),
              open(cmp_dir / f"{b}_{args.tag}.json", "w"), indent=2)
    out = RESULTS_DIR / "comparison" / "convergence_timing"
    out.mkdir(exist_ok=True)
    json.dump(dict(benchmark=b, solver=args.tag, seed=args.seed, device=device, params=int(n_params),
                   epochs=args.adam, lbfgs=args.lbfgs, train_seconds=total_s, probe_seconds=state["probe_s"],
                   probes=rows), open(out / f"{b}_{args.tag}.json", "w"))


if __name__ == "__main__":
    main()
