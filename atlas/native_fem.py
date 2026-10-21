"""Cross-platform built-in quadratic (P2) Finite Element Method solver
(`numpy` + `scipy.sparse`) providing exact plane-stress reference solutions and
inverse-design checks on machines without WSL2 / legacy FEniCS (`dolfin + mshr`).

NOTE: `fem/` (`fem_live.py`, `fem_parametric.py`, `fem/reference/`) remains
100% genuine FEniCS 2019.1.0 and untouched. Users can switch freely between
`auto`, `fenics` (native Linux / WSL2), and `builtin` (this solver) in Settings.
"""
from __future__ import annotations

import json
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

MAX_CELLS = 40000
N_EVAL = 10000

# Degree-4 exact 6-point symmetric triangle quadrature on {(xi, eta): xi>=0, eta>=0, xi+eta<=1}
# Area of reference triangle = 0.5 (weights sum to 0.5)
_A1, _B1 = 0.445948490915965, 0.108103018168070
_A2, _B2 = 0.091576213509771, 0.816847572980459
_W1, _W2 = 0.1116907948390055, 0.0549758718276610
_QUAD_PTS = np.array([
    [_A1, _A1], [_B1, _A1], [_A1, _B1],
    [_A2, _A2], [_B2, _A2], [_A2, _B2],
], dtype=np.float64)
_QUAD_W = np.array([_W1, _W1, _W1, _W2, _W2, _W2], dtype=np.float64)

# 3-point Gauss-Legendre rule on [0, 1]
_GL_PTS = 0.5 * (1.0 + np.array([-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)], dtype=np.float64))
_GL_W = 0.5 * np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)


def _p2_shape(xi: np.ndarray, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate 6-node P2 basis (N, dN/dxi, dN/deta) at arrays `xi, eta` of shape `(Q,)`."""
    l0 = 1.0 - xi - eta
    l1 = xi
    l2 = eta
    N = np.stack([
        l0 * (2.0 * l0 - 1.0),
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        4.0 * l0 * l1,
        4.0 * l1 * l2,
        4.0 * l2 * l0,
    ], axis=-1)
    dN_xi = np.stack([
        1.0 - 4.0 * l0,
        4.0 * l1 - 1.0,
        np.zeros_like(xi),
        4.0 * (l0 - l1),
        4.0 * l2,
        -4.0 * l2,
    ], axis=-1)
    dN_eta = np.stack([
        1.0 - 4.0 * l0,
        np.zeros_like(eta),
        4.0 * l2 - 1.0,
        -4.0 * l1,
        4.0 * l1,
        4.0 * (l0 - l2),
    ], axis=-1)
    return N, dN_xi, dN_eta


_N_Q, _DN_XI_Q, _DN_ETA_Q = _p2_shape(_QUAD_PTS[:, 0], _QUAD_PTS[:, 1])  # (6, 6)
# P1 basis at quadrature points for local DG(1) stress projection
_P1_Q = np.stack([1.0 - _QUAD_PTS[:, 0] - _QUAD_PTS[:, 1], _QUAD_PTS[:, 0], _QUAD_PTS[:, 1]], axis=-1)  # (6, 3)
_P1_MASS_INV = np.linalg.inv(np.einsum("q,qi,qj->ij", _QUAD_W, _P1_Q, _P1_Q))  # (3, 3)


def _graded_01(n: int, power_start: float = 1.0, power_end: float = 1.0) -> np.ndarray:
    """Monotone 1D grid on [0, 1] with optional polynomial clustering at 0 (`power_start > 1`)
    and/or 1 (`power_end > 1`)."""
    t = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    if power_start != 1.0 and power_end == 1.0:
        return t ** power_start
    if power_end != 1.0 and power_start == 1.0:
        return 1.0 - (1.0 - t) ** power_end
    if power_start != 1.0 and power_end != 1.0:
        a = t ** power_start
        b = 1.0 - (1.0 - t) ** power_end
        w = t * t * (3.0 - 2.0 * t)
        return (1.0 - w) * a + w * b
    return t


class _PatchGrid:
    """Builds a multi-patch conforming isoparametric P2 triangular mesh by stitching
    structured `(2*nu + 1) x (2*nv + 1)` P2 node grids across shared patch interfaces."""

    def __init__(self, merge_tol: float = 1e-11):
        self.merge_tol = merge_tol
        self._coords_list: List[np.ndarray] = []
        self._cells_list: List[np.ndarray] = []
        self._mat_list: List[int] = []
        self._edges: Dict[str, List[np.ndarray]] = {}

    def add_patch(
        self,
        u_grid: np.ndarray,
        v_grid: np.ndarray,
        mapper: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
        material: int = 0,
        edge_tags: Optional[Dict[str, str]] = None,
    ) -> None:
        nu = len(u_grid) - 1
        nv = len(v_grid) - 1
        # Fine sub-grid including mid-edge parameters for exact curved P2 geometry!
        u2 = np.empty(2 * nu + 1, dtype=np.float64)
        u2[0::2] = u_grid
        u2[1::2] = 0.5 * (u_grid[:-1] + u_grid[1:])
        v2 = np.empty(2 * nv + 1, dtype=np.float64)
        v2[0::2] = v_grid
        v2[1::2] = 0.5 * (v_grid[:-1] + v_grid[1:])

        UU, VV = np.meshgrid(u2, v2, indexing="ij")
        X, Y = mapper(UU, VV)
        pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
        offset = sum(len(c) for c in self._coords_list)
        self._coords_list.append(pts)
        idx = (offset + np.arange(len(pts), dtype=np.int64)).reshape(2 * nu + 1, 2 * nv + 1)

        # Split each quad `(2i..2i+2, 2j..2j+2)` along alternating/symmetric diagonals into 2 P2 triangles
        I = 2 * np.arange(nu)[:, None]
        J = 2 * np.arange(nv)[None, :]
        # Triangle 1: (0,0) -> (2,0) -> (2,2), midpoints (1,0), (2,1), (1,1)
        t1 = np.stack([
            idx[I, J], idx[I + 2, J], idx[I + 2, J + 2],
            idx[I + 1, J], idx[I + 2, J + 1], idx[I + 1, J + 1],
        ], axis=-1).reshape(-1, 6)
        # Triangle 2: (0,0) -> (2,2) -> (0,2), midpoints (1,1), (1,2), (0,1)
        t2 = np.stack([
            idx[I, J], idx[I + 2, J + 2], idx[I, J + 2],
            idx[I + 1, J + 1], idx[I + 1, J + 2], idx[I, J + 1],
        ], axis=-1).reshape(-1, 6)
        cells = np.concatenate([t1, t2], axis=0)
        self._cells_list.append(cells)
        self._mat_list.extend([material] * len(cells))

        if edge_tags:
            for side, tag in edge_tags.items():
                if side == "u0":
                    k = np.arange(nv)
                    e = np.stack([idx[0, 2 * k], idx[0, 2 * k + 1], idx[0, 2 * k + 2]], axis=-1)
                elif side == "u1":
                    k = np.arange(nv)
                    e = np.stack([idx[2 * nu, 2 * k], idx[2 * nu, 2 * k + 1], idx[2 * nu, 2 * k + 2]], axis=-1)
                elif side == "v0":
                    k = np.arange(nu)
                    e = np.stack([idx[2 * k, 0], idx[2 * k + 1, 0], idx[2 * k + 2, 0]], axis=-1)
                elif side == "v1":
                    k = np.arange(nu)
                    e = np.stack([idx[2 * k, 2 * nv], idx[2 * k + 1, 2 * nv], idx[2 * k + 2, 2 * nv]], axis=-1)
                else:
                    raise ValueError(f"Unknown patch side: {side}")
                self._edges.setdefault(tag, []).append(e)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        raw_coords = np.concatenate(self._coords_list, axis=0)
        scale = 1.0 / self.merge_tol
        keys = np.round(raw_coords * scale).astype(np.int64)
        _, first_idx, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        order = np.argsort(first_idx)
        remap = np.empty_like(order)
        remap[order] = np.arange(len(first_idx), dtype=np.int64)
        node_map = remap[inv]
        coords = raw_coords[first_idx[order]]
        cells = node_map[np.concatenate(self._cells_list, axis=0)]
        mats = np.asarray(self._mat_list, dtype=np.int64)
        edges = {tag: node_map[np.concatenate(arrs, axis=0)] for tag, arrs in self._edges.items()}
        return coords, cells, mats, edges


def _solve_p2_elasticity(
    coords: np.ndarray,
    cells: np.ndarray,
    c11: np.ndarray,
    c12: np.ndarray,
    c33: np.ndarray,
    dirichlet: List[Tuple[np.ndarray, int, float]],
    load_edges: np.ndarray,
    traction_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Assemble and solve isoparametric 2D plane-stress P2 FEM system + DG(1) stress projection.

    Returns `(u_nodal, dg1_stress, compliance)` where:
      - `u_nodal`: `(N_nodes, 2)` displacement at every P2 node
      - `dg1_stress`: `(N_cells, 3, 3)` linear P1 stress `[sxx, syy, sxy]` per triangle
      - `compliance`: `float` exact boundary work `∫_{Γ_N} t · u ds`
    """
    n_nodes = len(coords)
    n_cells = len(cells)
    n_dofs = 2 * n_nodes

    xy = coords[cells]  # (E, 6, 2)
    # Isoparametric Jacobian J = [dx/dxi, dy/dxi; dx/deta, dy/deta] at all 6 quadrature points:
    dx_dxi = np.einsum("qi,eix->eqx", _DN_XI_Q, xy)    # (E, 6, 2)
    dx_deta = np.einsum("qi,eix->eqx", _DN_ETA_Q, xy)  # (E, 6, 2)
    detJ = dx_dxi[:, :, 0] * dx_deta[:, :, 1] - dx_dxi[:, :, 1] * dx_deta[:, :, 0]  # (E, 6)
    absJ = np.abs(detJ)
    invJ00 = dx_deta[:, :, 1] / detJ
    invJ01 = -dx_dxi[:, :, 1] / detJ
    invJ10 = -dx_deta[:, :, 0] / detJ
    invJ11 = dx_dxi[:, :, 0] / detJ

    # Physical shape gradients dN/dx, dN/dy of shape (E, 6_q, 6_node)
    dN_dx = invJ00[:, :, None] * _DN_XI_Q[None, :, :] + invJ01[:, :, None] * _DN_ETA_Q[None, :, :]
    dN_dy = invJ10[:, :, None] * _DN_XI_Q[None, :, :] + invJ11[:, :, None] * _DN_ETA_Q[None, :, :]

    # Strain-displacement matrix B of shape (E, 6_q, 3, 12) for DOFs [u0,v0, u1,v1, ..., u5,v5]
    B = np.zeros((n_cells, 6, 3, 12), dtype=np.float64)
    B[:, :, 0, 0::2] = dN_dx
    B[:, :, 1, 1::2] = dN_dy
    B[:, :, 2, 0::2] = dN_dy
    B[:, :, 2, 1::2] = dN_dx

    # Plane-stress constitutive matrix D of shape (E, 3, 3)
    D = np.zeros((n_cells, 3, 3), dtype=np.float64)
    D[:, 0, 0] = c11
    D[:, 0, 1] = c12
    D[:, 1, 0] = c12
    D[:, 1, 1] = c11
    D[:, 2, 2] = c33

    wJ = _QUAD_W[None, :] * absJ  # (E, 6)
    DB = np.einsum("eab,eqbj->eqaj", D, B)  # (E, 6, 3, 12)
    Ke = np.einsum("eq,eqai,eqaj->eij", wJ, B, DB)  # (E, 12, 12)

    edofs = np.empty((n_cells, 12), dtype=np.int64)
    edofs[:, 0::2] = 2 * cells
    edofs[:, 1::2] = 2 * cells + 1
    rows = np.repeat(edofs, 12, axis=1).ravel()
    cols = np.tile(edofs, (1, 12)).ravel()
    K = sparse.csc_matrix((Ke.ravel(), (rows, cols)), shape=(n_dofs, n_dofs))

    # Assemble quadratic P2 traction vector F on `load_edges` (shape `(M, 3)` = `[n0, mid, n2]`)
    F = np.zeros(n_dofs, dtype=np.float64)
    s = _GL_PTS  # (3,)
    Ne1 = np.stack([
        (1.0 - s) * (1.0 - 2.0 * s),
        4.0 * s * (1.0 - s),
        s * (2.0 * s - 1.0),
    ], axis=-1)  # (3_q, 3_node)
    dNe1 = np.stack([
        4.0 * s - 3.0,
        4.0 - 8.0 * s,
        4.0 * s - 1.0,
    ], axis=-1)  # (3_q, 3_node)

    exy = coords[load_edges]  # (M, 3, 2)
    px = np.einsum("qi,mix->mqx", Ne1, exy)   # (M, 3, 2)
    dp = np.einsum("qi,mix->mqx", dNe1, exy)  # (M, 3, 2)
    ds_len = np.hypot(dp[:, :, 0], dp[:, :, 1])  # (M, 3)
    tx, ty = traction_fn(px[:, :, 0], px[:, :, 1])
    fe_x = np.einsum("q,mq,mq,qi->mi", _GL_W, ds_len, tx, Ne1)
    fe_y = np.einsum("q,mq,mq,qi->mi", _GL_W, ds_len, ty, Ne1)
    np.add.at(F, 2 * load_edges, fe_x)
    np.add.at(F, 2 * load_edges + 1, fe_y)

    # Strong Dirichlet BC elimination on symmetric system
    fixed_mask = np.zeros(n_dofs, dtype=bool)
    u_full = np.zeros(n_dofs, dtype=np.float64)
    for node_ids, comp, val in dirichlet:
        dids = 2 * node_ids + comp
        fixed_mask[dids] = True
        u_full[dids] = val

    free = np.flatnonzero(~fixed_mask)
    rhs = F[free] - K[free, :][:, fixed_mask] @ u_full[fixed_mask]
    u_full[free] = spsolve(K[free, :][:, free], rhs)
    u_nodal = u_full.reshape(n_nodes, 2)

    # Compliance = ∫_{Γ_N} t · u ds (identical to `df.assemble(df.dot(t, uh) * ds(1))`)
    compliance = float(np.dot(F, u_full))

    # Local element-wise DG(1) stress projection (matching FEniCS `FunctionSpace(mesh, "DG", 1)`)
    ue = u_full[edofs]  # (E, 12)
    sig_q = np.einsum("eqaj,ej->eqa", DB, ue)  # (E, 6_q, 3_comp)
    rhs_dg = np.einsum("q,qk,eqa->eka", _QUAD_W, _P1_Q, sig_q)  # (E, 3, 3)
    dg1_stress = np.einsum("ik,eka->eia", _P1_MASS_INV, rhs_dg)  # (E, 3_basis, 3_comp)
    return u_nodal, dg1_stress, compliance


def _evaluate_field(
    coords: np.ndarray,
    cells: np.ndarray,
    u_nodal: np.ndarray,
    dg1_stress: np.ndarray,
    pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast vectorized evaluation of `(u, v, von_mises, valid_mask)` at arbitrary query points `pts`."""
    verts = coords[cells[:, :3]]  # (E, 3, 2)
    centroids = verts.mean(axis=1)
    tree = cKDTree(centroids)
    k_cand = min(24, len(cells))
    _, cand = tree.query(pts, k=k_cand)  # (P, K)
    if cand.ndim == 1:
        cand = cand[:, None]

    # Solve 2-step Newton inversion on the isoparametric P2 triangle for each candidate cell
    xy_cand = coords[cells[cand]]  # (P, K, 6, 2)
    # Initial guess via affine vertex inverse on (v0, v1, v2)
    v0 = xy_cand[:, :, 0, :]
    e1 = xy_cand[:, :, 1, :] - v0
    e2 = xy_cand[:, :, 2, :] - v0
    det0 = e1[:, :, 0] * e2[:, :, 1] - e1[:, :, 1] * e2[:, :, 0]
    rel = pts[:, None, :] - v0
    xi = (rel[:, :, 0] * e2[:, :, 1] - rel[:, :, 1] * e2[:, :, 0]) / det0
    eta = (-rel[:, :, 0] * e1[:, :, 1] + rel[:, :, 1] * e1[:, :, 0]) / det0

    # 3 Newton iterations for curved isoparametric edges (exact in 1 step on straight triangles!)
    flat_n = xi.size
    for _ in range(3):
        N, dN_xi, dN_eta = _p2_shape(xi.ravel(), eta.ravel())
        N = N.reshape(xi.shape + (6,))
        dN_xi = dN_xi.reshape(xi.shape + (6,))
        dN_eta = dN_eta.reshape(xi.shape + (6,))
        pos = np.einsum("pki,pkix->pkx", N, xy_cand)
        res = pos - pts[:, None, :]
        J00 = np.einsum("pki,pki->pk", dN_xi, xy_cand[:, :, :, 0])
        J01 = np.einsum("pki,pki->pk", dN_eta, xy_cand[:, :, :, 0])
        J10 = np.einsum("pki,pki->pk", dN_xi, xy_cand[:, :, :, 1])
        J11 = np.einsum("pki,pki->pk", dN_eta, xy_cand[:, :, :, 1])
        det = J00 * J11 - J01 * J10
        dxi = (J11 * res[:, :, 0] - J01 * res[:, :, 1]) / det
        deta = (-J10 * res[:, :, 0] + J00 * res[:, :, 1]) / det
        xi = xi - dxi
        eta = eta - deta

    violation = np.maximum.reduce([
        -xi, -eta, xi + eta - 1.0,
    ])
    best_slot = np.argmin(violation, axis=1)
    row_idx = np.arange(len(pts))
    best_cell = cand[row_idx, best_slot]
    best_xi = np.clip(xi[row_idx, best_slot], 0.0, 1.0)
    best_eta = np.clip(eta[row_idx, best_slot], 0.0, 1.0)
    valid = violation[row_idx, best_slot] <= 1e-4

    N_best, _, _ = _p2_shape(best_xi, best_eta)
    u_cell = u_nodal[cells[best_cell]]  # (P, 6, 2)
    uv = np.einsum("pi,pix->px", N_best, u_cell)

    p1 = np.stack([1.0 - best_xi - best_eta, best_xi, best_eta], axis=-1)
    sig = np.einsum("pi,pia->pa", p1, dg1_stress[best_cell])
    sx, sy, tau = sig[:, 0], sig[:, 1], sig[:, 2]
    vm = np.sqrt(np.maximum(sx * sx - sx * sy + sy * sy + 3.0 * tau * tau, 0.0))
    return uv[:, 0], uv[:, 1], vm, valid


def _plane_stress_constants(E: np.ndarray, nu: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c11 = E / (1.0 - nu ** 2)
    c12 = nu * E / (1.0 - nu ** 2)
    c33 = E / (2.0 * (1.0 + nu))
    return c11, c12, c33


def _eval_grid(x0: float, x1: float, y0: float, y1: float, area: float) -> np.ndarray:
    step = math.sqrt(area / N_EVAL)
    xs = np.arange(x0, x1 + 1e-12, step)
    ys = np.arange(y0, y1 + 1e-12, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], axis=1)


# ---------------------------------------------------------------------------
# Body-fitted transfinite P2 meshes matching every benchmark in fem_live.py
# and fem_parametric.py
# ---------------------------------------------------------------------------

def _mesh_cantilever(L: float, H: float, nx: int = 88, ny: int = 28):
    g = _PatchGrid()
    u = _graded_01(nx, power_start=1.6, power_end=1.0)
    v = _graded_01(ny, power_start=1.35, power_end=1.35)
    g.add_patch(
        u, v,
        lambda uu, vv: (L * uu, H * vv),
        edge_tags={"u0": "left", "u1": "right"},
    )
    return g.finalize()


def _mesh_l_bracket(s: float, fillet: float = 0.0, n_arm: int = 34):
    """3-patch conforming L-bracket mesh on `[0, 2s]^2 \\ [0, s) x (s, 2s]` with exact
    circular inner fillet of radius `fillet >= 0` and dense radial grading near `(s, s)`."""
    g = _PatchGrid()
    f = max(0.0, float(fillet))
    cx, cy = s - f, s + f

    # Radial coordinate `u` in [0, 1]: `u=0` is at the inner corner / fillet arc, `u=1` is outer corner `(2s, 0)`
    u_in = _graded_01(n_arm, power_start=2.2, power_end=1.0)
    t_mid = _graded_01(n_arm, power_start=1.0, power_end=1.0)

    def inner_arc_lower(tt):
        # tt in [0, 1]: maps from midpoint of fillet (-pi/4) along lower arm toward (s - f, s) -> (0, s)
        # Wait: split inner boundary cleanly at theta = -pi/4!
        # Lower half of fillet: theta in [-pi/4, -pi/2] for small segment, plus straight segment to x=0?
        # Even cleaner: classic 3-quad L-shape split around hub `(s, 0)` / `(2s, s)`!
        raise NotImplementedError

    # Classic 3-patch transfinite Coons decomposition of filleted/sharp L-bracket:
    # Patch A (left horizontal leg): `x in [0, s-f]` (or `[0, s]` when f=0), `y in [0, s]`
    # Patch C (top vertical leg):    `x in [s, 2s]`, `y in [s+f, 2s]`
    # Patch B1 & B2 (corner knee split along diagonal from fillet midpoint to `(2s, 0)`):
    #   - B1 connects lower-left leg interface `x = s - f, y in [0, s]` & bottom `[s - f, 2s]`
    #     and lower half of fillet `theta in [-pi/2, -pi/4]`!
    #   - B2 connects upper-right leg interface `y = s + f, x in [s, 2s]` & right `[0, s + f]`
    #     and upper half of fillet `theta in [-pi/4, 0]`!
    # EVERY SINGLE INTERFACE SHARES IDENTICAL 1D PARAMETER ARRAYS -> 100% C^0 CONFORMING!
    n_leg = n_arm
    n_rad = n_arm + 12
    s_leg = _graded_01(n_leg, power_start=1.0, power_end=1.9)       # clusters toward corner knee
    r_knee = _graded_01(n_rad, power_start=2.75 if f < 1e-9 else 2.2, power_end=1.0)  # clusters strongly at sharp corner / fillet!
    q_half = _graded_01(n_leg, power_start=1.6, power_end=1.0)      # clusters toward corner tip `q=0`

    x_cut = s - f
    y_cut = s + f

    # Patch A: `[0, x_cut] x [0, s]`, `u = s_leg` (0->x_cut), `v = 1 - r_knee` (`v=1` at `y=s`, `v=0` at `y=0`!)
    # Notice at `u = 1` (`x = x_cut`), `y(r) = s * (1 - r)` for `r in r_knee`!
    g.add_patch(
        s_leg, r_knee,
        lambda uu, rr: (x_cut * uu, s * (1.0 - rr)),
        edge_tags={"u0": "left"},
    )

    # Patch B1 (lower-right knee): `r in r_knee` (0 = inner wall, 1 = outer wall), `q in q_half` (0 = diagonal, 1 = Patch A interface)
    # Inner curve `r = 0`: for `f == 0`, constant `(s, s)`; for `f > 0`, arc `theta(q) = -pi/4 - (pi/4)*q` -> `(cx + f cos th, cy + f sin th)`
    # Outer curve `r = 1`: straight bottom segment from `(2*s, 0)` at `q=0` to `(x_cut, 0)` at `q=1`!
    # Side `q = 1`: linear from `(x_cut, s)` at `r=0` to `(x_cut, 0)` at `r=1` -> `(x_cut, s*(1 - r))` — EXACT MATCH WITH PATCH A `u=1`!
    # Side `q = 0`: linear from `(cx + f/sqrt(2), cy - f/sqrt(2))` at `r=0` to `(2*s, 0)` at `r=1`!
    def map_b1(rr, qq):
        th = -0.25 * math.pi - 0.25 * math.pi * qq
        x_in = cx + f * np.cos(th)
        y_in = cy + f * np.sin(th)
        x_out = 2.0 * s + (x_cut - 2.0 * s) * qq
        y_out = np.zeros_like(qq)
        return (1.0 - rr) * x_in + rr * x_out, (1.0 - rr) * y_in + rr * y_out

    g.add_patch(r_knee, q_half, map_b1)

    # Patch B2 (upper-right knee): `r in r_knee` (0 = inner wall, 1 = outer wall), `q in q_half` (0 = diagonal, 1 = Patch C interface)
    # Inner curve `r = 0`: `theta(q) = -pi/4 + (pi/4)*q` -> at `q=0`: `-pi/4` (EXACT MATCH WITH B1 `q=0`!); at `q=1`: `0` -> `(s, s + f)`!
    # Outer curve `r = 1`: straight right boundary from `(2*s, 0)` at `q=0` to `(2*s, y_cut)` at `q=1`!
    # Side `q = 1`: linear from `(s, y_cut)` at `r=0` to `(2*s, y_cut)` at `r=1` -> `(s + s*r, y_cut)`!
    def map_b2(rr, qq):
        th = -0.25 * math.pi + 0.25 * math.pi * qq
        x_in = cx + f * np.cos(th)
        y_in = cy + f * np.sin(th)
        x_out = np.full_like(qq, 2.0 * s)
        y_out = y_cut * qq
        # Note: traction in fem_live / fem_parametric acts ONLY on `x = 2*s` WITH `y >= s`!
        return (1.0 - rr) * x_in + rr * x_out, (1.0 - rr) * y_in + rr * y_out

    # Wait: if `f > 0`, `y_cut = s + f > s`, so part of `x = 2*s` (`y in [s, s + f]`) lies on B2's `r = 1` edge!
    # Even better: split `x = 2*s` right at `y = s` when `f == 0` OR include ALL `x = 2*s` in `"right_all"` and let
    # `traction_fn(x, y)` return `0` for `y < s` OR split `q_half` cleanly right at `y_out = s` (`q_split = s / y_cut`)!!
    # SPLITTING `q_half` RIGHT AT `q_split = s / y_cut` MAKES THE TRACTION DISCONTINUITY AT `y = s` FALL 100% ON A MESH VERTEX!!
    if f > 1e-12:
        q_split = s / y_cut
        n1 = max(8, round(n_leg * 0.72))
        n2 = max(4, n_leg - n1)
        qa = q_split * _graded_01(n1, power_start=1.35, power_end=1.0)
        qb = q_split + (1.0 - q_split) * np.linspace(0.0, 1.0, n2 + 1)[1:]
        q_b2 = np.concatenate([qa, qb])
    else:
        q_b2 = q_half
    g.add_patch(r_knee, q_b2, map_b2, edge_tags={"u1": "right_knee"})

    # Patch C (upper vertical arm): `x in [s, 2*s]` via `r in r_knee` (`s + s*r` — EXACT MATCH WITH B2 `q=1`!),
    # `y in [y_cut, 2*s]` via `u in s_leg_top`!
    s_top = _graded_01(n_leg, power_start=1.6, power_end=1.0)
    g.add_patch(
        r_knee, s_top,
        lambda rr, uu: (s + s * rr, y_cut + (2.0 * s - y_cut) * uu),
        edge_tags={"u1": "right_top"},
    )
    coords, cells, mats, edges = g.finalize()
    # Filter `"right_knee"` edges strictly to `y >= s - 1e-9` (exact since `q_split` placed a node right at `y = s`!)
    rk = edges["right_knee"]
    ymid = coords[rk[:, 1], 1]
    edges["load"] = np.concatenate([rk[ymid >= s - 1e-9], edges["right_top"]], axis=0)
    return coords, cells, mats, edges


def _mesh_plate_hole(r: float, W: float, H: float, n_rad: int = 40, n_ang_half: int = 28):
    """2-patch body-fitted quarter plate with circular hole `D(0, r)`."""
    g = _PatchGrid()
    th_diag = math.atan2(H, W)
    rr = _graded_01(n_rad, power_start=2.0, power_end=1.0)
    ta = np.linspace(0.0, th_diag, n_ang_half + 1)
    tb = np.linspace(th_diag, 0.5 * math.pi, n_ang_half + 1)

    def map_right_sector(u_r, th):
        r_out = W / np.cos(th)
        rad = r + (r_out - r) * u_r
        return rad * np.cos(th), rad * np.sin(th)

    def map_top_sector(u_r, th):
        r_out = H / np.sin(th)
        rad = r + (r_out - r) * u_r
        return rad * np.cos(th), rad * np.sin(th)

    g.add_patch(rr, ta, map_right_sector, edge_tags={"v0": "bottom", "u1": "right"})
    g.add_patch(rr, tb, map_top_sector, edge_tags={"v1": "left"})
    return g.finalize()


def _mesh_inclusion(a: float, b: float, W: float, H: float, n_ang_half: int = 26):
    """5-patch 100% interface-conforming quarter-plate mesh with exact elliptical inclusion
    `(x/a)^2 + (y/b)^2 = 1`:
      - Inner inclusion (`material = 1`): 3 O-grid quad patches (center box `[0, 0.45a] x [0, 0.45b]` +
        2 curvilinear collar patches out to `(a cos th, b sin th)`) -> zero polar degeneracy at `(0,0)`!
      - Outer matrix (`material = 0`): 2 curvilinear patches from `(a cos th, b sin th)` out to `x = W`
        and `y = H`, clustered on BOTH sides of the material interface `(x/a)^2 + (y/b)^2 = 1`!
    """
    g = _PatchGrid()
    # Choose interface split angle matching outer corner ray so rays never invert for any (a, b, W, H)
    th_mid = 0.25 * math.pi
    t1 = np.linspace(0.0, th_mid, n_ang_half + 1)          # u in [0, 1] along bottom -> mid
    t2 = np.linspace(th_mid, 0.5 * math.pi, n_ang_half + 1) # u in [0, 1] along mid -> top
    s_core = np.linspace(0.0, 1.0, n_ang_half + 1)

    c_frac = 0.42
    r_in = _graded_01(20, power_start=1.0, power_end=1.85)   # clusters sharply toward ellipse from inside!
    r_out = _graded_01(34, power_start=1.85, power_end=1.0)  # clusters sharply at ellipse from outside!

    # Patch I0: core rectangle `[0, c_frac*a] x [0, c_frac*b]`
    g.add_patch(
        s_core, s_core,
        lambda u, v: (c_frac * a * u, c_frac * b * v),
        material=1,
        edge_tags={"v0": "bottom_in", "u0": "left_in"},
    )

    # Patch I1: lower-right collar connecting core `x = c_frac*a, y in [0, c_frac*b]` (`r=0`)
    # to ellipse `theta in [0, pi/4]` (`r=1`)
    def map_in1(rr, s):
        th = th_mid * s
        x0 = np.full_like(s, c_frac * a)
        y0 = c_frac * b * s
        x1 = a * np.cos(th)
        y1 = b * np.sin(th)
        return (1.0 - rr) * x0 + rr * x1, (1.0 - rr) * y0 + rr * y1

    g.add_patch(r_in, s_core, map_in1, material=1, edge_tags={"v0": "bottom_collar"})

    # Patch I2: upper-left collar connecting core `x in [c_frac*a -> 0], y = c_frac*b` (`r=0`)
    # to ellipse `theta in [pi/4 -> pi/2]` (`r=1`)
    def map_in2(rr, s):
        th = th_mid + (0.5 * math.pi - th_mid) * s
        x0 = c_frac * a * (1.0 - s)
        y0 = np.full_like(s, c_frac * b)
        x1 = a * np.cos(th)
        y1 = b * np.sin(th)
        return (1.0 - rr) * x0 + rr * x1, (1.0 - rr) * y0 + rr * y1

    g.add_patch(r_in, s_core, map_in2, material=1, edge_tags={"v1": "left_collar"})

    # Patch O1: outer matrix lower-right (`theta in [0, pi/4]` on ellipse -> `x = W, y in [0, H]`!)
    def map_out1(rr, s):
        th = th_mid * s
        x0 = a * np.cos(th)
        y0 = b * np.sin(th)
        x1 = np.full_like(s, W)
        y1 = H * s
        return (1.0 - rr) * x0 + rr * x1, (1.0 - rr) * y0 + rr * y1

    g.add_patch(r_out, s_core, map_out1, material=0, edge_tags={"v0": "bottom_out", "u1": "right"})

    # Patch O2: outer matrix upper-left (`theta in [pi/4, pi/2]` on ellipse -> `x in [W -> 0], y = H`!)
    def map_out2(rr, s):
        th = th_mid + (0.5 * math.pi - th_mid) * s
        x0 = a * np.cos(th)
        y0 = b * np.sin(th)
        x1 = W * (1.0 - s)
        y1 = np.full_like(s, H)
        return (1.0 - rr) * x0 + rr * x1, (1.0 - rr) * y0 + rr * y1

    g.add_patch(r_out, s_core, map_out2, material=0, edge_tags={"v1": "left_out"})

    coords, cells, mats, edges = g.finalize()
    edges["bottom"] = np.concatenate([edges["bottom_in"], edges["bottom_collar"], edges["bottom_out"]], axis=0)
    edges["left"] = np.concatenate([edges["left_in"], edges["left_collar"], edges["left_out"]], axis=0)
    return coords, cells, mats, edges


# ---------------------------------------------------------------------------
# Public APIs matching `fem/fem_live.py` and `fem/fem_parametric.py` 1-for-1
# ---------------------------------------------------------------------------

def solve_forward(benchmark: str = "cantilever", params: Optional[dict] = None) -> dict:
    """Solve any named benchmark with optional parameter overrides using Built-in P2 FEM."""
    from atlas import live
    return solve_live_spec(live.fem_spec(benchmark, params))


def solve_live_spec(spec: dict) -> dict:
    """Drop-in pure-Python/SciPy counterpart to `fem/fem_live.py` returning
    `dict(coords, u, v, vm, compliance, cells, engine)`."""
    cfg = dict(spec)
    kind = cfg.pop("kind")

    if kind == "cantilever":
        L, H, P, E, nu = float(cfg["L"]), float(cfg["H"]), float(cfg["P"]), float(cfg["E"]), float(cfg["nu"])
        coords, cells, _, edges = _mesh_cantilever(L, H)
        c11, c12, c33 = _plane_stress_constants(np.full(len(cells), E), np.full(len(cells), nu))
        left_nodes = np.unique(edges["left"].ravel())
        bcs = [(left_nodes, 0, 0.0), (left_nodes, 1, 0.0)]
        trac = lambda x, y: (np.zeros_like(y), -6.0 * (P / H) * (y / H) * (1.0 - y / H))
        u_nodal, dg1, C = _solve_p2_elasticity(coords, cells, c11, c12, c33, bcs, edges["right"], trac)
        pts = _eval_grid(0.0, L, 0.0, H, L * H)

    elif kind == "l_bracket":
        s, P, E, nu = float(cfg["s"]), float(cfg["P"]), float(cfg["E"]), float(cfg["nu"])
        coords, cells, _, edges = _mesh_l_bracket(s, fillet=0.0)
        c11, c12, c33 = _plane_stress_constants(np.full(len(cells), E), np.full(len(cells), nu))
        left_nodes = np.unique(edges["left"].ravel())
        bcs = [(left_nodes, 0, 0.0), (left_nodes, 1, 0.0)]
        trac = lambda x, y: (6.0 * P * ((y - s) / s) * (1.0 - (y - s) / s), np.zeros_like(y))
        u_nodal, dg1, C = _solve_p2_elasticity(coords, cells, c11, c12, c33, bcs, edges["load"], trac)
        pts = _eval_grid(0.0, 2.0 * s, 0.0, 2.0 * s, 3.0 * s * s)
        pts = pts[~((pts[:, 0] < s - 1e-12) & (pts[:, 1] > s + 1e-12))]

    elif kind == "plate_hole":
        r, W, H, sigma, E, nu = (float(cfg["r"]), float(cfg["W"]), float(cfg["H"]),
                                 float(cfg["sigma"]), float(cfg["E"]), float(cfg["nu"]))
        coords, cells, _, edges = _mesh_plate_hole(r, W, H)
        c11, c12, c33 = _plane_stress_constants(np.full(len(cells), E), np.full(len(cells), nu))
        left_nodes = np.unique(edges["left"].ravel())
        bot_nodes = np.unique(edges["bottom"].ravel())
        bcs = [(left_nodes, 0, 0.0), (bot_nodes, 1, 0.0)]
        trac = lambda x, y: (np.full_like(x, sigma), np.zeros_like(y))
        u_nodal, dg1, C = _solve_p2_elasticity(coords, cells, c11, c12, c33, bcs, edges["right"], trac)
        pts = _eval_grid(0.0, W, 0.0, H, W * H)
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) > r + 1e-3]

    elif kind == "inclusion":
        a, b, W, H, sigma = (float(cfg["a"]), float(cfg["b"]), float(cfg["W"]),
                             float(cfg["H"]), float(cfg["sigma"]))
        E1, nu1, E2, nu2 = float(cfg["E1"]), float(cfg["nu1"]), float(cfg["E2"]), float(cfg["nu2"])
        coords, cells, mats, edges = _mesh_inclusion(a, b, W, H)
        E_arr = np.where(mats == 1, E2, E1)
        nu_arr = np.where(mats == 1, nu2, nu1)
        c11, c12, c33 = _plane_stress_constants(E_arr, nu_arr)
        left_nodes = np.unique(edges["left"].ravel())
        bot_nodes = np.unique(edges["bottom"].ravel())
        bcs = [(left_nodes, 0, 0.0), (bot_nodes, 1, 0.0)]
        trac = lambda x, y: (np.full_like(x, sigma), np.zeros_like(y))
        u_nodal, dg1, C = _solve_p2_elasticity(coords, cells, c11, c12, c33, bcs, edges["right"], trac)
        dist = lambda m: np.abs(
            np.hypot(m[:, 0], m[:, 1]) - a * b / np.sqrt(
                (b * np.cos(np.arctan2(m[:, 1], m[:, 0]))) ** 2
                + (a * np.sin(np.arctan2(m[:, 1], m[:, 0]))) ** 2
            )
        )
        pts = _eval_grid(0.0, W, 0.0, H, W * H)
        pts = pts[dist(pts) > 1e-3]
    else:
        raise ValueError(f"Unsupported native FEM benchmark kind: {kind!r}")

    u, v, vm, ok = _evaluate_field(coords, cells, u_nodal, dg1, pts)
    ok = ok & np.isfinite(vm)
    print(f"  Built-in P2 FEM reference ({kind}): {len(cells)} cells, compliance {C:.6e}", flush=True)
    print(f"  evaluated at {int(ok.sum())} points", flush=True)
    return {
        "coords": pts[ok],
        "u": u[ok],
        "v": v[ok],
        "vm": vm[ok],
        "compliance": np.array(C, dtype=np.float64),
        "cells": np.array(len(cells), dtype=np.int64),
        "engine": "builtin_scipy_p2",
    }


def solve_parametric_check(spec: dict) -> List[dict]:
    """Drop-in pure-Python/SciPy counterpart to `fem/fem_parametric.py --check`."""
    bench = spec["benchmark"]
    kw = dict(spec.get("kwargs", {}))
    out: List[dict] = []

    for p in spec["points"]:
        pval = float(p)
        if bench == "cantilever":
            L = float(kw.get("L", 4.0))
            P = float(kw.get("P", -1e6))
            E = float(kw.get("E", 210e9))
            nu = float(kw.get("nu", 0.3))
            coords, cells, _, edges = _mesh_cantilever(L, pval, nx=120, ny=max(16, int(round(32 * pval))))
            c11, c12, c33 = _plane_stress_constants(np.full(len(cells), E), np.full(len(cells), nu))
            left_nodes = np.unique(edges["left"].ravel())
            bcs = [(left_nodes, 0, 0.0), (left_nodes, 1, 0.0)]
            trac = lambda x, y: (np.zeros_like(y), 6.0 * (P / pval) * (y / pval) * (1.0 - y / pval))
            _, _, C = _solve_p2_elasticity(coords, cells, c11, c12, c33, bcs, edges["right"], trac)
            out.append(dict(p=pval, compliance=C, volume=L * pval))

        elif bench == "plate_hole":
            W = float(kw.get("W", 2.0))
            H = float(kw.get("H", 2.0))
            sigma_inf = float(kw.get("sigma_inf", 1e6))
            E = float(kw.get("E", 210e9))
            nu = float(kw.get("nu", 0.3))
            coords, cells, _, edges = _mesh_plate_hole(pval, W, H, n_rad=46, n_ang_half=32)
            c11, c12, c33 = _plane_stress_constants(np.full(len(cells), E), np.full(len(cells), nu))
            left_nodes = np.unique(edges["left"].ravel())
            bot_nodes = np.unique(edges["bottom"].ravel())
            bcs = [(left_nodes, 0, 0.0), (bot_nodes, 1, 0.0)]
            trac = lambda x, y: (np.full_like(x, sigma_inf), np.zeros_like(y))
            u_nodal, dg1, C = _solve_p2_elasticity(coords, cells, c11, c12, c33, bcs, edges["right"], trac)
            th = np.linspace(0.0, 0.5 * math.pi, 181)
            arc = np.stack([(pval * (1.0 + 3e-4)) * np.cos(th), (pval * (1.0 + 3e-4)) * np.sin(th)], axis=-1)
            _, _, vm_arc, ok = _evaluate_field(coords, cells, u_nodal, dg1, arc)
            out.append(dict(
                p=pval,
                compliance=C,
                peak_vm=float(np.max(vm_arc[ok])),
                volume=W * H - math.pi * pval * pval / 4.0,
            ))

        elif bench == "l_bracket":
            s = float(kw.get("s", 1.0))
            P = float(kw.get("P", 1e6))
            E = float(kw.get("E", 210e9))
            nu = float(kw.get("nu", 0.3))
            coords, cells, _, edges = _mesh_l_bracket(s, fillet=pval, n_arm=40)
            c11, c12, c33 = _plane_stress_constants(np.full(len(cells), E), np.full(len(cells), nu))
            left_nodes = np.unique(edges["left"].ravel())
            bcs = [(left_nodes, 0, 0.0), (left_nodes, 1, 0.0)]
            trac = lambda x, y: (6.0 * P * ((y - s) / s) * (1.0 - (y - s) / s), np.zeros_like(y))
            u_nodal, dg1, C = _solve_p2_elasticity(coords, cells, c11, c12, c33, bcs, edges["load"], trac)
            cx, cy = s - pval, s + pval
            ang = np.linspace(-0.5 * math.pi, 0.0, 181)
            k = 1.0 + 3e-4
            p_arc = np.stack([cx + pval * k * np.cos(ang), cy + pval * k * np.sin(ang)], axis=-1)
            p_h = np.stack([np.linspace(s - 0.5, s - pval, 60), np.full(60, s + 1e-5)], axis=-1)
            p_v = np.stack([np.full(60, s + 1e-5), np.linspace(s + pval, s + 0.5, 60)], axis=-1)
            probe = np.concatenate([p_arc, p_h, p_v], axis=0)
            _, _, vm_p, ok = _evaluate_field(coords, cells, u_nodal, dg1, probe)
            out.append(dict(
                p=pval,
                compliance=C,
                peak_vm=float(np.max(vm_p[ok])),
                volume=3.0 * s * s + pval * pval * (1.0 - math.pi / 4.0),
            ))
        else:
            raise ValueError(f"Unsupported parametric benchmark: {bench!r}")
    return out


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "--check":
        with open(sys.argv[2]) as fh:
            spec = json.load(fh)
        res = solve_parametric_check(spec)
        with open(sys.argv[3], "w") as fh:
            json.dump(res, fh)
        print(json.dumps(res), flush=True)
    else:
        with open(sys.argv[1]) as fh:
            spec = json.load(fh)
        res = solve_live_spec(spec)
        np.savez_compressed(
            sys.argv[2],
            coords=res["coords"], u=res["u"], v=res["v"], vm=res["vm"],
            compliance=res["compliance"], cells=res["cells"],
        )
