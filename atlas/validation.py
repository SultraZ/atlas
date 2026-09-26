"""validation.py
FEM validation module for ATLAS
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import json
import warnings
from atlas.paths import FEM_REFERENCE_DIR


SINGULAR_LOCATION_CONFIG = {
    'cantilever': [
        ('clamped_corner_x0_y0', lambda g: (0.0, 0.0)),
        ('clamped_corner_x0_yH', lambda g: (0.0, g['H'])),
    ],
    'l_bracket': [
        ('reentrant_corner', lambda g: (g['size'], g['size'])),
    ],
    'plate_hole': [
        ('hole_on_loading_axis', lambda g: (g['r'], 0.0)),
    ],
}

SINGULARITY_EXCLUSION_FACTOR = 2.5
SINGULARITY_LOCAL_SPACING_NNEAR = 24

VON_MISES_EXCL_TARGET = 0.10

VON_MISES_ROI_HALF_WIDTH = 0.06  # plate_hole: band |dist-to-center - r| < 0.06

VON_MISES_ROI_CONFIG = {
    'cantilever': {
        'desc': 'clamp corners (fixed-radius 0.10 disks)',
        'kind': 'disks',
        'centers_fn': lambda g: [(0.0, 0.0), (0.0, g['H'])],
        'radius': 0.10,
    },
    'l_bracket': {
        'desc': 're-entrant corner (fixed-radius 0.12 disk)',
        'kind': 'disks',
        'centers_fn': lambda g: [(g['size'], g['size'])],
        'radius': 0.12,
    },
    'plate_hole': {
        'desc': f'hole-arc band (half-width {VON_MISES_ROI_HALF_WIDTH})',
        'kind': 'ring',
        'r_fn': lambda g: g['r'],
        'half_width': VON_MISES_ROI_HALF_WIDTH,
    },
}


def relative_l2(pred, ref) -> float:
    """Classic relative L2 error (module-level so metrics helpers can reuse
    it without an FEMValidator instance).
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))


def singular_locations(benchmark: str, geometry: Dict[str, float],
                       centers: Optional[List[Tuple[float, float]]] = None
                       ) -> List[Tuple[str, Tuple[float, float]]]:
    """Resolve this benchmark's known singular locations from SINGULAR_LOCATION_CONFIG.
    A caller can override the config by passing explicit `centers`.
    Unresolvable entries (missing geometry key) produce a warning, not a crash.
    """
    resolved: List[Tuple[str, Tuple[float, float]]] = []
    if centers is not None:
        return [(f'singular_{i}', (float(c[0]), float(c[1]))) for i, c in enumerate(centers)]
    for name, loc_fn in SINGULAR_LOCATION_CONFIG.get(benchmark, []):
        try:
            x, y = loc_fn(geometry)
            resolved.append((name, (float(x), float(y))))
        except (KeyError, TypeError, ValueError) as e:
            warnings.warn(
                f"[FEM validation] cannot resolve singular location '{name}' "
                f"for '{benchmark}' from geometry {geometry}: {e}", stacklevel=2)
    return resolved


def estimate_local_point_spacing(coords, center: Tuple[float, float],
                                 n_near: int = SINGULARITY_LOCAL_SPACING_NNEAR
                                 ) -> Optional[float]:
    """Median nearest-neighbor distance among the n_near evaluation points
    closest to `center`. Pure numpy (this repo has no scipy dependency),
    so it works identically on CPU/GPU-loaded reference arrays.
    Returns None for an empty coordinate set.
    """
    coords = np.asarray(coords)
    if coords is None or coords.shape[0] == 0:
        return None
    d2 = (coords[:, 0] - center[0]) ** 2 + (coords[:, 1] - center[1]) ** 2
    k = min(int(n_near), coords.shape[0])
    idx = np.argpartition(d2, k - 1)[:k]
    sub = coords[idx]
    dists = []
    for p in sub:
        dd = np.hypot(sub[:, 0] - p[0], sub[:, 1] - p[1])
        dd = dd[dd > 1e-15]
        if dd.size:
            dists.append(float(dd.min()))
    if not dists:
        return None
    return float(np.median(dists))


def exclusion_mask(coords, centers, r_excl: float) -> np.ndarray:
    """Boolean keep-mask over evaluation points: True where the point is
    farther than r_excl from EVERY singular center. Empty/invalid r_excl
    keeps everything.
    """
    coords = np.asarray(coords)
    if r_excl is None or r_excl <= 0 or not centers:
        return np.ones(coords.shape[0], dtype=bool)
    keep = np.ones(coords.shape[0], dtype=bool)
    for cx, cy in centers:
        keep &= ~(np.hypot(coords[:, 0] - cx, coords[:, 1] - cy) < r_excl)
    return keep


def compute_von_mises_metrics(vm_pred, vm_fem, coords, centers, r_excl: Optional[float]):
    """Both von Mises L2 variants in one place (used by FEMValidator and by
    plot_fem_comparison so the printed/JSON numbers and the plot always agree):
      - von_mises_l2_global : relative L2 over all evaluation points
      - von_mises_l2_excl   : relative L2 excluding points within r_excl of any center
    Raises no exceptions on degenerate inputs - returns None exclusion
    numbers when exclusion isn't possible/meaningful.
    """
    vm_pred = np.asarray(vm_pred).reshape(-1, 1)
    vm_fem = np.asarray(vm_fem).reshape(-1, 1)
    coords = np.asarray(coords)
    n_total = int(coords.shape[0])
    err_global = relative_l2(vm_pred, vm_fem)
    base = {
        'von_mises_l2_global': err_global,
        'von_mises_l2_excl': None,
        'von_mises_excluded_points': 0,
        'von_mises_total_points': n_total,
    }
    if r_excl is None or r_excl <= 0 or not centers:
        return base
    keep = exclusion_mask(coords, centers, r_excl)
    n_excluded = int((~keep).sum())
    if not keep.any():
        base['von_mises_excluded_points'] = n_excluded
        return base
    err_excl = relative_l2(vm_pred[keep], vm_fem[keep])
    return {
        'von_mises_l2_global': err_global,
        'von_mises_l2_excl': err_excl,
        'von_mises_excluded_points': n_excluded,
        'von_mises_total_points': n_total,
    }


def default_exclusion_radius(coords, centers, factor: float = SINGULARITY_EXCLUSION_FACTOR
                             ) -> Optional[float]:
    """Auto r_excl: SINGULARITY_EXCLUSION_FACTOR x the local point spacing
    near each singular center (max across centers, so a single radius covers
    every location). Tied to the actual evaluation-point density of THIS
    benchmark's mesh, not a hardcoded number.
    """
    if not centers:
        return None
    spacings = [estimate_local_point_spacing(coords, c) for c in centers]
    spacings = [s for s in spacings if s is not None]
    if not spacings:
        return None
    return float(factor * max(spacings))


def roi_mask(coords, benchmark: str, geometry: Dict[str, float]) -> np.ndarray:
    """Keep-mask for the FIXED physical singular-ROI metric: True where the
    evaluation point is OUTSIDE this benchmark's configured singular region
    (fixed-radius disks around clamp/re-entrant corners, a fixed half-width
    band hugging the hole arc). Unconfigured benchmark / unresolvable geometry
    keeps everything.
    """
    coords = np.asarray(coords)
    keep = np.ones(coords.shape[0], dtype=bool)
    spec = VON_MISES_ROI_CONFIG.get(benchmark)
    if spec is None:
        return keep
    try:
        if spec['kind'] == 'disks':
            centers = spec['centers_fn'](geometry)
            r = spec['radius']
            for cx, cy in centers:
                keep &= ~(np.hypot(coords[:, 0] - cx, coords[:, 1] - cy) < r)
        elif spec['kind'] == 'ring':
            r_hole = float(spec['r_fn'](geometry))
            keep &= ~(np.abs(np.hypot(coords[:, 0], coords[:, 1]) - r_hole)
                      < spec['half_width'])
    except (KeyError, TypeError, ValueError) as e:
        warnings.warn(f"[FEM validation] cannot build ROI mask for "
                      f"'{benchmark}' from geometry {geometry}: {e}", stacklevel=2)
        return np.ones(coords.shape[0], dtype=bool)
    return keep


def compute_von_mises_roi_metric(vm_pred, vm_fem, coords, benchmark: str,
                                 geometry: Dict[str, float]) -> Dict:
    """Fixed singular-ROI relative L2 von Mises error `von_mises_l2_roi`,
    additive to the existing metrics (nothing here changes the keys the
    validator already produces). Returns None sentinel values when the
    benchmark/geometry isn't resolved or the ROI removes every point.
    """
    base = {
        'von_mises_l2_roi': None,
        'von_mises_roi_excluded_points': 0,
        'von_mises_roi_desc': VON_MISES_ROI_CONFIG.get(benchmark, {}).get(
            'desc', 'n/a'),
    }
    vm_pred = np.asarray(vm_pred).reshape(-1, 1)
    vm_fem = np.asarray(vm_fem).reshape(-1, 1)
    coords = np.asarray(coords)
    if coords.shape[0] == 0 or vm_pred.shape[0] != coords.shape[0]:
        return base
    keep = roi_mask(coords, benchmark, geometry)
    n_excluded = int((~keep).sum())
    base['von_mises_roi_excluded_points'] = n_excluded
    if not keep.any():
        return base
    base['von_mises_l2_roi'] = relative_l2(vm_pred[keep], vm_fem[keep])
    return base


def sampler_geometry_dict(sampler, benchmark: str) -> Dict[str, float]:
    """Extract the geometry parameters that actually matter for comparing
    a sampler's problem to a stored FEM reference. Used both when *writing*
    a reference (fenics_scripts write a matching {benchmark}_meta.json) and
    when *reading* one (FEMValidator.load_fem_reference checks the two
    agree) - see check_geometry_match().
    """
    if benchmark == 'cantilever':
        return {'L': float(sampler.L), 'H': float(sampler.H)}
    elif benchmark == 'l_bracket':
        geom = {'size': float(sampler.size)}
        if getattr(sampler, 'fillet_enabled', False):
            geom['fillet'] = float(sampler.fillet.detach().cpu().item())
        return geom
    elif benchmark == 'plate_hole':
        r_val = float(sampler.r.detach().cpu().item()) if torch.is_tensor(sampler.r) else float(sampler.r)
        return {'W': float(sampler.W), 'H': float(sampler.H), 'r': r_val}
    else:
        return {}


def check_geometry_match(meta: Optional[Dict], current: Dict[str, float],
                          rel_tol: float = 1e-3) -> Optional[str]:
    """Compare a loaded {benchmark}_meta.json (written by the FEniCS
    reference scripts) against the geometry of the sampler you're actually
    validating right now. Returns None if they match (or no meta file was
    found and the caller should decide how to treat that), otherwise a
    human-readable mismatch message.
    """
    if meta is None:
        return ("no geometry metadata found alongside this FEM reference "
                "(older reference file, or not generated by the provided "
                "fenics_scripts/*.py) - cannot verify it matches the current "
                "run's geometry. Treat this validation result with caution.")
    mismatches = []
    for key, current_val in current.items():
        ref_val = meta.get(key)
        if ref_val is None:
            mismatches.append(f"{key}: reference metadata missing this field")
            continue
        denom = max(abs(ref_val), 1e-12)
        if abs(current_val - ref_val) / denom > rel_tol:
            mismatches.append(f"{key}: reference={ref_val}, current run={current_val}")
    if mismatches:
        return ("FEM reference geometry does NOT match the current run:\n    "
                + "\n    ".join(mismatches)
                + "\n    -> this comparison is against the WRONG problem. "
                  "Regenerate the FEM reference at the current geometry "
                  "before trusting this error number.")
    return None


class FEMValidator:
    def __init__(self, physics, device: str = 'cpu'):
        self.physics = physics
        self.device = device
        self.history = {}

    def load_fem_reference(self, benchmark: str, data_dir: str = str(FEM_REFERENCE_DIR)):
        # try to load real FEM data if the student already generated it
        try:
            coords = np.load(os.path.join(data_dir, f'{benchmark}_coords.npy'))
            u = np.load(os.path.join(data_dir, f'{benchmark}_u.npy'))
            v = np.load(os.path.join(data_dir, f'{benchmark}_v.npy'))
            meta = None
            meta_path = os.path.join(data_dir, f'{benchmark}_meta.json')
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            vonmises = None
            vm_path = os.path.join(data_dir, f'{benchmark}_vonmises.npy')
            if os.path.isfile(vm_path):
                vonmises = np.load(vm_path)
            return {'coords': coords, 'u': u, 'v': v, 'meta': meta, 'vonmises': vonmises}
        except FileNotFoundError:
            return None

    def generate_mock_fem_reference(self, sampler, n_points: int = 2000):
        # super rough placeholder so we can test the whole pipeline
        # replace this later with actual FEM results
        if hasattr(sampler, 'L'):  # cantilever
            x = np.random.uniform(0, sampler.L, (n_points, 1))
            y = np.random.uniform(0, sampler.H, (n_points, 1))
        elif hasattr(sampler, 'W'):  # plate with hole
            x = np.random.uniform(0, sampler.W, (n_points, 1))
            y = np.random.uniform(0, sampler.H, (n_points, 1))
        else:  # L-bracket
            s = sampler.size
            x = np.random.uniform(0, 2 * s, (n_points, 1))
            y = np.random.uniform(0, 2 * s, (n_points, 1))

        # fake displacement field just so the numbers aren't zero
        u = 1e-5 * x * (1.0 - y / (y.max() + 1e-8))
        v = -2e-5 * (x / (x.max() + 1e-8)) ** 2
        return {'coords': np.hstack([x, y]), 'u': u, 'v': v}

    def evaluate_xpinn_on_points(self, model, sampler, coords: np.ndarray):
        # reuse the same subdomain routing we already fixed in visualization
        from atlas.visualization import _evaluate_model_on_grid
        device = next(model.parameters()).device
        x_flat = coords[:, 0:1]
        y_flat = coords[:, 1:2]
        u, v = _evaluate_model_on_grid(model, sampler, x_flat, y_flat, device)
        return u.reshape(-1, 1), v.reshape(-1, 1)

    def evaluate_xpinn_vonmises_on_points(self, model, sampler, coords: np.ndarray):
        from atlas.visualization import evaluate_von_mises_on_points
        device = next(model.parameters()).device
        return evaluate_von_mises_on_points(model, self.physics, sampler, coords, device=device)

    def compute_l2_error(self, pred, ref):
        # classic relative L2 (delegates to the module-level helper so the
        # metric helpers and the validator provably use the same formula)
        return relative_l2(pred, ref)

    def compute_max_pointwise_error(self, pred, ref):
        """        Absolute value of the single worst pointwise discrepancy, plus
        where it occurred. The paper's Methodology (Validation Module)
        specifically calls for this alongside the L2 error: "the maximum
        pointwise error at stress concentration locations" - an L2 (average
        -ish) error can look good overall while completely missing a sharp
        local spike right at a stress concentration, which is exactly the
        failure mode a single blended error number would hide.
        """
        diff = np.abs(pred.flatten() - ref.flatten())
        idx = int(np.argmax(diff))
        return float(diff[idx]), idx

    DISPLACEMENT_L2_TARGET = 0.05

    def validate(self, model, sampler, benchmark: str = 'cantilever',
                 use_mock: bool = True, data_dir: str = str(FEM_REFERENCE_DIR),
                 r_excl: Optional[float] = None):
        print("=" * 60)
        print(f"FEM VALIDATION — {benchmark.upper()}")
        print("=" * 60)

        fem_data = self.load_fem_reference(benchmark, data_dir)
        used_mock_data = fem_data is None

        geometry_warning = None
        if fem_data is None:
            if use_mock:
                print("[Warning] No real FEM files found. Using mock data for now.")
                fem_data = self.generate_mock_fem_reference(sampler)
            else:
                raise FileNotFoundError(
                    f"Missing FEM reference for '{benchmark}'. "
                    "Generate it with fenics_scripts/ and put the .npy files in fem/reference/"
                )
        else:
            current_geom = sampler_geometry_dict(sampler, benchmark)
            geometry_warning = check_geometry_match(fem_data.get('meta'), current_geom)
            if geometry_warning:
                warnings.warn(f"[FEM validation] {geometry_warning}", stacklevel=2)
                print(f"  [WARNING] {geometry_warning}")

        coords = fem_data['coords']
        u_fem = fem_data['u']
        v_fem = fem_data['v']
        vm_fem = fem_data.get('vonmises')

        u_pred, v_pred = self.evaluate_xpinn_on_points(model, sampler, coords)

        err_u = self.compute_l2_error(u_pred, u_fem)
        err_v = self.compute_l2_error(v_pred, v_fem)
        err_disp = self.compute_l2_error(
            np.sqrt(u_pred**2 + v_pred**2),
            np.sqrt(u_fem**2 + v_fem**2)
        )

        meaningful = (not used_mock_data) and (geometry_warning is None)
        passed_5pct = (err_disp < self.DISPLACEMENT_L2_TARGET) if meaningful else None

        results = {
            'relative_L2_u': err_u,
            'relative_L2_v': err_v,
            'relative_L2_displacement': err_disp,
            'displacement_within_5pct_target': passed_5pct,
            'used_mock_data': used_mock_data,
            'geometry_warning': geometry_warning,  # None if verified consistent
        }

        print(f"  Relative L2 error (u)      : {err_u:.6e}")
        print(f"  Relative L2 error (v)      : {err_v:.6e}")
        print(f"  Relative L2 error (|disp|) : {err_disp:.6e}  ({err_disp:.2%})")
        if passed_5pct is None:
            reason = "mock data" if used_mock_data else "geometry mismatch"
            print(f"  [Paper target: <5% displacement L2] N/A - not meaningful against {reason}")
        elif passed_5pct:
            print(f"  [Paper target: <5% displacement L2] PASS ({err_disp:.2%} < 5%)")
        else:
            print(f"  [Paper target: <5% displacement L2] FAIL ({err_disp:.2%} >= 5%)")

        if vm_fem is not None:
            vm_pred = self.evaluate_xpinn_vonmises_on_points(model, sampler, coords)
            err_vm = self.compute_l2_error(vm_pred, vm_fem)
            max_err, max_idx = self.compute_max_pointwise_error(vm_pred, vm_fem)
            max_loc = coords[max_idx].tolist()

            geom = sampler_geometry_dict(sampler, benchmark)
            centers = [loc for _, loc in singular_locations(benchmark, geom)]
            r_used = r_excl if r_excl is not None else default_exclusion_radius(coords, centers)
            vm_metrics = compute_von_mises_metrics(vm_pred, vm_fem, coords, centers, r_used)
            err_vm_excl = vm_metrics['von_mises_l2_excl']

            # The 10% stress target gates ONLY on the excluded-region metric
            # (see the VON_MISES_EXCL_TARGET TODO above); global is context.
            vm_within_10pct = (err_vm_excl is not None and err_vm_excl < VON_MISES_EXCL_TARGET) \
                if meaningful else None

            results['relative_L2_von_mises'] = err_vm
            results['von_mises_l2_global'] = err_vm
            results['von_mises_l2_excl'] = err_vm_excl
            results['von_mises_exclusion_radius'] = r_used
            results['von_mises_exclusion_factor'] = SINGULARITY_EXCLUSION_FACTOR
            results['von_mises_excluded_points'] = vm_metrics['von_mises_excluded_points']
            results['von_mises_total_points'] = vm_metrics['von_mises_total_points']
            results['von_mises_l2_excl_within_10pct'] = vm_within_10pct
            results['max_pointwise_von_mises_error'] = max_err
            results['max_pointwise_von_mises_error_location'] = max_loc

            # Fixed singular-region reading (additive; see
            # VON_MISES_ROI_CONFIG for the per-benchmark physical ROIs).
            roi_metrics = compute_von_mises_roi_metric(
                vm_pred, vm_fem, coords, benchmark, geom)
            results['von_mises_l2_roi'] = roi_metrics['von_mises_l2_roi']
            results['von_mises_roi_excluded_points'] = \
                roi_metrics['von_mises_roi_excluded_points']
            results['von_mises_roi_desc'] = roi_metrics['von_mises_roi_desc']
            vm_roi = results['von_mises_l2_roi']
            vm_roi_within_10pct = (vm_roi is not None and vm_roi < VON_MISES_EXCL_TARGET) \
                if meaningful else None
            results['von_mises_l2_roi_within_10pct'] = vm_roi_within_10pct
            vm_roi_total = results['von_mises_total_points']
            vm_roi_note = (f", {vm_roi_total - results['von_mises_roi_excluded_points']}/"
                           f"{vm_roi_total} pts kept") if vm_roi_total else ""

            print(f"  Relative L2 error (von Mises, GLOBAL)   : {err_vm:.6e} ({err_vm:.2%})")
            excl_label = f"{err_vm_excl:.6e} ({err_vm_excl:.2%})" if err_vm_excl is not None else "n/a"
            excl_note = (f" [excluded {vm_metrics['von_mises_excluded_points']}/"
                         f"{vm_metrics['von_mises_total_points']} pts")
            if err_vm_excl is not None and r_used is not None:
                excl_note += f", r_excl={r_used:.6g}"
            excl_note += "]"
            print(f"  Relative L2 error (von Mises, EXCL.)    : {excl_label}{excl_note}")
            roi_label = f"{vm_roi:.6e} ({vm_roi:.2%})" if vm_roi is not None else "n/a"
            print(f"  Relative L2 error (von Mises, ROI)      : {roi_label}{vm_roi_note}"
                  f"  [{results['von_mises_roi_desc']}]")
            if r_used is not None:
                print(f"  r_excl = {r_used:.6g}  (= {SINGULARITY_EXCLUSION_FACTOR:.1f}x local "
                      f"point spacing near {len(centers)} singular location(s); "
                      f"override with r_excl=<value>)")
            if vm_within_10pct is None:
                reason = "mock data" if used_mock_data else "geometry mismatch"
                print(f"  [Stress target: <10% von Mises L2 (excluded region)] "
                      f"N/A - not meaningful against {reason}")
            elif vm_within_10pct:
                print(f"  [Stress target: <10% von Mises L2 (excluded region)] "
                      f"PASS ({err_vm_excl:.2%} < 10%)")
            else:
                print(f"  [Stress target: <10% von Mises L2 (excluded region)] "
                      f"FAIL ({err_vm_excl:.2%} >= 10%)")
            print(f"  Max pointwise von Mises error       : {max_err:.6e} Pa "
                  f"at (x={max_loc[0]:.4f}, y={max_loc[1]:.4f})")
        else:
            results['relative_L2_von_mises'] = None
            results['von_mises_l2_global'] = None
            results['von_mises_l2_excl'] = None
            results['von_mises_exclusion_radius'] = None
            results['von_mises_exclusion_factor'] = SINGULARITY_EXCLUSION_FACTOR
            results['von_mises_excluded_points'] = None
            results['von_mises_total_points'] = None
            results['von_mises_l2_excl_within_10pct'] = None
            results['max_pointwise_von_mises_error'] = None
            results['von_mises_l2_roi'] = None
            results['von_mises_roi_excluded_points'] = None
            results['von_mises_roi_desc'] = VON_MISES_ROI_CONFIG.get(
                benchmark, {}).get('desc', 'n/a')
            results['von_mises_l2_roi_within_10pct'] = None
            print("  [Info] No von Mises reference in this FEM data "
                  "(older reference, or mock data) - stress error not computed. "
                  "Regenerate with the updated fenics_scripts/*.py to get this.")

        print("=" * 60)

        self.history[benchmark] = results
        return results


def run_validation(model, sampler, physics, benchmark: str,
                   device: str = 'cpu', use_mock: bool = True,
                   data_dir: str = str(FEM_REFERENCE_DIR),
                   r_excl: Optional[float] = None):
    # thin wrapper so atlas_main stays clean
    validator = FEMValidator(physics, device=device)
    return validator.validate(model, sampler, benchmark, use_mock=use_mock,
                              data_dir=data_dir, r_excl=r_excl)