"""Run from the project root:
    python tests/test_validation_exclusion.py
"""

import os
import sys
import traceback
import warnings

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

import numpy as np

from atlas.validation import (SINGULARITY_EXCLUSION_FACTOR, VON_MISES_EXCL_TARGET,
                              singular_locations, estimate_local_point_spacing,
                              exclusion_mask, compute_von_mises_metrics,
                              default_exclusion_radius)

ok = []
bad = []


def check(name, fn):
    try:
        fn()
        ok.append(name)
        print(f"PASS  {name}")
    except Exception as e:
        bad.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


def _grid(step=0.1, n=20, ox=0.0, oy=0.0):
    xs = np.linspace(ox, ox + step * (n - 1), n)
    ys = np.linspace(oy, oy + step * (n - 1), n)
    X, Y = np.meshgrid(xs, ys)
    return np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])


def test_config_resolution():
    c = singular_locations('cantilever', {'L': 4.0, 'H': 1.0})
    assert [(n, (round(x, 6), round(y, 6))) for n, (x, y) in c] == \
        [('clamped_corner_x0_y0', (0.0, 0.0)), ('clamped_corner_x0_yH', (0.0, 1.0))], c
    c = singular_locations('l_bracket', {'size': 1.0})
    assert [(n, (round(x, 6), round(y, 6))) for n, (x, y) in c] == \
        [('reentrant_corner', (1.0, 1.0))], c
    c = singular_locations('plate_hole', {'W': 2.0, 'H': 2.0, 'r': 0.3})
    assert [(n, (round(x, 6), round(y, 6))) for n, (x, y) in c] == \
        [('hole_on_loading_axis', (0.3, 0.0))], c


def test_centers_override():
    c = singular_locations('cantilever', {}, centers=[(0.5, 0.25)])
    assert len(c) == 1 and c[0][1] == (0.5, 0.25), c


def test_missing_geometry_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = singular_locations('cantilever', {'L': 4.0})
    assert len(c) == 1, c
    assert any("singular location" in str(x.message) for x in w), "expected a warning"


def test_local_spacing_grid():
    coords = _grid(step=0.1, n=20)
    s = estimate_local_point_spacing(coords, (0.2, 0.2))
    assert s is not None and abs(s - 0.1) < 1e-6, s


def test_local_spacing_empty():
    assert estimate_local_point_spacing(np.empty((0, 2)), (0.0, 0.0)) is None


def test_exclusion_mask():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    keep = exclusion_mask(coords, [(0.0, 0.0)], 0.5)
    assert keep.tolist() == [False, True, True], keep
    keep = exclusion_mask(coords, [(0.0, 0.0)], None)
    assert keep.tolist() == [True, True, True]
    assert exclusion_mask(coords, [], 0.3).all()


def test_default_exclusion_radius():
    coords = _grid(step=0.1, n=20)
    r = default_exclusion_radius(coords, [(0.0, 0.0), (1.9, 1.9)])
    assert r is not None and abs(r - SINGULARITY_EXCLUSION_FACTOR * 0.1) < 1e-6, r


def test_von_mises_metrics_identity():
    coords = _grid(step=0.1, n=20)
    pred = coords[:, 0:1]
    ref = coords[:, 0:1] * 1.0
    m = compute_von_mises_metrics(pred, ref, coords, [(0.0, 0.0)], 0.3)
    assert abs(m['von_mises_l2_global']) < 1e-12
    assert abs(m['von_mises_l2_excl']) < 1e-12
    assert m['von_mises_total_points'] == len(coords)
    assert m['von_mises_excluded_points'] > 0


def test_von_mises_metrics_excl_isolates_bad_region():
    coords = _grid(step=0.05, n=30)
    ref = coords[:, 0:1]
    pred = ref.copy()
    inside = np.hypot(coords[:, 0] - 0.0, coords[:, 1] - 0.0) < 0.3
    pred[inside] += 50.0
    m = compute_von_mises_metrics(pred, ref, coords, [(0.0, 0.0)], 0.3)
    assert m['von_mises_l2_global'] > 1.0
    assert m['von_mises_l2_excl'] is not None and m['von_mises_l2_excl'] < 1e-9, m


def test_von_mises_metrics_empty_keep():
    coords = _grid(step=0.1, n=10)
    m = compute_von_mises_metrics(coords[:, 0:1], coords[:, 0:1], coords, [(0.0, 0.0)], 99.0)
    assert m['von_mises_l2_excl'] is None
    assert m['von_mises_excluded_points'] == len(coords)


def test_von_mises_metrics_no_centers():
    coords = _grid(step=0.1, n=10)
    m = compute_von_mises_metrics(coords[:, 0:1] * 2, coords[:, 0:1], coords, [], 0.5)
    assert m['von_mises_l2_excl'] is None and m['von_mises_excluded_points'] == 0


def test_validate_mock_graceful_new_keys():
    import tempfile
    import torch
    from atlas.sampling import PlateWithHoleSampler
    from atlas.networks import create_xpinn_for_benchmark
    from atlas.physics import PlaneStressElasticity
    from atlas.validation import FEMValidator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = PlateWithHoleSampler(W=2.0, H=2.0, r_init=0.3, n_interior=60, n_boundary=20, n_interface=10, device=device)
    model = create_xpinn_for_benchmark("plate_hole", device=device)
    phys = PlaneStressElasticity(E=2.1e-2, nu=0.3)
    with tempfile.TemporaryDirectory() as td:
        res = FEMValidator(phys, device=device).validate(model, sam, benchmark="plate_hole",
                                                         use_mock=True, data_dir=td, r_excl=0.1)
    assert res['used_mock_data'] is True
    assert res['von_mises_l2_global'] is None
    assert res['von_mises_l2_excl'] is None
    assert res['von_mises_l2_excl_within_10pct'] is None
    assert res['von_mises_excluded_points'] is None
    assert 'relative_L2_displacement' in res and 'displacement_within_5pct_target' in res


check("config resolution per benchmark", test_config_resolution)
check("explicit centers override config", test_centers_override)
check("missing geometry key warns, does not crash", test_missing_geometry_warns)
check("local point spacing = grid step", test_local_spacing_grid)
check("local point spacing empty -> None", test_local_spacing_empty)
check("exclusion_mask", test_exclusion_mask)
check("default_exclusion_radius = factor x spacing", test_default_exclusion_radius)
check("von Mises metrics: identity -> ~0", test_von_mises_metrics_identity)
check("von Mises excl isolates bad region", test_von_mises_metrics_excl_isolates_bad_region)
check("von Mises excl: all points excluded -> None", test_von_mises_metrics_empty_keep)
check("von Mises excl: no centers -> None", test_von_mises_metrics_no_centers)
check("validate() mock path reports new keys gracefully", test_validate_mock_graceful_new_keys)

print(f"\npassed: {len(ok)}  failed: {len(bad)}  {bad if bad else ''}")
assert VON_MISES_EXCL_TARGET == 0.10, "provisional target must stay 10%"
print("DONE")