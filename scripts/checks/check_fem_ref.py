import numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from atlas.paths import FEM_REFERENCE_DIR
from atlas.sampling import CantileverBeamSampler, LBracketSampler, PlateWithHoleSampler
from atlas.validation import sampler_geometry_dict, check_geometry_match

for b in ["cantilever", "l_bracket", "plate_hole"]:
    for f in ["coords", "u", "v", "vonmises"]:
        p = os.path.join(FEM_REFERENCE_DIR, f"{b}_{f}.npy")
        if not os.path.isfile(p):
            print(f"{b:12s} {f:10s} MISSING")
            continue
        arr = np.load(p)
        print(f"{b:12s} {f:10s} shape={arr.shape} finite={np.isfinite(arr).all()} "
              f"range=[{arr.min():.4g},{arr.max():.4g}]")
    with open(os.path.join(FEM_REFERENCE_DIR, f"{b}_meta.json")) as fh:
        print(f"{b:12s} meta={json.load(fh)}")

cases = [
    ("cantilever", CantileverBeamSampler(L=4.0, H=1.0), FEM_REFERENCE_DIR),
    ("l_bracket", LBracketSampler(size=1.0), FEM_REFERENCE_DIR),
    ("plate_hole", PlateWithHoleSampler(W=2.0, H=2.0, r_init=0.3), FEM_REFERENCE_DIR),
]
for bench, sampler, data_dir in cases:
    meta_path = os.path.join(data_dir, f"{bench}_meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    current = sampler_geometry_dict(sampler, bench)
    warn = check_geometry_match(meta, current)
    print(f"\n{bench}: current geom = {current}")
    print(f"{bench}: geometry check -> {warn if warn else 'MATCH (OK)'}")