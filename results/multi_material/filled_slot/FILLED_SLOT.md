# Epoxy-filled elliptical slot in a steel plate: ATLAS energy vs ATLAS-XPINN

An elliptical slot (semi-axes 0.2 m along the load, 0.4 m across it) filled with epoxy (3.5 GPa,
nu 0.35) in a steel plate (210 GPa, nu 0.30), contrast 60. Quarter plate [0, 2]^2, symmetry rollers
on x = 0 and y = 0, 1 MPa tension on x = 2. Protocol fixed before any run: [PROTOCOL.md](PROTOCOL.md).

**Result: passes the protocol (lower 5-seed means on both metrics), but the von Mises gain is
small and not statistically significant.** The displacement gain is large and holds in every seed.

## Reference
FEniCS P2 on an ellipse-conforming mesh (108k cells), element-wise DG1 stresses. Mesh convergence
between the last two levels: compliance 6.9e-9, interface-band von Mises 4.7e-4, peak 6.2e-6.
Stress concentration Kt = 4.77 at the top of the slot (an open slot of this shape: 5.0).

## Seed 42 (2000 Adam + 2000 L-BFGS, same 12,960 quadrature points)

| solver | params | train time | displacement L2 | von Mises L2 | band vM, filler side | band vM, plate side | compliance vs FEM |
|---|---|---|---|---|---|---|---|
| ATLAS energy (single 4x64) | 12,802 | 326 s | 0.201% | 0.543% | 27.1% | 3.09% | -0.034% |
| ATLAS energy (single 4x91, parameter-matched) | 25,573 | 556 s | 0.182% | 0.495% | 26.0% | 2.37% | -0.036% |
| **ATLAS-XPINN (filler + plate)** | 25,604 | 363 s | **0.066%** | **0.433%** | **0.32%** | 3.03% | **-0.002%** |

## Five seeds (42-46): ATLAS-XPINN vs the better single network

| solver | displacement L2 | von Mises L2 | band vM, filler side | band vM, plate side | train time |
|---|---|---|---|---|---|
| ATLAS energy (single 4x91) | 0.222 +- 0.037% | 0.531 +- 0.064% | 28.8 +- 3.3% | **2.89 +- 0.48%** | 557 s |
| **ATLAS-XPINN** | **0.058 +- 0.009%** | **0.467 +- 0.104%** | **0.55 +- 0.36%** | 3.13 +- 0.79% | **360 s** |

- Displacement: 3.8x lower, in all five seeds.
- von Mises: 12% lower on average, but lower in only 3 of 5 seeds; paired difference -0.064
  percentage points, t = -1.24 with 4 degrees of freedom (p about 0.3). Treat it as a tie.
- Stress next to the interface: 53x lower on the filler side; on the plate side the single network
  is slightly better (2.9% against 3.1%).
- Peak von Mises at the top of the slot, predicted / FEM: ATLAS-XPINN 0.976 on average (0.954 to
  0.994, slightly non-conservative), the 4x91 network 1.005 (0.961 to 1.051, scattered both ways).

## Why the stress gain is smaller here than for the insert and the inclusion
With a filler 60x softer than the plate, the slot behaves almost like an open hole: the dominant
stress error sits in the steel at the Kt = 4.8 concentration, on the plate side of the interface,
which a decomposition along the interface does not help. ATLAS-XPINN removes the error inside the
filler (28.8% -> 0.55%) and the displacement error, but the filler carries almost no stress, so the
whole-plate von Mises metric barely moves. The steel insert (3x stiffer than its plate) is the
opposite case: the interface error dominates both sides, and ATLAS-XPINN wins by 10x.

## Files
- `atlas/`, `atlas_wide/`, `xpinn/`: `seed*.json`, `seed*.npz`, `seed*_model.pt`; `table.md`,
  `convergence.png`, `fields.png`, `profiles.png`; `logs/`
- code: `atlas/benchmarks/filled_slot.py`, `fem/fenics_scripts/filled_slot_fenics.py`, `scripts/multi_material/run_filled_slot.py`

Reproduce (one GPU job at a time):

    bash scripts/multi_material/run_protocol.sh filled_slot vm_l2
    python scripts/multi_material/make_report.py --bench filled_slot
