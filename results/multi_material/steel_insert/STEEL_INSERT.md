# Steel insert in an aluminium plate: ATLAS energy vs ATLAS-XPINN

A bonded steel insert (210 GPa, nu 0.30, radius 0.3 m) in an aluminium plate (70 GPa, nu 0.33):
the inclusion benchmark's geometry and quadrature with a STIFF inclusion instead of a soft one.
Quarter plate [0, 2]^2, symmetry rollers on x = 0 and y = 0, 1 MPa tension on x = 2. Protocol,
including the decision rule, fixed before any run: [PROTOCOL.md](PROTOCOL.md).

**Result: ATLAS-XPINN beats both single networks on every metric and every seed** - the second
benchmark where domain decomposition pays off.

## Reference
FEniCS P2 on the interface-conforming mesh of the inclusion benchmark (110k cells), element-wise
DG1 stresses. Mesh convergence between the last two levels: compliance 1.2e-10, interface-band von
Mises 4.6e-5. The insert carries a uniform 1.27 sigma (a stiff insert attracts load).

## Seed 42 (2000 Adam + 2000 L-BFGS, same 12,960 quadrature points)

| solver | params | train time | displacement L2 | von Mises L2 | band vM, insert side | band vM, plate side | compliance vs FEM |
|---|---|---|---|---|---|---|---|
| ATLAS energy (single 4x64) | 12,802 | 330 s | 0.061% | 1.245% | 10.93% | 9.45% | -0.020% |
| ATLAS energy (single 4x91, parameter-matched) | 25,573 | 564 s | 0.050% | 1.044% | 8.78% | 9.06% | -0.005% |
| **ATLAS-XPINN (insert + plate)** | 25,604 | 378 s | **0.021%** | **0.117%** | **0.36%** | **0.56%** | **-0.002%** |

"Band" = the 0.0025 m grid within 0.05 m of the interface.

## Five seeds (42-46): ATLAS-XPINN vs the better single network

| solver | displacement L2 | von Mises L2 | band vM, insert side | band vM, plate side | train time |
|---|---|---|---|---|---|
| ATLAS energy (single 4x91) | 0.063 +- 0.014% | 1.219 +- 0.217% | 10.6 +- 1.6% | 9.73 +- 1.23% | 599 s |
| **ATLAS-XPINN** | **0.021 +- 0.003%** | **0.118 +- 0.004%** | **0.24 +- 0.09%** | **0.66 +- 0.08%** | **377 s** |

Lower displacement AND von Mises error in all five seeds: 3x lower displacement error, 10x lower
von Mises error, 15-44x lower stress error at the interface, in 63% of the time, and a von Mises
spread over seeds 50x narrower (+-0.004% against +-0.22%).

## Why
The same mechanism as the soft inclusion ([../bimaterial/BIMATERIAL.md](../bimaterial/BIMATERIAL.md)):
a smooth single network cannot put a kink in the displacement at the interface, so it oscillates
around it - here between 1.0 and 1.5 MPa where the FEM stress is smooth
([profiles.png](profiles.png), [fields.png](fields.png)). With the subdomain interface on the
material boundary, each subnetwork carries a smooth field (the insert's is uniform) and ATLAS-XPINN
follows the FEM profile through the interface. Unlike the soft inclusion, no local peak is
under- or over-estimated.

## Files
- `atlas/`, `atlas_wide/`, `xpinn/`: `seed*.json` (metrics, error-vs-time probes), `seed*.npz`,
  `seed*_model.pt`; `table.md`, `convergence.png`, `fields.png`, `profiles.png`; `logs/`
- code: `atlas/benchmarks/steel_insert.py`, `fem/fenics_scripts/steel_insert_fenics.py`, `scripts/multi_material/run_steel_insert.py`

Reproduce (one GPU job at a time):

    bash scripts/multi_material/run_protocol.sh steel_insert vm_l2
    python scripts/multi_material/make_report.py --bench steel_insert
