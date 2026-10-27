# ATLAS-XPINN inverse design, seeds 42, 43, 44, 45, 46

FEM = FEniCS 2019.1.0. The FEM target is the target ratio times the FEM value at the starting shape.

## Per seed

| Benchmark | Seed | Parameter found | FEM optimum | Parameter vs FEM optimum | FEM at design vs target | ATLAS vs FEM | Steps | Status | Time |
|---|---|---|---|---|---|---|---|---|---|
| Cantilever | 42 | 1.1928 | 1.1930 | 0.01 % | +0.03 % | -0.14 % | 5 | converged | 9.0 min |
| Cantilever | 43 | 1.1930 | 1.1930 | 0.00 % | -0.01 % | -0.12 % | 5 | converged | 9.2 min |
| Cantilever | 44 | 1.1929 | 1.1930 | 0.00 % | +0.01 % | -0.16 % | 5 | converged | 9.0 min |
| Cantilever | 45 | 1.1928 | 1.1930 | 0.01 % | +0.02 % | -0.13 % | 5 | converged | 9.0 min |
| Cantilever | 46 | 1.1931 | 1.1930 | 0.01 % | -0.04 % | -0.11 % | 5 | converged | 9.1 min |
| Plate with a hole | 42 | 0.4809 | 0.4816 | 0.13 % | -0.05 % | -0.06 % | 5 | converged | 13.1 min |
| Plate with a hole | 43 | 0.4822 | 0.4816 | 0.12 % | +0.04 % | -0.01 % | 5 | converged | 13.2 min |
| Plate with a hole | 44 | 0.4819 | 0.4816 | 0.07 % | +0.02 % | -0.02 % | 5 | converged | 13.2 min |
| Plate with a hole | 45 | 0.4807 | 0.4816 | 0.18 % | -0.06 % | -0.03 % | 5 | converged | 13.2 min |
| Plate with a hole | 46 | 0.4804 | 0.4816 | 0.24 % | -0.08 % | -0.03 % | 5 | converged | 13.2 min |
| L-bracket | 42 | 0.2391 | 0.2152 | 11.14 % | -3.73 % | -1.46 % | 5 | converged | 21.2 min |
| L-bracket | 43 | 0.2280 | 0.2152 | 5.95 % | -2.12 % | -1.25 % | 5 | converged | 21.1 min |
| L-bracket | 44 | 0.2220 | 0.2152 | 3.20 % | -1.25 % | -1.09 % | 5 | converged | 21.0 min |
| L-bracket | 45 | 0.2185 | 0.2152 | 1.54 % | -0.52 % | -2.17 % | 5 | converged | 25.3 min |
| L-bracket | 46 | 0.2339 | 0.2152 | 8.70 % | -2.82 % | -0.80 % | 6 | converged | 24.3 min |
| Steel insert | 42 | 0.5271 | 0.5269 | 0.04 % | -0.00 % | -0.00 % | 4 | converged | 9.8 min |
| Steel insert | 43 | 0.5264 | 0.5269 | 0.08 % | +0.01 % | -0.00 % | 4 | converged | 10.1 min |
| Steel insert | 44 | 0.5270 | 0.5269 | 0.02 % | -0.00 % | -0.00 % | 4 | converged | 10.1 min |
| Steel insert | 45 | 0.5272 | 0.5269 | 0.05 % | -0.00 % | -0.00 % | 4 | converged | 10.0 min |
| Steel insert | 46 | 0.5267 | 0.5269 | 0.03 % | +0.00 % | -0.00 % | 4 | converged | 10.1 min |
| Soft inclusion | 42 | 0.5159 | 0.5150 | 0.18 % | +0.05 % | +0.00 % | 4 | converged | 10.2 min |
| Soft inclusion | 43 | 0.5153 | 0.5150 | 0.06 % | +0.01 % | -0.00 % | 4 | converged | 10.1 min |
| Soft inclusion | 44 | 0.5155 | 0.5150 | 0.09 % | +0.02 % | -0.00 % | 4 | converged | 10.2 min |
| Soft inclusion | 45 | 0.5153 | 0.5150 | 0.05 % | +0.01 % | +0.00 % | 4 | converged | 9.9 min |
| Soft inclusion | 46 | 0.5151 | 0.5150 | 0.02 % | +0.00 % | -0.00 % | 4 | converged | 9.8 min |
| Epoxy-filled slot | 42 | 0.3333 | 0.3146 | 5.96 % | -3.40 % | +0.93 % | 6 | converged | 13.3 min |
| Epoxy-filled slot | 43 | 0.3199 | 0.3146 | 1.70 % | -1.01 % | +0.08 % | 6 | converged | 13.4 min |
| Epoxy-filled slot | 44 | 0.3612 | 0.3146 | 14.81 % | -7.72 % | +1.09 % | 6 | converged | 12.3 min |
| Epoxy-filled slot | 45 | 0.3462 | 0.3146 | 10.06 % | -5.53 % | -0.05 % | 6 | converged | 12.1 min |
| Epoxy-filled slot | 46 | 0.3299 | 0.3146 | 4.88 % | -2.70 % | -0.02 % | 6 | converged | 13.2 min |

## Summary across seeds

| Benchmark | Parameter | Goal metric | Seeds | Parameter, mean ± sd | Range | FEM optimum | FEM vs target, mean | FEM vs target, worst | ATLAS vs FEM, worst | Within 10 % |
|---|---|---|---|---|---|---|---|---|---|---|
| Cantilever | beam height | compliance | 5 | 1.1929 ± 0.0001 | 1.1928–1.1931 | 1.1930 | +0.00 % | 0.04 % | 0.16 % | 5/5 |
| Plate with a hole | hole radius | compliance | 5 | 0.4812 ± 0.0008 | 0.4804–0.4822 | 0.4816 | -0.03 % | 0.08 % | 0.06 % | 5/5 |
| L-bracket | fillet radius | peak_vm | 5 | 0.2283 ± 0.0084 | 0.2185–0.2391 | 0.2152 | -2.09 % | 3.73 % | 2.17 % | 5/5 |
| Steel insert | insert radius | compliance | 5 | 0.5269 ± 0.0003 | 0.5264–0.5272 | 0.5269 | +0.00 % | 0.01 % | 0.00 % | 5/5 |
| Soft inclusion | inclusion radius | compliance | 5 | 0.5154 ± 0.0003 | 0.5151–0.5159 | 0.5150 | +0.02 % | 0.05 % | 0.00 % | 5/5 |
| Epoxy-filled slot | slot semi-axis along the load | peak_vm | 5 | 0.3381 ± 0.0160 | 0.3199–0.3612 | 0.3146 | -4.07 % | 7.72 % | 1.09 % | 5/5 |
