# ATLAS forward-solver phase summary

Metrics from the unchanged validation path. Displacement gate: <5% (relative L2). Stress gate: <10% von Mises L2 on the excluded region (provisional). Global von Mises is context only.

| snapshot | benchmark | epochs | warm-EB | L-BFGS | disp L2% | disp | vm excl% | vm excl | vm global% |
|---|---|---|:---:|:---:|---|:---:|---|:---:|---|
| phase0/adam | cantilever | 2000 | yes | - | 8.79% | FAIL | 131.69% | FAIL | 131.25% |
| phase0/adam | l_bracket | 2000 | - | - | 99.38% | FAIL | 95.85% | FAIL | 95.91% |
| phase0/adam | plate_hole | 2000 | - | - | 34.66% | FAIL | 28.99% | FAIL | 29.00% |
| phase0/lbfgs | cantilever | 2000 | yes | yes(15) | 6.31% | FAIL | 116.42% | FAIL | 116.12% |
| phase0/lbfgs | l_bracket | 2000 | - | yes(15) | 99.14% | FAIL | 95.44% | FAIL | 95.51% |
| phase0/lbfgs | plate_hole | 2000 | - | yes(15) | 36.89% | FAIL | 30.47% | FAIL | 30.48% |
| phase0_full/lbfgs | cantilever | 11000 | yes | yes(15) | 6.36% | FAIL | 116.26% | FAIL | 115.99% |
| phase0_full/lbfgs | l_bracket | 5000 | - | yes(15) | 99.02% | FAIL | 95.29% | FAIL | 95.36% |
| phase0_full/lbfgs | plate_hole | 5000 | - | yes(15) | 29.12% | FAIL | 26.91% | FAIL | 26.91% |
| phase1/lbfgs | cantilever | 2000 | yes | yes(15) | 6.31% | FAIL | 116.42% | FAIL | 116.12% |
| phase1/lbfgs | l_bracket | 2000 | - | yes(15) | 99.28% | FAIL | 96.13% | FAIL | 96.19% |
| phase1/lbfgs | plate_hole | 2000 | - | yes(15) | 31.99% | FAIL | 26.52% | FAIL | 26.52% |
| phase1_full/lbfgs | l_bracket | 5000 | - | yes(15) | 97.04% | FAIL | 84.35% | FAIL | 84.60% |
| phase1_full/lbfgs | plate_hole | 5000 | - | yes(15) | 27.32% | FAIL | 25.51% | FAIL | 25.51% |
| phase1b/lbfgs | l_bracket | 2000 | - | yes(15) | 96.87% | FAIL | 84.64% | FAIL | 84.88% |
| phase1c/lbfgs | l_bracket | 2000 | - | yes(15) | 68.76% | FAIL | 71.37% | FAIL | 71.88% |
| phase1d/lbfgs | l_bracket | 2000 | - | yes(15) | 67.60% | FAIL | 76.26% | FAIL | 76.66% |
| phase1d_full/lbfgs | l_bracket | 5000 | - | yes(15) | 70.93% | FAIL | 76.77% | FAIL | 77.15% |
| phase1e/adam | l_bracket | 2000 | - | - | 88.91% | FAIL | 78.07% | FAIL | 78.42% |
| phase2/lbfgs | l_bracket | 2000 | - | yes(15) | 71.67% | FAIL | 79.34% | FAIL | 79.68% |
| phase2/lbfgs | plate_hole | 2000 | - | yes(15) | 35.73% | FAIL | 30.39% | FAIL | 30.39% |
| phase2b/adam | l_bracket | 2000 | - | - | 90.35% | FAIL | 80.20% | FAIL | 80.53% |
| phase3/kirsch | plate_hole | 2000 | - | yes(15) | 33.82% | FAIL | 27.77% | FAIL | 27.77% |
