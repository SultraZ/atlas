# Convergence and processing time: DeepXDE PINN vs ATLAS energy vs ATLAS-XPINN energy

Same seed (42), GPU, one job at a time. ATLAS and ATLAS-XPINN: 2000 Adam + 2000 L-BFGS iterations; PINN (DeepXDE): library defaults, 10000 Adam + 2000 L-BFGS iterations, float32. Errors are scored against the FEM reference during training (monitoring only, never in the loss); probing time is excluded from the clock. Displacement error is the vector L2 (stricter than the |u| metric in validation.py). "Time to X" = first probe after which the error stays below X.

| benchmark | solver | params | train time [s] | s / iteration | final disp L2 | final vM L2 (excl.) | time to 5% disp [s] | time to 1% disp [s] | iters to 5% | iters to 1% |
|---|---|---|---|---|---|---|---|---|---|---|
| Cantilever | PINN (DeepXDE library) | 12802 | 695 | 0.058 | 88.09% | 89.92% | - | - | - | - |
| Cantilever | ATLAS energy (single network) | 12802 | 257 | 0.064 | 0.14% | 2.05% | 83 | 120 | 1700 | 2300 |
| Cantilever | ATLAS-XPINN energy (decomposed) | 25604 | 336 | 0.084 | 0.64% | 2.97% | 143 | 173 | 2100 | 2400 |
| Plate with hole | PINN (DeepXDE library) | 12802 | 666 | 0.055 | 51.42% | 31.66% | - | - | - | - |
| Plate with hole | ATLAS energy (single network) | 12802 | 270 | 0.067 | 0.20% | 2.58% | 108 | 123 | 2100 | 2300 |
| Plate with hole | ATLAS-XPINN energy (decomposed) | 38406 | 495 | 0.124 | 0.95% | 3.52% | 253 | 468 | 2300 | 3800 |
| L-bracket | PINN (DeepXDE library) | 12802 | 746 | 0.062 | 115.18% | 81.97% | - | - | - | - |
| L-bracket | ATLAS energy (single network) | 12826 | 605 | 0.151 | 0.26% | 4.16% | 57 | 283 | 400 | 2100 |
| L-bracket | ATLAS-XPINN energy (decomposed) | 38478 | 974 | 0.244 | 1.44% | 5.30% | 477 | - | 2100 | - |
