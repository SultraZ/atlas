# Convergence and processing time: DeepXDE PINN (defaults / configured) vs ATLAS-XPINN

Same seed (42), GPU, one job at a time. ATLAS-XPINN: 2000 Adam + 2000 L-BFGS iterations. PINN (DeepXDE): 10000 Adam + 2000 L-BFGS iterations, library defaults (float32) or the configured run (exact essential BCs via an output transform, float64, extra collocation points at the stress concentration, loss weights balanced from the initial losses; never chosen by the FEM error). Errors are scored against the FEM reference during training (monitoring only, never in the loss); probing time is excluded from the clock. Displacement error is the vector L2 (stricter than the |u| metric in validation.py). "Time to X" = first probe after which the error stays below X.

| benchmark | solver | params | train time [s] | s / iteration | final disp L2 | final vM L2 (excl.) | time to 5% disp [s] | time to 1% disp [s] | iters to 5% | iters to 1% |
|---|---|---|---|---|---|---|---|---|---|---|
| Cantilever | PINN (DeepXDE, library defaults) | 12802 | 695 | 0.058 | 88.09% | 89.92% | - | - | - | - |
| Cantilever | PINN (DeepXDE, configured) | 12802 | 3479 | 0.290 | 102.57% | 93.30% | - | - | - | - |
| Cantilever | ATLAS-XPINN | 25604 | 311 | 0.078 | 0.27% | 2.10% | 124 | 143 | 2100 | 2300 |
| Plate with hole | PINN (DeepXDE, library defaults) | 12802 | 666 | 0.055 | 51.42% | 31.66% | - | - | - | - |
| Plate with hole | PINN (DeepXDE, configured) | 12802 | 3009 | 0.251 | 50.50% | 33.60% | - | - | - | - |
| Plate with hole | ATLAS-XPINN | 38406 | 455 | 0.114 | 0.81% | 3.39% | 261 | 455 | 2600 | 4000 |
| L-bracket | PINN (DeepXDE, library defaults) | 12802 | 746 | 0.062 | 115.18% | 81.97% | - | - | - | - |
| L-bracket | PINN (DeepXDE, configured) | 12802 | 3277 | 0.273 | 100.52% | 94.45% | - | - | - | - |
| L-bracket | ATLAS-XPINN | 38478 | 875 | 0.219 | 0.30% | 4.55% | 169 | 495 | 1200 | 2600 |
