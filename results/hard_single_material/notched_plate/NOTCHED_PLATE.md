# Notched plate (h/r = 10): ATLAS energy vs ATLAS-XPINN vs DeepXDE PINN

Plate 1 x 2 m, U-notch depth 0.25 m / root radius 0.025 m, clamped at y = 0, uniform tension at y = 2. Seed 42, one GPU job at a time. Errors vs the FEniCS reference (never used in training). Displacement error is the vector L2; von Mises excludes the notch apex and clamp corners; 'notch zone' is the von Mises L2 on a dense grid around the root (excluding 0.005 m around the apex); 'root peak' is the predicted / FEM peak von Mises there (1 = exact). The FEM peak itself is converged to about 4 % (change between the last two mesh levels).

| solver | params | train time [s] | displacement L2 | von Mises L2 (excl.) | notch-zone von Mises L2 | root peak vs FEM | time to 5 % disp [s] |
|---|---|---|---|---|---|---|---|
| PINN (DeepXDE library) | 12802 | 1160 | 209.58% | 57.81% | 92.27% | 0.039 | - |
| ATLAS energy (single network) | 12802 | 339 | 0.08% | 0.73% | 1.59% | 1.000 | 182 |
| ATLAS-XPINN, patch layout (2000+2000) | 25604 | 598 | 49.87% | 29.63% | 66.50% | 0.115 | - |
| ATLAS-XPINN, patch layout (2000+8000) | 25604 | 1609 | 2.36% | 1.43% | 2.66% | 0.997 | 916 |
| ATLAS-XPINN, split layout, penalty | 25604 | 483 | 1.71% | 3.53% | 14.81% | 0.536 | 301 |
| ATLAS-XPINN, split layout + AL | 25604 | 703 | 0.89% | 3.49% | 11.07% | 0.757 | 517 |
| ATLAS-XPINN, split + AL + warm start | 25604 | 407 | 14.63% | 15.18% | 46.84% | 0.333 | - |
