| solver | params | train time | displacement L2 | von Mises L2 | band vM, inclusion side | band vM, matrix side | compliance vs FEM |
|---|---|---|---|---|---|---|---|
| ATLAS energy (single 4x64) | 12,802 | 325 s | 0.20% | 0.70% | 21.64% | 4.08% | -0.071% |
| ATLAS energy (single 4x91, parameter-matched) | 25,573 | 545 s | 0.13% | 0.50% | 18.71% | 3.48% | -0.023% |
| ATLAS-XPINN (inclusion + matrix) | 25,604 | 352 s | 0.04% | 0.32% | 1.38% | 2.24% | -0.000% |

Seeds (mean +- std over the seeds found):

| solver | seeds | displacement L2 | von Mises L2 |
|---|---|---|---|
| ATLAS energy (single 4x91, parameter-matched) | 5 | 0.15 +- 0.02% | 0.63 +- 0.11% |
| ATLAS-XPINN (inclusion + matrix) | 5 | 0.05 +- 0.02% | 0.32 +- 0.09% |
