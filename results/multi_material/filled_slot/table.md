| solver | params | train time | displacement L2 | von Mises L2 | band vM, inclusion side | band vM, matrix side | compliance vs FEM |
|---|---|---|---|---|---|---|---|
| ATLAS energy (single 4x64) | 12,802 | 326 s | 0.20% | 0.54% | 27.12% | 3.09% | -0.034% |
| ATLAS energy (single 4x91, parameter-matched) | 25,573 | 556 s | 0.18% | 0.50% | 25.98% | 2.37% | -0.036% |
| ATLAS-XPINN (filler + plate) | 25,604 | 363 s | 0.07% | 0.43% | 0.32% | 3.03% | -0.002% |

Seeds (mean +- std over the seeds found):

| solver | seeds | displacement L2 | von Mises L2 |
|---|---|---|---|
| ATLAS energy (single 4x91, parameter-matched) | 5 | 0.22 +- 0.04% | 0.53 +- 0.06% |
| ATLAS-XPINN (filler + plate) | 5 | 0.06 +- 0.01% | 0.47 +- 0.10% |
