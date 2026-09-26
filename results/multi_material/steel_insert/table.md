| solver | params | train time | displacement L2 | von Mises L2 | band vM, inclusion side | band vM, matrix side | compliance vs FEM |
|---|---|---|---|---|---|---|---|
| ATLAS energy (single 4x64) | 12,802 | 330 s | 0.06% | 1.25% | 10.93% | 9.45% | -0.020% |
| ATLAS energy (single 4x91, parameter-matched) | 25,573 | 564 s | 0.05% | 1.04% | 8.77% | 9.06% | -0.005% |
| ATLAS-XPINN (insert + plate) | 25,604 | 378 s | 0.02% | 0.12% | 0.36% | 0.56% | -0.002% |

Seeds (mean +- std over the seeds found):

| solver | seeds | displacement L2 | von Mises L2 |
|---|---|---|---|
| ATLAS energy (single 4x91, parameter-matched) | 5 | 0.06 +- 0.01% | 1.22 +- 0.22% |
| ATLAS-XPINN (insert + plate) | 5 | 0.02 +- 0.00% | 0.12 +- 0.00% |
