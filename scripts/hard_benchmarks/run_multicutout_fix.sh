#!/bin/bash
# Run from the project root:
#   bash scripts/hard_benchmarks/run_multicutout_fix.sh
cd /home/sultra/atlas_main
PY=/home/sultra/miniconda3/envs/Atlas/bin/python
L=results/hard_single_material/multicutout/logs; rm -f $L/FIXDONE
$PY scripts/hard_benchmarks/run_multicutout.py --solver xpinn --seed 42 --lbfgs 8000 --al-keep-memory --tag xpinn_long_keepmem > $L/xpinn_long_keepmem_seed42.log 2>&1
$PY scripts/hard_benchmarks/run_multicutout.py --solver xpinn --seed 42 --lbfgs 8000 --al-keep-memory --warm-start-lbfgs 2000 --tag xpinn_long_keepmem_coarse > $L/xpinn_long_keepmem_coarse_seed42.log 2>&1
echo done > $L/FIXDONE
