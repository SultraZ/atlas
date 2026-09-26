#!/bin/bash
# Run from the project root:
#   bash scripts/hard_benchmarks/run_multicutout_long.sh
cd /home/sultra/atlas_main
PY=/home/sultra/miniconda3/envs/Atlas/bin/python
L=results/hard_single_material/multicutout/logs; rm -f $L/LONGDONE
$PY scripts/hard_benchmarks/run_multicutout.py --solver atlas --seed 42 --lbfgs 8000 --tag atlas_long > $L/atlas_long_seed42.log 2>&1
$PY scripts/hard_benchmarks/run_multicutout.py --solver xpinn --seed 42 --lbfgs 8000 --tag xpinn_long > $L/xpinn_long_seed42.log 2>&1
echo done > $L/LONGDONE
