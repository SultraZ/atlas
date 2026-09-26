#!/bin/bash
# Run from the project root:
#   bash scripts/hard_benchmarks/run_multicutout_seed42.sh
cd /home/sultra/atlas_main
PY=/home/sultra/miniconda3/envs/Atlas/bin/python
L=results/hard_single_material/multicutout/logs; rm -f $L/SEED42DONE
$PY scripts/hard_benchmarks/run_multicutout.py --solver atlas --seed 42 > $L/atlas_seed42.log 2>&1
$PY scripts/hard_benchmarks/run_multicutout.py --solver xpinn --seed 42 > $L/xpinn_seed42.log 2>&1
echo done > $L/SEED42DONE
