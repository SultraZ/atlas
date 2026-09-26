#!/bin/bash
# Run from the project root:
#   bash scripts/experiments/run_deepxde_configured.sh
cd /home/sultra/atlas_main
PY=/home/sultra/miniconda3/envs/Atlas/bin/python
L=results/comparison/logs; mkdir -p $L; rm -f $L/CFGDONE
for b in cantilever plate_hole l_bracket; do
  $PY scripts/deepxde_baseline.py --benchmark $b --configured --tag deepxde_configured > $L/${b}_configured.log 2>&1
done
echo done > $L/CFGDONE
