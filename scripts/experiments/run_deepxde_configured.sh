#!/bin/bash
# DeepXDE "configured" runs on the three benchmarks (settings fixed in advance from standard
# DeepXDE practice, never from the FEM error). One GPU job at a time.
cd /home/sultra/atlas_main
PY=/home/sultra/miniconda3/envs/Atlas/bin/python
L=results/comparison/logs; mkdir -p $L; rm -f $L/CFGDONE
for b in cantilever plate_hole l_bracket; do
  $PY scripts/deepxde_baseline.py --benchmark $b --configured --tag deepxde_configured > $L/${b}_configured.log 2>&1
done
echo done > $L/CFGDONE
