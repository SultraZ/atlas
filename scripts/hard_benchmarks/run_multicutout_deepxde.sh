#!/bin/bash
# Run from the project root:
#   bash scripts/hard_benchmarks/run_multicutout_deepxde.sh
cd /home/sultra/atlas_main
PY=/home/sultra/miniconda3/envs/Atlas/bin/python
L=results/hard_single_material/multicutout/logs; rm -f $L/DDEDONE
$PY scripts/deepxde_baseline.py --benchmark multicutout --n-domain 8000 --n-boundary 4000 \
    --tag deepxde > $L/deepxde_seed42.log 2>&1
$PY scripts/deepxde_baseline.py --benchmark multicutout --n-domain 8000 --n-boundary 4000 \
    --configured --tag deepxde_configured > $L/deepxde_configured_seed42.log 2>&1
echo done > $L/DDEDONE
