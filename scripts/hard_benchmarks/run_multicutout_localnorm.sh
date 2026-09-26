#!/bin/bash
# Run from the project root:
#   bash scripts/hard_benchmarks/run_multicutout_localnorm.sh
cd /home/sultra/atlas_main
L=results/hard_single_material/multicutout/logs
while [ ! -f $L/DDEDONE ]; do sleep 30; done
/home/sultra/miniconda3/envs/Atlas/bin/python scripts/hard_benchmarks/run_multicutout.py --solver xpinn --seed 42 \
  --lbfgs 8000 --al-keep-memory --warm-start-lbfgs 2000 --local-norm \
  --tag xpinn_long_coarse_localnorm > $L/xpinn_long_coarse_localnorm_seed42.log 2>&1
echo done > $L/LOCALNORMDONE
