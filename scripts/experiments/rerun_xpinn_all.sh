#!/bin/bash
# Run from the project root:
#   bash scripts/experiments/rerun_xpinn_all.sh
cd /home/sultra/atlas_main
PY=/home/sultra/miniconda3/envs/Atlas/bin/python
L=results/atlas_xpinn_energy/logs
step() { echo "$(date +%H:%M) START $1" >> $L/PROGRESS; shift; "$@"; echo "$(date +%H:%M) END rc=$?" >> $L/PROGRESS; }
for b in cantilever plate_hole l_bracket; do
  step "forward $b" $PY scripts/compare_solvers.py --benchmark $b --variant xpinn_energy > $L/forward_$b.log 2>&1
done
for b in cantilever plate_hole l_bracket; do
  step "convergence $b" $PY scripts/run_convergence_timing.py --benchmark $b --solver xpinn > $L/convergence_$b.log 2>&1
done
for b in cantilever plate_hole l_bracket; do
  step "5-seed $b" $PY scripts/run_repeatability_study.py --benchmark $b --formulation energy_xpinn --epochs 2000 > $L/seeds_$b.log 2>&1
done
for b in cantilever plate_hole l_bracket; do
  step "inverse $b" $PY -c "from atlas.atlas_main import run_inverse_design_all; run_inverse_design_all('$b', 'cuda', decomposed=True)" > $L/inverse_$b.log 2>&1
done
for b in cantilever plate_hole l_bracket; do
  step "fem-check $b" env "PATH=/home/sultra/miniconda3/envs/Atlas/bin:$PATH" $PY scripts/verify_inverse_design.py --benchmark $b --solver atlas_xpinn_energy > $L/femcheck_$b.log 2>&1
done
step "notched split" $PY scripts/hard_benchmarks/run_notched_plate.py --solver xpinn --layout split --tag xpinn_split > $L/notched_split.log 2>&1
step "reports" bash -c "$PY scripts/make_comparison_report.py && $PY scripts/make_comparison_report.py --only atlas xpinn_energy --out-sub atlas_vs_atlas_xpinn && $PY scripts/make_comparison_report.py --only deepxde xpinn_energy --out-sub standard_pinn_vs_atlas_xpinn && $PY scripts/make_convergence_timing_report.py" > $L/reports.log 2>&1
echo ALLDONE >> $L/PROGRESS
