#!/usr/bin/env bash
# The benchmark protocol, one GPU job at a time:
#   FEM reference (with a free-memory guard) -> seed 42: atlas, xpinn
#   -> atlas_wide only if xpinn beats atlas on displacement AND von Mises
#   -> seeds 43-46 for xpinn and the better single network only if xpinn beats both.
#
#   bash scripts/multi_material/run_protocol.sh <bench> <vm_key> [--skip-fem]
#   e.g. bash scripts/multi_material/run_protocol.sh filled_slot vm_l2
# expects fem/fenics_scripts/<bench>_fenics.py and scripts/multi_material/run_<bench>.py; results and logs go
# to results/multi_material/<bench>/ (or failed_candidates/<bench>/ if that folder already exists).
set -u
cd "$(dirname "$0")/../.."
B=$1; VM=$2; SKIP_FEM=${3:-}
PY=/home/sultra/miniconda3/envs/Atlas/bin/python
R=results/multi_material/$B
[ -d results/multi_material/failed_candidates/$B ] && R=results/multi_material/failed_candidates/$B
L=$R/logs; mkdir -p "$L"

if [ "$SKIP_FEM" != "--skip-fem" ]; then
  env "PATH=/home/sultra/miniconda3/envs/Atlas/bin:$PATH" python fem/fenics_scripts/${B}_fenics.py > $L/fem.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    a=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
    if [ "$a" -lt 300 ]; then kill $pid; echo "KILLED_LOW_MEMORY ($a MB)"; fi
    sleep 2
  done
  wait $pid || { echo FEM_FAILED; tail -n 3 $L/fem.log; exit 1; }
  grep -E "change|stress|Kt|saved" $L/fem.log
fi

run() {   # solver seed
  $PY scripts/multi_material/run_${B}.py --solver $1 --seed $2 > $L/${1}_seed$2.log 2>&1
  grep DONE $L/${1}_seed$2.log || { echo "FAILED $1 seed $2"; tail -n 3 $L/${1}_seed$2.log; exit 1; }
}
beats() {   # does xpinn beat solver $1 at seed 42 on both metrics?
  $PY -c "
import json,sys
x=json.load(open('$R/xpinn/seed42.json'))['final']; a=json.load(open('$R/$1/seed42.json'))['final']
sys.exit(0 if x['disp_vec_l2']<a['disp_vec_l2'] and x['$VM']<a['$VM'] else 1)"
}

run atlas 42
run xpinn 42
beats atlas || { echo "ATLAS_BEATS_XPINN -> $B stops here"; exit 0; }
echo "XPINN_BEATS_ATLAS -> running atlas_wide"
run atlas_wide 42
beats atlas_wide || { echo "CONTROL_BEATS_XPINN -> $B stops here"; exit 0; }
better=$($PY -c "
import json
a=json.load(open('$R/atlas/seed42.json'))['final']; w=json.load(open('$R/atlas_wide/seed42.json'))['final']
print('atlas_wide' if w['disp_vec_l2']+w['$VM'] < a['disp_vec_l2']+a['$VM'] else 'atlas')")
echo "PASSED_SEED42 -> 5 seeds for xpinn and $better"
for seed in 43 44 45 46; do run xpinn $seed; run $better $seed; done
echo ALL_SEEDS_FINISHED
