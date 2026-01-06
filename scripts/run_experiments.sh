#!/usr/bin/env bash
set -euo pipefail

# Paper-aligned batch runner
# Runs 10 seeds for each dataset with MI acquisition.

SEEDS=($(seq 0 9))
TASKS=(reuters neurips webkb)
ACQ=mi
MODE=rl
MODEL=${MODEL:-dkl_gp}
OUT_ROOT=${OUT_ROOT:-results}
STAMP=$(date +%Y%m%d_%H%M%S)

for task in "${TASKS[@]}"; do
  base_dir="${OUT_ROOT}/${MODE}/${MODEL}/${ACQ}/${task}/${STAMP}"
  mkdir -p "${base_dir}"
  for seed in "${SEEDS[@]}"; do
    echo "Running task=${task} seed=${seed} acquisition=${ACQ}"
    python -m meta.main task=${task} acquisition=${ACQ} model=${MODEL} train.mode=${MODE} train.seed=${seed} hydra.run.dir=${base_dir}/seed_${seed}
  done
done
