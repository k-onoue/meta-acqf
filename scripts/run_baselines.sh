#!/usr/bin/env bash
set -euo pipefail

# Baseline runner: greedy (no RL) for RBF GP and DKL GP.
SEEDS=($(seq 0 9))
TASKS=(reuters neurips webkb)
MODE=greedy
ACQ=mi
REFIT=true
OUT_ROOT=${OUT_ROOT:-results}
STAMP=$(date +%Y%m%d_%H%M%S)

run() {
  local model="$1"
  for task in "${TASKS[@]}"; do
    local base_dir="${OUT_ROOT}/${MODE}/${model}/${ACQ}/${task}/${STAMP}"
    mkdir -p "${base_dir}"
    for seed in "${SEEDS[@]}"; do
      echo "Running baseline model=${model} task=${task} seed=${seed}"
      python -m meta.main \
        model=${model} \
        acquisition=${ACQ} \
        train.mode=${MODE} \
        acquisition.use_softmax=false \
        train.seed=${seed} \
        task=${task} \
        model.pretrain.enabled=false \
        train.refit_mll=${REFIT} \
        hydra.run.dir=${base_dir}/seed_${seed}
    done
  done
}

run rbf_gp
run dkl_gp
