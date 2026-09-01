#!/bin/bash
set -euo pipefail

TRAIN_ENVS=${1:-1}
EVAL_ENVS=${2:-4}
BASE_PORT=${3:-2334}
SMOKE_PORT=$((BASE_PORT + 1))
EVAL_PORT=$((BASE_PORT + 2))

SMOKE_EXP=e9_r2r_setwise_grpo_20
MAIN_EXP=e9_r2r_setwise_grpo_500
MAIN_CKPT=data/logs/checkpoints/${MAIN_EXP}/ckpt.iter500.pth

echo "[1/4] E9 objective smoke"
python tools/smoke_e9_objective.py 2>&1 | tee e9_objective_smoke.log

echo "[2/4] E9 20-iteration trainer smoke"
E9_EXP_NAME=${SMOKE_EXP} \
    bash run_r2r/e9_grpo_1gpu.bash 20 "${TRAIN_ENVS}" "${BASE_PORT}" \
    2>&1 | tee e9_grpo_smoke20.log

echo "[3/4] E9 500-iteration training"
E9_EXP_NAME=${MAIN_EXP} \
    bash run_r2r/e9_grpo_1gpu.bash 500 "${TRAIN_ENVS}" "${SMOKE_PORT}" \
    2>&1 | tee e9_grpo_500.log

if [ ! -f "${MAIN_CKPT}" ]; then
    echo "Missing E9 checkpoint after training: ${MAIN_CKPT}" >&2
    exit 1
fi

echo "[4/4] E9 full val_unseen evaluation"
bash run_r2r/e9_eval_1gpu.bash \
    "${MAIN_CKPT}" -1 "${EVAL_ENVS}" "${EVAL_PORT}" \
    2>&1 | tee e9_eval_full.log

echo "E9 closed loop completed: ${MAIN_CKPT}"
