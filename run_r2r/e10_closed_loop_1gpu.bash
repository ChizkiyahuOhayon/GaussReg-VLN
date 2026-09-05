#!/bin/bash
set -euo pipefail

TRAIN_ENVS=${1:-1}
EVAL_ENVS=${2:-4}
BASE_PORT=${3:-2334}
MAIN_PORT=$((BASE_PORT + 1))
EVAL_PORT=$((BASE_PORT + 2))

SMOKE_EXP=e10_r2r_frontier_advantage_20
MAIN_EXP=e10_r2r_frontier_advantage_500
SMOKE_CKPT=data/logs/checkpoints/${SMOKE_EXP}/ckpt.iter20.pth
MAIN_CKPT=data/logs/checkpoints/${MAIN_EXP}/ckpt.iter500.pth

echo "[1/6] E10 objective smoke"
python tools/smoke_e10_objective.py 2>&1 | tee e10_objective_smoke.log

echo "[2/6] E10 20-iteration trainer smoke"
E10_EXP_NAME=${SMOKE_EXP} \
    bash run_r2r/e10_grpo_1gpu.bash 20 "${TRAIN_ENVS}" "${BASE_PORT}" \
    2>&1 | tee e10_grpo_smoke20.log

echo "[3/6] Check E10 smoke checkpoint"
python tools/check_e10_checkpoint.py "${SMOKE_CKPT}"

echo "[4/6] E10 500-iteration training"
E10_EXP_NAME=${MAIN_EXP} \
    bash run_r2r/e10_grpo_1gpu.bash 500 "${TRAIN_ENVS}" "${MAIN_PORT}" \
    2>&1 | tee e10_grpo_500.log

echo "[5/6] Check E10 main checkpoint"
python tools/check_e10_checkpoint.py "${MAIN_CKPT}"

echo "[6/6] E10 full val_unseen evaluation"
bash run_r2r/e10_eval_1gpu.bash \
    "${MAIN_CKPT}" -1 "${EVAL_ENVS}" "${EVAL_PORT}" \
    2>&1 | tee e10_eval_full.log

echo "E10 closed loop completed: ${MAIN_CKPT}"
