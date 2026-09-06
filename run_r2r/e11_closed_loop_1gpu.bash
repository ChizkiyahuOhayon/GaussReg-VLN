#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/habitat_env.bash"

TRAIN_ENVS=${1:-1}
EVAL_ENVS=${2:-4}
BASE_PORT=${3:-2334}
MAIN_PORT=$((BASE_PORT + 1))
EVAL_PORT=$((BASE_PORT + 2))

SMOKE_EXP=e11_r2r_geo_token_20
MAIN_EXP=e11_r2r_geo_token_500
SMOKE_CKPT=data/logs/checkpoints/${SMOKE_EXP}/ckpt.iter20.pth
MAIN_CKPT=data/logs/checkpoints/${MAIN_EXP}/ckpt.iter500.pth

echo "[1/6] E11 model smoke"
python tools/smoke_e11_model.py 2>&1 | tee e11_model_smoke.log

echo "[2/6] E11 20-iteration trainer smoke"
E11_EXP_NAME=${SMOKE_EXP} \
    bash run_r2r/e11_grpo_1gpu.bash 20 "${TRAIN_ENVS}" "${BASE_PORT}" \
    2>&1 | tee e11_grpo_smoke20.log

echo "[3/6] Check E11 smoke checkpoint"
python tools/check_e11_checkpoint.py "${SMOKE_CKPT}"

echo "[4/6] E11 500-iteration training"
E11_EXP_NAME=${MAIN_EXP} \
    bash run_r2r/e11_grpo_1gpu.bash 500 "${TRAIN_ENVS}" "${MAIN_PORT}" \
    2>&1 | tee e11_grpo_500.log

echo "[5/6] Check E11 main checkpoint"
python tools/check_e11_checkpoint.py "${MAIN_CKPT}"

echo "[6/6] E11 full val_unseen evaluation"
bash run_r2r/e11_eval_1gpu.bash \
    "${MAIN_CKPT}" -1 "${EVAL_ENVS}" "${EVAL_PORT}" \
    2>&1 | tee e11_eval_full.log

echo "E11 closed loop completed: ${MAIN_CKPT}"
