#!/bin/bash
set -euo pipefail

# One training GPU and one evaluation GPU, used sequentially. Each process sees 0.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source run_r2r/habitat_env.bash
TRAIN_GPU=${1:-1}
EVAL_GPU=${2:-2}
TRAIN_ENVS=${3:-1}
EVAL_ENVS=${4:-4}
PORT=${5:-2434}
RUN_ID=${E12_RUN_ID:-e12_r2r_successor}
if [[ ! "${RUN_ID}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo 'E12_RUN_ID must contain only letters, digits, underscores or hyphens.' >&2
    exit 2
fi
BASE=data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth
PRETRAINED=pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt
LOG_DIR=data/logs/${RUN_ID}
SMOKE_EXP=${RUN_ID}_20
MAIN_EXP=${RUN_ID}_500
EVAL_EXP=${RUN_ID}_eval
SMOKE_CKPT=data/logs/checkpoints/${SMOKE_EXP}/ckpt.iter20.pth
MAIN_CKPT=data/logs/checkpoints/${MAIN_EXP}/ckpt.iter500.pth
RESULTS=data/logs/checkpoints/${EVAL_EXP}/eval_results
for directory in "${LOG_DIR}" "data/logs/checkpoints/${SMOKE_EXP}" \
                 "data/logs/checkpoints/${MAIN_EXP}" "data/logs/checkpoints/${EVAL_EXP}"; do
    if [ -e "${directory}" ]; then
        echo "Refusing to overwrite existing evidence: ${directory}" >&2
        echo 'For an infrastructure retry, use a new E12_RUN_ID; training still starts from E0.' >&2
        exit 1
    fi
done
mkdir -p "${LOG_DIR}"
phase=preflight
trap 'code=$?; printf "phase=%s exit=%s\n" "${phase}" "${code}" > "${LOG_DIR}/exit_status.txt"' EXIT
export GLOG_minloglevel=2 MAGNUM_LOG=quiet
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

COMMON=(
    --exp-config run_r2r/iter_train.yaml
    SIMULATOR_GPU_IDS '[0]' TORCH_GPU_IDS '[0]' GPU_NUMBERS 1
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
    MODEL.gauss_feat_size 0 MODEL.gauss_residual_scale 0.0
    MODEL.candidate_scorer_hidden_size 0 MODEL.gaussian_bev_hidden_size 0
    MODEL.anchor_repair_hidden_size 0 MODEL.hindsight_stop_hidden_size 0
    MODEL.terminal_commit_hidden_size 0 MODEL.geo_token_hidden_size 0
    MODEL.successor_hidden_size 256 MODEL.pretrained_path "${PRETRAINED}"
)

echo '[1/7] Record assets, dataset identities and code revision'
CUDA_VISIBLE_DEVICES=${TRAIN_GPU} python tools/e12_protocol.py preflight \
    --baseline "${BASE}" --pretrained "${PRETRAINED}" \
    --output "${LOG_DIR}/manifest.json" 2>&1 | tee "${LOG_DIR}/preflight.log"

phase=model_smoke
echo '[2/7] Production model, strict E0 load and gradient smoke'
CUDA_VISIBLE_DEVICES=${TRAIN_GPU} python tools/smoke_e12_model.py \
    --baseline "${BASE}" --pretrained "${PRETRAINED}" \
    2>&1 | tee "${LOG_DIR}/model_smoke.log"

train_from_e0() {
    local iterations=$1 experiment=$2 master_port=$3
    CUDA_VISIBLE_DEVICES=${TRAIN_GPU} python -m torch.distributed.launch \
        --nproc_per_node=1 --master_port "${master_port}" run.py \
        --exp_name "${experiment}" --run-type grpo "${COMMON[@]}" \
        TRAINER_NAME GRPO-R1 NUM_ENVIRONMENTS "${TRAIN_ENVS}" \
        ONLY_LAST_SAVEALL True GRPO.iters "${iterations}" GRPO.log_every 10 \
        GRPO.lr 0.0001 GRPO.warmup_iters 0 GRPO.min_lr_ratio 0.25 \
        GRPO.load_from_ckpt True GRPO.ckpt_to_load "${BASE}" GRPO.is_requeue False \
        GRPO.successor_only True GRPO.back_algo control \
        GRPO.gauss_only False GRPO.candidate_scorer_only False \
        GRPO.gaussian_bev_only False GRPO.anchor_repair_only False \
        GRPO.hindsight_stop_only False GRPO.terminal_commit_only False \
        GRPO.success_set_commit False GRPO.setwise_group_policy False \
        GRPO.frontier_advantage False GRPO.geo_token_only False \
        GRPO.sample_num 8 GRPO.update_epochs 1 GRPO.grpo_beta 0.0 \
        GRPO.enable_amp False GRPO.enable_all_dropouts False \
        GRPO.dropout_in_sampling False GRPO.waypoint_aug False \
        GRPO.max_grad_norm 2.0 TASK_CONFIG.DATASET.SPLIT train \
        TASK_CONFIG.DATASET.SUFFIX _10
}

phase=train20
echo '[3/7] Independent 20-iteration training smoke'
train_from_e0 20 "${SMOKE_EXP}" "${PORT}" 2>&1 | tee "${LOG_DIR}/train20.log"
phase=check20
echo '[4/7] Require real updates, changed decoder and bitwise unchanged E0'
python tools/e12_protocol.py checkpoint "${SMOKE_CKPT}" --baseline "${BASE}" \
    --iteration 20 2>&1 | tee "${LOG_DIR}/check20.log"

phase=train500
echo '[5/7] Independent 500-iteration training from strict E0'
train_from_e0 500 "${MAIN_EXP}" "$((PORT + 1))" 2>&1 | tee "${LOG_DIR}/train500.log"
phase=check500
echo '[6/7] Verify the fixed iteration-500 checkpoint'
python tools/e12_protocol.py checkpoint "${MAIN_CKPT}" --baseline "${BASE}" \
    --iteration 500 2>&1 | tee "${LOG_DIR}/check500.log"

phase=full_eval
echo '[7/7] Full R2R val_unseen inference and simultaneous six-metric gate'
CUDA_VISIBLE_DEVICES=${EVAL_GPU} python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port "$((PORT + 2))" run.py \
    --exp_name "${EVAL_EXP}" --run-type eval "${COMMON[@]}" \
    TRAINER_NAME SS-ETP-R1 NUM_ENVIRONMENTS "${EVAL_ENVS}" \
    IL.back_algo control EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT -1 \
    EVAL.fast_eval False EVAL.EPISODE_ID None EVAL.SAVE_RESULTS True \
    EVAL.CKPT_PATH_DIR "${MAIN_CKPT}" TASK_CONFIG.DATASET.SUFFIX '' \
    2>&1 | tee "${LOG_DIR}/eval_full.log"
python tools/e12_protocol.py results "${RESULTS}" --checkpoint "${MAIN_CKPT}" \
    --output "${LOG_DIR}/full_result.json" 2>&1 | tee "${LOG_DIR}/gate.log"
phase=completed
echo "E12 execution completed. Performance decision: ${LOG_DIR}/full_result.json"
