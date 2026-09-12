#!/bin/bash
set -euo pipefail

# Training and evaluation GPUs are used sequentially; each process sees GPU 0.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source run_r2r/habitat_env.bash
TRAIN_GPU=${1:-0}
EVAL_GPU=${2:-0}
TRAIN_ENVS=${3:-1}
EVAL_ENVS=${4:-4}
PORT=${5:-3138}
RUN_ID=${E18_RUN_ID:-e18_r2r_route_attention}
if [[ ! "${RUN_ID}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo 'E18_RUN_ID must contain only letters, digits, underscores or hyphens.' >&2
    exit 2
fi
BASE=data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth
PRETRAINED=pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt
LOG_DIR=data/logs/${RUN_ID}
for directory in "${LOG_DIR}" \
    "data/logs/checkpoints/${RUN_ID}_control_20" "data/logs/checkpoints/${RUN_ID}_control_500" \
    "data/logs/checkpoints/${RUN_ID}_control_eval" "data/logs/checkpoints/${RUN_ID}_route_20" \
    "data/logs/checkpoints/${RUN_ID}_route_500" "data/logs/checkpoints/${RUN_ID}_route_eval"; do
    if [ -e "${directory}" ]; then
        echo "Refusing to overwrite existing evidence: ${directory}" >&2
        exit 1
    fi
done
mkdir -p "${LOG_DIR}"
phase=preflight
trap 'code=$?; printf "phase=%s exit=%s\n" "${phase}" "${code}" > "${LOG_DIR}/exit_status.txt"' EXIT
export GLOG_minloglevel=2 MAGNUM_LOG=quiet
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
run() {
    printf '%s ' "${phase}" >> "${LOG_DIR}/commands.txt"
    printf '%q ' "$@" >> "${LOG_DIR}/commands.txt"
    printf '\n' >> "${LOG_DIR}/commands.txt"
    local started=${SECONDS}
    E18_PHASE=${phase} "$@"
    printf '%s\t%s\n' "${phase}" "$((SECONDS - started))" >> "${LOG_DIR}/phase_seconds.tsv"
}
printf 'train_gpu=%s eval_gpu=%s train_envs=%s eval_envs=%s port=%s\n' \
    "${TRAIN_GPU}" "${EVAL_GPU}" "${TRAIN_ENVS}" "${EVAL_ENVS}" "${PORT}" > "${LOG_DIR}/hardware.txt"

COMMON=(
    --exp-config run_r2r/iter_train.yaml
    SIMULATOR_GPU_IDS '[0]' TORCH_GPU_IDS '[0]' GPU_NUMBERS 1
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
    MODEL.gauss_feat_size 0 MODEL.gauss_residual_scale 0.0
    MODEL.candidate_scorer_hidden_size 0 MODEL.gaussian_bev_hidden_size 0
    MODEL.anchor_repair_hidden_size 0 MODEL.hindsight_stop_hidden_size 0
    MODEL.terminal_commit_hidden_size 0 MODEL.geo_token_hidden_size 0
    MODEL.successor_hidden_size 0 MODEL.instruction_coverage_hidden_size 0
    MODEL.landmark_transport_size 0 MODEL.factorized_landmark_size 0
    MODEL.factorized_landmark_budgeted False
    MODEL.factorized_landmark_monotonic False
    MODEL.route_attention True
    MODEL.pretrained_path "${PRETRAINED}"
)

echo '[1/7] Record assets, dataset identities and code revision'
CUDA_VISIBLE_DEVICES=${TRAIN_GPU} run python tools/e18_protocol.py preflight \
    --baseline "${BASE}" --pretrained "${PRETRAINED}" \
    --output "${LOG_DIR}/manifest.json" 2>&1 | tee "${LOG_DIR}/preflight.log"

phase=model_smoke
echo '[2/7] Production model, strict E0 load and cloned-layer / route-mask smoke'
CUDA_VISIBLE_DEVICES=${TRAIN_GPU} run python tools/smoke_e18_model.py \
    --baseline "${BASE}" --pretrained "${PRETRAINED}" --device cuda \
    --output "${LOG_DIR}/model_smoke.json" 2>&1 | tee "${LOG_DIR}/model_smoke.log"

train_from_e0() {
    local iterations=$1 experiment=$2 master_port=$3
    CUDA_VISIBLE_DEVICES=${TRAIN_GPU} run python -m torch.distributed.launch \
        --nproc_per_node=1 --master_port "${master_port}" run.py \
        --exp_name "${experiment}" --run-type grpo "${COMMON[@]}" MODEL.route_attention_full_graph "${FULL_GRAPH}" \
        TRAINER_NAME GRPO-R1 NUM_ENVIRONMENTS "${TRAIN_ENVS}" \
        ONLY_LAST_SAVEALL True GRPO.iters "${iterations}" GRPO.log_every 10 \
        GRPO.lr 0.00002 GRPO.warmup_iters 0 GRPO.min_lr_ratio 0.25 \
        GRPO.load_from_ckpt True GRPO.ckpt_to_load "${BASE}" \
        GRPO.is_requeue False \
        GRPO.monotonic_factorized_landmark_only False \
        GRPO.route_attention_only True \
        GRPO.budgeted_factorized_landmark_only False \
        GRPO.factorized_landmark_only False \
        GRPO.landmark_transport_only False GRPO.gauss_only False \
        GRPO.candidate_scorer_only False GRPO.gaussian_bev_only False \
        GRPO.anchor_repair_only False GRPO.hindsight_stop_only False \
        GRPO.terminal_commit_only False GRPO.success_set_commit False \
        GRPO.setwise_group_policy False GRPO.frontier_advantage False \
        GRPO.geo_token_only False GRPO.successor_only False \
        GRPO.instruction_coverage_only False GRPO.back_algo control \
        GRPO.batch_size 1 GRPO.sample_num 8 GRPO.update_epochs 1 \
        GRPO.grpo_beta 0.04 GRPO.enable_amp False \
        GRPO.enable_all_dropouts False GRPO.dropout_in_sampling False \
        GRPO.waypoint_aug False GRPO.max_grad_norm 2.0 \
        TASK_CONFIG.DATASET.SPLIT train TASK_CONFIG.DATASET.SUFFIX _10
}

for ARM in control route; do
    FULL_GRAPH=False
    if [ "${ARM}" = control ]; then FULL_GRAPH=True; fi
    SMOKE_EXP=${RUN_ID}_${ARM}_20
    MAIN_EXP=${RUN_ID}_${ARM}_500
    EVAL_EXP=${RUN_ID}_${ARM}_eval
    SMOKE_CKPT=data/logs/checkpoints/${SMOKE_EXP}/ckpt.iter20.pth
    MAIN_CKPT=data/logs/checkpoints/${MAIN_EXP}/ckpt.iter500.pth
    RESULTS=data/logs/checkpoints/${EVAL_EXP}/eval_results
phase=${ARM}_train20
echo '[3/7] Independent 20-iteration training smoke'
train_from_e0 20 "${SMOKE_EXP}" "${PORT}" 2>&1 | tee "${LOG_DIR}/${ARM}_train20.log"
phase=${ARM}_check20
echo '[4/7] Require changed E18 branch weights and bitwise unchanged E0'
run python tools/e18_protocol.py checkpoint "${SMOKE_CKPT}" --baseline "${BASE}" \
    --iteration 20 --arm "${ARM}" --output "${LOG_DIR}/${ARM}_check20.json" 2>&1 | tee "${LOG_DIR}/${ARM}_check20.log"

phase=${ARM}_train500
echo '[5/7] Independent 500-iteration training from strict E0'
train_from_e0 500 "${MAIN_EXP}" "$((PORT + 1))" 2>&1 | tee "${LOG_DIR}/${ARM}_train500.log"
phase=${ARM}_check500
echo '[6/7] Verify the fixed iteration-500 checkpoint'
run python tools/e18_protocol.py checkpoint "${MAIN_CKPT}" --baseline "${BASE}" \
    --iteration 500 --arm "${ARM}" --output "${LOG_DIR}/${ARM}_check500.json" 2>&1 | tee "${LOG_DIR}/${ARM}_check500.log"

phase=${ARM}_full_eval
echo '[7/7] Full R2R val_unseen inference and six-metric E15 and SPL >56.85 gate'
CUDA_VISIBLE_DEVICES=${EVAL_GPU} run python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port "$((PORT + 2))" run.py \
    --exp_name "${EVAL_EXP}" --run-type eval "${COMMON[@]}" MODEL.route_attention_full_graph "${FULL_GRAPH}" \
    TRAINER_NAME SS-ETP-R1 NUM_ENVIRONMENTS "${EVAL_ENVS}" \
    IL.back_algo control EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT -1 \
    EVAL.fast_eval False EVAL.EPISODE_ID None EVAL.SAVE_RESULTS True \
    EVAL.CKPT_PATH_DIR "${MAIN_CKPT}" TASK_CONFIG.DATASET.SUFFIX '' \
    2>&1 | tee "${LOG_DIR}/${ARM}_eval_full.log"
phase=${ARM}_gate
run python tools/e18_protocol.py results "${RESULTS}" \
    --checkpoint "${MAIN_CKPT}" --arm "${ARM}" \
    --check-report "${LOG_DIR}/${ARM}_check500.json" --output "${LOG_DIR}/${ARM}_full_result.json" \
    2>&1 | tee "${LOG_DIR}/${ARM}_gate.log"
done
phase=compare
run python tools/e18_protocol.py compare --control "${LOG_DIR}/control_full_result.json" \
    --route "${LOG_DIR}/route_full_result.json" --output "${LOG_DIR}/full_result.json" \
    2>&1 | tee "${LOG_DIR}/compare.log"
phase=completed
echo "E18 execution completed. Performance decision: ${LOG_DIR}/full_result.json"
