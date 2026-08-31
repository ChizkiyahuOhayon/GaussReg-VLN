#!/bin/bash
set -euo pipefail

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

ITERS=${1:-20}
NENV=${2:-1}
PORT=${3:-2334}
EXP_NAME=${E8_EXP_NAME:-e8_r2r_success_set_commit_${ITERS}}
BASE_CKPT=${E8_BASE_CKPT:-data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth}
PRETRAINED=${E8_PRETRAINED:-pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt}

if [ ! -f "${BASE_CKPT}" ]; then
    echo "Missing R2R GRPO checkpoint: ${BASE_CKPT}" >&2
    exit 1
fi
if [ ! -f "${PRETRAINED}" ]; then
    echo "Missing joint-pretraining checkpoint: ${PRETRAINED}" >&2
    exit 1
fi

echo "E8 success-set commit | ${ITERS} iterations | ${NENV} environment(s)"
python -m torch.distributed.launch --nproc_per_node=1 --master_port "${PORT}" run.py \
    --exp_name "${EXP_NAME}" \
    --run-type grpo \
    --exp-config run_r2r/iter_train.yaml \
    SIMULATOR_GPU_IDS "[0]" \
    TORCH_GPU_IDS "[0]" \
    GPU_NUMBERS 1 \
    NUM_ENVIRONMENTS "${NENV}" \
    ONLY_LAST_SAVEALL True \
    TRAINER_NAME GRPO-R1 \
    GRPO.iters "${ITERS}" \
    GRPO.lr 0.0001 \
    GRPO.warmup_iters 0 \
    GRPO.min_lr_ratio 0.25 \
    GRPO.log_every 10 \
    GRPO.load_from_ckpt True \
    GRPO.ckpt_to_load "${BASE_CKPT}" \
    GRPO.is_requeue False \
    GRPO.gauss_only False \
    GRPO.candidate_scorer_only False \
    GRPO.gaussian_bev_only False \
    GRPO.anchor_repair_only False \
    GRPO.hindsight_stop_only False \
    GRPO.terminal_commit_only True \
    GRPO.success_set_commit True \
    GRPO.waypoint_aug False \
    GRPO.sample_num 1 \
    GRPO.update_epochs 1 \
    GRPO.grpo_beta 0.0 \
    GRPO.enable_amp False \
    GRPO.enable_all_dropouts False \
    GRPO.dropout_in_sampling False \
    GRPO.max_grad_norm 2.0 \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
    TASK_CONFIG.DATASET.SUFFIX "_10" \
    MODEL.gauss_feat_size 0 \
    MODEL.gauss_residual_scale 0.0 \
    MODEL.candidate_scorer_hidden_size 0 \
    MODEL.gaussian_bev_hidden_size 0 \
    MODEL.anchor_repair_hidden_size 0 \
    MODEL.hindsight_stop_hidden_size 0 \
    MODEL.terminal_commit_hidden_size 128 \
    MODEL.pretrained_path "${PRETRAINED}"
