#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/habitat_env.bash"

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 CHECKPOINT [EPISODE_COUNT] [NUM_ENVIRONMENTS] [PORT]" >&2
    exit 2
fi

CKPT=$1
EPISODES=${2:--1}
NENV=${3:-4}
PORT=${4:-2335}
PRETRAINED=${E11_PRETRAINED:-pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt}

if [ ! -f "${CKPT}" ]; then
    echo "Missing E11 checkpoint: ${CKPT}" >&2
    exit 1
fi
if [ ! -f "${PRETRAINED}" ]; then
    echo "Missing joint-pretraining checkpoint: ${PRETRAINED}" >&2
    exit 1
fi

echo "E11 GeoToken-R1 evaluation | ${EPISODES} episode(s) | ${NENV} environment(s)"
python -m torch.distributed.launch --nproc_per_node=1 --master_port "${PORT}" run.py \
    --exp_name e11_r2r_geo_token \
    --run-type eval \
    --exp-config run_r2r/iter_train.yaml \
    SIMULATOR_GPU_IDS "[0]" \
    TORCH_GPU_IDS "[0]" \
    GPU_NUMBERS 1 \
    NUM_ENVIRONMENTS "${NENV}" \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
    EVAL.SPLIT val_unseen \
    EVAL.EPISODE_COUNT "${EPISODES}" \
    EVAL.CKPT_PATH_DIR "${CKPT}" \
    IL.back_algo control \
    MODEL.gauss_feat_size 0 \
    MODEL.gauss_residual_scale 0.0 \
    MODEL.candidate_scorer_hidden_size 0 \
    MODEL.gaussian_bev_hidden_size 0 \
    MODEL.anchor_repair_hidden_size 0 \
    MODEL.hindsight_stop_hidden_size 0 \
    MODEL.terminal_commit_hidden_size 0 \
    MODEL.geo_token_hidden_size 64 \
    MODEL.pretrained_path "${PRETRAINED}"
