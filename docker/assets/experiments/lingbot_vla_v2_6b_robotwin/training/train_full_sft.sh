#!/usr/bin/env bash
set -euo pipefail

# Full-parameter SFT smoke/production launcher for LingBot-VLA-v2.
# GPU_COUNT supports 1, 2, or 4 and defaults to 4. The effective global batch
# remains 4 by adjusting gradient accumulation to 4, 2, or 1 respectively.

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/RoboTwin}"
MODEL_ENV="${MODEL_ENV:-/opt/robotwin-env}"
GPU_COUNT="${GPU_COUNT:-4}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-${MAX_STEPS}}"
ENABLE_FULL_SHARD="${ENABLE_FULL_SHARD:-true}"
DATA_PARALLEL_MODE="${DATA_PARALLEL_MODE:-fsdp2}"
ENABLE_FSDP_OFFLOAD="${ENABLE_FSDP_OFFLOAD:-false}"
ENABLE_ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-false}"
ACTIVATION_GPU_LIMIT="${ACTIVATION_GPU_LIMIT:-0.0}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/runtime/outputs/full_sft_${GPU_COUNT}gpu_${MAX_STEPS}steps}"
DATA_LIST="${DATA_LIST:-${ROBOTWIN_ROOT}/data/robotwin_demo_clean_joint_v30.txt}"
CONFIG="${CONFIG:-${ROBOTWIN_ROOT}/experiments/lingbot_vla_v2_6b_robotwin/training/reproduction_100steps/lingbotvla_cli.yaml}"

SOURCE_DIR="${ROBOTWIN_ROOT}/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2"
LOG_DIR="/workspace/runtime/outputs/logs"

case "${GPU_COUNT}" in
  1)
    DEFAULT_GPU_IDS="0"
    DEFAULT_GRADIENT_ACCUMULATION_STEPS=4
    ;;
  2)
    DEFAULT_GPU_IDS="0,1"
    DEFAULT_GRADIENT_ACCUMULATION_STEPS=2
    ;;
  4)
    DEFAULT_GPU_IDS="0,1,2,3"
    DEFAULT_GRADIENT_ACCUMULATION_STEPS=1
    ;;
  *)
    echo "GPU_COUNT must be 1, 2, or 4; got ${GPU_COUNT}" >&2
    exit 2
    ;;
esac
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_GRADIENT_ACCUMULATION_STEPS}}"
GPU_IDS="${GPU_IDS:-${DEFAULT_GPU_IDS}}"
if [[ "$(awk -F, '{print NF}' <<<"${GPU_IDS}")" -ne "${GPU_COUNT}" ]]; then
  echo "GPU_IDS (${GPU_IDS}) must contain exactly ${GPU_COUNT} device IDs" >&2
  exit 2
fi

test -x "${MODEL_ENV}/bin/python"
test -f "${CONFIG}"
test -f "${DATA_LIST}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Manifests produced before the repository moved from /workspace/RoboTwin to
# /RoboTwin contain stale absolute paths. Keep persistent data immutable and
# build a corrected runtime copy instead.
if grep -q '/workspace/RoboTwin/' "${DATA_LIST}"; then
  NORMALIZED_DATA_LIST="/workspace/runtime/robotwin_demo_clean_joint_v30.txt"
  sed 's#/workspace/RoboTwin/#/RoboTwin/#g' "${DATA_LIST}" > "${NORMALIZED_DATA_LIST}"
  DATA_LIST="${NORMALIZED_DATA_LIST}"
fi

export HIP_VISIBLE_DEVICES="${GPU_IDS}"
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

cd "${SOURCE_DIR}"
"${MODEL_ENV}/bin/python" -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${GPU_COUNT}" \
  -m tasks.vla.train_lingbotvla \
  "${CONFIG}" \
  --data.train_path "${DATA_LIST}" \
  --train.output_dir "${OUTPUT_DIR}" \
  --train.use_lora false \
  --train.train_expert_only false \
  --train.data_parallel_mode "${DATA_PARALLEL_MODE}" \
  --train.data_parallel_replicate_size 1 \
  --train.data_parallel_shard_size "${GPU_COUNT}" \
  --train.micro_batch_size "${MICRO_BATCH_SIZE}" \
  --train.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --train.global_batch_size "${GLOBAL_BATCH_SIZE}" \
  --train.enable_gradient_checkpointing true \
  --train.enable_fsdp_offload "${ENABLE_FSDP_OFFLOAD}" \
  --train.enable_activation_offload "${ENABLE_ACTIVATION_OFFLOAD}" \
  --train.activation_gpu_limit "${ACTIVATION_GPU_LIMIT}" \
  --train.enable_full_shard "${ENABLE_FULL_SHARD}" \
  --train.enable_resume false \
  --train.max_steps "${MAX_STEPS}" \
  --train.save_steps "${SAVE_STEPS}" \
  2>&1 | tee "${LOG_DIR}/full_sft_${GPU_COUNT}gpu_${MAX_STEPS}steps.log"
