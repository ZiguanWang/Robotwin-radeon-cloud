#!/usr/bin/env bash
set -euo pipefail

# Full-parameter SFT smoke/production launcher for LingBot-VLA-v2.
# GPU_COUNT supports 4 or 8 and defaults to 4. Unless explicitly
# overridden, gradient accumulation is derived from the requested global batch,
# GPU count, and per-rank micro batch size.

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
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
OPTIMIZER="${OPTIMIZER:-adamw}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
TEACHER_MODE="${TEACHER_MODE:-full}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/runtime/outputs/full_sft_${GPU_COUNT}gpu_${MAX_STEPS}steps}"
DATA_LIST="${DATA_LIST:-${ROBOTWIN_ROOT}/data/robotwin_demo_clean_joint_v30.txt}"
CONFIG="${CONFIG:-${ROBOTWIN_ROOT}/experiments/lingbot_vla_v2_6b_robotwin/training/reproduction_100steps/lingbotvla_cli.yaml}"

SOURCE_DIR="${ROBOTWIN_ROOT}/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2"
MODEL_ROOT="${ROBOTWIN_ROOT}/experiments/lingbot_vla_v2_6b_robotwin/models"
BASE_MODEL="${MODEL_ROOT}/robbyant_lingbot-vla-v2-6b"
TOKENIZER_PATH="${MODEL_ROOT}/Qwen3-VL-4B-Instruct-config-tokenizer"
MOGE_PATH="${MODEL_ROOT}/moge-2-vitb-normal/model.pt"
LINGBOT_DEPTH_PATH="${BASE_MODEL}/depth/model.pt"
DINO_VIDEO_PATH="${BASE_MODEL}/dino_video/teacher_step_10000.pth"
DINO_VIDEO_CONFIG="${BASE_MODEL}/dino_video/config.yaml"
LOG_DIR="/workspace/runtime/outputs/logs"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/full_sft_${GPU_COUNT}gpu_${MAX_STEPS}steps.log}"

case "${GPU_COUNT}" in
  4)
    DEFAULT_GPU_IDS="0,1,2,3"
    DEFAULT_MICRO_BATCH_SIZE=16
    ;;
  8)
    DEFAULT_GPU_IDS="0,1,2,3,4,5,6,7"
    DEFAULT_MICRO_BATCH_SIZE=16
    ;;
  *)
    echo "GPU_COUNT must be 4 or 8; got ${GPU_COUNT}" >&2
    exit 2
    ;;
esac
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${DEFAULT_MICRO_BATCH_SIZE}}"

ALIGN_PARAMS=""
USE_FUTURE_IMAGE=false
case "${TEACHER_MODE}" in
  none)
    ;;
  current_depth)
    for required in "${BASE_MODEL}/model.safetensors.index.json" "${TOKENIZER_PATH}" "${MOGE_PATH}" "${LINGBOT_DEPTH_PATH}"; do
      test -e "${required}" || { echo "Missing teacher dependency: ${required}" >&2; exit 2; }
    done
    ALIGN_PARAMS=$(printf '{"mode":"query","num_task_tokens":8,"depth_loss_weight":0.004,"future_depth_loss_weight":0.004,"use_future_video":false,"llm":{"dim_out":2560,"image_token_size":8,"image_input_size":224},"depth":{"model_type":"MoRGBD","moge_path":"%s","morgbd_path":"%s","num_layers":1,"num_heads":4,"dim_head":32,"ff_mult":1,"num_backbone_tokens":256,"token_size":16,"dim_out":1024,"input_size":224,"use_future_depth":false,"block_future_depth_to_action":true,"detach_future_image_feats":true},"visual_steps":5000}' "${MOGE_PATH}" "${LINGBOT_DEPTH_PATH}")
    ;;
  full)
    USE_FUTURE_IMAGE=true
    for required in "${BASE_MODEL}/model.safetensors.index.json" "${TOKENIZER_PATH}" "${MOGE_PATH}" "${LINGBOT_DEPTH_PATH}" "${DINO_VIDEO_PATH}" "${DINO_VIDEO_CONFIG}"; do
      test -e "${required}" || { echo "Missing teacher dependency: ${required}" >&2; exit 2; }
    done
    ALIGN_PARAMS=$(printf '{"mode":"query","num_task_tokens":8,"depth_loss_weight":0.004,"future_depth_loss_weight":0.004,"use_future_video":true,"llm":{"dim_out":2560,"image_token_size":8,"image_input_size":224},"depth":{"model_type":"MoRGBD","moge_path":"%s","morgbd_path":"%s","num_layers":1,"num_heads":4,"dim_head":32,"ff_mult":1,"num_backbone_tokens":256,"token_size":16,"dim_out":1024,"input_size":224,"use_future_depth":true,"block_future_depth_to_action":true,"detach_future_image_feats":true},"video":{"ckpt_path":"%s","config_path":"%s","attention_mode":"flex_block_causal","input_size":256,"block_suffix_to_future_video":true,"block_warmup_steps":0,"block_warmup_gradual":false,"share_future_depth_query":true,"use_shared_future_task_proj":true,"use_current_shared_task_proj":true,"shared_query_head_type":"resampler","num_future_frames":1,"use_warmup_frame":true,"effective_fps":1.0,"n_blocks":1,"cls_pool":"last","head_type":"resampler","detach_image_feats":true,"num_layers":1,"num_heads":4,"dim_head":32,"ff_mult":1,"num_backbone_tokens":256,"dim_out":1024,"target_type":"absolute","future_video_loss_weight":0.004,"use_smooth_l1_loss":false,"use_mse_loss":true,"mse_loss_weight":1.0,"use_patch_loss":true,"use_current_patch_loss":true,"use_cosine_loss":false,"cosine_loss_weight":0.2,"use_cls_loss":false,"cls_loss_type":"mse","cls_loss_weight":0.2,"log_max_samples":32,"log_scale":16},"visual_steps":5000}' "${MOGE_PATH}" "${LINGBOT_DEPTH_PATH}" "${DINO_VIDEO_PATH}" "${DINO_VIDEO_CONFIG}")
    ;;
  *)
    echo "TEACHER_MODE must be none, current_depth, or full; got ${TEACHER_MODE}" >&2
    exit 2
    ;;
esac
LOCAL_BATCH_SIZE=$((GPU_COUNT * MICRO_BATCH_SIZE))
if (( GLOBAL_BATCH_SIZE % LOCAL_BATCH_SIZE != 0 )); then
  echo "GLOBAL_BATCH_SIZE (${GLOBAL_BATCH_SIZE}) must be divisible by GPU_COUNT * MICRO_BATCH_SIZE (${LOCAL_BATCH_SIZE})" >&2
  exit 2
fi
DEFAULT_GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / LOCAL_BATCH_SIZE))
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_GRADIENT_ACCUMULATION_STEPS}}"
if (( MICRO_BATCH_SIZE * GPU_COUNT * GRADIENT_ACCUMULATION_STEPS != GLOBAL_BATCH_SIZE )); then
  echo "GLOBAL_BATCH_SIZE must equal MICRO_BATCH_SIZE * GPU_COUNT * GRADIENT_ACCUMULATION_STEPS" >&2
  exit 2
fi
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
TRAIN_COMMAND=(
  "${MODEL_ENV}/bin/python" -m torch.distributed.run
  --standalone \
  --nproc-per-node="${GPU_COUNT}" \
  -m tasks.vla.train_lingbotvla \
  "${CONFIG}" \
  --data.train_path "${DATA_LIST}" \
  --data.num_workers "${NUM_WORKERS}" \
  --data.prefetch_factor "${PREFETCH_FACTOR}" \
  --train.output_dir "${OUTPUT_DIR}" \
  --train.optimizer "${OPTIMIZER}" \
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
  --train.save_steps "${SAVE_STEPS}"
)
if [[ -n "${ALIGN_PARAMS}" ]]; then
  TRAIN_COMMAND+=(
    --model.model_path "${BASE_MODEL}"
    --model.config_path "${BASE_MODEL}"
    --model.tokenizer_path "${TOKENIZER_PATH}"
    --model.post_training true
    --data.use_future_image "${USE_FUTURE_IMAGE}"
    --train.align_params "${ALIGN_PARAMS}"
  )
fi

if (( TIMEOUT_SECONDS > 0 )); then
  timeout --signal=INT --kill-after=60 "${TIMEOUT_SECONDS}" "${TRAIN_COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
  status=${PIPESTATUS[0]}
  case "${status}" in
    0|124|130|137) exit 0 ;;
    *) exit "${status}" ;;
  esac
else
  "${TRAIN_COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
fi
