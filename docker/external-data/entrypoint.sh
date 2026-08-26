#!/usr/bin/env bash
set -Eeuo pipefail

PERSISTENT_ROOT=/models/robotwin-persistent
required_paths=(
  "${PERSISTENT_ROOT}/assets/objects/objaverse/list.json"
  "${PERSISTENT_ROOT}/data/demo_clean"
  "${PERSISTENT_ROOT}/data/lerobot"
  "${PERSISTENT_ROOT}/data/robotwin_demo_clean_joint_v30.txt"
  "${PERSISTENT_ROOT}/models/robbyant_lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt/model.safetensors.index.json"
)

for required_path in "${required_paths[@]}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing persistent RoboTwin content: ${required_path}" >&2
    echo "Mount the external RoboTwin data at ${PERSISTENT_ROOT}" >&2
    exit 1
  fi
done

mkdir -p \
  /workspace/runtime/eval_result \
  /workspace/runtime/outputs \
  /workspace/runtime/.cache/huggingface

exec "$@"
