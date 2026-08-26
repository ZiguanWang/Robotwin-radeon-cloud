#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$script_dir/.." && pwd)
source_root="$root/source/lingbot-vla-v2"
python_bin=${LINGBOT_VLA_PYTHON:-"$root/env/.venv/bin/python"}
default_model_path="$root/models/robbyant_lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"
qwen_path="$root/models/Qwen3-VL-4B-Instruct-config-tokenizer"

hip_id=${1:?usage: launch_official_server.sh HIP_ID PORT LOG_FILE [USE_COMPILE]}
port=${2:?}
log_file=${3:?}
use_compile=${4:-False}
model_path=${5:-$default_model_path}

mkdir -p "$(dirname "$log_file")"
log_file=$(realpath -m "$log_file")
export HIP_VISIBLE_DEVICES="$hip_id"
unset ROCR_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export QWEN3VL_PATH="$qwen_path"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
export PYTHONPATH="$source_root${PYTHONPATH:+:$PYTHONPATH}"

cd "$source_root"
exec "$python_bin" -m deploy.lingbot_vla_v2_policy \
  --model_path "$model_path" \
  --use_length 25 \
  --use_bf16 True \
  --use_fp32 False \
  --use_compile "$use_compile" \
  --port "$port" >> "$log_file" 2>&1

