#!/usr/bin/env bash
# Run task-specific LoRA training using the supplied data manifest.
set -euo pipefail
: "${LORA_SOURCE:?Set LORA_SOURCE}"
: "${LORA_OUTPUT:?Set LORA_OUTPUT}"
: "${LORA_DATA_MANIFEST:?Set LORA_DATA_MANIFEST}"
: "${LORA_GPU_COUNT:?Set LORA_GPU_COUNT}"
: "${LORA_MICRO_BATCH:?Set LORA_MICRO_BATCH}"
: "${LORA_GRAD_ACCUM:?Set LORA_GRAD_ACCUM}"
: "${LORA_MAX_STEPS:?Set LORA_MAX_STEPS}"
global_batch=$((LORA_GPU_COUNT * LORA_MICRO_BATCH * LORA_GRAD_ACCUM))
printf 'LoRA batch run: GPUs=%s micro=%s accum=%s global=%s steps=%s\n' \
    "$LORA_GPU_COUNT" "$LORA_MICRO_BATCH" "$LORA_GRAD_ACCUM" "$global_batch" \
    "$LORA_MAX_STEPS"
mkdir -p "$LORA_OUTPUT"
cd "$LORA_SOURCE"
exec /opt/robotwin-env/bin/python -m torch.distributed.run --standalone \
    "--nproc-per-node=$LORA_GPU_COUNT" -m tasks.vla.train_lingbotvla \
    /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/training/lingbotvla_cli.yaml \
    --data.train_path "$LORA_DATA_MANIFEST" --data.num_workers 2 \
    --train.output_dir "$LORA_OUTPUT" \
    --train.use_lora true --train.lora_scope all_attention \
    --train.lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --train.lora_rank 16 --train.lora_alpha 32 \
    --train.data_parallel_shard_size "$LORA_GPU_COUNT" \
    --train.micro_batch_size "$LORA_MICRO_BATCH" \
    --train.gradient_accumulation_steps "$LORA_GRAD_ACCUM" \
    --train.global_batch_size "$global_batch" \
    --train.lr 0.0005 --train.vit_lr 0.00002 \
    --train.enable_resume false --train.max_steps "$LORA_MAX_STEPS" \
    --train.save_steps "${LORA_SAVE_STEPS:-0}"
