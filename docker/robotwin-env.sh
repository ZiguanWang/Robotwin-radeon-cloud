#!/usr/bin/env bash

# Runtime environment shared by interactive shells and non-interactive jobs.
# This file is safe to source repeatedly.

export AITER_TRITON_ONLY="${AITER_TRITON_ONLY:-1}"
export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}"
export HF_HOME="${HF_HOME:-/workspace/runtime/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/workspace/runtime/tmp}"

# ROCm compute visibility and Mesa/Vulkan render visibility are independent.
# The caller supplies HIP_VISIBLE_DEVICES and, for same-model GPUs, a PCI-form
# DRI_PRIME value. Do not let inherited compute masks filter a second time.
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HSA_OVERRIDE_GFX_VERSION
if [[ -n "${DRI_PRIME:-}" ]]; then
    unset MESA_VK_DEVICE_SELECT
fi

if [[ ":${PYTHONPATH:-}:" != *":/opt/aiter:"* ]]; then
    export PYTHONPATH="/opt/aiter${PYTHONPATH:+:${PYTHONPATH}}"
fi
