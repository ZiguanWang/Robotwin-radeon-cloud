#!/usr/bin/env bash
set -e

mkdir -p \
  /workspace/runtime/eval_result \
  /workspace/runtime/outputs \
  /workspace/runtime/.cache/huggingface

exec "$@"
