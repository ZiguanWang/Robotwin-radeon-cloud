#!/usr/bin/env bash
set -euo pipefail

# Run one task through RoboTwin's XPolicyLab evaluator and LingBot websocket.
# Usage:
#   run_robotwin_eval.sh <task|index> <none|320|640|1280x960|all> [task_config] [test_num] [port] [expert_check]
# Examples:
#   run_robotwin_eval.sh adjust_bottle none
#   run_robotwin_eval.sh 1 640 demo_clean 100 13400 true
#   run_robotwin_eval.sh 1 all demo_clean 100 13400 true

ROOT=${ROBOTWIN_ROOT:-/RoboTwin}
RUNTIME_DIR=${ROBOTWIN_RUNTIME:-/workspace/runtime}
PYTHON_BIN=${ROBOTWIN_PYTHON:-/opt/robotwin-env/bin/python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASK_SELECTOR=${1:-}
VIDEO_MODE=${2:-}
TASK_CONFIG=${3:-demo_clean}
TEST_NUM=${4:-100}
PORT=${5:-13400}
EXPERT_CHECK=${6:-true}

if [[ -z "$TASK_SELECTOR" || -z "$VIDEO_MODE" ]]; then
  echo "Usage: $0 <task-name|1-based-task-index> <none|320|640|1280x960|all> [task_config] [test_num] [port] [expert_check]" >&2
  exit 2
fi

if [[ ! "$TEST_NUM" =~ ^[1-9][0-9]*$ || ! "$EXPERT_CHECK" =~ ^(true|false)$ ]]; then
  echo "test_num must be positive and expert_check must be true or false" >&2
  exit 2
fi
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-${ROBOTWIN_GPU:-0}}"
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HSA_OVERRIDE_GFX_VERSION MESA_VK_DEVICE_SELECT
if [[ -z "${DRI_PRIME:-}" ]]; then
  if ! DRI_PRIME=$("$PYTHON_BIN" "$SCRIPT_DIR/resolve_dri_prime.py" --visible-one); then
    echo "Cannot map the selected HIP GPU to a render device; set DRI_PRIME explicitly" >&2
    exit 2
  fi
  export DRI_PRIME
fi
if [[ ! "$DRI_PRIME" =~ ^pci-[[:xdigit:]]{4}_[[:xdigit:]]{2}_[[:xdigit:]]{2}_[0-7]!$ ]]; then
  echo "DRI_PRIME must be a PCI selector such as pci-0000_63_00_0!" >&2
  exit 2
fi
echo "[robotwin-eval] HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES DRI_PRIME=$DRI_PRIME"

case "$VIDEO_MODE" in
  none|320|640|1280x960|all) ;;
  *) echo "Invalid video mode: $VIDEO_MODE (expected none, 320, 640, 1280x960, or all)" >&2; exit 2 ;;
esac

TASK_NAME=$($PYTHON_BIN - "$ROOT/env_cfg/eval/all_tasks.yml" "$TASK_SELECTOR" <<'PY'
import sys
import yaml

path, selector = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    tasks = list(yaml.safe_load(f)["tasks"])
if selector.isdigit():
    index = int(selector)
    if index < 1 or index > len(tasks):
        raise SystemExit(f"task index must be between 1 and {len(tasks)}")
    print(tasks[index - 1])
else:
    if selector not in tasks:
        raise SystemExit(f"unknown task {selector!r}; inspect all_tasks.yml")
    print(selector)
PY
)

CONFIG_TO_USE="$TASK_CONFIG"
CONFIG_DIR="$ROOT/env_cfg/task_config"
if [[ "$VIDEO_MODE" == "640" || "$VIDEO_MODE" == "1280x960" || "$VIDEO_MODE" == "all" ]]; then
  CONFIG_TO_USE="${TASK_CONFIG}_large_d435"
  SOURCE_CONFIG="$CONFIG_DIR/${TASK_CONFIG}.yml"
  LARGE_CONFIG="$CONFIG_DIR/${CONFIG_TO_USE}.yml"
  if [[ ! -f "$LARGE_CONFIG" ]]; then
    cp "$SOURCE_CONFIG" "$LARGE_CONFIG"
    sed -i \
      -e 's/head_camera_type: D435/head_camera_type: Large_D435/' \
      -e 's/wrist_camera_type: D435/wrist_camera_type: Large_D435/' \
      "$LARGE_CONFIG"
  fi
  if [[ "$VIDEO_MODE" == "1280x960" || "$VIDEO_MODE" == "all" ]]; then
    CAMERA_CONFIG="$CONFIG_DIR/_camera_config.yml"
    if ! grep -q '^XL_D435:' "$CAMERA_CONFIG"; then
      printf '\nXL_D435:\n  fovy: 37\n  w: 1280\n  h: 960\n' >> "$CAMERA_CONFIG"
    fi
    CONFIG_TO_USE="${TASK_CONFIG}_xl_d435"
    SOURCE_CONFIG="$CONFIG_DIR/${TASK_CONFIG}.yml"
    XL_CONFIG="$CONFIG_DIR/${CONFIG_TO_USE}.yml"
    if [[ ! -f "$XL_CONFIG" ]]; then
      cp "$SOURCE_CONFIG" "$XL_CONFIG"
      sed -i \
        -e 's/head_camera_type: D435/head_camera_type: XL_D435/' \
        -e 's/wrist_camera_type: D435/wrist_camera_type: XL_D435/' \
        "$XL_CONFIG"
    fi
  fi
fi

run_one() {
  local mode="$1"
  local config="$TASK_CONFIG"
  local video_arg="False"
  if [[ "$mode" == "320" ]]; then
    video_arg="True"
  elif [[ "$mode" == "640" ]]; then
    config="${TASK_CONFIG}_large_d435"
    video_arg="True"
  elif [[ "$mode" == "1280x960" ]]; then
    config="${TASK_CONFIG}_xl_d435"
    video_arg="True"
  fi

  local stamp start end elapsed log result_line
  stamp=$(date +%Y%m%d_%H%M%S)
  start=$(date +%s.%N)
  log="$RUNTIME_DIR/outputs/logs/robotwin_eval_${TASK_NAME}_${mode}_${stamp}.log"
  mkdir -p "$(dirname "$log")"

  echo "[robotwin-eval] task=$TASK_NAME mode=$mode config=$config test_num=$TEST_NUM"
  echo "[robotwin-eval] log=$log"
  set +e
  cd "$ROOT"
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "$ROOT/scripts/eval_policy_xpolicylab.py" \
    --task_name "$TASK_NAME" \
    --task_config "$config" \
    --policy_name LingBot-VLA-v2 \
    --protocol lingbot_vla_v2 \
    --host 127.0.0.1 \
    --port "$PORT" \
    --seed 0 \
    --test_num "$TEST_NUM" \
    --expert_check "$EXPERT_CHECK" \
    --eval_batch false \
    --additional_info "eval_video_log=${video_arg,,}" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  end=$(date +%s.%N)
  elapsed=$(awk -v s="$start" -v e="$end" 'BEGIN {printf "%.3f", e-s}')
  result_line=$(grep -E 'Final success rate:|Success rate:' "$log" | tail -1 || true)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$TASK_NAME" "$mode" "$TEST_NUM" "$elapsed" "$rc" "$result_line" \
    | tee -a "$RUNTIME_DIR/outputs/robotwin_eval_timing.tsv"
  return "$rc"
}

if [[ "$VIDEO_MODE" == "all" ]]; then
  run_one none
  run_one 320
  run_one 640
  run_one 1280x960
else
  run_one "$VIDEO_MODE"
fi
