#!/usr/bin/env bash
# Prepare external-data inputs on the host; no Docker container is started.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
TARGET=/models/robotwin-persistent
RUNTIME="${TARGET}/.preparation_runtime"
SOURCE="${RUNTIME}/robotwin-source"
DOWNLOAD_ENV="${RUNTIME}/download-env"
DATA_ENV="${RUNTIME}/lerobot-env"
ROBOTWIN_COMMIT=e154702de28cac4b9bfe0299044303f507236bb3
XPOLICYLAB_COMMIT=c37109c500be67d0dea6b36bf7337bbd26e763cd
DATA_REVISION=a967b852afa21a9cbf19a198f7e653109042e87c
MODEL_REVISION=11c703bf6a5c1f45b3b69168482da11fdbba53d7
QWEN_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17
MOGE_REVISION=ca5f0e07ff01d3e5a364c1d954ed12ee1814b368

usage() {
    cat <<'EOF'
Usage: bash scripts/data/prepare_external_data.sh [--dry-run]

Prepare /models/robotwin-persistent on the host: assets, 50 demo_clean tasks,
model files, LeRobot v3 task datasets, and the container-path training manifest.
Python dependencies are installed into isolated environments under
/models/robotwin-persistent/.preparation_runtime when needed. No Docker image
or GPU is required. Large downloads and conversions are skipped when complete.
EOF
}

case "${1:-}" in
    --dry-run)
        printf 'Target: %s\nSource/cache/venvs: %s\n' "${TARGET}" "${RUNTIME}"
        printf 'Stages: clone pinned RoboTwin/XPolicyLab; download assets, 50 clean tasks and models; convert missing LeRobot tasks; validate and write manifest.\n'
        exit 0 ;;
    -h|--help) usage; exit 0 ;;
    '') ;;
    *) usage >&2; exit 2 ;;
esac
if (($# > 1)); then usage >&2; exit 2; fi

command -v git >/dev/null
command -v python3 >/dev/null
[[ -d "${TARGET}" && -w "${TARGET}" ]] || {
    echo "Create a writable ${TARGET} before running this script" >&2
    exit 2
}
[[ "$(python3 -c 'import sys; print(sys.version_info >= (3, 12))')" == True ]] || {
    echo 'Python 3.12 or newer is required' >&2
    exit 2
}
mkdir -p "${RUNTIME}" "${TARGET}/assets" "${TARGET}/data" "${TARGET}/models"
export HF_HOME="${RUNTIME}/huggingface" XDG_CACHE_HOME="${RUNTIME}/.cache"
export TMPDIR="${RUNTIME}/tmp"
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}"

ensure_venv() {
    local destination=$1 check=$2
    shift 2
    if [[ ! -x "${destination}/bin/python" ]]; then
        python3 -m venv --without-pip "${destination}"
    fi
    if ! "${destination}/bin/python" -m pip --version >/dev/null 2>&1; then
        command -v curl >/dev/null
        if [[ ! -f "${RUNTIME}/get-pip.py" ]]; then
            curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "${RUNTIME}/get-pip.py"
        fi
        "${destination}/bin/python" "${RUNTIME}/get-pip.py"
    fi
    if ! "${destination}/bin/python" -c "${check}" >/dev/null 2>&1; then
        "${destination}/bin/python" -m pip install --upgrade pip
        "${destination}/bin/python" -m pip install "$@"
        "${destination}/bin/python" -c "${check}"
    fi
}

# Keep the pinned source in the preparation cache. It supplies the task list,
# archive downloader and LeRobot converter, but is not mounted as model data.
if [[ ! -d "${SOURCE}/.git" ]]; then
    git clone --filter=blob:none --no-checkout \
        https://github.com/RoboTwin-Platform/RoboTwin.git "${SOURCE}"
    git -C "${SOURCE}" checkout "${ROBOTWIN_COMMIT}"
fi
[[ "$(git -C "${SOURCE}" rev-parse HEAD)" == "${ROBOTWIN_COMMIT}" ]] || {
    echo "Unexpected RoboTwin source revision in ${SOURCE}" >&2; exit 1;
}
if [[ ! -f "${SOURCE}/XPolicyLab/scripts/transform_lerobot_v30_format.py" ]]; then
    git -C "${SOURCE}" submodule update --init XPolicyLab
fi
[[ "$(git -C "${SOURCE}/XPolicyLab" rev-parse HEAD)" == "${XPOLICYLAB_COMMIT}" ]] || {
    echo 'Unexpected XPolicyLab source revision' >&2; exit 1;
}
if git -C "${SOURCE}/XPolicyLab" apply --check \
    "${REPO_ROOT}/patches/xpolicylab-lerobot-v30.patch" >/dev/null 2>&1; then
    git -C "${SOURCE}/XPolicyLab" apply \
        "${REPO_ROOT}/patches/xpolicylab-lerobot-v30.patch"
elif ! git -C "${SOURCE}/XPolicyLab" apply --reverse --check \
    "${REPO_ROOT}/patches/xpolicylab-lerobot-v30.patch" >/dev/null 2>&1; then
    echo 'XPolicyLab source differs from the expected patched revision' >&2
    exit 1
fi

# The converter expects repository-relative data; point the cached source at
# the persistent data. Preserve the small tracked trees in *.repository.
for name in assets data; do
    if [[ ! -L "${SOURCE}/${name}" ]]; then
        if [[ -d "${SOURCE}/${name}" ]]; then
            cp -a --update=none "${SOURCE}/${name}/." "${TARGET}/${name}/"
            mv "${SOURCE}/${name}" "${SOURCE}/${name}.repository"
        fi
        ln -s "${TARGET}/${name}" "${SOURCE}/${name}"
    fi
    [[ "$(realpath "${SOURCE}/${name}")" == "${TARGET}/${name}" ]] || {
        echo "Unexpected ${SOURCE}/${name} link" >&2; exit 1;
    }
done

ensure_venv "${DOWNLOAD_ENV}" \
    "import yaml; from importlib.metadata import version; assert version('huggingface-hub') == '0.34.3'" \
    'huggingface-hub==0.34.3' 'pyyaml==6.0.2'
HF_CLI="${DOWNLOAD_ENV}/bin/huggingface-cli"

# 1. Simulation assets. Render embodiment templates with the path used later
# inside the external-data image, rather than the host's temporary source path.
if [[ ! -f "${TARGET}/assets/objects/objaverse/list.json" ||
      ! -d "${TARGET}/assets/embodiments/aloha-agilex" ||
      ! -d "${TARGET}/assets/background_texture" ]]; then
    echo '[prepare] downloading simulation assets'
    "${HF_CLI}" download TianxingChen/RoboTwin2.0 --repo-type dataset \
        --revision "${DATA_REVISION}" \
        --include background_texture.zip embodiments.zip objects.zip \
        --local-dir "${RUNTIME}/downloads/robotwin-assets"
    "${DOWNLOAD_ENV}/bin/python" - "${RUNTIME}/downloads/robotwin-assets" "${TARGET}/assets" <<'PY'
from pathlib import Path
from zipfile import ZipFile
import sys
source, target = map(Path, sys.argv[1:])
for name in ('background_texture.zip', 'embodiments.zip', 'objects.zip'):
    with ZipFile(source / name) as archive:
        archive.extractall(target)
PY
fi
"${DOWNLOAD_ENV}/bin/python" - "${TARGET}/assets/embodiments" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
templates = list(root.rglob('*_tmp.yml'))
if not templates:
    raise SystemExit('No embodiment YAML templates found')
for template in templates:
    text = template.read_text()
    target = template.with_name(template.name.replace('_tmp.yml', '.yml'))
    target.write_text(text.replace('${ASSETS_PATH}', '/RoboTwin').replace('$ASSETS_PATH', '/RoboTwin'))
PY
[[ -f "${TARGET}/assets/objects/objaverse/list.json" ]]

mapfile -t tasks < <("${DOWNLOAD_ENV}/bin/python" - "${SOURCE}/env_cfg/eval/all_tasks.yml" <<'PY'
from pathlib import Path
import sys, yaml
tasks = yaml.safe_load(Path(sys.argv[1]).read_text())['tasks']
if len(tasks) != 50 or len(set(tasks)) != 50:
    raise SystemExit('Expected 50 unique RoboTwin tasks')
print(*tasks, sep='\n')
PY
)
[[ "${#tasks[@]}" -eq 50 ]]

# 2. Native demo_clean archives. The upstream downloader normalizes the
# episode filenames and directory layout required by XPolicyLab.
raw_complete=true
raw_task_complete() {
    local folder="${TARGET}/data/demo_clean/$1"
    [[ -d "${folder}" ]] &&
        [[ "$(find "${folder}" -name '*.hdf5' | wc -l)" -eq 50 ]]
}
for task in "${tasks[@]}"; do
    if ! raw_task_complete "${task}"; then
        raw_complete=false; break
    fi
done
if [[ "${raw_complete}" == false ]]; then
    echo '[prepare] downloading clean demonstrations'
    (cd "${SOURCE}" && \
        PATH="${DOWNLOAD_ENV}/bin:${PATH}" \
        ROBOTWIN_DATA_ROOT="${TARGET}/data" \
        HF_ARCHIVE_CACHE="${RUNTIME}/downloads/demo-clean" \
        HF_REVISION="${DATA_REVISION}" \
        HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}" \
        HF_EXTRACT_WORKERS="${HF_EXTRACT_WORKERS:-4}" \
        HF_KEEP_ARCHIVES=0 \
        bash scripts/download_xpolicylab_data.sh "${tasks[@]}")
fi

# 3. Model files at the same paths expected by the external-data image.
base="${TARGET}/models/robbyant_lingbot-vla-v2-6b"
shards=0
if [[ -d "${base}" ]]; then
    shards=$(find "${base}" -maxdepth 1 -name 'model-*.safetensors' | wc -l)
fi
if [[ ! -f "${base}/model.safetensors.index.json" ||
      ! -f "${base}/depth/model.pt" ||
      ! -f "${base}/dino_video/teacher_step_10000.pth" ||
      ! -f "${base}/dino_video/config.yaml" || "${shards}" -ne 6 ]]; then
    "${HF_CLI}" download robbyant/lingbot-vla-v2-6b \
        --revision "${MODEL_REVISION}" --local-dir "${base}"
fi
qwen="${TARGET}/models/Qwen3-VL-4B-Instruct-config-tokenizer"
if [[ ! -f "${qwen}/config.json" || ! -f "${qwen}/tokenizer_config.json" ||
      ! -f "${qwen}/merges.txt" || ! -f "${qwen}/vocab.json" ]]; then
    "${HF_CLI}" download Qwen/Qwen3-VL-4B-Instruct --revision "${QWEN_REVISION}" \
        --include '*.json' '*.txt' '*.jinja' merges.txt vocab.json \
        --local-dir "${qwen}"
fi
moge="${TARGET}/models/moge-2-vitb-normal"
if [[ ! -f "${moge}/model.pt" ]]; then
    "${HF_CLI}" download Ruicheng/moge-2-vitb-normal \
        --revision "${MOGE_REVISION}" --local-dir "${moge}"
fi

# 4. Convert only missing/incomplete tasks, installing LeRobot on demand.
valid_dataset() {
    local info=$1
    [[ -f "${info}" ]] || return 1
    python3 - "${info}" <<'PY'
import json, sys
from pathlib import Path
try:
    info = json.loads(Path(sys.argv[1]).read_text())
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if info.get('total_episodes') == 50 else 1)
PY
}

need_conversion=false
for task in "${tasks[@]}"; do
    if ! valid_dataset "${TARGET}/data/lerobot/${task}_joint_v30/meta/info.json"; then
        need_conversion=true; break
    fi
done
if [[ "${need_conversion}" == true ]]; then
    ensure_venv "${DATA_ENV}" \
        "import av, cv2, h5py, lerobot, torch, yaml; from importlib.metadata import version; assert version('lerobot') == '0.6.0' and version('h5py') == '3.14.0'" \
        'lerobot[dataset]==0.6.0' 'h5py==3.14.0' 'pyyaml==6.0.2' 'opencv-python-headless' 'av==15.0.0'
    export HF_LEROBOT_HOME="${TARGET}/data/lerobot"
    export LEROBOT_VIDEO_BACKEND=pyav LEROBOT_VIDEO_CODEC=h264
    export PYTHONPATH="${SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
    mkdir -p "${HF_LEROBOT_HOME}" "${RUNTIME}/logs/conversion"
    for task in "${tasks[@]}"; do
        output="${HF_LEROBOT_HOME}/${task}_joint_v30"
        valid_dataset "${output}/meta/info.json" && continue
        if [[ -e "${output}" ]]; then
            mv "${output}" "${output}.incomplete.$(date -u +%Y%m%d_%H%M%S)"
        fi
        echo "[prepare] converting ${task}"
        if ! (cd "${SOURCE}" && "${DATA_ENV}/bin/python" \
            XPolicyLab/scripts/transform_lerobot_v30_format.py \
            "demo_clean.${task}.aloha_agilex" \
            --repo_id "${task}_joint_v30" --max_episode 50) \
            >"${RUNTIME}/logs/conversion/${task}.log" 2>&1 ||
            grep -q '^Failed files:' "${RUNTIME}/logs/conversion/${task}.log"; then
            tail -100 "${RUNTIME}/logs/conversion/${task}.log" >&2
            exit 1
        fi
        valid_dataset "${output}/meta/info.json" || {
            echo "Conversion did not create 50 valid episodes in ${output}" >&2; exit 1;
        }
    done
fi

# 5. Verify data before publishing a manifest with container-visible paths.
for task in "${tasks[@]}"; do
    raw_task_complete "${task}" || {
        echo "Missing clean demonstrations for ${task}" >&2; exit 1;
    }
    valid_dataset "${TARGET}/data/lerobot/${task}_joint_v30/meta/info.json" || {
        echo "Missing or incomplete LeRobot task: ${task}" >&2; exit 1;
    }
done
for required in \
    "${TARGET}/assets/objects/objaverse/list.json" \
    "${base}/model.safetensors.index.json" \
    "${base}/depth/model.pt" \
    "${base}/dino_video/teacher_step_10000.pth" \
    "${base}/dino_video/config.yaml" \
    "${qwen}/config.json" "${qwen}/tokenizer_config.json" \
    "${qwen}/merges.txt" "${qwen}/vocab.json" "${moge}/model.pt"; do
    [[ -f "${required}" ]] || { echo "Missing ${required}" >&2; exit 1; }
done
[[ "$(find "${base}" -maxdepth 1 -name 'model-*.safetensors' | wc -l)" -eq 6 ]]
manifest="${TARGET}/data/robotwin_demo_clean_joint_v30.txt"
temporary="${manifest}.tmp.$$"
for task in "${tasks[@]}"; do
    printf 'robotwin /RoboTwin/data/lerobot/%s_joint_v30\n' "${task}"
done >"${temporary}"
[[ "$(wc -l <"${temporary}")" -eq 50 ]]
mv "${temporary}" "${manifest}"
echo "[prepare] complete: ${TARGET}"
