#!/usr/bin/env bash
set -Eeuo pipefail

cd /RoboTwin

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/RoboTwin/data/lerobot}"
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
export LEROBOT_VIDEO_CODEC="${LEROBOT_VIDEO_CODEC:-h264}"
CONVERSION_JOBS="${CONVERSION_JOBS:-8}"
LEROBOT_PYTHON="${LEROBOT_PYTHON:-/opt/lerobot-env/bin/python}"
LOG_ROOT=/tmp/robotwin-lerobot-conversion
CACHE_ROOT="${CONVERSION_CACHE_ROOT:-/tmp/robotwin-lerobot-cache}"
MANIFEST=/RoboTwin/data/robotwin_demo_clean_joint_v30.txt

mkdir -p "${HF_LEROBOT_HOME}" "${LOG_ROOT}" "${CACHE_ROOT}"
mapfile -t tasks < <(
    find data/demo_clean -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
)
[[ "${#tasks[@]}" -eq 50 ]]

CONVERTER_KEY="$({
    printf '%s\n' 'lerobot=0.6.0' 'max_episode=50' \
        "data_revision=${ROBOTWIN_DATA_REVISION:-unknown}" \
        "video_backend=${LEROBOT_VIDEO_BACKEND}" \
        "video_codec=${LEROBOT_VIDEO_CODEC}"
    sha256sum /usr/local/bin/convert-all-robotwin-data
    find XPolicyLab -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum
} | sha256sum | cut -d' ' -f1)"

convert_task() {
    local task="$1"
    local repo_id="${task}_joint_v30"
    local log_file="${LOG_ROOT}/${task}.log"
    local source_key cache_key cache_dir cache_tmp output_dir

    source_key="$(find "data/demo_clean/${task}" -type f -printf '%P %s\n' | sort | sha256sum | cut -d' ' -f1)"
    cache_key="${CONVERTER_KEY}-${source_key}"
    cache_dir="${CACHE_ROOT}/${task}/${cache_key}"
    cache_tmp="${CACHE_ROOT}/${task}/.${cache_key}.tmp.$$"
    output_dir="${HF_LEROBOT_HOME}/${repo_id}"

    if [[ -f "${cache_dir}/.complete" && -f "${cache_dir}/dataset/meta/info.json" ]]; then
        rm -rf "${output_dir}"
        cp -a "${cache_dir}/dataset" "${output_dir}"
        echo "[cached] ${task} -> ${output_dir}"
        return 0
    fi

    rm -rf "${output_dir}"
    if "${LEROBOT_PYTHON}" \
        XPolicyLab/scripts/transform_lerobot_v30_format.py \
        "demo_clean.${task}.aloha_agilex" \
        --repo_id "${repo_id}" \
        --max_episode 50 >"${log_file}" 2>&1 && \
        ! grep -q '^Failed files:' "${log_file}" && \
        test -f "${output_dir}/meta/info.json"; then
        mkdir -p "${CACHE_ROOT}/${task}"
        rm -rf "${cache_tmp}"
        mkdir -p "${cache_tmp}"
        cp -a "${output_dir}" "${cache_tmp}/dataset"
        touch "${cache_tmp}/.complete"
        rm -rf "${cache_dir}"
        mv "${cache_tmp}" "${cache_dir}"
        echo "[converted] ${task} -> ${output_dir}"
    else
        echo "[failed] ${task}; converter output:" >&2
        tail -200 "${log_file}" >&2 || true
        return 1
    fi
}
export -f convert_task
export HF_LEROBOT_HOME LEROBOT_VIDEO_BACKEND LEROBOT_VIDEO_CODEC \
    LOG_ROOT CACHE_ROOT CONVERTER_KEY LEROBOT_PYTHON

printf '%s\n' "${tasks[@]}" | \
    xargs -r -P "${CONVERSION_JOBS}" -n 1 bash -c 'convert_task "$1"' _

: >"${MANIFEST}"
for task in "${tasks[@]}"; do
    printf 'robotwin %s/%s_joint_v30\n' "${HF_LEROBOT_HOME}" "${task}" \
        >>"${MANIFEST}"
done

[[ "$(find "${HF_LEROBOT_HOME}" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 50 ]]
[[ "$(wc -l <"${MANIFEST}")" -eq 50 ]]
rm -rf "${LOG_ROOT}"
