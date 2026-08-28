# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="RoboTwin LingBot-VLA-v2 ROCm runtime" \
      org.opencontainers.image.description="Reproducible ROCm environment for RoboTwin inference, evaluation, and LoRA training"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/workspace/runtime/.cache/huggingface \
    HF_LEROBOT_HOME=/RoboTwin/data/lerobot \
    LEROBOT_VIDEO_BACKEND=pyav \
    LEROBOT_VIDEO_CODEC=h264 \
    PYOPENGL_PLATFORM=egl \
    ROBOTWIN_DISABLE_CUROBO=1 \
    ROBOTWIN_EE_PLANNER=mplib \
    TOKENIZERS_PARALLELISM=false \
    AITER_USE_SYSTEM_TRITON=1 \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    FLASH_ATTENTION_USE_SYSTEM_AITER=TRUE \
    ROBOTWIN_ROOT=/RoboTwin \
    LINGBOT_VLA_SOURCE=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 \
    QWEN3VL_PATH=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/Qwen3-VL-4B-Instruct-config-tokenizer

ARG PIP_INDEX_URL=
ARG ROBOTWIN_COMMIT=266f3aadf505a4f7fe9af0faa41a20f5f47cd123
ARG LINGBOT_VLA_COMMIT=951475ae1b1d87553e7dc47c97b53a3d695c0d13
ARG AITER_COMMIT=9bab8388c35936814a659b4ebd245c491e1b940a
ARG FLASH_ATTENTION_COMMIT=bc76302fbb24c0158207978930db030ca1eca5ca
ARG AITER_GPU_ARCHS=gfx1100;gfx1201
ARG LINGBOT_MODEL_REVISION=0451855729ec904f970600e0aec8b84661423afe
ARG QWEN_CONFIG_REVISION=ebb281ec70b05090aa6165b016eac8ec08e71b17
ARG ROBOTWIN_DATA_REVISION=a967b852afa21a9cbf19a198f7e653109042e87c

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        ca-certificates git git-lfs curl wget ffmpeg \
        libgl1 libglib2.0-0 libvulkan1 vulkan-tools mesa-vulkan-drivers \
        build-essential ninja-build cmake pkg-config && \
    git lfs install --system

COPY requirements.txt requirements-no-deps.txt /tmp/robotwin-requirements/

# Keep the image ROCm torch stack in /opt/venv read-only and install model-side
# dependencies into a separate venv.  A .pth file makes the ROCm wheels visible
# without allowing pip to replace them with CUDA/PyPI wheels.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [[ -n "${PIP_INDEX_URL}" ]]; then \
        /opt/venv/bin/python -m pip config set global.index-url "${PIP_INDEX_URL}"; \
    fi && \
    /opt/venv/bin/python -m venv --system-site-packages /opt/robotwin-env && \
    MODEL_SITE="$(/opt/robotwin-env/bin/python -c 'import site; print(site.getsitepackages()[0])')" && \
    printf '%s\n' \
        /opt/venv/lib/python3.12/site-packages \
        /opt/venv/local/lib/python3.12/dist-packages \
        /opt/venv/lib/python3/dist-packages \
        /opt/venv/lib/python3.12/dist-packages \
        > "${MODEL_SITE}/rocm_image_venv.pth" && \
    /opt/robotwin-env/bin/python -m pip install --upgrade 'pip<26' 'setuptools<81' wheel && \
    /opt/robotwin-env/bin/python -m pip install --prefer-binary \
        -r /tmp/robotwin-requirements/requirements.txt && \
    /opt/robotwin-env/bin/python -m pip install --no-deps \
        -r /tmp/robotwin-requirements/requirements-no-deps.txt

# LeRobot 0.6 dataset conversion requires NumPy 2 and huggingface-hub >= 1,
# which conflict with the model runtime.  Keep conversion fully isolated.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/venv/bin/python -m venv --system-site-packages /opt/lerobot-env && \
    DATA_SITE="$(/opt/lerobot-env/bin/python -c 'import site; print(site.getsitepackages()[0])')" && \
    printf '%s\n' \
        /opt/venv/lib/python3.12/site-packages \
        /opt/venv/local/lib/python3.12/dist-packages \
        /opt/venv/lib/python3/dist-packages \
        /opt/venv/lib/python3.12/dist-packages \
        > "${DATA_SITE}/rocm_image_venv.pth" && \
    /opt/lerobot-env/bin/python -m pip install --upgrade 'pip<26' 'setuptools<81' wheel && \
    /opt/lerobot-env/bin/python -m pip install --prefer-binary \
        'lerobot[dataset]==0.6.0' 'h5py==3.14.0'

RUN /opt/robotwin-env/bin/python - <<'PY'
import torch, torchvision, numpy, sapien, mplib, lerobot
assert torch.__file__.startswith('/opt/venv/'), torch.__file__
assert torch.version.hip is not None
assert numpy.__version__ == '1.26.4'
assert lerobot.__version__ == '0.6.0'
print('model environment:', torch.__version__, torch.version.hip)
PY

# RoboTwin imports Open3D from envs.utils.save_file at evaluator startup even
# when video/point-cloud logging is disabled. Python 3.12 wheels start at 0.19.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/robotwin-env/bin/python -m pip install --prefer-binary 'open3d==0.19.0'

ENV PYTHONPATH=/opt/aiter

COPY patches/flash-attention-no-gpu-build.patch /tmp/

# Build the tested AMD FlashAttention2 stack without allowing either project
# to replace the ROCm image's PyTorch-matched Triton package.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    git clone https://github.com/ROCm/aiter.git /opt/aiter && \
    git -C /opt/aiter checkout "${AITER_COMMIT}" && \
    cd /opt/aiter && \
    GPU_ARCHS="${AITER_GPU_ARCHS}" AITER_USE_SYSTEM_TRITON=1 \
        /opt/robotwin-env/bin/python setup.py develop && \
    git clone https://github.com/ZiguanWang/flash-attention.git /tmp/flash-attention && \
    git -C /tmp/flash-attention checkout "${FLASH_ATTENTION_COMMIT}" && \
    cd /tmp/flash-attention && \
    git apply /tmp/flash-attention-no-gpu-build.patch && \
    GPU_ARCHS="${AITER_GPU_ARCHS}" \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    FLASH_ATTENTION_USE_SYSTEM_AITER=TRUE \
        /opt/robotwin-env/bin/python -m pip install \
        --no-build-isolation --no-deps . && \
    cd / && rm -rf /tmp/flash-attention

RUN /opt/robotwin-env/bin/python - <<'PY'
from importlib.metadata import version

import triton

assert version("flash-attn") == "2.8.4"
print(
    "FlashAttention2:", version("flash-attn"),
    "AITER:", version("amd-aiter"),
    "Triton:", triton.__version__,
)
PY

# Bake the exact GitHub revisions and the ROCm/LeRobot compatibility patches
# exercised by this reproduction into the image.
COPY patches/ /tmp/robotwin-patches/
RUN git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git /RoboTwin && \
    git -C /RoboTwin checkout "${ROBOTWIN_COMMIT}" && \
    git -C /RoboTwin submodule update --init --recursive && \
    test "$(git -C /RoboTwin rev-parse HEAD)" = "${ROBOTWIN_COMMIT}" && \
    git -C /RoboTwin apply /tmp/robotwin-patches/robotwin-rocm.patch && \
    git -C /RoboTwin/XPolicyLab apply /tmp/robotwin-patches/xpolicylab-lerobot-v30.patch && \
    mkdir -p /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source && \
    git clone https://github.com/robbyant/lingbot-vla-v2.git \
        /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 && \
    git -C /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 \
        checkout "${LINGBOT_VLA_COMMIT}" && \
    test "$(git -C /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 rev-parse HEAD)" = "${LINGBOT_VLA_COMMIT}" && \
    git -C /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 \
        apply /tmp/robotwin-patches/lingbot-vla-v2-rocm.patch && \
    /opt/robotwin-env/bin/python -m pip install -e /RoboTwin/XPolicyLab && \
    /opt/robotwin-env/bin/python -m pip install --no-deps -e \
        /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 && \
    rm -rf /tmp/robotwin-patches

# RoboTwin's Git repository does not contain the simulation assets. Bake the
# objects, embodiments and background textures required by closed-loop eval.
RUN --mount=type=cache,id=robotwin-sim-assets,target=/tmp/robotwin-sim-assets,sharing=locked \
    /opt/robotwin-env/bin/huggingface-cli download TianxingChen/RoboTwin2.0 \
        --repo-type dataset \
        --revision "${ROBOTWIN_DATA_REVISION}" \
        --include background_texture.zip embodiments.zip objects.zip \
        --local-dir /tmp/robotwin-sim-assets && \
    cd /RoboTwin/assets && \
    /opt/robotwin-env/bin/python -c \
        'from zipfile import ZipFile; from pathlib import Path; root=Path("/tmp/robotwin-sim-assets"); [ZipFile(root / p).extractall(".") for p in ("background_texture.zip", "embodiments.zip", "objects.zip")]' && \
    cd /RoboTwin && \
    /opt/robotwin-env/bin/python scripts/update_embodiment_config_path.py && \
    test -f assets/objects/objaverse/list.json

# Download public model artifacts at fixed revisions. Model weights are kept in
# /RoboTwin so the local and Radeon Cloud layouts are identical.
RUN --mount=type=cache,id=robotwin-models,target=/tmp/robotwin-model-cache,sharing=locked \
    MODEL_ROOT=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models && \
    CACHE_ROOT=/tmp/robotwin-model-cache && \
    mkdir -p "${MODEL_ROOT}" && \
    /opt/robotwin-env/bin/huggingface-cli download \
        robbyant/lingbot-vla-v2-6b-robotwin \
        --revision "${LINGBOT_MODEL_REVISION}" \
        --local-dir "${CACHE_ROOT}/robbyant_lingbot-vla-v2-6b-robotwin" && \
    /opt/robotwin-env/bin/huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
        --revision "${QWEN_CONFIG_REVISION}" \
        --include '*.json' '*.txt' '*.jinja' merges.txt vocab.json \
        --local-dir "${CACHE_ROOT}/Qwen3-VL-4B-Instruct-config-tokenizer" && \
    cp -a "${CACHE_ROOT}/robbyant_lingbot-vla-v2-6b-robotwin" "${MODEL_ROOT}/" && \
    cp -a "${CACHE_ROOT}/Qwen3-VL-4B-Instruct-config-tokenizer" "${MODEL_ROOT}/" && \
    rm -rf "${MODEL_ROOT}/robbyant_lingbot-vla-v2-6b-robotwin/.cache" \
        "${MODEL_ROOT}/Qwen3-VL-4B-Instruct-config-tokenizer/.cache" && \
    test -f "${MODEL_ROOT}/robbyant_lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt/model.safetensors.index.json" && \
    test "$(find "${MODEL_ROOT}/robbyant_lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt" -name 'model-*.safetensors' | wc -l)" -eq 6

# Bake all 50 unpacked demo_clean tasks into the image. Source ZIPs live only
# in a BuildKit cache and are retained there for later rebuilds, so the 23.4 GB
# archive set can be reused without entering an image layer.
RUN --mount=type=cache,target=/tmp/robotwin-demo-zips,sharing=locked \
    cd /RoboTwin && \
    ROBOTWIN_DATA_ROOT=/RoboTwin/data \
    HF_ARCHIVE_CACHE=/tmp/robotwin-demo-zips \
    HF_REVISION="${ROBOTWIN_DATA_REVISION}" \
    HF_KEEP_ARCHIVES=1 \
    HF_MAX_WORKERS=8 \
    HF_EXTRACT_WORKERS=8 \
    PATH=/opt/lerobot-env/bin:${PATH} bash scripts/download_xpolicylab_data.sh && \
    test "$(find data/demo_clean -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 50 && \
    test -z "$(find data -type f -name '*.zip' -print -quit)"

# Each task is an independent LeRobot dataset. Parallel conversion is faster,
# and LingBot-VLA's multi-dataset loader consumes the generated 50-line list.
COPY full/convert_all_data.sh /usr/local/bin/convert-all-robotwin-data
RUN --mount=type=cache,id=robotwin-lerobot-v30,target=/tmp/robotwin-lerobot-cache,sharing=locked \
    chmod 0755 /usr/local/bin/convert-all-robotwin-data && \
    ROBOTWIN_DATA_REVISION="${ROBOTWIN_DATA_REVISION}" \
    CONVERSION_JOBS=8 /usr/local/bin/convert-all-robotwin-data

# Runtime scripts and training configs change frequently, so copy them only
# after all large immutable asset/model/dataset layers have been produced.
COPY assets/experiments/lingbot_vla_v2_6b_robotwin/ \
    /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/

# Interactive/remote-access conveniences are deliberately late in the file so
# changing them does not invalidate models, simulation assets, or datasets.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        openssh-server vim unzip && \
    rm -f /etc/ssh/ssh_host_*

RUN mkdir -p /workspace/runtime && \
    rm -rf /RoboTwin/eval_result && \
    ln -s /workspace/runtime/eval_result /RoboTwin/eval_result

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/venv/bin/python -m pip install --prefer-binary --upgrade 'jupyterlab==4.6.3' && \
    test "$(/opt/venv/bin/jupyter-lab --version)" = "4.6.3" && \
    test -d /opt/venv/share/jupyter/lab/static

COPY full/entrypoint.sh /usr/local/bin/robotwin-entrypoint
RUN chmod 0755 /usr/local/bin/robotwin-entrypoint

ENV LINGBOT_VLA_PYTHON=/opt/robotwin-env/bin/python

WORKDIR /RoboTwin
ENTRYPOINT ["/usr/local/bin/robotwin-entrypoint"]
CMD ["bash"]
