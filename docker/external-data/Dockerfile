# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="RoboTwin LingBot-VLA-v2 ROCm external-data runtime" \
      org.opencontainers.image.description="ROCm runtime using an external mount for RoboTwin assets, datasets, and models"

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
    QWEN3VL_PATH=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/Qwen3-VL-4B-Instruct-config-tokenizer \
    LINGBOT_VLA_PYTHON=/opt/robotwin-env/bin/python

ARG PIP_INDEX_URL=
ARG ROBOTWIN_COMMIT=266f3aadf505a4f7fe9af0faa41a20f5f47cd123
ARG LINGBOT_VLA_COMMIT=951475ae1b1d87553e7dc47c97b53a3d695c0d13
ARG AITER_COMMIT=9bab8388c35936814a659b4ebd245c491e1b940a
ARG FLASH_ATTENTION_COMMIT=bc76302fbb24c0158207978930db030ca1eca5ca
ARG AITER_GPU_ARCHS=gfx1100;gfx1201

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        ca-certificates git git-lfs curl wget ffmpeg \
        libgl1 libglib2.0-0 libvulkan1 vulkan-tools mesa-vulkan-drivers \
        build-essential ninja-build cmake pkg-config && \
    git lfs install --system

COPY docker/requirements.txt docker/requirements-no-deps.txt /tmp/robotwin-requirements/

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

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/robotwin-env/bin/python -m pip install --prefer-binary 'open3d==0.19.0'

ENV PYTHONPATH=/opt/aiter

COPY docker/patches/flash-attention-no-gpu-build.patch /tmp/

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

COPY docker/patches/ /tmp/robotwin-patches/
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

COPY docker/assets/experiments/lingbot_vla_v2_6b_robotwin/ \
    /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        openssh-server vim unzip && \
    rm -f /etc/ssh/ssh_host_*

RUN rm -rf \
        /RoboTwin/assets \
        /RoboTwin/data \
        /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models && \
    mkdir -p \
        /models/robotwin-persistent \
        /workspace/runtime \
        /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin && \
    ln -s /models/robotwin-persistent/assets /RoboTwin/assets && \
    ln -s /models/robotwin-persistent/data /RoboTwin/data && \
    ln -s /models/robotwin-persistent/models \
        /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    /opt/venv/bin/python -m pip install --prefer-binary --upgrade 'jupyterlab==4.6.3' && \
    test "$(/opt/venv/bin/jupyter-lab --version)" = "4.6.3" && \
    test -d /opt/venv/share/jupyter/lab/static

COPY docker/external-data/entrypoint.sh /usr/local/bin/robotwin-external-entrypoint
COPY RoboTwin_ROCm_Reproduction.ipynb /RoboTwin/RoboTwin_ROCm_Reproduction.ipynb
RUN chmod 0755 /usr/local/bin/robotwin-external-entrypoint

WORKDIR /RoboTwin
ENTRYPOINT ["/usr/local/bin/robotwin-external-entrypoint"]
CMD ["bash"]
