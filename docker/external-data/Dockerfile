# syntax=docker/dockerfile:1.7
# Build the ROCm runtime; simulation assets, datasets, and model weights are mounted later.
ARG BASE_IMAGE=rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="RoboTwin LingBot-VLA-v2 ROCm runtime" \
      org.opencontainers.image.description="ROCm runtime with mounted RoboTwin assets, datasets, and models"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Runtime paths and ROCm settings shared by shells, notebooks, and launchers.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/workspace/runtime/.cache/huggingface \
    TMPDIR=/workspace/runtime/tmp \
    HF_LEROBOT_HOME=/RoboTwin/data/lerobot \
    PYOPENGL_PLATFORM=egl \
    ROBOTWIN_DISABLE_CUROBO=1 \
    ROBOTWIN_EE_PLANNER=mplib \
    TOKENIZERS_PARALLELISM=false \
    AITER_TRITON_ONLY=1 \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    ROBOTWIN_ROOT=/RoboTwin \
    LINGBOT_VLA_SOURCE=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 \
    QWEN3VL_PATH=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/Qwen3-VL-4B-Instruct-config-tokenizer \
    LINGBOT_VLA_PYTHON=/opt/robotwin-env/bin/python \
    PATH=/opt/robotwin-env/bin:${PATH} \
    PYTHONPATH=/opt/aiter

# Pin the upstream sources and target GPU architectures used in this image.
ARG PIP_INDEX_URL=
ARG ROBOTWIN_COMMIT=e154702de28cac4b9bfe0299044303f507236bb3
ARG XPOLICYLAB_COMMIT=c37109c500be67d0dea6b36bf7337bbd26e763cd
ARG LINGBOT_VLA_COMMIT=951475ae1b1d87553e7dc47c97b53a3d695c0d13
ARG AITER_COMMIT=9bab8388c35936814a659b4ebd245c491e1b940a
ARG FLASH_ATTENTION_COMMIT=9c6d9297e5d43653c2ea5837ed9e0d10f76ed231
ARG AITER_GPU_ARCHS=gfx1100;gfx1201

# Create writable runtime paths and install OS libraries needed by simulation,
# video, source builds, notebooks, and remote access. The base image already
# provides Python 3.12 headers. Do not bake shared SSH host keys into the image.
RUN mkdir -p /workspace/runtime/tmp /workspace/runtime/.cache/huggingface

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        ca-certificates git git-lfs curl wget ffmpeg \
        libgl1 libglib2.0-0 libvulkan1 vulkan-tools mesa-vulkan-drivers \
        build-essential ninja-build cmake pkg-config \
        openssh-server vim unzip && \
    git lfs install --system && \
    rm -f /etc/ssh/ssh_host_*

# Install the verified model, evaluator, and Jupyter package set into one venv
# without replacing the base image's ROCm PyTorch or Triton.
COPY docker/requirements.txt /tmp/robotwin-requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    if [[ -n "${PIP_INDEX_URL}" ]]; then \
        /opt/venv/bin/python -m pip config set global.index-url "${PIP_INDEX_URL}"; \
    fi && \
    /opt/venv/bin/python -m venv --system-site-packages /opt/robotwin-env && \
    MODEL_SITE="$(/opt/robotwin-env/bin/python -c 'import site; print(site.getsitepackages()[0])')" && \
    BASE_SITE="$(/opt/venv/bin/python -c 'import site; print(site.getsitepackages()[0])')" && \
    printf '%s\n' "${BASE_SITE}" > "${MODEL_SITE}/rocm_image_venv.pth" && \
    awk '/^(pip|setuptools|wheel)==/{print}' \
        /tmp/robotwin-requirements.txt > /tmp/robotwin-build-tools.txt && \
    /opt/robotwin-env/bin/python -m pip install --no-deps \
        -r /tmp/robotwin-build-tools.txt && \
    /opt/robotwin-env/bin/python -m pip install --no-deps --no-build-isolation \
        --prefer-binary -r /tmp/robotwin-requirements.txt

# Check ROCm, the LeRobot reader, and Jupyter before copying project code.
RUN /opt/robotwin-env/bin/python - <<'PY'
import torch, torchvision, numpy, sapien, mplib, lerobot, jupyterlab, tornado
import pipeline, utils3d
assert torch.__file__.startswith('/opt/venv/'), torch.__file__
assert torch.version.hip is not None
print('model environment:', torch.__version__, torch.version.hip)
PY

# Build AITER against the base Triton and install the pinned upstream AMD
# FlashAttention2 implementation using the already-installed AITER kernels.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    git init /opt/aiter && \
    git -C /opt/aiter remote add origin https://github.com/ROCm/aiter.git && \
    git -C /opt/aiter fetch --depth 1 origin "${AITER_COMMIT}" && \
    git -C /opt/aiter checkout FETCH_HEAD && \
    cd /opt/aiter && \
    sed -i '/flydsl==0.1.9.dev599/d' pyproject.toml && \
    GPU_ARCHS="${AITER_GPU_ARCHS}" AITER_USE_SYSTEM_TRITON=1 \
        /opt/robotwin-env/bin/python setup.py develop && \
    git init /tmp/flash-attention && \
    git -C /tmp/flash-attention remote add origin https://github.com/Dao-AILab/flash-attention.git && \
    git -C /tmp/flash-attention fetch --depth 1 origin "${FLASH_ATTENTION_COMMIT}" && \
    git -C /tmp/flash-attention checkout FETCH_HEAD && \
    cd /tmp/flash-attention && \
    AITER_TRITON_ONLY=1 \
    GPU_ARCHS="${AITER_GPU_ARCHS}" \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    FLASH_ATTENTION_USE_SYSTEM_AITER=TRUE \
        /opt/robotwin-env/bin/python -m pip install \
        --no-build-isolation --no-deps . && \
    cd / && rm -rf /tmp/flash-attention

# Fail during the build if the pinned attention packages cannot be imported.
RUN /opt/robotwin-env/bin/python - <<'PY'
from importlib.metadata import version

import aiter
import flash_attn
import triton

assert version("flash-attn") == "2.8.4"
print(
    "FlashAttention2:", version("flash-attn"),
    "AITER:", version("amd-aiter"),
    "Triton:", triton.__version__,
)
PY

# Clone the pinned RoboTwin, XPolicyLab, and LingBot source trees.
RUN git clone --no-checkout https://github.com/RoboTwin-Platform/RoboTwin.git /RoboTwin && \
    git -C /RoboTwin checkout "${ROBOTWIN_COMMIT}" && \
    git -C /RoboTwin submodule update --init --recursive && \
    test "$(git -C /RoboTwin rev-parse HEAD)" = "${ROBOTWIN_COMMIT}" && \
    test "$(git -C /RoboTwin/XPolicyLab rev-parse HEAD)" = "${XPOLICYLAB_COMMIT}" && \
    mkdir -p /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source && \
    git clone https://github.com/robbyant/lingbot-vla-v2.git \
        /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 && \
    git -C /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 \
        checkout "${LINGBOT_VLA_COMMIT}" && \
    test "$(git -C /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 rev-parse HEAD)" = "${LINGBOT_VLA_COMMIT}"

# Apply compatibility patches directly after checkout so editable installs
# discover the final source. Dependency and attention layers remain cached.
COPY patches/ /tmp/robotwin-patches/
RUN git -C /RoboTwin apply /tmp/robotwin-patches/robotwin-rocm-reproduction.patch && \
    git -C /RoboTwin/XPolicyLab apply /tmp/robotwin-patches/xpolicylab-lerobot-v30.patch && \
    git -C /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 \
        apply /tmp/robotwin-patches/lingbot-vla-v2-rocm.patch

RUN /opt/robotwin-env/bin/python -m pip install --no-deps --no-build-isolation \
        -e /RoboTwin/XPolicyLab \
        -e /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 && \
    /opt/robotwin-env/bin/python -m py_compile /RoboTwin/scripts/eval_policy_xpolicylab.py && \
    test -f /RoboTwin/XPolicyLab/XPolicyLab.py && \
    rm -rf /tmp/robotwin-patches

# Keep project scripts in one purpose-based tree. Copy them after dependency
# installation so edits to a script reuse the expensive build layers.
COPY scripts/ /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/
COPY configs/lingbot_vla_v2_6b_robotwin/lingbotvla_cli.yaml \
    /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/training/lingbotvla_cli.yaml

# Replace RoboTwin's asset and data directories with links to the mounted
# /models/robotwin-persistent tree; keep evaluation outputs under runtime.
RUN rm -rf \
        /RoboTwin/assets \
        /RoboTwin/data \
        /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models \
        /RoboTwin/eval_result && \
    mkdir -p \
        /models/robotwin-persistent \
        /workspace/runtime \
        /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin && \
    ln -s /models/robotwin-persistent/assets /RoboTwin/assets && \
    ln -s /models/robotwin-persistent/data /RoboTwin/data && \
    ln -s /models/robotwin-persistent/models \
        /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models && \
    ln -s /workspace/runtime/eval_result /RoboTwin/eval_result

# Make runtime settings available to login and non-interactive shells. Copy
# this late so editing the environment script does not rebuild Python packages.
COPY docker/robotwin-env.sh /usr/local/bin/robotwin-env
RUN chmod 0755 /usr/local/bin/robotwin-env && \
    ln -sf /usr/local/bin/robotwin-env /etc/profile.d/robotwin-env.sh

# Start through an entrypoint that checks the external mount and initializes
# writable runtime folders before launching the requested command.
COPY docker/external-data/entrypoint.sh /usr/local/bin/robotwin-external-entrypoint
RUN chmod 0755 /usr/local/bin/robotwin-external-entrypoint

# Add interactive notebooks last so notebook edits reuse all runtime layers.
COPY notebooks/ /RoboTwin/notebooks/

WORKDIR /RoboTwin
ENTRYPOINT ["/usr/local/bin/robotwin-external-entrypoint"]
CMD ["bash"]
