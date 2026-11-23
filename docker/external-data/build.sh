#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
BASE_IMAGE="${BASE_IMAGE:-rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
ROBOTWIN_COMMIT="${ROBOTWIN_COMMIT:-e154702de28cac4b9bfe0299044303f507236bb3}"
XPOLICYLAB_COMMIT="${XPOLICYLAB_COMMIT:-c37109c500be67d0dea6b36bf7337bbd26e763cd}"
LINGBOT_VLA_COMMIT="${LINGBOT_VLA_COMMIT:-951475ae1b1d87553e7dc47c97b53a3d695c0d13}"
AITER_COMMIT="${AITER_COMMIT:-9bab8388c35936814a659b4ebd245c491e1b940a}"
FLASH_ATTENTION_COMMIT="${FLASH_ATTENTION_COMMIT:-9c6d9297e5d43653c2ea5837ed9e0d10f76ed231}"
AITER_GPU_ARCHS="${AITER_GPU_ARCHS:-gfx1100;gfx1201}"

build_args=(
  --file "${SCRIPT_DIR}/Dockerfile"
  --tag "${IMAGE_NAME}"
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
  --build-arg "ROBOTWIN_COMMIT=${ROBOTWIN_COMMIT}"
  --build-arg "XPOLICYLAB_COMMIT=${XPOLICYLAB_COMMIT}"
  --build-arg "LINGBOT_VLA_COMMIT=${LINGBOT_VLA_COMMIT}"
  --build-arg "AITER_COMMIT=${AITER_COMMIT}"
  --build-arg "FLASH_ATTENTION_COMMIT=${FLASH_ATTENTION_COMMIT}"
  --build-arg "AITER_GPU_ARCHS=${AITER_GPU_ARCHS}"
  --progress plain
)

if [[ -n "${PIP_INDEX_URL:-}" ]]; then
  build_args+=(--build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}")
fi

echo "Building ${IMAGE_NAME} from ${BASE_IMAGE}"
DOCKER_BUILDKIT=1 docker build "${build_args[@]}" "${REPOSITORY_ROOT}"
echo "Built ${IMAGE_NAME}"
