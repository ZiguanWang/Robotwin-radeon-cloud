#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1-external-data}"
BASE_IMAGE="${BASE_IMAGE:-rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
ROBOTWIN_COMMIT="${ROBOTWIN_COMMIT:-266f3aadf505a4f7fe9af0faa41a20f5f47cd123}"
LINGBOT_VLA_COMMIT="${LINGBOT_VLA_COMMIT:-951475ae1b1d87553e7dc47c97b53a3d695c0d13}"

build_args=(
  --file "${SCRIPT_DIR}/Dockerfile"
  --tag "${IMAGE_NAME}"
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
  --build-arg "ROBOTWIN_COMMIT=${ROBOTWIN_COMMIT}"
  --build-arg "LINGBOT_VLA_COMMIT=${LINGBOT_VLA_COMMIT}"
  --progress plain
)

if [[ -n "${PIP_INDEX_URL:-}" ]]; then
  build_args+=(--build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}")
fi

echo "Building ${IMAGE_NAME} from ${BASE_IMAGE}"
DOCKER_BUILDKIT=1 docker build "${build_args[@]}" "${REPOSITORY_ROOT}"
echo "Built ${IMAGE_NAME}"
