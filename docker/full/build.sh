#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
BASE_IMAGE="${BASE_IMAGE:-rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1}"
ROBOTWIN_COMMIT="${ROBOTWIN_COMMIT:-266f3aadf505a4f7fe9af0faa41a20f5f47cd123}"
LINGBOT_VLA_COMMIT="${LINGBOT_VLA_COMMIT:-951475ae1b1d87553e7dc47c97b53a3d695c0d13}"
LINGBOT_MODEL_REVISION="${LINGBOT_MODEL_REVISION:-0451855729ec904f970600e0aec8b84661423afe}"
QWEN_CONFIG_REVISION="${QWEN_CONFIG_REVISION:-ebb281ec70b05090aa6165b016eac8ec08e71b17}"
ROBOTWIN_DATA_REVISION="${ROBOTWIN_DATA_REVISION:-a967b852afa21a9cbf19a198f7e653109042e87c}"

build_args=(
  --file "${SCRIPT_DIR}/Dockerfile"
  --tag "${IMAGE_NAME}"
  --build-arg "BASE_IMAGE=${BASE_IMAGE}"
  --build-arg "ROBOTWIN_COMMIT=${ROBOTWIN_COMMIT}"
  --build-arg "LINGBOT_VLA_COMMIT=${LINGBOT_VLA_COMMIT}"
  --build-arg "LINGBOT_MODEL_REVISION=${LINGBOT_MODEL_REVISION}"
  --build-arg "QWEN_CONFIG_REVISION=${QWEN_CONFIG_REVISION}"
  --build-arg "ROBOTWIN_DATA_REVISION=${ROBOTWIN_DATA_REVISION}"
  --progress plain
)

if [[ -n "${PIP_INDEX_URL:-}" ]]; then
  build_args+=(--build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}")
fi

echo "Building ${IMAGE_NAME} from ${BASE_IMAGE}"
DOCKER_BUILDKIT=1 docker build "${build_args[@]}" "${DOCKER_ROOT}"
echo "Built ${IMAGE_NAME}"
echo "Verify with: docker run --rm --device=/dev/kfd --device=/dev/dri ${IMAGE_NAME} /opt/robotwin-env/bin/python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())'"
