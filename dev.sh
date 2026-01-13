#!/usr/bin/env bash
set -euo pipefail

# ==== 配置区 ====（可选本地 docker 辅助脚本；推荐首选 VSCode Dev Container）
IMAGE_NAME="eesizer-dev:local"
CONTAINER_NAME="eesizer-dev"
DOCKERFILE_PATH=".devcontainer/Dockerfile"
WORKDIR="/workspaces/EEsizer"

# 项目根目录 = 脚本所在目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage: $0 {build|run|rebuild} [command...]

  build      Build the Docker image (${IMAGE_NAME}) using .devcontainer/Dockerfile
  run        Run a dev container with current project mounted, default command is 'bash'
             You can also pass a command, e.g.:
               $0 run python -m pytest

  rebuild    Remove existing image (if any) and build again

Examples:
  $0 build
  $0 run
  $0 run python -m pytest
  $0 rebuild
EOF
}

check_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker command not found. Please install Docker Desktop or Docker CLI." >&2
    exit 1
  fi
}

build_image() {
  check_docker
  echo ">> Building image ${IMAGE_NAME} from ${DOCKERFILE_PATH} ..."

  local uid
  uid="$(id -u)"

  docker build \
    -f "${PROJECT_ROOT}/${DOCKERFILE_PATH}" \
    -t "${IMAGE_NAME}" \
    --build-arg USER_UID="${uid}" \
    "${PROJECT_ROOT}"

  echo ">> Image ${IMAGE_NAME} built successfully."
}


run_container() {
  check_docker

  # 如果镜像不存在，自动先 build 一次
  if [[ -z "$(docker images -q "${IMAGE_NAME}" 2>/dev/null)" ]]; then
    echo ">> Image ${IMAGE_NAME} not found. Building it first..."
    build_image
  fi

  echo ">> Running container ${CONTAINER_NAME} with ${IMAGE_NAME} ..."
  echo "   Mount: ${PROJECT_ROOT} -> ${WORKDIR}"

  # 有附加命令就执行，没有就进 bash
  if [[ $# -gt 0 ]]; then
    local user_cmd
    user_cmd="$(printf '%q ' "$@")"
    user_cmd="${user_cmd% }"
    docker run --rm -it \
      --name "${CONTAINER_NAME}" \
      -v "${PROJECT_ROOT}:${WORKDIR}" \
      -w "${WORKDIR}" \
      "${IMAGE_NAME}" bash -lc "python -c 'import eesizer_core' >/dev/null 2>&1 || pip install -e '.[dev]'; ${user_cmd}"
  else
    docker run --rm -it \
      --name "${CONTAINER_NAME}" \
      -v "${PROJECT_ROOT}:${WORKDIR}" \
      -w "${WORKDIR}" \
      "${IMAGE_NAME}" bash
  fi
}

rebuild_image() {
  check_docker
  echo ">> Rebuilding image ${IMAGE_NAME} ..."
  # 尝试删旧镜像（如果存在）
  if [[ -n "$(docker images -q "${IMAGE_NAME}" 2>/dev/null)" ]]; then
    docker image rm "${IMAGE_NAME}" || true
  fi
  build_image
}

# ==== 主逻辑 ====
cmd="${1:-}"

case "${cmd}" in
  build)
    shift
    build_image
    ;;
  run)
    shift
    run_container "$@"
    ;;
  rebuild)
    shift
    rebuild_image
    ;;
  ""|-h|--help|help)
    usage
    ;;
  *)
    echo "Error: unknown command '${cmd}'" >&2
    usage
    exit 1
    ;;
esac
