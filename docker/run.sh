#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Host-side wrapper around the hipdnn-ep build container.
#
# Subcommands:
#   image   Build/refresh the Docker image (idempotent — layers cache).
#   build   One-shot full build inside a --rm container (no GPU attached).
#   shell   Open an interactive shell in a long-lived container with
#           /dev/kfd + /dev/dri/renderD* attached. Re-execs into an
#           already-running or stopped container of the same name.
#   stop    Stop + rm the long-lived shell container.
#
# Env knobs (override on the command line, e.g. `BUILD_OGA=1 ./run.sh build`):
#   IMAGE, CONTAINER_NAME, HIP_ARCHITECTURES, BUILD_OGA, SKIP_LIT,
#   FORCE_RECONFIGURE.

set -euo pipefail

# Path resolution. This script lives at <SOURCE_DIR>/docker/run.sh, so:
#   DOCKER_DIR = <repo>/docker
#   SOURCE_DIR = <repo>            (one level up — the onnx-hipdnn-ep checkout)
#   WORKSPACE  = <repo>/..         (two levels up — sibling-layout root)
# Use `pwd -P` to resolve symlinks (e.g. /home/$USER/hipdnn-ep ->
# /wrk/.../hipdnn-ep on autofs setups). Otherwise env vars handed to the
# container would name the symlinked path, which doesn't exist inside the
# container — only the bind-mount target path does.
#
# Workspace layout (sibling-of-source — mirrors Windows quick_start.md
# "Directory Layout" so the same cmake/curl invocations work on both OSes):
#   $WORKSPACE/onnx-hipdnn-ep/     source (= SOURCE_DIR)
#   $WORKSPACE/onnxruntime/        ORT source
#   $WORKSPACE/onnxruntime-genai/  OGA source (BUILD_OGA=1 only)
#   $WORKSPACE/prebuilt-local/     built deps (ORT, protobuf, flatbuffers)
#   $WORKSPACE/therock-dist/       TheRock SDK (~13 GB)
#   $WORKSPACE/build/              cmake build dirs (out-of-source)
#   $WORKSPACE/install/            cmake install prefix
DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_DIR="$(cd "$DOCKER_DIR/.." && pwd -P)"
WORKSPACE="$(cd "$SOURCE_DIR/.." && pwd -P)"

: "${IMAGE:=hipdnn-ep-build:llvm22-noble}"
: "${CONTAINER_NAME:=${USER}.hipdnn-ep.shell}"
: "${HIP_ARCHITECTURES:=gfx1151}"
: "${BUILD_OGA:=0}"
: "${SKIP_LIT:=0}"
: "${FORCE_RECONFIGURE:=0}"

host_uid=$(id -u)
host_gid=$(id -g)
host_user=$(id -un)
host_group=$(id -gn)

populate_common_args() {
    # $1 = name of array variable to append into.
    local -n _out=$1
    # Workspace bind-mount (bind-propagation=shared works around autofs/NFS
    # edge cases on some host filesystems).
    _out+=(
        --mount "type=bind,source=$WORKSPACE,target=$WORKSPACE,bind-propagation=shared"
        -e "HOST_UID=$host_uid"
        -e "HOST_GID=$host_gid"
        -e "HOST_USER=$host_user"
        -e "HOST_GROUP=$host_group"
        -e "HOST_HOME=$WORKSPACE/.docker-home"
        -e "SOURCE_DIR=$SOURCE_DIR"
        -e "HIP_ARCHITECTURES=$HIP_ARCHITECTURES"
        -e "BUILD_OGA=$BUILD_OGA"
        -e "SKIP_LIT=$SKIP_LIT"
        -e "FORCE_RECONFIGURE=$FORCE_RECONFIGURE"
        -w "$SOURCE_DIR"
        --cap-add SYS_PTRACE
        -v "/dev/shm:/dev/shm"
        --network=host
    )
    # Forward sccache + GitHub Actions cache env vars when present. The
    # mozilla-actions/sccache-action step in CI sets these on the host runner;
    # the in-container sccache binary (installed via the Dockerfile) needs
    # them to talk to the same GHA cache backend.
    local var
    for var in SCCACHE_GHA_ENABLED ACTIONS_CACHE_URL ACTIONS_RUNTIME_TOKEN; do
        if [ -n "${!var:-}" ]; then
            _out+=(-e "$var=${!var}")
        fi
    done
}

# Best-effort GPU passthrough. Only attach for `shell` — `build` doesn't need
# a GPU (compile-only; the GPU is only used by `ctest` Execute tests and
# downstream tools like hip-onnx-runner).
populate_gpu_args() {
    local -n _out=$1
    [ -e /dev/kfd ] && _out+=(--device=/dev/kfd)
    if [ -d /dev/dri ]; then
        local d
        for d in /dev/dri/renderD*; do
            [ -e "$d" ] && _out+=(--device="$d")
        done
    fi
}

cmd_image() {
    echo "[host] docker build -t $IMAGE $DOCKER_DIR"
    docker build -t "$IMAGE" "$DOCKER_DIR"
}

cmd_build() {
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        cmd_image
    fi
    mkdir -p "$WORKSPACE/.docker-home"
    local -a docker_args=()
    populate_common_args docker_args
    set -x
    docker run --rm -t \
        "${docker_args[@]}" \
        "$IMAGE" \
        bash "$DOCKER_DIR/build.sh"
}

cmd_shell() {
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        cmd_image
    fi
    mkdir -p "$WORKSPACE/.docker-home"

    local state
    state=$(docker container inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "")
    if [ "$state" = "running" ]; then
        echo "[host] $CONTAINER_NAME already running — exec into it"
        docker exec -it "$CONTAINER_NAME" gosu "$host_user" bash -l
        return $?
    fi
    if [ "$state" = "exited" ]; then
        echo "[host] $CONTAINER_NAME exists but stopped — starting"
        docker start "$CONTAINER_NAME" >/dev/null
        docker exec -it "$CONTAINER_NAME" gosu "$host_user" bash -l
        return $?
    fi

    local -a docker_args=()
    populate_common_args docker_args
    populate_gpu_args docker_args
    set -x
    docker run -dit \
        --name "$CONTAINER_NAME" \
        "${docker_args[@]}" \
        "$IMAGE" \
        bash
    set +x
    docker exec -it "$CONTAINER_NAME" gosu "$host_user" bash -l
}

cmd_stop() {
    local state
    state=$(docker container inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "")
    if [ -z "$state" ]; then
        echo "[host] no container named $CONTAINER_NAME"
        return 0
    fi
    docker rm -f "$CONTAINER_NAME"
}

cmd_help() {
    sed -n '2,14p' "$0"
}

case "${1:-help}" in
    image)          cmd_image ;;
    build)          cmd_build ;;
    shell)          cmd_shell ;;
    stop)           cmd_stop ;;
    -h|--help|help) cmd_help ;;
    *)
        echo "unknown subcommand: $1" >&2
        cmd_help
        exit 1
        ;;
esac
