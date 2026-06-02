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
#   IMAGE, CONTAINER_NAME, HIP_ARCHITECTURES, UBUNTU_USER_ID, BUILD_OGA,
#   SKIP_LIT, FORCE_RECONFIGURE.

set -euo pipefail

# `pwd -P` resolves autofs symlinks (e.g. /home/$USER/hipdnn-ep ->
# /wrk/.../hipdnn-ep); container bind-mounts only know the real path.
DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_DIR="$(cd "$DOCKER_DIR/.." && pwd -P)"
WORKSPACE="$(cd "$SOURCE_DIR/.." && pwd -P)"

: "${IMAGE:=hipdnn-ep-build:llvm22-noble}"
: "${CONTAINER_NAME:=${USER}.hipdnn-ep.shell}"
: "${HIP_ARCHITECTURES:=gfx1151}"
# Single integer used for both the renumbered UID and GID of the default
# `ubuntu` user inside the image (parks the squatter so the entrypoint can
# claim HOST_UID/HOST_GID, typically 1000/1000). Default matches CI; only
# change if 60001 is already taken on your system. Forwarded as a Docker
# `--build-arg` in cmd_image; consumed by the ARG of the same name in
# docker/Dockerfile.
: "${UBUNTU_USER_ID:=60001}"
: "${BUILD_OGA:=0}"
# OGA_REF intentionally not defaulted; build.sh owns the pin. Unset on the
# host means "use build.sh default"; setting it overrides.
: "${SKIP_LIT:=0}"
: "${FORCE_RECONFIGURE:=0}"

host_uid=$(id -u)
host_gid=$(id -g)
host_user=$(id -un)
host_group=$(id -gn)

populate_common_args() {
    local -n _out=$1
    # bind-propagation=shared works around autofs/NFS edge cases.
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
    )
    if [ -n "${OGA_REF:-}" ]; then
        _out+=(-e "OGA_REF=$OGA_REF")
    fi
    if [ -n "${ONNXRUNTIME_PR_PATCHES:-}" ]; then
        _out+=(-e "ONNXRUNTIME_PR_PATCHES=$ONNXRUNTIME_PR_PATCHES")
    fi
    _out+=(
        -w "$SOURCE_DIR"
        --cap-add SYS_PTRACE
        -v "/dev/shm:/dev/shm"
        --network=host
    )
}

# GPU passthrough — `shell` only, since `build` is compile-only.
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
    echo "[host] docker build -t $IMAGE --build-arg UBUNTU_USER_ID=$UBUNTU_USER_ID $DOCKER_DIR"
    docker build -t "$IMAGE" --build-arg "UBUNTU_USER_ID=$UBUNTU_USER_ID" "$DOCKER_DIR"
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
