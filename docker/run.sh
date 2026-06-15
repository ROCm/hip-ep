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
# Env knobs (override on the command line, e.g. `HIP_ARCHITECTURES=gfx942 ./run.sh build`):
#   IMAGE, CONTAINER_NAME, HIP_ARCHITECTURES.
#
# `build` runs the native driver build.py inside the container (docker
# is just an isolation wrapper). OGA + ONNX-Runtime-from-source patching are CI
# concerns (see .github/workflows/linux-build.yml), not local docker knobs.

set -euo pipefail

# `pwd -P` resolves autofs symlinks (e.g. /home/$USER/hipdnn-ep ->
# /wrk/.../hipdnn-ep); container bind-mounts only know the real path.
DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_DIR="$(cd "$DOCKER_DIR/.." && pwd -P)"
WORKSPACE="$(cd "$SOURCE_DIR/.." && pwd -P)"

# Auto-detect the host GPU arch from the amdgpu kernel driver's sysfs topology
# (gfx_target_version per KFD node), so a developer with a different GPU need
# not pass HIP_ARCHITECTURES. This is host-side on purpose: the compile-only
# `build` container has no /dev/kfd, so the GPU node's properties are
# unreadable inside it ("Operation not permitted"); and the ROCm/TheRock arch
# tools aren't available before the SDK is downloaded. Needs only the amdgpu
# kernel module (no ROCm userland). gfx_target_version encodes the arch as
# major*10000 + minor*100 + step (e.g. 110501 -> gfx1151, 90010 -> gfx90a);
# the CPU node reports 0 and is skipped.
detect_hip_arch() {
    local props v major minor step
    for props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        [ -r "$props" ] || continue
        v=$(awk '/^gfx_target_version /{print $2; exit}' "$props" 2>/dev/null) || continue
        [ -n "$v" ] && [ "$v" != "0" ] || continue
        major=$((v / 10000)); minor=$(((v / 100) % 100)); step=$((v % 100))
        printf 'gfx%d%x%x\n' "$major" "$minor" "$step"
        return 0
    done
    return 1
}

: "${IMAGE:=hipdnn-ep-build:llvm22-noble}"
: "${CONTAINER_NAME:=${USER}.hipdnn-ep.shell}"
if [ -z "${HIP_ARCHITECTURES:-}" ]; then
    if HIP_ARCHITECTURES="$(detect_hip_arch)"; then
        echo "[host] auto-detected HIP_ARCHITECTURES=$HIP_ARCHITECTURES (from /sys/class/kfd)"
    else
        HIP_ARCHITECTURES=gfx1151
        echo "[host] no AMD GPU detected in /sys/class/kfd; defaulting HIP_ARCHITECTURES=$HIP_ARCHITECTURES"
    fi
fi

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
    # docker is now just an isolation wrapper around the same native driver
    # build.py (no second build recipe). HIP_ARCHITECTURES is detected
    # host-side above and passed in (the compile-only container has no GPU node
    # to auto-detect from). LLVM/ORT/protobuf/flatbuffers/TheRock are resolved
    # by cmake/deps.cmake; OGA is a CI concern and is not built here.
    docker run --rm -t \
        "${docker_args[@]}" \
        "$IMAGE" \
        python3 "$SOURCE_DIR/build.py" --config Release --hip_arch "$HIP_ARCHITECTURES"
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
