#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Wrapper that runs a bench script inside the hipdnn-ep build container
# without needing an interactive shell. Use this from any host terminal that
# doesn't deliver a clean TTY to `docker exec -it` -- e.g. Cursor's embedded
# terminal, where `./docker/run.sh shell` SIGKILLs the docker-exec step and
# leaves zombie containers behind.
#
# This is the standalone copy that lives under tools/perf-report/. It is
# location-independent: the repo root is found by walking up from the
# script's directory looking for build.py (the canonical project build
# script per CLAUDE.md), so the same script works whether it sits at
# <repo>/oga/, <repo>/tools/perf-report/, or anywhere else inside the
# repo.
#
# Differences from `./docker/run.sh shell`:
#   - One-shot container: `docker run --rm` (no --name, no persistent state,
#     no zombies on Ctrl+C or TTY drop)
#   - No --interactive / --tty flags; streams stdout/stderr only
#   - Bypasses docker/entrypoint.sh entirely. The entrypoint has a latent bug
#     that surfaces when the host's render-device GID (gid of /dev/kfd, often
#     993 = `render` group on the host) does not exist as a group inside the
#     container image: `grp_name=$(getent group "$dev_gid" | cut -d: -f1)`
#     fails under `set -euo pipefail` (getent exits 2 -> pipefail propagates
#     -> errexit kills the script) and the container dies silently with exit
#     2 before the bench ever runs. The wrapper avoids this with --user +
#     --group-add: we run directly as the host UID/GID and add the device's
#     GID as a supplemental group via docker, no useradd / usermod needed.
#     Side benefit: anything the bench writes into the bind-mounted workspace
#     (e.g. tools/perf-report/_perf_logs/*.log) is already owned by you on
#     the host.
#
# Usage (all args forwarded to the in-container bench script):
#   tools/perf-report/docker_run_bench.sh                              # defaults
#   tools/perf-report/docker_run_bench.sh --oga --prompt 128 --gen 128 --reps 3 --warmup 1
#   tools/perf-report/docker_run_bench.sh --shape dynamic --seq 1 --time 30
#   tools/perf-report/docker_run_bench.sh --help                       # forwarded
#
# Env knobs (override on the command line):
#   WORKSPACE   parent dir bind-mounted into the container (default: auto-detected
#               as parent of the repo root, matching docker/run.sh's convention)
#   IMAGE       docker image tag (default: hipdnn-ep-build:llvm22-noble)
#   REL_BENCH   path of the bench script to exec inside the container, relative
#               to the repo root (default: tools/perf-report/run_bench.sh).
#               Override to invoke against a different bench entry point, e.g.
#               REL_BENCH=oga/run_bench.sh tools/perf-report/docker_run_bench.sh ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate the repo root by walking upward from SCRIPT_DIR looking for
# `build.py` -- the canonical project build script per CLAUDE.md. This
# decouples the wrapper from any specific placement under the repo (it
# works from oga/, tools/perf-report/, or anywhere else). We deliberately
# don't shell out to `git rev-parse --show-toplevel` so the script remains
# usable inside extracted source tarballs that have no .git/ directory.
# `build.py` is preferred over `CMakeLists.txt` as a marker because every
# sub-project under tools/ and 3rd-party/ has its own CMakeLists.txt, but
# build.py exists only at the repo root.
SOURCE_DIR="$SCRIPT_DIR"
while [ "$SOURCE_DIR" != "/" ] && [ ! -f "$SOURCE_DIR/build.py" ]; do
    SOURCE_DIR="$(dirname "$SOURCE_DIR")"
done
if [ "$SOURCE_DIR" = "/" ]; then
    echo "ERROR: could not locate repo root from $SCRIPT_DIR" >&2
    echo "       (walked up looking for build.py; none found)" >&2
    exit 1
fi

: "${WORKSPACE:=$(cd "$SOURCE_DIR/.." && pwd)}"
: "${IMAGE:=hipdnn-ep-build:llvm22-noble}"
: "${REL_BENCH:=tools/perf-report/run_bench.sh}"   # path inside the repo to invoke

# Fail fast if the bench script doesn't exist on disk. Without this check
# the container would still start, bash would fail to open the path, and
# the user would see only `bash: <REL_BENCH>: No such file or directory`
# with no hint that REL_BENCH was misset (or that the default script was
# deleted / moved). The default targets the tracked sibling bench
# (tools/perf-report/run_bench.sh) so a fresh clone has a working
# entry point without any local setup.
if [ ! -f "$SOURCE_DIR/$REL_BENCH" ]; then
    echo "ERROR: bench script not found at \$SOURCE_DIR/\$REL_BENCH" >&2
    echo "       \$SOURCE_DIR = $SOURCE_DIR" >&2
    echo "       \$REL_BENCH  = $REL_BENCH" >&2
    echo "       Set REL_BENCH to a bench script that exists under the repo" >&2
    echo "       (default: tools/perf-report/run_bench.sh)." >&2
    exit 1
fi

# Reap any leftover named container from a failed `./docker/run.sh shell`
# attempt. Idempotent + silent if no such container exists. Cursor's embedded
# terminal regularly leaves the persistent "${USER}.hipdnn-ep.shell" container
# in "Exited" state when the docker-exec gets SIGKILL'd; subsequent
# `./docker/run.sh shell` then errors with "container is not running". This
# wrapper doesn't use a named container (--rm), but reaping keeps the docker
# state clean for when you do drop into a real terminal later.
ZOMBIE_NAME="${USER}.hipdnn-ep.shell"
if docker container inspect -f '{{.State.Status}}' "$ZOMBIE_NAME" >/dev/null 2>&1; then
    state=$(docker container inspect -f '{{.State.Status}}' "$ZOMBIE_NAME")
    if [ "$state" != "running" ]; then
        echo "[reap] removing stale '$ZOMBIE_NAME' (status: $state)"
        docker rm -f "$ZOMBIE_NAME" >/dev/null
    fi
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: docker image '$IMAGE' not found." >&2
    echo "       Build it first: ./docker/run.sh image" >&2
    exit 1
fi

# Resolve the render-device GIDs on the host so we can add them as docker
# supplemental groups. Two GIDs matter: /dev/kfd (the AMDGPU compute
# interface) and /dev/dri/renderD128 (the render-node DRM interface used
# for buffer management). They usually agree (both belong to `render`),
# but a misconfigured host can split them; we add both as supplemental
# groups in that case so the in-container user can open either.
GPU_MISSING=0
if [ ! -e /dev/kfd ]; then
    echo "ERROR: /dev/kfd not found on host" >&2
    GPU_MISSING=1
fi
if [ ! -e /dev/dri/renderD128 ]; then
    echo "ERROR: /dev/dri/renderD128 not found on host" >&2
    GPU_MISSING=1
fi
if [ "$GPU_MISSING" = "1" ]; then
    echo "       This wrapper requires an AMD GPU exposed via the AMDGPU driver." >&2
    echo "       Check 'lsmod | grep amdgpu' and 'ls /dev/kfd /dev/dri/' on host." >&2
    exit 1
fi
KFD_GID=$(stat -c '%g' /dev/kfd)
DRI_GID=$(stat -c '%g' /dev/dri/renderD128)
if [ "$KFD_GID" != "$DRI_GID" ]; then
    echo "WARNING: /dev/kfd gid=$KFD_GID differs from /dev/dri/renderD128 gid=$DRI_GID" >&2
    echo "         Adding both as supplemental groups."                                  >&2
fi

# Build the docker args incrementally so the --group-add list is clean.
DOCKER_ARGS=(
    --rm
    --mount "type=bind,source=$WORKSPACE,target=$WORKSPACE,bind-propagation=shared"
    -w "$SOURCE_DIR"
    --user "$(id -u):$(id -g)"
    --group-add "$KFD_GID"
)
if [ "$KFD_GID" != "$DRI_GID" ]; then
    DOCKER_ARGS+=(--group-add "$DRI_GID")
fi
DOCKER_ARGS+=(
    --device=/dev/kfd --device=/dev/dri/renderD128
    --cap-add SYS_PTRACE
    -v /dev/shm:/dev/shm
    --network=host
    # Override HOME so anything the bench writes via $HOME (sccache cfg,
    # Python __pycache__) goes into a writable per-user dir under workspace
    # instead of the image's /root or a non-existent /home/<host-user>.
    -e "HOME=$WORKSPACE/.docker-home"
    # Skip docker/entrypoint.sh: --entrypoint bash + script-as-arg avoids
    # the broken getent pipeline (see header comment).
    --entrypoint bash
)

exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" "$REL_BENCH" "$@"
