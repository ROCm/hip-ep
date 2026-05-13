#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Container entrypoint.
#
# Creates a non-root user matching the host's UID/GID so that bind-mounted
# files written from inside the container are owned by the host user (no
# chown needed). Borrows the pattern from xdock-vitis-ai-sw/docker.sh, but
# trimmed: we only need build/iteration, not jupyter / board ssh / etc.
#
# Required env vars (set by run.sh on the host):
#   HOST_UID, HOST_GID, HOST_USER, HOST_GROUP
#
# Optional:
#   HOST_HOME — if set, it's the path that should be used as $HOME inside
#               the container (e.g. a bind-mounted host home for caches).
#               Defaults to /home/${HOST_USER}.
set -euo pipefail

: "${HOST_UID:?HOST_UID not set — run.sh should pass it via -e}"
: "${HOST_GID:?HOST_GID not set — run.sh should pass it via -e}"
: "${HOST_USER:?HOST_USER not set — run.sh should pass it via -e}"
: "${HOST_GROUP:?HOST_GROUP not set — run.sh should pass it via -e}"
: "${HOST_HOME:=/home/${HOST_USER}}"

# Create matching group + user if they don't already exist (idempotent for
# `docker exec` re-attaches into a long-lived container).
if ! getent group "$HOST_GID" >/dev/null; then
    groupadd -g "$HOST_GID" "$HOST_GROUP"
fi
if ! id -u "$HOST_USER" >/dev/null 2>&1; then
    useradd -m -d "$HOST_HOME" -u "$HOST_UID" -g "$HOST_GID" -s /bin/bash "$HOST_USER"
    # Add to `dev` group → NOPASSWD sudo (see Dockerfile).
    groupadd -f dev
    usermod -aG dev "$HOST_USER"
    # Render group / video / dialout for /dev/dri and /dev/kfd access if those
    # groups exist on the image; ignore if they don't.
    for grp in render video; do
        if getent group "$grp" >/dev/null; then
            usermod -aG "$grp" "$HOST_USER"
        fi
    done
fi

# When /dev/kfd or /dev/dri/* are passed in via --device, their group is the
# host's `render` GID, which probably doesn't match anything in the container.
# Add the user to the actual GID of those device nodes so HIP can open them.
for dev in /dev/kfd /dev/dri/renderD*; do
    if [ -e "$dev" ]; then
        dev_gid=$(stat -c '%g' "$dev")
        if [ "$dev_gid" != "0" ] && ! id -G "$HOST_USER" | tr ' ' '\n' | grep -qx "$dev_gid"; then
            # Resolve a name for the device's GID: prefer the existing group
            # name if one is already mapped (e.g. `render` from a base image),
            # otherwise create a synthetic one.
            grp_name=$(getent group "$dev_gid" | cut -d: -f1)
            if [ -z "$grp_name" ]; then
                grp_name="hostgid_${dev_gid}"
                groupadd -g "$dev_gid" "$grp_name"
            fi
            # `usermod -aG` accepts either a numeric GID or a group name, but
            # some shadow-utils builds silently no-op on numeric input when no
            # named group with that literal numeric name exists. Pass the name
            # to be safe.
            usermod -aG "$grp_name" "$HOST_USER"
        fi
    fi
done

# Drop into the host user. exec so signals propagate.
exec gosu "$HOST_USER" "$@"
