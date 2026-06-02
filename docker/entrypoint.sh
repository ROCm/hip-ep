#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Container entrypoint: re-create host UID/GID inside so bind-mounted files
# stay owned by the host user (no chown needed).
#
# Required env (from run.sh): HOST_UID, HOST_GID, HOST_USER, HOST_GROUP.
# Optional: HOST_HOME (defaults to /home/${HOST_USER}).
set -euo pipefail

: "${HOST_UID:?HOST_UID not set — run.sh should pass it via -e}"
: "${HOST_GID:?HOST_GID not set — run.sh should pass it via -e}"
: "${HOST_USER:?HOST_USER not set — run.sh should pass it via -e}"
: "${HOST_GROUP:?HOST_GROUP not set — run.sh should pass it via -e}"
: "${HOST_HOME:=/home/${HOST_USER}}"

# Idempotent for `docker exec` re-attaches into a long-lived container.
if ! getent group "$HOST_GID" >/dev/null; then
    groupadd -g "$HOST_GID" "$HOST_GROUP"
fi
if ! id -u "$HOST_USER" >/dev/null 2>&1; then
    useradd -m -d "$HOST_HOME" -u "$HOST_UID" -g "$HOST_GID" -s /bin/bash "$HOST_USER"
    # `dev` group -> NOPASSWD sudo (see Dockerfile).
    groupadd -f dev
    usermod -aG dev "$HOST_USER"
    for grp in render video; do
        if getent group "$grp" >/dev/null; then
            usermod -aG "$grp" "$HOST_USER"
        fi
    done
fi

# /dev/kfd + /dev/dri/* device nodes use the HOST's render GID, which
# typically doesn't match any in-container group. Map our user into the
# device's actual GID so HIP can open it. usermod takes either name or
# numeric, but some shadow-utils silently no-op on numeric — resolve a
# name (existing or synthetic) and pass that.
#
# `|| true` on the substitution is load-bearing: under `set -euo pipefail`,
# `getent group <missing_gid>` exits 2 and pipefail propagates that through
# `cut`. Without the suppressor, a `var=$(failing-pipe)` in this strict mode
# kills the whole script before the fallback `if [ -z "$grp_name" ]` below
# ever runs — which is exactly when we need it (host /dev/kfd's render-group
# GID is rarely present in the base image's /etc/group).
for dev in /dev/kfd /dev/dri/renderD*; do
    if [ -e "$dev" ]; then
        dev_gid=$(stat -c '%g' "$dev")
        if [ "$dev_gid" != "0" ] && ! id -G "$HOST_USER" | tr ' ' '\n' | grep -qx "$dev_gid"; then
            grp_name=$(getent group "$dev_gid" | cut -d: -f1 || true)
            if [ -z "$grp_name" ]; then
                grp_name="hostgid_${dev_gid}"
                groupadd -g "$dev_gid" "$grp_name"
            fi
            usermod -aG "$grp_name" "$HOST_USER"
        fi
    fi
done

exec gosu "$HOST_USER" "$@"
