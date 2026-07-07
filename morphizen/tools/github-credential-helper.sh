#!/usr/bin/env bash
##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

set -euo pipefail

# Read the JSON request from stdin to extract the URI
read -r request || true

uri="$(printf '%s' "${request}" | sed 's/.*"uri":"\([^"]*\)".*/\1/')"

# Determine which token to use based on hostname
if [[ "${uri}" == *"gitenterprise.xilinx.com"* ]]; then
    if [[ -n "${GHE_TOKEN:-}" ]]; then
        TOKEN="${GHE_TOKEN}"
    elif command -v gh &>/dev/null; then
        TOKEN="$(gh auth token --hostname gitenterprise.xilinx.com 2>/dev/null || true)"
    fi
else
    # github.com, raw.githubusercontent.com, codeload.github.com
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        TOKEN="${GITHUB_TOKEN}"
    elif command -v gh &>/dev/null; then
        TOKEN="$(gh auth token 2>/dev/null || true)"
    fi
fi

if [[ -z "${TOKEN:-}" ]]; then
    printf '{"headers":{}}\n'
    exit 0
fi

printf '{"headers":{"Authorization":["Bearer %s"]}}\n' "${TOKEN}"
