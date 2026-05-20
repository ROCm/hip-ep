#!/usr/bin/env bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

set -euo pipefail

# Read the JSON request from stdin to extract the URI
read -r request || true

uri="$(printf '%s' "${request}" | sed 's/.*"uri":"\([^"]*\)".*/\1/')"

# github.com, raw.githubusercontent.com, codeload.github.com
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    TOKEN="${GITHUB_TOKEN}"
elif command -v gh &>/dev/null; then
    TOKEN="$(gh auth token 2>/dev/null || true)"
fi

if [[ -z "${TOKEN:-}" ]]; then
    printf '{"headers":{}}\n'
    exit 0
fi

printf '{"headers":{"Authorization":["Bearer %s"]}}\n' "${TOKEN}"
