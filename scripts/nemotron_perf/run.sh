#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
# The SDK fix is a read-only container mount. It does not replace host files.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ZSTD=${NEMOTRON_SDK_ZSTD:-/scratch/fhanuman/RCOmEP/.ir_dump_fixed/meta-onyx-zstd-fix-20261007T211847Z_ee3d80cd/zstd-upstream-fix/lib/librocm_sysdeps_zstd.so.1.5.7}
exec python3 "$HERE/run.py" --sdk-zstd-library "$ZSTD" "$@"
