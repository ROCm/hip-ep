#!/bin/bash
##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
adduser --disabled-password --uid $HOST_USER_ID --home /github/workspace --shell /bin/bash --no-create-home --gecos "" $HOST_USER_NAME
printenv
chmod +x /entrypoint.user.sh
sudo --preserve-env -u "$HOST_USER_NAME" env \
     HOME=/github/workspace \
     PATH="$PATH:/github/workspace/.local/bin:/workspace/local/bin" \
     LD_LIBRARY_PATH=/workspace/local/lib \
     VAI_RT_BUILD_DIR=/workspace/build \
     VAI_RT_PREFIX=/workspace/local \
     /bin/bash /entrypoint.user.sh "$@"
