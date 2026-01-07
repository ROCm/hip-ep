#!/bin/bash
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

# Environment setup script for morphizen-hipdnn

# Set up build and install directories
export MORPHIZEN_HIPDNN_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MORPHIZEN_BUILD_DIR="${MORPHIZEN_HIPDNN_ROOT}/../build/morphizen-hipdnn"
export MORPHIZEN_INSTALL_PREFIX="${MORPHIZEN_HIPDNN_ROOT}/../local"

# MorphiZen debug flags
export MORPHIZEN_DEBUG_HIPDNN=1
export MORPHIZEN_DEBUG_PLUGIN=1
export MORPHIZEN_DEBUG_TAR_CACHE=1
export DEBUG_VAIP_PASS=1
export MORPHIZEN_DEBUG_DEINITIALIZE=1

# ONNX Runtime flags
export XLNX_ONNX_EP_VERBOSE=2
export DEBUG_LOG_LEVEL=info
export XLNX_ENABLE_EP_SHARED_CONTEXT=1
export XLNX_ENABLE_CACHE_CONTEXT=1
export XLNX_ENABLE_CACHE=0
export XLNX_USE_CACHE_DIR=/tmp/
export XLNX_USE_CACHE_KEY=morphizen-hipdnn

# Configuration file
export VITISAI_EP_JSON_CONFIG="${MORPHIZEN_HIPDNN_ROOT}/etc/vaip_config.json"

echo "MorphiZen HIP DNN environment configured"
echo "Root: ${MORPHIZEN_HIPDNN_ROOT}"
echo "Build: ${MORPHIZEN_BUILD_DIR}"
echo "Install: ${MORPHIZEN_INSTALL_PREFIX}"
