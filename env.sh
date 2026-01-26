#!/bin/bash
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

# Environment setup script for onnx-hipdnn-ep

# Set up build and install directories
export ONNX_HIPDNN_EP_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ONNX_HIPDNN_EP_BUILD_DIR="${ONNX_HIPDNN_EP_ROOT}/../build/onnx-hipdnn-ep"
export ONNX_HIPDNN_EP_INSTALL_PREFIX="${ONNX_HIPDNN_EP_ROOT}/../local"

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
export XLNX_USE_CACHE_KEY=onnx-hipdnn-ep

# Configuration file
export VITISAI_EP_JSON_CONFIG="${ONNX_HIPDNN_EP_ROOT}/etc/vaip_config.json"

echo "ONNX HIP DNN EP environment configured"
echo "Root: ${ONNX_HIPDNN_EP_ROOT}"
echo "Build: ${ONNX_HIPDNN_EP_BUILD_DIR}"
echo "Install: ${ONNX_HIPDNN_EP_INSTALL_PREFIX}"
