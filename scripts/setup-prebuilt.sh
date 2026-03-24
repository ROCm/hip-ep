#!/bin/bash
##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PREBUILT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)/../prebuilt-local
REPO=wcy123/llvm-mlir-prebuilt

LLVM_TAG=llvm-22.1.0-release
LLVM_ASSET=llvm-22.1.0-release-windows-x64.zip

# Protobuf is a required dependency of onnx-mlir: ONNX model files (.onnx) are
# protobuf-serialized, so onnx-mlir needs protobuf to parse them. The protobuf
# package also bundles abseil (absl), which protobuf 3.21+ depends on internally.
PROTO_TAG=protobuf-34.0-release
PROTO_ASSET=protobuf-34.0-release-windows-x64.zip

FLATBUFFERS_TAG=flatbuffers-25.12.19-release
FLATBUFFERS_ASSET=flatbuffers-25.12.19-release-windows-x64.zip

mkdir -p "$PREBUILT_DIR"

download_and_extract() {
    local tag=$1 asset=$2
    local sentinel="$PREBUILT_DIR/.extracted-$asset"
    if [ ! -f "$PREBUILT_DIR/$asset" ]; then
        echo "Downloading $asset ..."
        gh release download "$tag" --repo "$REPO" --pattern "$asset" --dir "$PREBUILT_DIR"
        # zip changed — force re-extraction
        rm -f "$sentinel"
    else
        echo "Already downloaded: $asset"
    fi
    if [ ! -f "$sentinel" ]; then
        echo "Extracting $asset ..."
        unzip -q -o "$PREBUILT_DIR/$asset" -d "$PREBUILT_DIR"
        touch "$sentinel"
    else
        echo "Already extracted: $asset"
    fi
}

download_and_extract "$LLVM_TAG" "$LLVM_ASSET"
download_and_extract "$PROTO_TAG" "$PROTO_ASSET"
download_and_extract "$FLATBUFFERS_TAG" "$FLATBUFFERS_ASSET"

echo ""
echo "Pre-built binaries ready at: $PREBUILT_DIR"
echo "CMake configs:"
ls "$PREBUILT_DIR/lib/cmake/"
