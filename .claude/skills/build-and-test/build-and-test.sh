#!/bin/bash
##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Build and test script for MorphiZen with automated dependency management

set -euo pipefail

# Setup paths
WORKSPACE=$(cd ../.. && pwd)
LOCAL_DIR="$WORKSPACE/local"
PROJECT_DIR=$(pwd)
PROJECT_NAME=$(basename "$PROJECT_DIR")
BUILD_DIR="$WORKSPACE/build/$PROJECT_NAME"

echo "🔧 MorphiZen Build and Test"
echo "Project: $PROJECT_DIR"
echo "Build dir: $BUILD_DIR"
echo "Install prefix: $LOCAL_DIR"
echo ""

# ============================================================================
# Step 1: MSVC Check (Windows Only)
# ============================================================================

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "🔍 Checking MSVC environment..."
    if ! which cl.exe >/dev/null 2>&1; then
        echo "❌ MSVC compiler not found"
        echo ""
        echo "Launch git-bash from \"Developer Command Prompt for VS 20XX\":"
        echo "1. Open \"Developer Command Prompt for VS 20XX\""
        echo "2. Run: bash"
        echo "3. Navigate to project and retry"
        echo ""
        echo "STATUS:MISSING_MSVC"
        exit 1
    fi
    echo "✅ MSVC found: $(cl.exe 2>&1 | head -1)"
fi

# ============================================================================
# Step 2: Auto-Build Missing Dependencies
# ============================================================================

echo ""
echo "📦 Checking dependencies..."

# Common CMake flags for all dependencies
COMMON_FLAGS=(
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>"
    "-DCMAKE_BUILD_TYPE=Debug"
    "-DCMAKE_INSTALL_PREFIX=$LOCAL_DIR"
    "-DBUILD_SHARED_LIBS=OFF"
)

# Function to build a dependency
build_dependency() {
    local dep_name=$1
    local dep_version=$2
    local git_url=$3
    shift 3
    local extra_flags=("$@")

    local source_dir="$WORKSPACE/../source/$dep_name"
    local build_dir_dep="$WORKSPACE/build/$dep_name"

    echo ""
    echo "🔨 Building $dep_name $dep_version..."

    # Clone if not exists
    if [ ! -d "$source_dir" ]; then
        echo "📥 Cloning $dep_name..."
        git clone --branch "$dep_version" --depth 1 "$git_url" "$source_dir"
    fi

    # Configure
    cmake -S "$source_dir" -B "$build_dir_dep" \
        "${COMMON_FLAGS[@]}" \
        "${extra_flags[@]}"

    # Build
    cmake --build "$build_dir_dep" --config Debug --parallel

    # Install
    cmake --install "$build_dir_dep" --config Debug

    echo "✅ $dep_name installed"
}

# Check protobuf v21.12
if ! ls "$LOCAL_DIR/lib/cmake/protobuf"/*.cmake >/dev/null 2>&1; then
    build_dependency "protobuf" "v21.12" "https://github.com/protocolbuffers/protobuf.git" \
        "-Dprotobuf_BUILD_TESTS=OFF"
else
    echo "✅ protobuf found"
fi

# Check gtest v1.15.0
if ! ls "$LOCAL_DIR/lib/cmake/GTest"/*.cmake >/dev/null 2>&1; then
    build_dependency "googletest" "v1.15.0" "https://github.com/google/googletest.git"
else
    echo "✅ gtest found"
fi

# Check glog v0.7.1
if ! ls "$LOCAL_DIR/lib/cmake/glog"/*.cmake >/dev/null 2>&1; then
    build_dependency "glog" "v0.7.1" "https://github.com/google/glog.git" \
        "-DBUILD_TESTING=OFF"
else
    echo "✅ glog found"
fi

# Check ONNX Runtime (cannot auto-build)
if ! ls "$LOCAL_DIR/lib/cmake/onnxruntime"/*.cmake >/dev/null 2>&1; then
    echo ""
    echo "❌ ONNX Runtime not found in $LOCAL_DIR"
    echo ""
    echo "ONNX Runtime requires VitisAI support and manual build."
    echo "See docs/developer-guide.md for build instructions."
    echo ""
    echo "STATUS:MISSING_ONNXRUNTIME"
    exit 1
fi
echo "✅ ONNX Runtime found"

# ============================================================================
# Step 3: Determine Build Type
# ============================================================================

echo ""
echo "🔍 Determining build type..."

CLEAN_BUILD=false
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    echo "📋 No existing build - clean build required"
    CLEAN_BUILD=true
else
    echo "📋 Existing build found - incremental build"
fi

# ============================================================================
# Step 4: Execute Build
# ============================================================================

echo ""
if [ "$CLEAN_BUILD" = true ]; then
    echo "🏗️  Configuring (clean build)..."

    cmake -S . -B "$BUILD_DIR" \
        -DBUILD_SHARED_LIBS=OFF \
        "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
        -DCMAKE_BUILD_TYPE=Debug \
        "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -Dmorphizen_ENABLE_UNIT_TEST=ON \
        --fresh

    if [ $? -ne 0 ]; then
        echo ""
        echo "STATUS:CONFIGURE_FAILED"
        exit 1
    fi
fi

echo ""
echo "🔨 Building..."
cmake --build "$BUILD_DIR" --config Debug --parallel

if [ $? -ne 0 ]; then
    echo ""
    echo "STATUS:BUILD_FAILED"
    exit 1
fi

echo ""
echo "✅ Build successful"

# ============================================================================
# Step 5: Run Tests
# ============================================================================

echo ""
echo "🧪 Running tests..."

ctest --test-dir "$BUILD_DIR" -C Debug --output-on-failure --timeout 600

if [ $? -ne 0 ]; then
    echo ""
    echo "STATUS:TEST_FAILED"
    exit 1
fi

echo ""
echo "✅ All tests passed"
echo ""
echo "STATUS:SUCCESS"
exit 0
