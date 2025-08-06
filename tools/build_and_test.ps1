##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
$ErrorActionPreference = "Stop"
$SCRIPT_DIR = $PSScriptRoot
. "$SCRIPT_DIR/run-external-command.ps1"
. "$SCRIPT_DIR/setup_msvc_env.ps1"

Run python -m pip install --user numpy==2.1.1 onnx==1.16.0

Run cmake -G Ninja -DBUILD_SHARED_LIBS=OFF `
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" `
    -S "${Env:VAI_RT_WORKSPACE}/morphizen-demo" -B "$Env:VAI_RT_BUILD_DIR/morphizen-demo" `
    "-DCMAKE_INSTALL_PREFIX=$Env:VAI_RT_PREFIX" `
    "-DFETCHCONTENT_BASE_DIR=$Env:VAI_RT_PREFIX/morphizen_deps_ninja" `
    "-DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON" `
    "-Dmorphizen_ENABLE_ORT_BRIDGE=ON" `
    "-DCMAKE_BUILD_TYPE=Debug" `
    --fresh

$jobs = [Environment]::ProcessorCount

Run ninja -C "$Env:VAI_RT_BUILD_DIR/morphizen-demo" -j $jobs install

Run ctest -j $jobs --test-dir "$Env:VAI_RT_BUILD_DIR/morphizen-demo" -C Debug --output-on-failure --timeout 600
