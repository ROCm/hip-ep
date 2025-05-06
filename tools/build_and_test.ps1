##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
$ErrorActionPreference = "Stop"
$SCRIPT_DIR = $PSScriptRoot
. "$SCRIPT_DIR/run-external-command.ps1"

Run python -m pip install --user numpy==2.1.1 onnx==1.16.0
Run cmake -DBUILD_SHARED_LIBS=OFF `
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" `
    -S "${Env:VAI_RT_WORKSPACE}/morphizen-demo" -B "$Env:VAI_RT_BUILD_DIR/morphizen-demo" `
    "-DCMAKE_INSTALL_PREFIX=$Env:VAI_RT_PREFIX" `
    "-DFETCHCONTENT_BASE_DIR=$Env:VAI_RT_PREFIX/morphizen_deps" `
    --fresh

$jobs = [Environment]::ProcessorCount

Run cmake  --build  "$Env:VAI_RT_BUILD_DIR/morphizen-demo" -j $jobs --target install

Run ctest -j $jobs --test-dir "$Env:VAI_RT_BUILD_DIR/morphizen-demo" -C Debug --output-on-failure
