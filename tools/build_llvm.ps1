##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
$ErrorActionPreference = "Stop"
$SCRIPT_DIR = $PSScriptRoot
. "$SCRIPT_DIR/run-external-command.ps1"
. "$SCRIPT_DIR/setup_msvc_env.ps1"

Set-Location "$Env:VAI_RT_WORKSPACE/llvm"
Run cmake -G Ninja -DBUILD_SHARED_LIBS=OFF `
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" `
    -S "${Env:VAI_RT_WORKSPACE}/llvm/llvm" -B "$Env:VAI_RT_BUILD_DIR/llvm" `
    "-DCMAKE_INSTALL_PREFIX=$Env:VAI_RT_PREFIX" `
    "-DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON" `
    "-DCMAKE_BUILD_TYPE=Debug" `
    "-DLLVM_ENABLE_PROJECTS=mlir" `
    "-DLLVM_TARGETS_TO_BUILD=host" `
    "-DLLVM_ENABLE_ASSERTIONS=ON" `
    "-DLLVM_ENABLE_RTTI=ON" `
    "-DLLVM_ENABLE_LIBEDIT=OFF" `
    "-DLLVM_BUILD_TOOLS=OFF" `
    "-DLLVM_INSTALL_UTILS=ON" `
    "-DLLVM_INCLUDE_TESTS=OFF" `
    "-DZLIB_USE_STATIC_LIBS=ON" `
    --fresh

$jobs = [Environment]::ProcessorCount

Run ninja -C "$Env:VAI_RT_BUILD_DIR/llvm" -j $jobs install
