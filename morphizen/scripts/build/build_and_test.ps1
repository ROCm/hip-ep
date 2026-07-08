##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
$ErrorActionPreference = "Stop"
$SCRIPT_DIR = $PSScriptRoot
. "$SCRIPT_DIR/run-external-command.ps1"
. "$SCRIPT_DIR/setup_msvc_env.ps1"

Run python -m pip install  numpy==2.1.1 onnx==1.16.0

# use "-DWIN24_BUILD=ON", otherwise get_provider_option("enable_cache_file_io_in_mem") has no effect.
# `-DWIN24_BUILD=OFF` to be deprecated in the future
Run cmake -G Ninja -DBUILD_SHARED_LIBS=OFF `
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" `
    -S "${Env:VAI_RT_WORKSPACE}/morphizen-demo" -B "$Env:VAI_RT_BUILD_DIR/morphizen-demo" `
    "-DCMAKE_INSTALL_PREFIX=$Env:VAI_RT_PREFIX" `
    "-DFETCHCONTENT_BASE_DIR=$Env:VAI_RT_PREFIX/morphizen_deps_ninja" `
    "-DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON" `
    "-Dmorphizen_ENABLE_ORT_BRIDGE=ON" `
    "-Dmorphizen_ENABLE_MLIR_BACKEND=ON" `
    "-DCMAKE_BUILD_TYPE=Debug" `
    "-DWIN24_BUILD=ON"  `
    "-DZLIB_USE_STATIC_LIBS=ON" `
    --fresh

$jobs = [Environment]::ProcessorCount

Run ninja -C "$Env:VAI_RT_BUILD_DIR/morphizen-demo" -j $jobs install

$ENV:MORPHIZEN_EP_ENABLE_CPU_DEVICE="1"
$ENV:ENABLE_CACHE_FILE_IO_IN_MEM="1"  # for stable CI tests
# print current timestamp
Write-Output "current timestamp: $(Get-Date)"
Run ctest --test-dir "$Env:VAI_RT_BUILD_DIR/morphizen-demo" -C Debug --output-on-failure --timeout 600 -N
Run ctest -j $jobs --test-dir "$Env:VAI_RT_BUILD_DIR/morphizen-demo" -C Debug --output-on-failure --timeout 600

Write-Output "Run unittests with mlir-backend"
$ENV:MORPHIZEN_ORT_BRIDGE_BACKEND="mlir-backend"
Run ctest -j $jobs --test-dir "$Env:VAI_RT_BUILD_DIR/morphizen-demo" -C Debug --output-on-failure --timeout 600
