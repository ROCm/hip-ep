##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
$morphizenDemoPath = "$Env:VAI_RT_WORKSPACE/morphizen-demo"
if (-Not (Test-Path -Path $morphizenDemoPath)) {
    Write-Host "morphizen-demo Directory does not exist. please download it with download_demo.ps1"
    exit 0
}
cmake -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF `
    -DCMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY=ON `
    -Dmorphizen_ENABLE_UNIT_TEST=ON   `
    -DBUILD_SHARED_LIBS=OFF `
    -S $morphizenDemoPath -B "$Env:VAI_RT_BUILD_DIR/morphizen-demo" `
    "-DCMAKE_INSTALL_PREFIX=$Env:VAI_RT_PREFIX" `
    -DFETCHCONTENT_BASE_DIR="$Env:VAI_RT_BUILD_DIR/morphizen_deps" `
    --fresh
$job = [Environment]::ProcessorCount
cmake --build "$Env:VAI_RT_BUILD_DIR/morphizen-demo" -j $job --target install
