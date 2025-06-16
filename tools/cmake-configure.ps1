##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
cmake -S "${env:VAI_RT_WORKSPACE}/morphizen-demo" -B "${env:VAI_RT_BUILD_DIR}/morphizne-demo" "-DCMAKE_INSTALL_PREFIX=${ENV:VAI_RT_PREFIX}"
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
} else {
    Write-Host "CMake configuration completed successfully."
}

cmake -G Ninja -S "${env:VAI_RT_WORKSPACE}/morphizen-demo" -B "${env:VAI_RT_BUILD_DIR}.ninja/morphizen-demo" `
   "-DCMAKE_INSTALL_PREFIX=${ENV:VAI_RT_PREFIX}" `
   "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration with Ninja generator failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
} else {
    Write-Host "CMake configuration with Ninja generator completed successfully."
}

Copy-Item -Path "${env:VAI_RT_BUILD_DIR}.ninja/morphizen-demo/compile_commands.json" `
   -Destination "${env:VAI_RT_WORKSPACE}/morphizen-demo/compile_commands.json" -Force

Copy-Item -Path "${env:VAI_RT_BUILD_DIR}.ninja/morphizne-demo/compile_commands.json" `
   -Destination "${env:VAI_RT_WORKSPACE}/MorphiZen/compile_commands.json" -Force
