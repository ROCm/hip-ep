##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##


# get number of cpus
$jobs = [Environment]::ProcessorCount
cmake --build "${env:VAI_RT_BUILD_DIR}/morphizen-demo" --target install -j $jobs

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake install failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
} else {
    Write-Host "CMake install completed successfully."
}
