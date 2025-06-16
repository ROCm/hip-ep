##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##


# get number of cpus
$jobs = [Environment]::ProcessorCount
Set-Location "${env:VAI_RT_BUILD_DIR}/morphizen-demo"

# if $Env:MORPHIZEN_TEST_CASE is set
if ($Env:MORPHIZEN_TEST_CASE) {
    Write-Host "Running CTest with MORPHIZEN_TEST_CASE: $Env:MORPHIZEN_TEST_CASE"
    # Run ctest with the specified test case
    ctest --build-config Debug -j $jobs -R $Env:MORPHIZEN_TEST_CASE
} else {
    Write-Host "Running CTest without specific test case."
    # Run ctest without any specific test case
    ctest --build-config Debug --show-only -j $jobs
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "CTest failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
} else {
    Write-Host "CTest completed successfully."
}
