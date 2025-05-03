##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
Write-Host "Download ONNXRuntime on Windows..."
$ErrorActionPreference = "Stop"
New-Item -Path $Env:VAI_RT_WORKSPACE -ItemType Directory -Force | Out-Null
$directoryPath = "$Env:VAI_RT_WORKSPACE/onnxruntime"
if (-Not (Test-Path -Path $directoryPath)) {
    Write-Host "Directory does not exist. Cloning the repository..."
    git clone https://github.com/Microsoft/onnxruntime.git --branch main --single-branch --depth 1 $directoryPath
}
else {
    # Directory exists, skip cloning
    Write-Host "Directory already exists. Skipping clone."
}
# Restore the directory to its original state
# save the current dirctory
$currentDirectory = Get-Location
Set-Location -Path $directoryPath
git fetch origin main
$originalCommit = git rev-parse HEAD
$wednesdayDate = (Get-Date).AddDays( - (7 + [int](Get-Date).DayOfWeek - 3) % 7).ToString("yyyy-MM-dd")
Write-Host "Last Wednesday was $wednesdayDate"
$firstCommit = git rev-list --reverse --since="$wednesdayDate 00:00" --until="$wednesdayDate 23:59" origin/main | Select-Object -First 1
if ([string]::IsNullOrEmpty($firstCommit)) {
    Write-Host "No need to update ONNXRuntime commit ID , no commits found on last Wednesday. "
}
elseif ($originalCommit -eq $firstCommit) {
    Write-Host "No need to update ONNXRuntime commit ID ,  ORT commit ID is $originalCommit"
}
else {
    Write-Host "Upgrade Onnxruntime commit ID from $originalCommit to $firstCommit "
    git checkout $firstCommit --force
}
git clean -fdx
$HEAD = git rev-parse HEAD
Write-Host "using onnxruntime commit ID $HEAD"
# restore to old directory
Set-Location -Path $currentDirectory