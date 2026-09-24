# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

param(
    [string]$HipMlirOpt = "$PSScriptRoot\..\..\..\build\hip-ep\bin\hip-mlir-opt.exe",
    [string]$InputPath = "$PSScriptRoot\..\..\sample_hip_add_compiler.hip.mlir",
    [string]$OutputDirectory = "$PSScriptRoot\artifacts"
)

$ErrorActionPreference = 'Stop'
$HipMlirOpt = [IO.Path]::GetFullPath($HipMlirOpt)
$InputPath = [IO.Path]::GetFullPath($InputPath)

foreach($path in @($HipMlirOpt, $InputPath)) {
    if(-not (Test-Path $path)) {
        throw "Missing required path: $path"
    }
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$HostLlvm = Join-Path $OutputDirectory 'sample_hip_add.host-llvm.mlir'

& $HipMlirOpt $InputPath --convert-hip-to-llvm -o $HostLlvm
if($LASTEXITCODE -ne 0) {
    throw "HIP-to-LLVM lowering failed (exit $LASTEXITCODE)."
}

$Help = & $HipMlirOpt --help 2>&1 | Out-String
$AmdgpuPasses = $Help | Select-String -Pattern 'rocdl|amdgpu|gpu-to-rocdl|gpu-to-llvm' -CaseSensitive:$false

Write-Host "Wrote host LLVM dialect IR: $HostLlvm"
if($AmdgpuPasses) {
    Write-Host 'This hip-mlir-opt build exposes AMDGPU/ROCDL passes:'
    $AmdgpuPasses | ForEach-Object { Write-Host $_.Line }
    Write-Host 'A device-kernel lowering can be added on top of those passes.'
} else {
    Write-Warning 'This hip-mlir-opt build has no registered AMDGPU/ROCDL lowering passes.'
    Write-Warning 'The output is host LLVM IR with HIP runtime calls, not an AMDGPU device kernel.'
}