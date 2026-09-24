# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.

param(
    [string]$InputPath = "$PSScriptRoot\..\..\sample_hip_add_compiler.hip.mlir",
    [string]$Architecture = 'gfx1151',
    [string]$OutputDirectory = "$PSScriptRoot\artifacts",
    [switch]$Relocatable
)

$ErrorActionPreference = 'Stop'
$InputPath = [IO.Path]::GetFullPath($InputPath)
if(-not (Test-Path $InputPath)) {
    throw "Missing HIP MLIR input: $InputPath"
}

$Hipcc = (Get-Command hipcc.bat -ErrorAction SilentlyContinue).Source
if(-not $Hipcc) {
    $Hipcc = 'C:\Program Files\AMD\ROCm\7.1\bin\hipcc.bat'
}
if(-not (Test-Path $Hipcc)) {
    throw 'hipcc was not found. Set hipcc.bat on PATH or install ROCm 7.1.'
}
$Bundler = Join-Path (Split-Path $Hipcc) 'clang-offload-bundler.exe'
if(-not (Test-Path $Bundler)) {
    throw "clang-offload-bundler is missing next to hipcc: $Bundler"
}

$InputText = Get-Content $InputPath -Raw
if($InputText -notmatch 'hip\.add\s*\(%ctx\)') {
    throw 'This prototype only supports a HIP dialect hip.add operation.'
}

# This prototype handles a static, non-broadcasting f32 add only.
$Matches = [regex]::Matches($InputText, 'memref<((?:\d+x)+)f32')
if($Matches.Count -lt 3) {
    throw 'Expected static f32 memrefs for lhs, rhs, and output.'
}
$Shapes = @($Matches | ForEach-Object { $_.Groups[1].Value })
if($Shapes[0] -ne $Shapes[1] -or $Shapes[0] -ne $Shapes[2]) {
    throw "This prototype does not support broadcasting: shapes are $($Shapes[0]), $($Shapes[1]), $($Shapes[2])."
}

$Dimensions = $Shapes[0].TrimEnd('x').Split('x') | ForEach-Object { [UInt64]$_ }
[UInt64]$ElementCount = 1
$Dimensions | ForEach-Object { $ElementCount *= $_ }

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$Stem = [IO.Path]::GetFileNameWithoutExtension($InputPath)
$HipSource = Join-Path $OutputDirectory "$Stem.dx12-device.hip"
$LlvmIr = Join-Path $OutputDirectory "$Stem.$Architecture.amdgcn.ll"
$Bundle = Join-Path $OutputDirectory "$Stem.$Architecture.bundle"
$CodeObject = Join-Path $OutputDirectory "$Stem.$Architecture.elf"

@"
#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" __global__ void hip_ep_add_f32(const float* __restrict__ lhs,
                                           const float* __restrict__ rhs,
                                           float* __restrict__ output,
                                           int64_t num_elements)
{
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(index < num_elements)
        output[index] = lhs[index] + rhs[index];
}
"@ | Set-Content -Path $HipSource -NoNewline

& $Hipcc --offload-arch=$Architecture --cuda-device-only -S -emit-llvm $HipSource -o $LlvmIr
if($LASTEXITCODE -ne 0) {
    throw "hipcc failed while emitting AMDGPU LLVM IR (exit $LASTEXITCODE)."
}

if($Relocatable) {
    & $Hipcc --offload-arch=$Architecture --cuda-device-only -fhip-emit-relocatable -c $HipSource -o $CodeObject
} else {
    & $Hipcc --offload-arch=$Architecture --cuda-device-only -c $HipSource -o $Bundle
}
if($LASTEXITCODE -ne 0) {
    throw "hipcc failed while emitting the AMDGPU code object (exit $LASTEXITCODE)."
}

if(-not $Relocatable) {
& $Bundler -type o -unbundle -targets "hipv4-amdgcn-amd-amdhsa--$Architecture" -input $Bundle -output $CodeObject
if($LASTEXITCODE -ne 0) {
    throw "Failed to extract the raw AMDGPU ELF (exit $LASTEXITCODE)."
}
Remove-Item $Bundle -Force
}


Write-Host "Lowered HIP MLIR static f32 add ($ElementCount elements) to:"
Write-Host "  HIP device source: $HipSource"
Write-Host "  AMDGPU LLVM IR:    $LlvmIr"
Write-Host "  HSA ELF:           $CodeObject (relocatable=$Relocatable)"