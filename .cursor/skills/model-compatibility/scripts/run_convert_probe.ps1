##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Run the EP-input MLIR through the ONNX-to-HIP conversion, which is the
# support oracle for the compatibility report: whatever is still an onnx op
# afterwards has no working conversion.
#
# The pipeline stops at convert-onnx-to-hip. It is the prefix of
# buildOnnxToHipPipeline (lib/Dialect/Transforms/Pipelines.cpp) up to and
# including that pass; the tail (bufferization, pooling, LLVM lowering) needs
# no GPU but says nothing about operator support.
#
# Both dumps are printed with locations. Re-parsing the input file gives every
# op a location into it, and the conversion carries that location to the ops it
# creates, which is how analyze_conversion.py pairs the two sides.

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputMlir,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$OutputDir,

    [string]$HipEpPackageRoot = "",
    [string]$HipMlirOptPath = ""
)

$ErrorActionPreference = "Stop"

$InputMlir = (Resolve-Path -LiteralPath $InputMlir).ProviderPath
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).ProviderPath

if ([string]::IsNullOrWhiteSpace($HipMlirOptPath)) {
    if ([string]::IsNullOrWhiteSpace($HipEpPackageRoot) -and $env:HIP_EP_PACKAGE_ROOT) {
        $HipEpPackageRoot = $env:HIP_EP_PACKAGE_ROOT
    }
    if ([string]::IsNullOrWhiteSpace($HipEpPackageRoot)) {
        $msg = '[HIP_EP_NOT_CONFIGURED] No HipEpPackageRoot supplied for hip-mlir-opt. Pass -HipEpPackageRoot <dir> or -HipMlirOptPath <exe>.'
        Write-Output $msg
        exit 10
    }
    # Executables carry no suffix outside Windows.
    $PackageBin = Join-Path $HipEpPackageRoot "bin"
    $HipMlirOptPath = @('hip-mlir-opt.exe', 'hip-mlir-opt') |
        ForEach-Object { Join-Path $PackageBin $_ } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
}
if ([string]::IsNullOrWhiteSpace($HipMlirOptPath) -or -not (Test-Path -LiteralPath $HipMlirOptPath)) {
    # Never fall back to a source-scanning approximation: a missing tool must
    # be visible, not silently downgrade the oracle.
    throw "hip-mlir-opt not found under $HipEpPackageRoot"
}
$HipMlirOptPath = (Resolve-Path -LiteralPath $HipMlirOptPath).ProviderPath

$InputLoc = Join-Path $OutputDir "ep_input_loc.mlir"
$Converted = Join-Path $OutputDir "converted.mlir"
$ConvertLog = Join-Path $OutputDir "convert_log.txt"
$ReprintLog = Join-Path $OutputDir "reprint_log.txt"

# Pass order matches buildOnnxToHipPipeline (lib/Dialect/Transforms/
# Pipelines.cpp) up to convert-onnx-to-hip. It is duplicated here because the
# compiler registers only the whole pipeline, so the check below fails loudly
# if a pass is renamed or dropped rather than letting the probe drift into
# reporting a different graph than the one production compiles.
$ConvertPasses = @(
    '--onnx-dialect=stub',
    '--simplify-onnx',
    '--hip-add-context-arg',
    '--onnx-loop-outline',
    '--onnx-if-outline',
    '--hip-infer-loop-body-shapes',
    '--convert-onnx-to-hip'
)

$optHelp = & $HipMlirOptPath --help 2>&1 | Out-String
$missingPasses = $ConvertPasses |
    ForEach-Object { ($_ -split '=')[0] } |
    Where-Object { $optHelp -notmatch [regex]::Escape($_) }
if ($missingPasses) {
    throw ("hip-mlir-opt does not know: $($missingPasses -join ', '). " +
        "The probe pipeline no longer matches buildOnnxToHipPipeline in " +
        "lib/Dialect/Transforms/Pipelines.cpp; update `$ConvertPasses.")
}

function Invoke-Opt {
    param([string[]]$OptArgs, [string]$LogPath, [string]$Label)

    Write-Host $Label -ForegroundColor Yellow
    $proc = Start-Process -FilePath $HipMlirOptPath -ArgumentList $OptArgs `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$LogPath.out" -RedirectStandardError $LogPath
    # hip-mlir-opt writes the module with -o, so its stdout only carries noise.
    Remove-Item -LiteralPath "$LogPath.out" -ErrorAction SilentlyContinue
    # An empty log is the normal case; keeping it suggests there is something
    # to read.
    if ((Test-Path -LiteralPath $LogPath) -and (Get-Item -LiteralPath $LogPath).Length -eq 0) {
        Remove-Item -LiteralPath $LogPath -ErrorAction SilentlyContinue
    }
    return $proc.ExitCode
}

# 1. Re-print the input with locations, in the same coordinate space the
#    conversion will report.
$exit = Invoke-Opt -Label '  re-print input with locations' -LogPath $ReprintLog -OptArgs @(
    $InputMlir, '--onnx-dialect=stub', '--mlir-print-debuginfo', '-o', $InputLoc
)
if ($exit -ne 0 -or -not (Test-Path -LiteralPath $InputLoc)) {
    if (Test-Path -LiteralPath $ReprintLog) {
        Get-Content -LiteralPath $ReprintLog -Tail 20 | Write-Host
    }
    throw "hip-mlir-opt could not re-print the input MLIR (exit $exit)"
}

# 2. Convert. HIPDNN_EP_DEBUG makes convert-onnx-to-hip list the op types it
#    could not convert, which is a cross-check on the leftover scan.
$prevDebug = $env:HIPDNN_EP_DEBUG
try {
    $env:HIPDNN_EP_DEBUG = "1"
    $exit = Invoke-Opt -Label '  convert-onnx-to-hip' -LogPath $ConvertLog -OptArgs (
        @($InputMlir) + $ConvertPasses + @('--mlir-print-debuginfo', '-o', $Converted)
    )
}
finally {
    if ($null -eq $prevDebug) {
        Remove-Item Env:HIPDNN_EP_DEBUG -ErrorAction SilentlyContinue
    } else {
        $env:HIPDNN_EP_DEBUG = $prevDebug
    }
}

$haveConvertLog = Test-Path -LiteralPath $ConvertLog

if (-not (Test-Path -LiteralPath $Converted)) {
    if ($haveConvertLog) { Get-Content -LiteralPath $ConvertLog -Tail 30 | Write-Host }
    throw "convert-onnx-to-hip produced no output (exit $exit)"
}
if ($exit -ne 0) {
    # A pass can fail after writing partial output; the report would then
    # describe a graph the compiler never accepted.
    if ($haveConvertLog) { Get-Content -LiteralPath $ConvertLog -Tail 30 | Write-Host }
    throw "convert-onnx-to-hip failed with exit code $exit"
}

if ($haveConvertLog -and (Select-String -LiteralPath $ConvertLog -Pattern 'unconverted onnx op type' -Quiet)) {
    Write-Host "  compiler reported unconverted ops:" -ForegroundColor Yellow
    Get-Content -LiteralPath $ConvertLog | Where-Object { $_ -match '^\s+onnx\.' } | Write-Host
}

Write-Host ""
Write-Host "OK: conversion probe complete"
Write-Host "    input (located): $InputLoc"
Write-Host "    converted:       $Converted"
if ($haveConvertLog) { Write-Host "    log:             $ConvertLog" }
