# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# Safe-entry repro harness for FP16 EP validation.
#
# IMPORTANT:
# - Do not run this on a display-attached workstation that has shown
#   LiveKernelEvent 141 / WATCHDOG resets.
# - Run only on a remote/non-display test machine.
# - Explicit opt-in is required so accidental local runs fail before touching
#   amdhip64/MIOpen/hipBLASLt.

param(
    [string]$DemosRoot = "D:\jam\demos",
    [string]$PrebuiltBin = "D:\jam\prebuilt-local\bin",
    [string]$WorkDir = "$env:TEMP\kokoro_fp16_repro"
)

$ErrorActionPreference = "Stop"

if ($env:HIPDNN_EP_ALLOW_GPU_RUNTIME -ne "1") {
    throw "Refusing to run: set HIPDNN_EP_ALLOW_GPU_RUNTIME=1 only on a safe/non-display test machine."
}

$python = Join-Path $DemosRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing venv python: $python"
}

$runner = Join-Path $PrebuiltBin "hip-onnx-runner.exe"
if (-not (Test-Path $runner)) {
    throw "Missing hip-onnx-runner.exe: $runner"
}

$rocm = & $python -c "import _rocm_sdk_devel as r, os; print(os.path.dirname(r.__file__))"
$env:THEROCK_DIST = $rocm
$env:MORPHIZEN_NO_BUFFER_OPT = "1"
$env:HIP_CUSTOM_KERNELS_DIR = "D:\jam\prebuilt-local\lib"
$env:HIPDNN_EP_SYNC_OPS = "1"
$env:HIPDNN_EP_AUTOTUNE = "0"
$env:HIPDNN_EP_ENABLE_GRAPHS = ""
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PATH = "$rocm\bin;$rocm\lib;$PrebuiltBin;D:\jam\prebuilt-local\lib;$env:PATH"

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

Write-Host "FP16 repro environment:"
Write-Host "  DemosRoot   = $DemosRoot"
Write-Host "  PrebuiltBin = $PrebuiltBin"
Write-Host "  WorkDir     = $WorkDir"
Write-Host "  THEROCK_DIST= $env:THEROCK_DIST"
Write-Host ""

Write-Host "Next manual steps on the safe machine:"
Write-Host "  1. Run isolated FP16 Sigmoid."
Write-Host "  2. Run isolated FP16 ReduceSum."
Write-Host "  3. Run isolated FP16 Cast(f16 -> i64)."
Write-Host "  4. Run duration chain."
Write-Host "  5. Run full Kokoro FP16 endpoint."
Write-Host ""
Write-Host "This script intentionally only prepares the environment; it does not launch"
Write-Host "hip-onnx-runner itself. Use the specific test command after confirming the"
Write-Host "machine is safe to reset if the driver wedges."
