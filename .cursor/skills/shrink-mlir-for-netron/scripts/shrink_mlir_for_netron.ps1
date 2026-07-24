##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Shrink huge MLIR and apply Netron fixes -> <stem>.netron.mlir only (intermediate in temp)
param(
  [Parameter(Mandatory = $true, Position = 0)]
  [string]$InputMlir,

  [Parameter(Mandatory = $false, Position = 1)]
  [string]$OutputDir = "",

  [switch]$KeepConstants
)

$ErrorActionPreference = "Stop"

$skillRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$shrinkPy = Join-Path $skillRoot "scripts/shrink_mlir_for_viewer.py"
$netronPy = Join-Path $skillRoot "scripts/netron_fixes.py"

$InputMlir = (Resolve-Path -LiteralPath $InputMlir).Path
$dir = if ([string]::IsNullOrWhiteSpace($OutputDir)) {
  [System.IO.Path]::GetDirectoryName($InputMlir)
} else {
  (Resolve-Path -LiteralPath $OutputDir).Path
}

$stem = [System.IO.Path]::GetFileNameWithoutExtension($InputMlir)
$netronMlir = Join-Path $dir "$stem.netron.mlir"
$viewMlir = Join-Path ([System.IO.Path]::GetTempPath()) ("$stem.view." + [Guid]::NewGuid().ToString("N") + ".mlir")

try {
  & python $shrinkPy $InputMlir -o $viewMlir 1>$null
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

  $netronArgs = @($netronPy, $viewMlir, "-o", $netronMlir)
  if ($KeepConstants) { $netronArgs += "--keep-constants" }

  & python @netronArgs
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
  if (Test-Path -LiteralPath $viewMlir) {
    Remove-Item -LiteralPath $viewMlir -Force -ErrorAction SilentlyContinue
  }
}

Write-Host ""
Write-Host "Netron file: $netronMlir"
exit 0
