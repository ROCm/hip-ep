##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Shrink huge MLIR for Netron / diff (streaming Python tool).
param(
  [Parameter(Mandatory = $true, Position = 0)]
  [string]$InputMlir,

  [Parameter(Mandatory = $false, Position = 1)]
  [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

$skillRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$pyScript = Join-Path $skillRoot "scripts/shrink_mlir_for_viewer.py"

$InputMlir = (Resolve-Path -LiteralPath $InputMlir).Path

$args = @($pyScript, $InputMlir)
if (-not [string]::IsNullOrWhiteSpace($OutputPath)) {
  $args += @("-o", $OutputPath)
}

& python @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
exit 0
