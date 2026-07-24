##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Write hip-compiler input MLIR next to an ONNX model (<stem>.mlir).

# Stops hip-onnx-runner once mlir_bytecode_dump.mlir size is stable (skips hip-compiler).

param(

  [Parameter(Mandatory = $true, Position = 0)]

  [string]$OnnxPath,



  [Parameter(Mandatory = $false, Position = 1)]

  [string]$OutputDir = ""

)



$ErrorActionPreference = "Stop"



$workspace = if ($env:WORKSPACE) { $env:WORKSPACE } else { Join-Path $env:USERPROFILE "workspace" }

$buildDir = if ($env:BUILD_DIR) { $env:BUILD_DIR } else { Join-Path $workspace "build\hip-ep" }

$config = if ($env:CONFIG) { $env:CONFIG } else { "RelWithDebInfo" }

$runner = Join-Path $buildDir "bin\$config\hip-onnx-runner.exe"



$pollSec = if ($env:ONNX_TO_MLIR_POLL_SEC) { [int]$env:ONNX_TO_MLIR_POLL_SEC } else { 2 }

$stablePolls = if ($env:ONNX_TO_MLIR_STABLE_POLLS) { [int]$env:ONNX_TO_MLIR_STABLE_POLLS } else { 3 }

$timeoutSec = if ($env:ONNX_TO_MLIR_TIMEOUT_SEC) { [int]$env:ONNX_TO_MLIR_TIMEOUT_SEC } else { 7200 }



$OnnxPath = (Resolve-Path -LiteralPath $OnnxPath).Path



if (-not (Test-Path -LiteralPath $runner)) {

  Write-Error "hip-onnx-runner not found: $runner. Build target hip-onnx-runner first."

}



$stem = [System.IO.Path]::GetFileNameWithoutExtension($OnnxPath)

if ([string]::IsNullOrWhiteSpace($OutputDir)) {

  $OutputDir = [System.IO.Path]::GetDirectoryName($OnnxPath)

}



New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null



if (-not $env:MORPHIZEN_EP_ENABLE_CPU_DEVICE) { $env:MORPHIZEN_EP_ENABLE_CPU_DEVICE = "1" }

if (-not $env:XLNX_ABI_2_0_CLONE_EXTERNAL_DATA_THRESHOLD) {

  $env:XLNX_ABI_2_0_CLONE_EXTERNAL_DATA_THRESHOLD = "1073741824"

}



$finalMlir = Join-Path $OutputDir "$stem.mlir"

$rawMlir = Join-Path $OutputDir "mlir_bytecode_dump.mlir"



Write-Host "ONNX:   $OnnxPath"

Write-Host "Output: $finalMlir"

Write-Host "Runner: $runner"

Write-Host "Stop after stable: $rawMlir (poll ${pollSec}s, $stablePolls stable checks, timeout ${timeoutSec}s)"

Write-Host ""



$runnerArgs = @(

  "-m", $OnnxPath,

  "--dump-compiler-mlir",

  "--mlir-dump-dir", $OutputDir

)

$proc = Start-Process -FilePath $runner -ArgumentList $runnerArgs -PassThru -NoNewWindow



function Stop-RunnerProcess {

  param([System.Diagnostics.Process]$Process)

  if ($null -eq $Process -or $Process.HasExited) { return }

  Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue

  if (-not $Process.WaitForExit(120000)) {

    Write-Warning "hip-onnx-runner (pid $($Process.Id)) did not exit within 120s after Stop-Process"

  }

}



$deadline = (Get-Date).AddSeconds($timeoutSec)

$lastSize = [int64]-1

$stableCount = 0

$dumpReady = $false

$stoppedEarly = $false



while ($true) {

  if ((Get-Date) -gt $deadline) {

    Stop-RunnerProcess -Process $proc

    Write-Error "Timeout waiting for stable dump: $rawMlir"

  }



  if (Test-Path -LiteralPath $rawMlir) {

    $size = (Get-Item -LiteralPath $rawMlir).Length

    if ($size -gt 0) {

      if ($size -eq $lastSize) {

        $stableCount++

        if ($stableCount -ge $stablePolls) {

          $dumpReady = $true

          $stoppedEarly = -not $proc.HasExited

          Stop-RunnerProcess -Process $proc

          break

        }

      } else {

        $stableCount = 0

        $lastSize = $size

      }

    }

  }



  if ($proc.HasExited) {

    if ((Test-Path -LiteralPath $rawMlir) -and (Get-Item -LiteralPath $rawMlir).Length -gt 0) {

      $dumpReady = $true

    }

    break

  }



  Start-Sleep -Seconds $pollSec

}



$runnerExit = if ($proc.HasExited) { $proc.ExitCode } else { -1 }



if (-not $dumpReady) {

  if (-not $proc.HasExited) {

    Stop-RunnerProcess -Process $proc

    $runnerExit = $proc.ExitCode

  }

  Write-Error "Dump not found or empty: $rawMlir (runner exit $runnerExit)"

}



Move-Item -LiteralPath $rawMlir -Destination $finalMlir -Force



$size = (Get-Item -LiteralPath $finalMlir).Length

Write-Host ""

Write-Host "Output MLIR: $finalMlir"

Write-Host "Size: $size bytes"

if ($stoppedEarly) {

  Write-Host "(stopped hip-onnx-runner after stable pre-compiler dump; skipped post-dump hip-compiler)"

} elseif ($runnerExit -ne 0) {

  Write-Host "(hip-onnx-runner exited $runnerExit; MLIR dump succeeded)"

}



exit 0
