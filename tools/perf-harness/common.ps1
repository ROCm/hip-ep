##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
## Shared environment resolution for the perf-harness scripts.
##
## Everything is an environment variable with a discovery fallback, so the same
## scripts run on any machine. Dot-source this, then use the $HarnessEnv fields.

Set-StrictMode -Version Latest

function Resolve-HarnessPath {
  param([string]$EnvName, [string[]]$Candidates, [switch]$Required, [string]$What)
  $v = [Environment]::GetEnvironmentVariable($EnvName)
  if ($v -and (Test-Path $v)) { return (Resolve-Path $v).Path }
  foreach ($c in $Candidates) {
    if ($c -and (Test-Path $c)) { return (Resolve-Path $c).Path }
  }
  if ($Required) {
    throw "Cannot locate $What. Set `$env:$EnvName."
  }
  return $null
}

# $PSScriptRoot is tools/perf-harness even when dot-sourced from a subdirectory,
# so the repo root is two levels up.
$script:HarnessRoot = $PSScriptRoot
$script:RepoRoot    = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# The RGP panel CLI ships in the Radeon Developer Tool Suite, which is a zip with
# a versioned folder name, so probe PATH before guessing at C:\tools.
$rgpCli = $null
$onPath = Get-Command RadeonDeveloperPanelCLI.exe -EA SilentlyContinue
if ($onPath) { $rgpCli = $onPath.Source }
if (-not $rgpCli) {
  $dir = [Environment]::GetEnvironmentVariable('RGP_DIR')
  if ($dir -and (Test-Path "$dir\RadeonDeveloperPanelCLI.exe")) {
    $rgpCli = "$dir\RadeonDeveloperPanelCLI.exe"
  }
}
if (-not $rgpCli) {
  $guess = Get-ChildItem 'C:\tools\RadeonDeveloperToolSuite*' -Directory -EA SilentlyContinue |
           Sort-Object Name -Descending |
           ForEach-Object { Join-Path $_.FullName 'RadeonDeveloperPanelCLI.exe' } |
           Where-Object { Test-Path $_ } | Select-Object -First 1
  if ($guess) { $rgpCli = $guess }
}

$py = [Environment]::GetEnvironmentVariable('HIPEP_PY')
if (-not $py -or -not (Test-Path $py)) {
  $c = Get-Command python.exe -EA SilentlyContinue
  $py = if ($c) { $c.Source } else { 'python' }
}

$script:HarnessEnv = [PSCustomObject]@{
  RepoRoot = $RepoRoot
  # Directory holding model_benchmark.exe and the EP DLLs under test. For the vlm
  # driver there is no model_benchmark.exe, so point this at the venv's
  # onnxruntime\capi, which is where the EP DLLs actually live.
  Bin      = Resolve-HarnessPath -EnvName 'HIPEP_BIN' -Candidates @() -Required -What 'the test package bin directory (model_benchmark.exe + EP DLLs)'
  # ONNX model directory (genai_config.json lives here).
  Model    = Resolve-HarnessPath -EnvName 'HIPEP_MODEL' -Candidates @() -Required -What 'the model directory'
  Python   = $py
  RgpCli   = $rgpCli
  Parser   = Join-Path $RepoRoot 'tools\rgp_parser'
  Harness  = $HarnessRoot
  # VLM driver: vision-language models have no text-only equivalent of
  # model_benchmark.exe, and their TTFT includes the vision encoder, so the
  # measurement has to run through vlm_benchmark.py with a real image.
  VlmBench   = Resolve-HarnessPath -EnvName 'HIPEP_VLM_BENCH' -Candidates @() -What 'vlm_benchmark.py'
  Image      = Resolve-HarnessPath -EnvName 'HIPEP_IMAGE' -Candidates @() -What 'the benchmark image'
  PromptFile = Resolve-HarnessPath -EnvName 'HIPEP_PROMPT_FILE' -Candidates @() -What 'the prompt file'
  OutRoot  = $(
      $o = [Environment]::GetEnvironmentVariable('HIPEP_OUT')
      if (-not $o) { $o = Join-Path $env:TEMP 'hipep-perf' }
      $o)
}

# Extra directories the EP needs on PATH (ROCm SDK runtime libraries, etc).
# Semicolon-separated; prepended ahead of the system PATH.
function Set-HarnessPath {
  $extra = [Environment]::GetEnvironmentVariable('HIPEP_PATH_EXTRA')
  $parts = @($HarnessEnv.Bin)
  if ($extra) { $parts += ($extra -split ';' | Where-Object { $_ }) }
  $env:PATH = ($parts -join ';') + ';' + $env:PATH
}

# Throughput must never be measured with the EP's own instrumentation: PERF
# alone costs ~4%. SQTT carries the timing instead.
#
# HIPDNN_EP_TRACE_FILE has to go too: hipdnn_ep_perf_enabled() is true for
# either variable, so a trace path left over from an earlier shell turns the
# profiler on with none of PERF's console output to give it away. That leak
# inflated a 16K VLM baseline from 14,610 ms to 17,545 ms before it was found.
function Clear-HarnessProfilingEnv {
  Remove-Item Env:HIPDNN_EP_PERF, Env:HIPDNN_EP_DEBUG, Env:HIPDNN_EP_TRACE_FILE `
    -EA SilentlyContinue
}

function Stop-HarnessProcesses {
  # -IncludePython only when the vlm driver is in use: killing every python on
  # the box would be unacceptable collateral otherwise.
  param([switch]$IncludePython)
  $names = @('model_benchmark', 'RadeonDeveloper*')
  if ($IncludePython) { $names += 'python' }
  Get-Process $names -EA SilentlyContinue |
    Stop-Process -Force -EA SilentlyContinue
  Start-Sleep 2
}
