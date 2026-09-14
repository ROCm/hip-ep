##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
## One clean decode-throughput (TPS) measurement, appended to a CSV.
##
## The sibling bench_ttft.ps1 measures the prefill; this measures the token
## generation that follows it. They are separate scripts because they want
## opposite workloads: TTFT wants -g 1 so nothing but the prefill is timed,
## while TPS needs enough generated tokens for the steady state to dominate the
## first-token transient.
##
## Defaults mirror the Jenkins genai case exactly (-g 128 -r 3 -w 1 -b 1 -ml 0
## with a real prompt file), so a number produced here is comparable to the CI
## baseline without a translation step. A local build that cannot reproduce CI
## is the first thing to find out, and it is cheap to check.
##
## Deliberately no HIPDNN_EP_PERF / HIPDNN_EP_DEBUG: PERF's per-inference stream
## sync serialises consecutive inferences and lowers measured TPS specifically.

param(
  [Parameter(Mandatory = $true)][string]$Tag,
  # A real prompt file is the CI workload. --use_random_tokens is the fallback
  # when the Run arithmetic has to be exact (see rgp_capture.ps1 -DecodeStep).
  [string]$PromptFile,
  [int]$SeqLen = 128,
  [int]$Gen    = 128,
  [int]$Reps   = 3,
  [int]$Warmup = 1,
  [string[]]$SetEnv = @(),
  [string]$OutDir
)

$ErrorActionPreference = 'Continue'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'common.ps1')

if (-not $OutDir) { $OutDir = Join-Path $HarnessEnv.OutRoot 'tps' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$log = Join-Path $OutDir "tps_$Tag.log"
$csv = Join-Path $OutDir 'tps_summary.csv'

Set-HarnessPath
Clear-HarnessProfilingEnv
# A fence left armed in the shell idles the GPU mid-run and destroys the
# measurement, so clear it here as well as in the capture script.
Remove-Item Env:RGP_FENCE, Env:RGP_FENCE_SKIP, Env:RGP_FENCE_MS,
            Env:RGP_FENCE_AFTER_INFERENCES -EA SilentlyContinue
$env:HIPDNN_EP_AUTOTUNE = '1'
$env:HIPDNN_EP_MATMUL_CUSTOM_WMMA = '1'
foreach ($kv in $SetEnv) {
  $k, $v = $kv -split '=', 2
  Set-Item -Path "Env:$k" -Value $v
  Write-Host "    env $k=$v"
}

Stop-HarnessProcesses

$workload = if ($PromptFile) { "prompt=$(Split-Path -Leaf $PromptFile)" } else { "seqlen=$SeqLen (random)" }
Write-Host ">>> TPS [$Tag] $workload -g $Gen -r $Reps -w $Warmup"
Get-Item (Join-Path $HarnessEnv.Bin 'custom_kernels_*.dll'), (Join-Path $HarnessEnv.Bin 'hipgpu.dll') -EA SilentlyContinue |
  ForEach-Object { Write-Host ("      {0}  {1:N2} MB  {2}" -f $_.LastWriteTime, ($_.Length / 1MB), $_.Name) }

$margs = @('-i', $HarnessEnv.Model, '-g', "$Gen", '-r', "$Reps", '-w', "$Warmup", '-b', '1', '-ml', '0', '-v')
if ($PromptFile) { $margs += @('--prompt_file', $PromptFile) }
else             { $margs += @('-l', "$SeqLen", '--use_random_tokens') }

Push-Location $HarnessEnv.Bin
& (Join-Path $HarnessEnv.Bin 'model_benchmark.exe') @margs *>&1 |
  Tee-Object -FilePath $log | Out-Null
$rc = $LASTEXITCODE
Pop-Location

# model_benchmark prints three timing blocks with identical field names, so the
# numbers have to be read from inside the right block rather than by grepping
# for "tokens/s" -- the first match is the prefill and the last is the sampler.
function Get-Block {
  param([string[]]$Lines, [string]$Header)
  for ($i = 0; $i -lt $Lines.Count; $i++) {
    if ($Lines[$i] -match $Header) {
      $o = [ordered]@{}
      foreach ($j in ($i + 1)..([Math]::Min($i + 7, $Lines.Count - 1))) {
        if ($Lines[$j] -match 'avg \(us\):\s+([\d.eE+\-]+)')        { $o.avg_us    = [double]$matches[1] }
        if ($Lines[$j] -match 'avg \(tokens/s\):\s+([\d.eE+\-]+)')  { $o.tps       = [double]$matches[1] }
        if ($Lines[$j] -match 'p50 \(us\):\s+([\d.eE+\-]+)')        { $o.p50_us    = [double]$matches[1] }
        if ($Lines[$j] -match 'stddev \(us\):\s+([\d.eE+\-]+)')     { $o.stddev_us = [double]$matches[1] }
        if ($Lines[$j] -match 'n:\s+(\d+)')                         { $o.n         = [int]$matches[1] }
      }
      return [PSCustomObject]$o
    }
  }
  return $null
}

$lines = Get-Content $log
$gen  = Get-Block -Lines $lines -Header 'Token generation'
$pre  = Get-Block -Lines $lines -Header 'Prompt processing \(time to first token\)'
$prompt = if ($lines -match 'prompt tokens: (\d+)') {
  [int]([regex]::Match(($lines -match 'prompt tokens: \d+')[0], 'prompt tokens: (\d+)').Groups[1].Value)
} else { $SeqLen }

if ($gen -and $gen.tps) {
  Write-Host ("`n=== TPS [$Tag] = {0:N2} tok/s   ({1:N2} ms/token, p50 {2:N2}, stddev {3:N2})   n={4} (exit={5})" -f `
              $gen.tps, ($gen.avg_us / 1000), ($gen.p50_us / 1000), ($gen.stddev_us / 1000), $gen.n, $rc)
  if ($pre) {
    Write-Host ("    TTFT {0:N0} ms ({1:N1} tok/s prefill, {2} prompt tokens)" -f `
                ($pre.avg_us / 1000), $pre.tps, $prompt)
  }
  [PSCustomObject]@{
    tag = $Tag
    prompt_tokens = $prompt
    tps = [Math]::Round($gen.tps, 2)
    ms_per_token = [Math]::Round($gen.avg_us / 1000, 3)
    p50_ms = [Math]::Round($gen.p50_us / 1000, 3)
    stddev_ms = [Math]::Round($gen.stddev_us / 1000, 3)
    ttft_ms = if ($pre) { [Math]::Round($pre.avg_us / 1000, 1) } else { $null }
    gen = $Gen; reps = $Reps; n = $gen.n
    when = (Get-Date -Format s)
  } | Export-Csv -Path $csv -NoTypeInformation -Append
  Write-Host "    appended -> $csv"
} else {
  Write-Host "`n=== TPS [$Tag] PARSE FAILED (exit=$rc) -- inspect $log"
  Get-Content $log -Tail 25
}
