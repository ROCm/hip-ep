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
## with a real prompt file, and none of the EP env flags CI does not set), so a
## number produced here is comparable to the CI baseline without a translation
## step. A local build that cannot reproduce CI is the first thing to find out,
## and it is cheap to check.
##
## Deliberately no HIPDNN_EP_PERF / HIPDNN_EP_DEBUG: PERF's per-inference stream
## sync serialises consecutive inferences and lowers measured TPS specifically.
##
## -PerfOps is the one exception, and it measures something else: per-family GPU
## time from event pairs around one op family (HIPDNN_EP_PERF_OPS), one family
## per process. Its TPS is not valid and is not appended to the CSV; the family
## ms per decode step are the output, and calibrate_sqtt.py turns them into
## per-family SQTT factors.

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
  # 'vlm' drives vlm_benchmark.py with an image, for multimodal exports that
  # model_benchmark cannot run. Needs a prompt file, like bench_ttft.ps1.
  [ValidateSet('model_benchmark', 'vlm')]
  [string]$Driver = 'model_benchmark',
  [int]$MaxLength,                  # vlm only: KV cache size; defaults to SeqLen + Gen
  [string]$ExecutionProvider = 'follow_config',   # vlm only; see bench_ttft.ps1
  # Op families (OP_PROFILE names, e.g. matmul_nbits, qmoe, gqa) to event-time,
  # one process each. See the header.
  [string[]]$PerfOps = @(),
  [string]$OutDir
)

$ErrorActionPreference = 'Continue'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'common.ps1')

if (-not $OutDir) { $OutDir = Join-Path $HarnessEnv.OutRoot 'tps' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$log = Join-Path $OutDir "tps_$Tag.log"
$csv = Join-Path $OutDir 'tps_summary.csv'

if ($Driver -eq 'vlm') {
  foreach ($n in 'VlmBench', 'Image') {
    if (-not $HarnessEnv.$n) { throw "Driver 'vlm' needs `$env:HIPEP_$($n.ToUpper()); see common.ps1." }
  }
  if (-not $PromptFile) { $PromptFile = $HarnessEnv.PromptFile }
  if (-not $PromptFile) { throw "Driver 'vlm' needs -PromptFile or `$env:HIPEP_PROMPT_FILE; see common.ps1." }
  if (-not $MaxLength) { $MaxLength = $SeqLen + $Gen }
}
if ($PromptFile) {
  if (-not (Test-Path $PromptFile)) { throw "Prompt file not found: $PromptFile" }
  $PromptFile = (Resolve-Path $PromptFile).Path
}

Set-HarnessPath
Clear-HarnessProfilingEnv
# A fence left armed in the shell idles the GPU mid-run and destroys the
# measurement, so clear it here as well as in the capture script.
Remove-Item Env:RGP_FENCE, Env:RGP_FENCE_SKIP, Env:RGP_FENCE_MS,
            Env:RGP_FENCE_AFTER_INFERENCES -EA SilentlyContinue
$savedEnv = Set-HarnessEnv $SetEnv
try {

# One driver run into $LogPath; returns the exit code. The vlm driver also
# writes its own summary JSON, which is the parse target on that path.
function Invoke-Driver {
  param([string]$LogPath, [string]$JsonPath)
  Stop-HarnessProcesses
  if ($Driver -eq 'vlm') {
    Remove-Item $JsonPath -EA SilentlyContinue
    & $HarnessEnv.Python '-u' (Join-Path $PSScriptRoot 'vlm_driver.py') `
      '--bin' $HarnessEnv.Bin '--prompt-file' $PromptFile $HarnessEnv.VlmBench `
      '-m' $HarnessEnv.Model '-i' $HarnessEnv.Image `
      '--max_tokens' "$Gen" '--max_length' "$MaxLength" `
      '-e' $ExecutionProvider '-n' "$Reps" '-w' "$Warmup" '-o' $JsonPath *>&1 |
      Tee-Object -FilePath $LogPath | Out-Null
    return $LASTEXITCODE
  }
  $margs = @('-i', $HarnessEnv.Model, '-g', "$Gen", '-r', "$Reps", '-w', "$Warmup", '-b', '1', '-ml', '0', '-v')
  if ($PromptFile) { $margs += @('--prompt_file', $PromptFile) }
  else             { $margs += @('-l', "$SeqLen", '--use_random_tokens') }
  Push-Location $HarnessEnv.Bin
  & (Join-Path $HarnessEnv.Bin 'model_benchmark.exe') @margs *>&1 |
    Tee-Object -FilePath $LogPath | Out-Null
  $rc = $LASTEXITCODE
  Pop-Location
  return $rc
}

$workload = if ($PromptFile) { "prompt=$(Split-Path -Leaf $PromptFile)" } else { "seqlen=$SeqLen (random)" }
Get-Item (Join-Path $HarnessEnv.Bin 'custom_kernels_*.dll'), (Join-Path $HarnessEnv.Bin 'hipgpu.dll') -EA SilentlyContinue |
  ForEach-Object { Write-Host ("      {0}  {1:N2} MB  {2}" -f $_.LastWriteTime, ($_.Length / 1MB), $_.Name) }

if ($PerfOps) {
  $logs = @()
  foreach ($op in $PerfOps) {
    Write-Host ">>> PERF_OPS [$Tag] op=$op driver=$Driver $workload -g $Gen -r $Reps -w $Warmup (TPS not valid)"
    $opLog = Join-Path $OutDir "tps_$Tag.perfops_$op.log"
    $env:HIPDNN_EP_PERF_OPS = $op
    $rc = Invoke-Driver -LogPath $opLog -JsonPath (Join-Path $OutDir "tps_$Tag.perfops_$op.json")
    Remove-Item Env:HIPDNN_EP_PERF_OPS -EA SilentlyContinue
    Write-Host "    exit=$rc -> $opLog"
    $logs += @('--perf-log', $opLog)
  }
  & $HarnessEnv.Python (Join-Path $HarnessEnv.Harness 'analysis\calibrate_sqtt.py') @logs
  return
}

Write-Host ">>> TPS [$Tag] driver=$Driver $workload -g $Gen -r $Reps -w $Warmup"
$json = Join-Path $OutDir "tps_$Tag.json"
$rc = Invoke-Driver -LogPath $log -JsonPath $json

# model_benchmark prints three timing blocks with identical field names, so the
# numbers have to be read from inside the right block rather than by grepping
# for "tokens/s" -- the first match is the prefill and the last is the sampler.
function Get-Block {
  param([string[]]$Lines, [string]$Header)
  for ($i = 0; $i -lt $Lines.Count; $i++) {
    if ($Lines[$i] -notmatch $Header) { continue }
    $o = [ordered]@{}
    # Stop at the next header rather than reading a fixed number of lines. The
    # three blocks are adjacent and share field names, so a window wide enough
    # for one of them reaches into the next and the later "avg (us)" -- the
    # sampler's 14 us -- silently overwrites token generation's 16,600.
    for ($j = $i + 1; $j -lt $Lines.Count; $j++) {
      if ($Lines[$j] -notmatch '^\s') { break }
      if ($Lines[$j] -match 'avg \(us\):\s+([\d.eE+\-]+)')        { $o.avg_us    = [double]$matches[1] }
      if ($Lines[$j] -match 'avg \(tokens/s\):\s+([\d.eE+\-]+)')  { $o.tps       = [double]$matches[1] }
      if ($Lines[$j] -match 'p50 \(us\):\s+([\d.eE+\-]+)')        { $o.p50_us    = [double]$matches[1] }
      if ($Lines[$j] -match 'stddev \(us\):\s+([\d.eE+\-]+)')     { $o.stddev_us = [double]$matches[1] }
      if ($Lines[$j] -match 'n:\s+(\d+)')                         { $o.n         = [int]$matches[1] }
    }
    return [PSCustomObject]$o
  }
  return $null
}

# Not $gen: PowerShell variable names are case-insensitive, so $gen IS the
# [int]$Gen parameter, and assigning a parsed object to it fails the cast.
$genBlk = $null; $preBlk = $null; $prompt = $SeqLen
if ($Driver -eq 'vlm') {
  if (Test-Path $json) {
    $s = (Get-Content $json -Raw | ConvertFrom-Json).summary
    $tg = $s.token_generation_ms
    $genBlk = [PSCustomObject]@{ avg_us = $tg.avg * 1000; tps = $tg.tps_avg
                                 p50_us = $tg.p50 * 1000; stddev_us = $tg.std * 1000
                                 n = $s.total_runs }
    $preBlk = [PSCustomObject]@{ avg_us = $s.ttft_ms.avg * 1000; tps = $s.prefill_throughput_tps.avg }
    $prompt = $s.token_breakdown.prompt_tokens
  }
} else {
  $lines  = Get-Content $log
  $genBlk = Get-Block -Lines $lines -Header 'Token generation'
  $preBlk = Get-Block -Lines $lines -Header 'Prompt processing \(time to first token\)'
  if ($lines -match 'prompt tokens: (\d+)') {
    $prompt = [int]([regex]::Match(($lines -match 'prompt tokens: \d+')[0], 'prompt tokens: (\d+)').Groups[1].Value)
  }
}

if ($genBlk -and $genBlk.tps) {
  Write-Host ("`n=== TPS [$Tag] = {0:N2} tok/s   ({1:N2} ms/token, p50 {2:N2}, stddev {3:N2})   n={4} (exit={5})" -f `
              $genBlk.tps, ($genBlk.avg_us / 1000), ($genBlk.p50_us / 1000), ($genBlk.stddev_us / 1000), $genBlk.n, $rc)
  if ($preBlk) {
    Write-Host ("    TTFT {0:N0} ms ({1:N1} tok/s prefill, {2} prompt tokens)" -f `
                ($preBlk.avg_us / 1000), $preBlk.tps, $prompt)
  }
  [PSCustomObject]@{
    tag = $Tag
    prompt_tokens = $prompt
    tps = [Math]::Round($genBlk.tps, 2)
    ms_per_token = [Math]::Round($genBlk.avg_us / 1000, 3)
    p50_ms = [Math]::Round($genBlk.p50_us / 1000, 3)
    stddev_ms = [Math]::Round($genBlk.stddev_us / 1000, 3)
    ttft_ms = if ($preBlk) { [Math]::Round($preBlk.avg_us / 1000, 1) } else { $null }
    gen = $Gen; reps = $Reps; n = $genBlk.n
    when = (Get-Date -Format s)
  } | Export-Csv -Path $csv -NoTypeInformation -Append
  Write-Host "    appended -> $csv"
} else {
  Write-Host "`n=== TPS [$Tag] PARSE FAILED (exit=$rc) -- inspect $log"
  Get-Content $log -Tail 25
}

} finally { Restore-HarnessEnv $savedEnv }
