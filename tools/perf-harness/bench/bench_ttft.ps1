##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
## One clean TTFT measurement, appended to a CSV.
##
## Deliberately no HIPDNN_EP_PERF / HIPDNN_EP_DEBUG: CLAUDE.md forbids measuring
## throughput with either (PERF alone costs ~4%).
##
## Warmup reps are not optional. Autotune caches are per-process and partly
## on-disk keyed by build timestamp, so iteration 1 of a fresh build always pays
## tuning cost and is not a measurement of the kernel.

param(
  [Parameter(Mandatory = $true)][string]$Tag,
  [int]$SeqLen = 16384,
  [int]$Reps   = 4,
  [int]$Warmup = 1,
  [string[]]$SetEnv = @(),          # e.g. -SetEnv 'HIPDNN_EP_GQA_NO_EXPAND_PREFILL=1'
  [string]$OutDir,
  # 'vlm' drives vlm_benchmark.py so the vision encoder and image-token splice
  # are inside TTFT, which is the whole number for a VLM. 'model_benchmark' is
  # the text-only path and cannot represent a multimodal prefill.
  [ValidateSet('model_benchmark', 'vlm')]
  [string]$Driver = 'model_benchmark',
  [int]$MaxTokens = 4,              # vlm only: keep decode short, TTFT is the target
  [int]$MaxLength,                  # vlm only: KV cache size; defaults to prompt + headroom
  # vlm only. 'follow_config' leaves the model's own genai_config provider list
  # alone; naming a provider overrides it, which is what an export pinned to
  # another EP (a -dml directory, say) needs to run here.
  [string]$ExecutionProvider = 'follow_config'
)

$ErrorActionPreference = 'Continue'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'common.ps1')

if (-not $OutDir) { $OutDir = Join-Path $HarnessEnv.OutRoot 'ttft' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$log = Join-Path $OutDir "ttft_$Tag.log"
$csv = Join-Path $OutDir 'ttft_summary.csv'

Set-HarnessPath
Clear-HarnessProfilingEnv
Remove-Item Env:RGP_FENCE, Env:RGP_FENCE_SKIP, Env:RGP_FENCE_MS -EA SilentlyContinue
$env:HIPDNN_EP_AUTOTUNE = '1'
$env:HIPDNN_EP_MATMUL_CUSTOM_WMMA = '1'
foreach ($kv in $SetEnv) {
  $k, $v = $kv -split '=', 2
  Set-Item -Path "Env:$k" -Value $v
  Write-Host "    env $k=$v"
}

# Serial only: concurrent GPU runs invalidate results.
Stop-HarnessProcesses -IncludePython:($Driver -eq 'vlm')

Write-Host ">>> TTFT [$Tag] driver=$Driver seqlen=$SeqLen -r $Reps -w $Warmup"
Get-Item (Join-Path $HarnessEnv.Bin 'custom_kernels_*.dll'), (Join-Path $HarnessEnv.Bin 'hipgpu.dll') -EA SilentlyContinue |
  ForEach-Object { Write-Host ("      {0}  {1:N2} MB  {2}" -f $_.LastWriteTime, ($_.Length / 1MB), $_.Name) }

# TTFT in microseconds, so both drivers land in the same CSV units.
$ttft = $null; $p50 = $null; $sd = $null
$tokens = ''

if ($Driver -eq 'vlm') {
  foreach ($n in 'VlmBench', 'Image', 'PromptFile') {
    if (-not $HarnessEnv.$n) { throw "Driver 'vlm' needs `$env:HIPEP_$($n.ToUpper()); see common.ps1." }
  }
  # vlm_benchmark's own --output_json is the parse target: scraping the pretty
  # printed block would break on every formatting change.
  $json = Join-Path $OutDir "ttft_$Tag.json"
  Remove-Item $json -EA SilentlyContinue
  if (-not $MaxLength) { $MaxLength = $SeqLen + 128 }

  & $HarnessEnv.Python '-u' $HarnessEnv.VlmBench `
    '-m' $HarnessEnv.Model '-i' $HarnessEnv.Image '--prompt_file' $HarnessEnv.PromptFile `
    '--max_tokens' "$MaxTokens" '--max_length' "$MaxLength" `
    '-e' $ExecutionProvider `
    '-n' "$Reps" '-w' "$Warmup" '-o' $json *>&1 |
    Tee-Object -FilePath $log | Out-Null
  $rc = $LASTEXITCODE

  if (Test-Path $json) {
    $j = Get-Content $json -Raw | ConvertFrom-Json
    $ttft = $j.summary.ttft_ms.avg * 1000
    $p50  = $j.summary.ttft_ms.p50 * 1000
    $sd   = $j.summary.ttft_ms.std * 1000
    $tb   = $j.summary.token_breakdown
    $SeqLen = $tb.prompt_tokens
    $tokens = "prompt=$($tb.prompt_tokens) text=$($tb.text_tokens) image=$($tb.image_tokens)"
  }
} else {
  Push-Location $HarnessEnv.Bin
  & (Join-Path $HarnessEnv.Bin 'model_benchmark.exe') `
    -i $HarnessEnv.Model -l $SeqLen --use_random_tokens -g 1 -r $Reps -w $Warmup -b 1 -ml 0 -v *>&1 |
    Tee-Object -FilePath $log | Out-Null
  $rc = $LASTEXITCODE
  Pop-Location

  # Pull the TTFT block: "Prompt processing (time to first token):" then "avg (us):".
  # model_benchmark switches to scientific notation once the fused paths get fast
  # enough (1.00924e+07), so the number pattern must accept both forms.
  $lines = Get-Content $log
  for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match 'Prompt processing \(time to first token\)') {
      foreach ($j in ($i + 1)..([Math]::Min($i + 6, $lines.Count - 1))) {
        if ($lines[$j] -match 'avg \(us\):\s+([\d.eE+\-]+)')    { $ttft = [double]$matches[1] }
        if ($lines[$j] -match 'p50 \(us\):\s+([\d.eE+\-]+)')    { $p50  = [double]$matches[1] }
        if ($lines[$j] -match 'stddev \(us\):\s+([\d.eE+\-]+)') { $sd   = [double]$matches[1] }
      }
      break
    }
  }
}

if ($ttft) {
  Write-Host ("`n=== TTFT [$Tag] = {0:N0} ms   p50 {1:N0} ms   stddev {2:N0} ms   (exit={3})" -f `
              ($ttft / 1000), ($p50 / 1000), ($sd / 1000), $rc)
  if ($tokens) { Write-Host "    $tokens" }
  [PSCustomObject]@{
    tag = $Tag; ttft_ms = [Math]::Round($ttft / 1000, 1); p50_ms = [Math]::Round($p50 / 1000, 1)
    stddev_ms = [Math]::Round($sd / 1000, 1); seqlen = $SeqLen; reps = $Reps
    when = (Get-Date -Format s)
  } | Export-Csv -Path $csv -NoTypeInformation -Append
  Write-Host "    appended -> $csv"
} else {
  Write-Host "`n=== TTFT [$Tag] PARSE FAILED (exit=$rc) -- inspect $log"
  Get-Content $log -Tail 25
}
