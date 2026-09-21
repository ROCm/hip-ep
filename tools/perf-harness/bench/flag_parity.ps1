##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

## Prove a default-off flag changes nothing but speed, on one model or on a set.
##
## Runs flag_parity.py twice per model -- once with the flag unset, once with it
## set -- in separate processes, because the runtime latches these flags in a
## function-local static on first read and a single process therefore cannot
## observe both states. Then compares the greedy token chain and the per-step
## logits.
##
## The EP DLLs the Python driver loads are the ones in the venv's
## onnxruntime\capi, NOT $HIPEP_BIN, so -SyncDlls copies the current $HIPEP_BIN
## set over them first. Skipping that silently tests whatever was there before,
## which is the same class of failure build_deploy.ps1 exists to prevent.

param(
  [Parameter(Mandatory = $true)][string]$Flag,       # e.g. HIPDNN_EP_GQA_FUSE_APPEND
  [string]$OnValue  = '1',
  # Model directories to check. Default: the five-model regression guard.
  [string[]]$Models,
  [int]$Steps = 24,
  [int]$PromptRepeat = 1,
  [double]$MinCos = 0.9999,
  [string]$Capi,                                     # venv onnxruntime\capi holding the EP DLLs
  [string]$Image,                                    # vision models need one
  [switch]$SyncDlls,
  # Let the autotuners run. Off by default -- see the note below; leaving them
  # free makes a correct change look wrong.
  [switch]$FreeAutotune,
  [string]$OutDir
)

$ErrorActionPreference = 'Stop'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'common.ps1')

if (-not $Models) {
  $Models = @(
    'C:\Users\zyq\Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-dml-pruned_lm_head'
    'C:\Users\zyq\gpt-oss-20b-webgpu-int4-rtn-block-32-pruned_lm_head'
    'C:\Users\zyq\gemma3-4b-it-rtn-int4-128gs-fp16-onnx-gpu-pruned_lm_head'
    'C:\Users\zyq\Qwen3.6-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml'
    'C:\Users\zyq\Qwen3.8-27B-fp16-ve-fp16-int4-k_quant-gs128-text-dml-onnx'
  )
}
# gemma3 is a vision export and refuses a text-only prefill, so the guard set
# cannot be driven without one.
if (-not $Image) { $Image = 'C:\Users\zyq\dog.jpg' }
if (-not $Capi) { $Capi = 'C:\Users\zyq\bench-venv\Lib\site-packages\onnxruntime\capi' }
# The interpreter has to be the one whose onnxruntime-genai knows the AMDGPU
# provider. A stock PyPI onnxruntime-genai does not, and fails at og.Model with
# "Unknown provider name 'AMDGPU'" -- which reads like a model problem, so say
# it here instead. $HIPEP_PY, or the venv that owns $Capi.
if (-not (Test-Path $HarnessEnv.Python)) {
  throw "HIPEP_PY does not point at an interpreter: $($HarnessEnv.Python)"
}
$ogCheck = & $HarnessEnv.Python -c "import onnxruntime_genai" 2>&1
if ($LASTEXITCODE -ne 0) {
  throw "$($HarnessEnv.Python) cannot import onnxruntime_genai. Set `$env:HIPEP_PY to the venv that owns $Capi.`n$ogCheck"
}
if (-not $OutDir) { $OutDir = Join-Path $HarnessEnv.OutRoot 'parity' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$py = Join-Path $PSScriptRoot 'flag_parity.py'

if ($SyncDlls) {
  Write-Host ">>> sync EP DLLs $($HarnessEnv.Bin) -> $Capi"
  foreach ($n in 'hipgpu.dll', 'custom_kernels_*.dll', 'HipFusionPatterns.pdl.mlir') {
    Get-Item (Join-Path $HarnessEnv.Bin $n) -EA SilentlyContinue | ForEach-Object {
      Copy-Item $_.FullName (Join-Path $Capi $_.Name) -Force
      Write-Host ("    {0,-32} {1}" -f $_.Name,
                  (Get-FileHash (Join-Path $Capi $_.Name) -Algorithm SHA256).Hash.Substring(0, 16))
    }
  }
}

Set-HarnessPath
Clear-HarnessProfilingEnv

# Pin the GQA flash-decode config for BOTH arms.
#
# This is not a convenience. The decode autotuner picks a split-K count by
# timing candidates, and split-K fixes the order the fp32 partials are reduced
# in. Any change that adds or removes dispatches shifts those timings, so it can
# tip a marginal split choice -- and a different reduction order is a different
# fp16 rounding, which shows up as a logit difference that has nothing to do
# with the code under test.
#
# Measured on HIPDNN_EP_GQA_FUSE_APPEND: free-running, off-vs-on gave a worst
# per-step cosine of 0.9970 and looked like a kernel bug. Pinned, the same two
# arms are bit-identical (1.00000000 at every step). The control that made this
# readable was off-vs-off, which is exactly 1.0 either way -- so the pipeline is
# deterministic run to run, and the only thing free autotune adds is a
# difference correlated with the flag.
#
# Pinning the GQA tuner is necessary but not sufficient, and the gap is why
# this script also runs a control. gemma3-4b reaches only ~0.9983 worst-step
# cosine when compared against *itself* -- two processes, identical code, both
# arms off. HIPDNN_EP_AUTOTUNE=0 does not close it either, so hipBLASLt's
# per-process kernel choice is not the whole story; something in that model's
# path is nondeterministic run to run.
#
# A fixed 0.9999 bar is therefore unreadable for it: the model fails on a
# comparison where there is nothing to find, on every item, and the only way to
# clear it is to argue the number away. So the verdict below is relative -- each
# model's own off-vs-off divergence is measured first, and the flag has to not
# be materially worse than it. A model that is bit-identical to itself still
# gets held to bit-identical.
if (-not $FreeAutotune) {
  $env:HIPDNN_GQA_DECODE_NOAUTOTUNE = '1'
  Write-Host "    pinned: HIPDNN_GQA_DECODE_NOAUTOTUNE=1 (both arms)"
} else {
  Remove-Item Env:HIPDNN_GQA_DECODE_NOAUTOTUNE -EA SilentlyContinue
  Write-Host "    WARNING: autotune free; a config flip will read as a numerical difference"
}

$results = @()
foreach ($model in $Models) {
  $name = Split-Path -Leaf $model
  if (-not (Test-Path (Join-Path $model 'genai_config.json'))) {
    Write-Host "--- $name : SKIP (no genai_config.json)"
    continue
  }
  Write-Host "`n=== $name"

  # One shared, pre-warmed autotune cache for both arms.
  #
  # Unlike the A/B, parity wants the arms to agree on kernel CONFIG so the only
  # thing left to differ is the code under test. Left alone they do not: the
  # first process to touch a model tunes by timing and writes a cache the second
  # then reads, so whichever arm ran first is the cold one. That artefact put
  # gpt-oss-20b at 0.99976 and looked exactly like a kernel bug -- rerun with the
  # cache already warm, the same two arms are bit-identical.
  #
  # Priming runs with the flag OFF, for the SAME number of steps as the
  # measurement, and its output is discarded. Full length matters: a short prime
  # leaves shapes the later steps reach still untuned, and gpt-oss-20b was still
  # 0.99976 after a two-step prime. The control that settled it ran the flag-OFF
  # arm twice against two different cold caches and reproduced 0.99975781
  # exactly -- same arm, same flag, so the divergence was never the flag.
  # Twice, because one pass is not always enough: gemma3-4b's first run against
  # a freshly primed cache still differed from its second by 0.9983 with the
  # flag OFF on both, and that lands on whichever arm is recorded first.
  $temp = Join-Path $OutDir "tune.$name"
  New-Item -ItemType Directory -Force -Path $temp | Out-Null
  $env:TEMP = $temp; $env:TMP = $temp
  Remove-Item "Env:$Flag" -EA SilentlyContinue
  for ($p = 0; $p -lt 2; $p++) {
    Stop-HarnessProcesses
    & $HarnessEnv.Python '-u' $py --model $model --steps $Steps `
        --prompt-repeat $PromptRepeat --image $Image `
        --out (Join-Path $temp "prime$p.npz") 2>&1 | Out-Null
  }
  Write-Host "    primed x2 (autotune settled, cache shared by both arms)"

  # Three arms, in this order: off, the control (off again, a third process),
  # then on. The control is what makes a sub-bar result readable -- without it,
  # a model that cannot reproduce itself is indistinguishable from a model the
  # flag broke, and the only way to tell them apart is to run this by hand.
  # 'off' is recorded first so both comparisons share the same reference.
  $dumps = @{}
  foreach ($state in 'off', 'control', 'on') {
    if ($state -eq 'on') { Set-Item "Env:$Flag" $OnValue } else { Remove-Item "Env:$Flag" -EA SilentlyContinue }
    $dump = Join-Path $OutDir "$name.$Flag.$state.npz"
    $dumps[$state] = $dump
    Stop-HarnessProcesses
    & $HarnessEnv.Python '-u' $py --model $model --steps $Steps `
        --prompt-repeat $PromptRepeat --image $Image --out $dump 2>&1 |
      Where-Object { $_ -match 'recorded|tokens:|Error|error|Traceback' } |
      ForEach-Object { Write-Host "    [$state] $_" }
  }
  Remove-Item "Env:$Flag" -EA SilentlyContinue

  if (-not (Test-Path $dumps['off']) -or -not (Test-Path $dumps['on'])) {
    Write-Host "    RESULT: FAIL (a dump is missing)"
    $results += [PSCustomObject]@{ model = $name; verdict = 'FAIL (no dump)' }
    continue
  }
  $ctlArg = if (Test-Path $dumps['control']) { @('--control', $dumps['control']) } else { @() }
  $out = & $HarnessEnv.Python $py --compare $dumps['off'] $dumps['on'] `
      --min-cos $MinCos @ctlArg 2>&1
  $out | ForEach-Object { Write-Host "    $_" }
  $verdict = if ($out -match 'PARITY OK') { 'OK' } else { 'FAIL' }
  # Distinguish "identical" from "inside this model's noise" in the summary --
  # they are different strengths of evidence and the table should not blur them.
  if ($verdict -eq 'OK' -and ($out -match 'noise budget')) { $verdict = 'OK (noise)' }
  elseif ($verdict -eq 'OK' -and ($out -match 'bitwise equal')) { $verdict = 'OK (bitwise)' }
  $results += [PSCustomObject]@{ model = $name; verdict = $verdict }
}

Remove-Item Env:HIPDNN_GQA_DECODE_NOAUTOTUNE -EA SilentlyContinue
Remove-Item Env:HIPDNN_EP_AUTOTUNE -EA SilentlyContinue

Write-Host "`n=== $Flag parity summary"
$results | Format-Table -AutoSize | Out-String -Width 160 | Write-Host
if (($results.verdict | Where-Object { $_ -like 'FAIL*' })) {
  Write-Host "PARITY FAILED -- do not proceed to the A/B."
  exit 1
}
Write-Host "All models clear."
