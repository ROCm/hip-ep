##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
## Interleaved, order-reversed A/B (or A/B/C/...) of build variants, by TTFT
## (-Metric ttft, the default) or by decode throughput (-Metric tps).
##
## Why not just run arm A a few times, then arm B a few times:
##
##  - Run-to-run spread on a thermally managed APU is comparable to the effects
##    worth shipping (~0.8% vs ~1-2%), so block-sequential runs let slow drift
##    land entirely on one arm. Interleaving and pairing by round cancels it.
##  - Position within a round is itself a bias. Always run at least one block
##    with -Reverse and check the delta survives; if it flips, you measured the
##    order, not the change.
##  - Discard rounds taken while the box is shedding heat from a build or a test
##    suite. They report a different answer -- in one measured case the opposite
##    sign -- and they are recognisable by sitting well above the known baseline.
##  - Do NOT compare absolute numbers across invocations, and never add two
##    deltas measured in different ones. The baseline drifts DOWNWARD over a
##    session as the OS page cache and the driver's shader cache warm: one
##    measured case walked 2112 -> 1937 -> 1745 ms on the same model, prompt and
##    settings, 21% with nothing changed. That is larger than most effects worth
##    shipping, and because it moves downward the "sits well above the baseline"
##    test above does not catch it. An A/A run (two identically configured arms
##    under different names) came out at -0.29% with the interval spanning zero,
##    so the interleaving does cancel it -- which is exactly why a delta has to
##    come from one invocation, and why stacking two of them has to be measured
##    as its own arm rather than inferred.
##
## Each arm gets its OWN TEMP. This no longer isolates an autotune cache -- the
## tuners memoize in-process only and the %TEMP% cache file is gone (see the
## header of matmul_nbits_kernel.hip; the offline LUT is the durable source
## now). What it still isolates is HIP's JIT/code-object cache, which is worth
## keeping separate across a DLL swap.
##
## Arms are defined in a JSON manifest. An arm may swap a DLL, set environment
## variables, or both:
##
##   { "base":  { "dll": "D:\\builds\\base\\custom_kernels_gfx1151.dll" },
##     "cand":  { "dll": "D:\\builds\\cand\\custom_kernels_gfx1151.dll" } }
##
##   { "off":   { "env": { "HIPDNN_EP_GQA_FUSE_APPEND": "0" } },
##     "on":    { "env": { "HIPDNN_EP_GQA_FUSE_APPEND": "1" } } }
##
## The named DLL is copied over the same filename in $env:HIPEP_BIN before each
## run, so every arm is measured through one identical harness.
##
## The env form exists because every optimisation here lands behind a default-off
## flag, and A/B-ing a flag on ONE binary removes the last way a DLL-swap A/B can
## lie: the two DLLs cannot differ in anything but the change, because there is
## only one of them. Arms still get their own TEMP even in that form -- a flag
## can change which autotune config wins, so a shared cache would carry the
## other arm's answer.
##
## Env keys are unioned across all arms and every key is written for every arm
## (absent ones removed), because Env: is process-wide: without that, arm A's
## variables survive into arm B and quietly make it a second copy of arm A.

param(
  [Parameter(Mandatory = $true)][string]$Manifest,
  [string[]]$Arms,                    # subset + order; default: every arm in the manifest
  # ttft -> bench_ttft.ps1, summarised on ttft_ms.
  # tps  -> bench_tps.ps1,  summarised on ms_per_token.
  [ValidateSet('ttft', 'tps')]
  [string]$Metric  = 'ttft',
  [int]$SeqLen     = 128,             # prompt length; decode cost is seqlen-dependent via the KV read
  [int]$Gen        = 128,             # tps only: tokens generated per rep
  # Forwarded verbatim to the bench script. A multimodal export cannot be driven
  # by model_benchmark, so without -Driver vlm an A/B on one of those models
  # measures a text-only prefill or does not run at all -- and the vision encoder
  # it leaves out is part of the number being compared.
  #
  # -PromptFile reaches both metrics. The rest are ttft only and are rejected
  # under -Metric tps below rather than dropped, because dropping them would
  # quietly measure a different workload than the one asked for.
  [ValidateSet('model_benchmark', 'vlm')]
  [string]$Driver,
  [string]$PromptFile,
  [int]$MaxTokens,
  [int]$MaxLength,
  [string]$ExecutionProvider,
  [int]$Rounds     = 3,
  [int]$Reps       = 4,
  [int]$StartRound = 1,
  [switch]$Reverse,
  [switch]$SkipPrime,                 # caches survive while the DLLs are unchanged
  [string]$OutDir
)

$ErrorActionPreference = 'Stop'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'common.ps1')

# Only the keys actually passed, so an unsupplied one leaves the bench script's
# own default in place instead of overwriting it with this script's.
$passThru = @{}
foreach ($k in 'Driver', 'PromptFile', 'MaxTokens', 'MaxLength', 'ExecutionProvider') {
  if ($PSBoundParameters.ContainsKey($k)) { $passThru[$k] = $PSBoundParameters[$k] }
}
if ($Metric -eq 'tps') {
  $ttftOnly = @($passThru.Keys | Where-Object { $_ -ne 'PromptFile' } | Sort-Object)
  if ($ttftOnly) {
    $verb = if ($ttftOnly.Count -gt 1) { 'apply' } else { 'applies' }
    throw "-$($ttftOnly -join ', -') $verb to -Metric ttft only."
  }
}

$armDefs = Get-Content $Manifest -Raw | ConvertFrom-Json
$allArms = $armDefs.PSObject.Properties.Name
if (-not $Arms) { $Arms = $allArms }

function Get-ArmField {
  param($Arm, [string]$Field)
  $p = $Arm.PSObject.Properties[$Field]
  if ($p) { return $p.Value }
  return $null
}

# Union of every key any arm sets, so each arm can clear the ones it does not.
$envKeys = @()
foreach ($a in $Arms) {
  if ($a -notin $allArms) { throw "Arm '$a' is not in $Manifest (have: $($allArms -join ', '))" }
  $dll = Get-ArmField $armDefs.$a 'dll'
  $armEnv = Get-ArmField $armDefs.$a 'env'
  if (-not $dll -and -not $armEnv) { throw "Arm '$a' defines neither 'dll' nor 'env'." }
  if ($dll -and -not (Test-Path $dll)) { throw "Arm '$a': DLL not found: $dll" }
  if ($armEnv) { $envKeys += $armEnv.PSObject.Properties.Name }
}
$envKeys = $envKeys | Sort-Object -Unique

$metricDir = if ($Metric -eq 'tps') { 'tps' } else { 'ttft' }
if (-not $OutDir) { $OutDir = Join-Path $HarnessEnv.OutRoot $metricDir }
$cacheRoot = Join-Path $HarnessEnv.OutRoot 'tunecache'
$benchScript = Join-Path $PSScriptRoot "bench_$Metric.ps1"
$echoRe      = if ($Metric -eq 'tps') { '=== TPS \[' } else { 'TTFT \[' }

function Invoke-Arm {
  param([string]$Name, [string]$Tag, [int]$RunReps)
  $dll = Get-ArmField $armDefs.$Name 'dll'
  if ($dll) {
    Copy-Item $dll (Join-Path $HarnessEnv.Bin (Split-Path -Leaf $dll)) -Force
  }
  $armEnv = Get-ArmField $armDefs.$Name 'env'

  # Clear the whole HIPDNN_ namespace, not just the keys this manifest names.
  #
  # The union below makes an arm hermetic against the OTHER ARMS of the same
  # manifest. It cannot make it hermetic against the shell, because Env: is
  # process-wide and survives between invocations of this script: a variable
  # left by an earlier sweep -- a score budget, a pinned config, a logging
  # switch -- is not in this manifest's union, so nothing removes it and it is
  # applied to every arm equally. That does not look like a failure. It looks
  # like a clean A/B of the wrong thing, and it reads as "no difference"
  # whenever the leaked variable dominates what the arms were varying.
  #
  # Measured: a leaked HIPDNN_EP_GQA_SCORE_BUDGET_MB=224 forced both arms of a
  # DLL swap to the same chunking and reported a real -4.5% win as +0.05%
  # (interval spanning zero). A later leaked HIPDNN_EP_GQA_NO_EXPAND_PREFILL=1
  # disabled the feature under test outright and made four different builds
  # measure identical. DLL-only arms are the exposed case, since they declare
  # no env keys at all and the union is then empty.
  Get-ChildItem Env: | Where-Object { $_.Name -like 'HIPDNN_*' } |
    ForEach-Object { Remove-Item "Env:$($_.Name)" -EA SilentlyContinue }

  $setEnv = @()
  foreach ($k in $envKeys) {
    $v = if ($armEnv) { Get-ArmField $armEnv $k } else { $null }
    if ($null -ne $v) { $setEnv += "$k=$v" }
    else { Remove-Item "Env:$k" -EA SilentlyContinue }
  }
  $temp = Join-Path $cacheRoot $Name
  New-Item -ItemType Directory -Force -Path $temp | Out-Null
  $env:TEMP = $temp; $env:TMP = $temp

  $common = @{ Tag = $Tag; Reps = $RunReps; Warmup = 1; SeqLen = $SeqLen
               SetEnv = $setEnv; OutDir = $OutDir } + $passThru
  if ($Metric -eq 'tps') { $common.Gen = $Gen }
  & $benchScript @common 2>&1 | Where-Object { $_ -match $echoRe }
}

if (-not $SkipPrime) {
  foreach ($name in $Arms) {
    Write-Host "--- priming $name (discarded: rebuilds that arm's autotune cache)"
    Invoke-Arm -Name $name -Tag "prime_$name" -RunReps 1 | Out-Null
  }
}

$order = if ($Reverse) { $Arms[($Arms.Count - 1)..0] } else { $Arms }
foreach ($r in $StartRound..($StartRound + $Rounds - 1)) {
  foreach ($name in $order) {
    Write-Host "--- round $r arm $name"
    Invoke-Arm -Name $name -Tag "ab_${name}_r$r" -RunReps $Reps
  }
}

Write-Host "`nSummarise with:"
Write-Host ("  {0} {1} {2} --metric {3} --baseline {4}" -f
            $HarnessEnv.Python,
            (Join-Path $PSScriptRoot 'ab_summary.py'),
            (Join-Path $OutDir "${metricDir}_summary.csv"),
            $Metric, $Arms[0])
