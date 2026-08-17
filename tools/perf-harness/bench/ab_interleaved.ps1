##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Interleaved, order-reversed A/B (or A/B/C/...) of DLL variants by TTFT.
#
# Why not just run arm A a few times, then arm B a few times:
#
#  - Run-to-run spread on a thermally managed APU is comparable to the effects
#    worth shipping (~0.8% vs ~1-2%), so block-sequential runs let slow drift
#    land entirely on one arm. Interleaving and pairing by round cancels it.
#  - Position within a round is itself a bias. Always run at least one block
#    with -Reverse and check the delta survives; if it flips, you measured the
#    order, not the change.
#  - Discard rounds taken while the box is shedding heat from a build or a test
#    suite. They report a different answer -- in one measured case the opposite
#    sign -- and they are recognisable by sitting well above the known baseline.
#
# Each arm gets its OWN TEMP, and therefore its own autotune cache file: the
# on-disk WMMA cache holds a single build timestamp and is discarded when it
# does not match, so a shared TEMP would make every DLL swap a cold-tune run.
#
# Arms are defined in a JSON manifest:
#
#   { "base":  { "dll": "D:\\builds\\base\\custom_kernels_gfx1151.dll" },
#     "cand":  { "dll": "D:\\builds\\cand\\custom_kernels_gfx1151.dll" } }
#
# The named DLL is copied over the same filename in $env:HIPEP_BIN before each
# run, so every arm is measured through one identical harness.
#
# Use "dlls" instead when an arm spans more than one artifact:
#
#   { "base": { "dlls": ["D:\\b\\base\\hipgpu.dll",
#                        "D:\\b\\base\\custom_kernels_gfx1151.dll"] }, ... }
#
# Anything touching lib/Conversion or lib/Runtime/real lands in hipgpu.dll, and
# the two share the extern "C" kernel ABI, so swapping only one of the pair
# measures a mix of both arms. Every arm must list the same file names, or the
# unlisted one silently persists from whichever arm ran last.

param(
  [Parameter(Mandatory = $true)][string]$Manifest,
  [string[]]$Arms,                    # subset + order; default: every arm in the manifest
  [int]$Rounds     = 3,
  [int]$Reps       = 4,
  [int]$StartRound = 1,
  [switch]$Reverse,
  [switch]$SkipPrime,                 # caches survive while the DLLs are unchanged
  [string]$OutDir,
  # Everything below only shapes the workload and is forwarded verbatim to
  # bench_ttft.ps1, whose defaults these mirror. Without them an A/B silently
  # measures bench_ttft's defaults rather than the operating point the arms were
  # built for -- on a VLM that means the text-only driver, a 16k sequence, and
  # whatever provider the model's genai_config happens to name.
  [ValidateSet('model_benchmark', 'vlm')]
  [string]$Driver = 'model_benchmark',
  [int]$SeqLen = 16384,
  [int]$MaxTokens = 4,
  [int]$MaxLength,
  [string]$ExecutionProvider = 'follow_config',
  [string[]]$SetEnv = @()
)

$ErrorActionPreference = 'Stop'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'common.ps1')

$armDefs = Get-Content $Manifest -Raw | ConvertFrom-Json
$allArms = $armDefs.PSObject.Properties.Name
if (-not $Arms) { $Arms = $allArms }

function Get-ArmDlls {
  param([string]$Name)
  $def = $armDefs.$Name
  $has = $def.PSObject.Properties.Name
  if ($has -contains 'dlls') { return @($def.dlls) }
  if ($has -contains 'dll')  { return @($def.dll) }
  throw "Arm '$Name' declares neither 'dll' nor 'dlls'."
}

foreach ($a in $Arms) {
  if ($a -notin $allArms) { throw "Arm '$a' is not in $Manifest (have: $($allArms -join ', '))" }
  foreach ($d in (Get-ArmDlls $a)) {
    if (-not (Test-Path $d)) { throw "Arm '$a': DLL not found: $d" }
  }
}

# A file swapped by one arm but not another keeps that arm's build for every
# later run, which reads as a real effect and is invisible in the output.
$armFileSets = @{}
foreach ($a in $Arms) {
  $armFileSets[$a] = @(Get-ArmDlls $a | ForEach-Object { Split-Path -Leaf $_ } | Sort-Object)
}
$reference = $armFileSets[$Arms[0]]
foreach ($a in $Arms) {
  if (Compare-Object $reference $armFileSets[$a]) {
    throw ("Arms swap different files, so one arm's build would leak into the other. " +
           "'$($Arms[0])' has [$($reference -join ', ')], '$a' has [$($armFileSets[$a] -join ', ')].")
  }
}

if (-not $OutDir) { $OutDir = Join-Path $HarnessEnv.OutRoot 'ttft' }
$cacheRoot = Join-Path $HarnessEnv.OutRoot 'tunecache'
$benchTtft = Join-Path $PSScriptRoot 'bench_ttft.ps1'

function Invoke-Arm {
  param([string]$Name, [string]$Tag, [int]$RunReps)
  foreach ($dll in (Get-ArmDlls $Name)) {
    Copy-Item $dll (Join-Path $HarnessEnv.Bin (Split-Path -Leaf $dll)) -Force
  }
  $temp = Join-Path $cacheRoot $Name
  New-Item -ItemType Directory -Force -Path $temp | Out-Null
  $env:TEMP = $temp; $env:TMP = $temp
  $fwd = @{
    Driver           = $Driver
    SeqLen           = $SeqLen
    MaxTokens        = $MaxTokens
    ExecutionProvider = $ExecutionProvider
    SetEnv           = $SetEnv
  }
  if ($PSBoundParameters.ContainsKey('MaxLength')) { $fwd.MaxLength = $MaxLength }
  & $benchTtft -Tag $Tag -Reps $RunReps -Warmup 1 -OutDir $OutDir @fwd 2>&1 |
    Where-Object { $_ -match 'TTFT \[' }
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
Write-Host "  $($HarnessEnv.Python) $(Join-Path $PSScriptRoot 'ab_summary.py') $(Join-Path $OutDir 'ttft_summary.csv') --baseline $($Arms[0])"
