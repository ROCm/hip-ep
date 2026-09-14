##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##
## Drive one RGP dispatch-mode capture of a model_benchmark run, then decode it
## with tools/rgp_parser.
##
## Positioning a capture on a specific op is the hard part. Two modes:
##
##   -Op <name>   Fence mode. The runtime's rgp_capture_fence (op_profile.cpp)
##                drains the GPU and idles right before that op's Nth instance,
##                printing [RGP_FENCE_ARMED]. This script watches for the marker
##                and triggers RGP inside the idle window, so the very next
##                dispatches are the op you asked for. Deterministic.
##                -DecodeStep / -AfterInferences pick the phase; -Skip cannot,
##                because every layer requests the same op within one Run, so it
##                selects a layer. A long prompt is fed as several Runs, so
##                decode step 1 at 16K is Run 32 -- -DecodeStep converts.
##
##   -Trigger     Auto-capture mode: RGP's own dispatch:<delay>:<count> trigger.
##                No runtime support needed, but it positions blind by dispatch
##                index, which drifts run to run. Use when the build under test
##                has no fence.
##
## Deliberately no HIPDNN_EP_PERF / HIPDNN_EP_DEBUG: CLAUDE.md forbids measuring
## throughput with either. SQTT already carries the timing.

[CmdletBinding(DefaultParameterSetName = 'Fence')]
param(
  [Parameter(ParameterSetName = 'Fence', Mandatory = $true)]
  [string]$Op,                                   # RGP_FENCE op name, e.g. qmoe | gqa | matmul_nbits
  [Parameter(ParameterSetName = 'Fence')]
  [int]$Skip = 0,                                # arm on the (Skip+1)-th instance; selects a LAYER, not a phase
  # Which ORT Run to arm on. Run 0 is the first prefill chunk. -Skip cannot name
  # a phase: every layer requests the same op within one Run, so it selects a
  # layer. Prefer -DecodeStep for decode, which does the Run arithmetic.
  [Parameter(ParameterSetName = 'Fence')]
  [int]$AfterInferences = 0,
  # Arm on decode step N (1 = first generated token). The generator feeds a long
  # prompt in -PrefillChunk-token Runs, so decode step N is Run
  # ceil(SeqLen/PrefillChunk)+N-1, not Run N. Overrides -AfterInferences.
  [Parameter(ParameterSetName = 'Fence')]
  [int]$DecodeStep = 0,
  [Parameter(ParameterSetName = 'Fence')]
  [int]$PrefillChunk = 512,
  [Parameter(ParameterSetName = 'Fence')]
  [int]$FenceMs = 15000,                         # idle window; must exceed detect + trigger + arm latency

  [Parameter(ParameterSetName = 'Trigger', Mandatory = $true)]
  [string]$Trigger,                              # e.g. dispatch:200:3000

  [string]$Tag,
  [int]$SeqLen = 16384,                          # with -PromptFile unset, drives --use_random_tokens
  [string]$PromptFile,
  # The fence lives in hipgpu.dll, not in the benchmark, so it fires the same way
  # under a Python driver. 'vlm' is required for multimodal models, whose vision
  # encoder model_benchmark.exe never touches.
  [ValidateSet('model_benchmark', 'vlm')]
  [string]$Driver = 'model_benchmark',
  [int]$MaxTokens = 2,                           # vlm only
  [int]$MaxLength,                               # vlm only
  # vlm only. 'follow_config' leaves the model's own genai_config provider list
  # alone; naming a provider overrides it, which is what an export pinned to
  # another EP (a -dml directory, say) needs to run here.
  [string]$ExecutionProvider = 'follow_config',
  [int]$Gen = 1,
  [int]$Reps = 2,                                # RGP streams the dump from the LIVE process; see note below
  [int]$OpCount = 4000,                          # --rgp-render-op-count: window size, not usability
  [ValidateSet('minimum', 'default', 'maximum')]
  [string]$Buf = 'default',
  [switch]$Counters,                             # SPM: required for memory/compute bound classification
  [int]$Dwell = 30,                              # seconds to let the .rgp finish dumping
  # How long to wait for the fence to arm. A 16K VLM prefill under the panel
  # needs far more than the 10 minutes that suffices for a 2K one: model load,
  # the per-process prefill autotune sweep and a cold first rep all land before
  # the fence can fire.
  [int]$ArmTimeoutSec = 600,
  [string]$OutDir
)

$ErrorActionPreference = 'Continue'
. (Join-Path (Split-Path -Parent $PSScriptRoot) 'common.ps1')

if (-not $HarnessEnv.RgpCli) {
  throw "RadeonDeveloperPanelCLI.exe not found. Set `$env:RGP_DIR to the Radeon Developer Tool Suite folder."
}
if (-not $Tag) {
  $Tag = if ($PSCmdlet.ParameterSetName -eq 'Fence') {
    "cap_${Op}" + $(if ($Skip -gt 0) { "_skip$Skip" } else { '' })
  } else { 'cap_trigger' }
}
# A fence that can never arm produces a 0-byte capture 20 minutes later, which
# is indistinguishable from the trigger-miss failure mode. Catch it up front.
if ($PSCmdlet.ParameterSetName -eq 'Fence' -and $DecodeStep -gt 0) {
  if ($Gen -lt $DecodeStep) {
    throw "-DecodeStep $DecodeStep needs -Gen at least $DecodeStep (have $Gen)."
  }
  if ($PromptFile) {
    # The token count of a prompt file is unknown here, so the Run index cannot
    # be derived from a byte count. Random tokens keep the arithmetic exact.
    throw '-DecodeStep needs -SeqLen (no -PromptFile); pass -AfterInferences instead.'
  }
  $chunks = [Math]::Ceiling($SeqLen / [double]$PrefillChunk)
  $AfterInferences = $chunks + $DecodeStep - 1
  Write-Host ("decode step $DecodeStep at seqlen $SeqLen = $chunks prefill " +
              "Runs + $DecodeStep -> -AfterInferences $AfterInferences")
}
if (-not $OutDir) { $OutDir = Join-Path $HarnessEnv.OutRoot 'captures' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$OutRgp   = Join-Path $OutDir "$Tag.rgp"
$decBase  = Join-Path $OutDir $Tag
$benchOut = Join-Path $OutDir "bench_$Tag.log"
$benchErr = Join-Path $OutDir "bench_$Tag.err"
$panelLog = Join-Path $OutDir "panel_$Tag.log"

$isVlm = ($Driver -eq 'vlm')
if ($isVlm) {
  foreach ($n in 'VlmBench', 'Image', 'PromptFile') {
    if (-not $HarnessEnv.$n) { throw "Driver 'vlm' needs `$env:HIPEP_$($n.ToUpper()); see common.ps1." }
  }
}

Stop-HarnessProcesses -IncludePython:$isVlm
Remove-Item $OutRgp, $benchOut, $benchErr, $panelLog -EA SilentlyContinue
'' | Set-Content $benchErr

# --- arm the panel BEFORE the HIP context exists, so the driver hooks on connect
$panelArgs = @(
  '-m', 'profiling', '-o', $OutRgp,
  '--rgp-capture-mode', 'dispatch',
  '--rgp-sqtt-buffer-size', $Buf,
  '--rgp-render-op-count', "$OpCount")
if ($Counters) { $panelArgs += '--rgp-counter-collection' }
if ($PSCmdlet.ParameterSetName -eq 'Trigger') { $panelArgs += @('--rgp-auto-capture', $Trigger) }
$targetProcess = if ($isVlm) { 'python' } else { 'model_benchmark' }
$panelArgs += @('-p', $targetProcess, '--verbose')

$pi = New-Object System.Diagnostics.ProcessStartInfo
$pi.FileName  = $HarnessEnv.RgpCli
$pi.Arguments = ($panelArgs | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
$pi.UseShellExecute        = $false
$pi.RedirectStandardInput  = $true
$pi.RedirectStandardOutput = $true
$pi.RedirectStandardError  = $true
$panel = [System.Diagnostics.Process]::Start($pi)
$pOut = $panel.StandardOutput; $pErr = $panel.StandardError
$pOutTask = $pOut.ReadLineAsync(); $pErrTask = $pErr.ReadLineAsync()
$posDesc = if ($PSCmdlet.ParameterSetName -eq 'Fence') {
  " op=$Op run=$AfterInferences skip=$Skip"
} else { '' }
Write-Host "panel PID=$($panel.Id) mode=$($PSCmdlet.ParameterSetName) buf=$Buf ops=$OpCount counters=$($Counters.IsPresent) tag=$Tag$posDesc"
Start-Sleep 6

Set-HarnessPath
Clear-HarnessProfilingEnv
$env:HIPDNN_EP_AUTOTUNE = '1'
$env:HIPDNN_EP_MATMUL_CUSTOM_WMMA = '1'
if ($PSCmdlet.ParameterSetName -eq 'Fence') {
  $env:RGP_FENCE = $Op; $env:RGP_FENCE_SKIP = "$Skip"; $env:RGP_FENCE_MS = "$FenceMs"
  $env:RGP_FENCE_AFTER_INFERENCES = "$AfterInferences"
} else {
  Remove-Item Env:RGP_FENCE, Env:RGP_FENCE_SKIP, Env:RGP_FENCE_MS,
              Env:RGP_FENCE_AFTER_INFERENCES -EA SilentlyContinue
}

if ($isVlm) {
  if (-not $PromptFile) { $PromptFile = $HarnessEnv.PromptFile }
  if (-not $MaxLength)  { $MaxLength  = $SeqLen + 128 }
  $exe   = $HarnessEnv.Python
  $margs = @('-u', $HarnessEnv.VlmBench, '-m', $HarnessEnv.Model, '-i', $HarnessEnv.Image,
             '--prompt_file', $PromptFile, '--max_tokens', "$MaxTokens",
             '--max_length', "$MaxLength", '-e', $ExecutionProvider,
             '-n', "$Reps", '-w', '0')
  $wd    = Split-Path -Parent $HarnessEnv.VlmBench
} else {
  $exe   = Join-Path $HarnessEnv.Bin 'model_benchmark.exe'
  $margs = @('-i', $HarnessEnv.Model, '-g', "$Gen", '-r', "$Reps", '-w', '0', '-b', '1', '-ml', '0', '-v')
  if ($PromptFile) { $margs += @('--prompt_file', $PromptFile) }
  else             { $margs += @('-l', "$SeqLen", '--use_random_tokens') }
  $wd    = $HarnessEnv.Bin
}

$bench = Start-Process -FilePath $exe `
  -ArgumentList $margs -WorkingDirectory $wd `
  -RedirectStandardOutput $benchOut -RedirectStandardError $benchErr `
  -PassThru -WindowStyle Hidden
Write-Host "model PID=$($bench.Id) launched"

# In fence mode, watch stderr for the armed marker and trigger inside the idle
# window. In trigger mode there is nothing to wait for; the panel fires itself.
$triggered = ($PSCmdlet.ParameterSetName -ne 'Fence')
$deadline = (Get-Date).AddSeconds($ArmTimeoutSec)
while (-not $bench.HasExited -and (Get-Date) -lt $deadline) {
  if ($pOutTask.IsCompleted) { if ($pOutTask.Result) { Add-Content $panelLog $pOutTask.Result }; $pOutTask = $pOut.ReadLineAsync() }
  if ($pErrTask.IsCompleted) { if ($pErrTask.Result) { Add-Content $panelLog $pErrTask.Result }; $pErrTask = $pErr.ReadLineAsync() }
  if (-not $triggered) {
    $hit = Select-String -Path $benchErr, $benchOut -Pattern 'RGP_FENCE_ARMED' -EA SilentlyContinue | Select-Object -First 1
    if ($hit) {
      Start-Sleep -Milliseconds 700
      Write-Host ">>> $($hit.Line.Trim()) -> triggering capture"
      $panel.StandardInput.WriteLine('capture'); $panel.StandardInput.Flush()
      $triggered = $true
    }
  }
  Start-Sleep -Milliseconds 250
}

# RGP streams the trace out of the LIVE process. A late-positioned fence can
# leave too little runtime for a large dump, which then stalls and writes
# nothing -- that is what -Reps buys, not extra measurement.
Write-Host "model exited=$($bench.HasExited) triggered=$triggered ; dwell ${Dwell}s for the dump"
Start-Sleep $Dwell
# The panel writes the .rgp as it shuts down, not when the transfer finishes, so
# it has to be allowed to exit on its own. Killing it a few seconds after 'quit'
# loses the whole capture: the trace transfers to 100%, the panel dies
# mid-write, and no file is ever created.
try { $panel.StandardInput.WriteLine('quit'); $panel.StandardInput.Flush() } catch {}
$sz = -1
for ($i = 0; $i -lt 100; $i++) {
  if ($panel.HasExited) { break }
  Start-Sleep 3
  if (Test-Path $OutRgp) {
    $now = (Get-Item $OutRgp).Length
    if ($now -gt 0 -and $now -eq $sz) { break }   # written and no longer growing
    $sz = $now
  }
}
Stop-HarnessProcesses -IncludePython:$isVlm
Remove-Item Env:RGP_FENCE, Env:RGP_FENCE_SKIP, Env:RGP_FENCE_MS,
            Env:RGP_FENCE_AFTER_INFERENCES -EA SilentlyContinue

if (-not (Test-Path $OutRgp)) {
  Write-Host "RESULT: $OutRgp NOT created"
  # A stalled dump and a trigger miss both end with no file, but they need
  # opposite fixes, so name which one happened. The panel prints "x / y MB" as
  # it drains the buffer out of the live process; a partial figure there means
  # the workload ended first and the answer is more runtime after the fence
  # (-Gen/-Reps), not a different fence position.
  $prog = Select-String -Path $panelLog -Pattern '([\d.]+) / ([\d.]+) MB' -EA SilentlyContinue |
          Select-Object -Last 1
  if ($prog) {
    $got = [double]$prog.Matches[0].Groups[1].Value
    $tot = [double]$prog.Matches[0].Groups[2].Value
    Write-Host ("!! dump stalled at {0:N1} / {1:N1} MB ({2:N0}%)." -f $got, $tot, (100 * $got / $tot))
    Write-Host "   RGP streams the trace out of the LIVE process and the model exited first."
    Write-Host "   Fix: raise -Gen (more decode steps after the fence) and/or -Reps,"
    Write-Host "   and/or lower -OpCount so there is less to transfer."
  }
  Get-Content $panelLog -Tail 20 -EA SilentlyContinue
  exit 1
}
Write-Host ("RESULT $OutRgp bytes: " + (Get-Item $OutRgp).Length)

# Gate on chunk inventory before trusting anything decoded from the file.
# --require-spm is an argparse store_true, so it must be present or absent; the
# PowerShell "-switch:$bool" form would pass the literal "--require-spm:True".
$verifyArgs = @($OutRgp)
if ($Counters) { $verifyArgs += '--require-spm' }
& $HarnessEnv.Python (Join-Path $PSScriptRoot 'verify_rgp.py') @verifyArgs
if ($LASTEXITCODE -ne 0) { Write-Host '!! capture unusable; not decoding it'; exit 1 }

Push-Location $HarnessEnv.Parser
& $HarnessEnv.Python main.py $OutRgp $decBase 2>&1 | Out-Null
Pop-Location

$ops = "${decBase}_operators.csv"
if (Test-Path $ops) {
  $rows = Import-Csv $ops
  $disp = if (Test-Path "${decBase}_dispatches.csv") { (Import-Csv "${decBase}_dispatches.csv").Count } else { 0 }
  Write-Host "PARSED dispatches=$disp op_rows=$($rows.Count) -> $decBase*"
  Write-Host '--- top 20 kernels by total GPU time ---'
  $rows | Sort-Object { [double]$_.total_us } -Descending | Select-Object -First 20 |
    Format-Table @{N = 'kernel'; E = { ($_.kernel -replace '<.*', '') -replace '\(.*', '' } },
                 count, mean_us, total_us, pct_gpu, occ_pct, bound_class, avg_mem_gbps -AutoSize |
    Out-String -Width 400 | Write-Host
} else {
  Write-Host 'PARSE FAILED (no operators.csv)'
  exit 1
}
