<#
  bottleneck_probe.ps1 - rigorous decode/prefill/cold-start bottleneck battery
  for the HIPDNN (MorphiZen) EP on Windows.

  Runs a fixed battery against ONE OGA model dir and captures raw logs plus a
  manifest.json under perf-results/probe_<ts>/. The companion analyzer
  (bottleneck_report.py) turns the logs into a single ranked, roofline-attributed
  report. This script only orchestrates + captures -- it computes no metrics.

  Batteries:
    1. Headline    : model_benchmark at each prompt length (clean throughput).
    2. Decode prof : model_benchmark p128 with HIPDNN_EP_PERF (per-op [PERF] tables).
    3. Launch A/B  : p128 decode with vs without HIP_LAUNCH_BLOCKING=1.
    4. Per-shape   : onnxruntime_perf_test -I over single_op seq1 graphs + HIPDNN_EP_PERF.
    5. Cold start  : coarse process wall of a minimal (g=1) load-dominated run.

  Transient "invalid resource handle" GPU-state failures are detected and the
  affected run is retried up to -Retries times.
#>
[CmdletBinding()]
param(
  [string]$Model    = "D:\Qwen3.5-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml",
  [int[]] $Prompts  = @(128, 512, 2048),
  [int]   $Gen      = 128,
  [int]   $Reps     = 5,
  [int]   $Warmup   = 2,
  [string]$Workspace = "C:\Users\Administrator\workspace",
  [string]$SingleOpDir = "D:\Qwen3.5-35B-A3B_int4_rtn_128gs_cuda_space\text\space_opt\single_op",
  [int]   $Retries  = 2
)

$ErrorActionPreference = "Stop"
$pkg     = Join-Path $Workspace "gpu-test-package\bin"
$ibin    = Join-Path $Workspace "install\bin"
$rock    = Join-Path $Workspace "therock-keep\bin"
$cfg     = Join-Path $Workspace "onnx-hipdnn-ep\etc\morphizen_config.json"
$ts      = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir  = Join-Path $Workspace "perf-results\probe_$ts"
New-Item -ItemType Directory -Force $outDir | Out-Null
$env:PATH = "$pkg;$ibin;$rock;" + $env:PATH

$modelName = Split-Path $Model -Leaf
Write-Host "[probe] model=$modelName out=$outDir"

$manifest = [ordered]@{
  model      = $Model
  model_name = $modelName
  timestamp  = $ts
  gen        = $Gen
  reps       = $Reps
  warmup     = $Warmup
  peak_gbps  = 256.0   # LPDDR5X roofline peak (Strix Halo) -- confirm exact spec
  runs       = @()
}

function Test-KernelError($logPath) {
  return [bool](Select-String -Path $logPath -Pattern 'invalid resource|launch failed|hipMalloc failed|Compilation failed' -Quiet)
}

# Run a command line (cmd /c) with env, capture to log, retry on transient GPU error.
function Invoke-Battery($label, $cmdLine, $logName, [switch]$AllowKernelErr) {
  $logPath = Join-Path $outDir $logName
  for ($attempt = 1; $attempt -le ($Retries + 1); $attempt++) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    cmd /c "$cmdLine > `"$logPath`" 2>&1"
    $sw.Stop()
    $wallMs = [math]::Round($sw.Elapsed.TotalMilliseconds, 1)
    $hasErr = Test-KernelError $logPath
    if ($hasErr -and -not $AllowKernelErr -and $attempt -le $Retries) {
      Write-Host "[probe]   $label attempt $attempt hit transient GPU error; retrying..."
      Start-Sleep -Seconds 3
      continue
    }
    Write-Host ("[probe]   {0} -> {1} (wall={2} ms, kernel_errors={3})" -f $label, $logName, $wallMs, $hasErr)
    return @{ label = $label; log = $logName; wall_ms = $wallMs; kernel_errors = [bool]$hasErr }
  }
}

$mb = "model_benchmark.exe"

# --- Battery 1: headline throughput per prompt length ---
foreach ($p in $Prompts) {
  $r = Invoke-Battery "headline p$p" "$mb -i `"$Model`" -l $p -g $Gen -r $Reps -w $Warmup" "headline_p$p.log"
  $r.type = "headline"; $r.prompt = $p
  $manifest.runs += $r
}

# --- Battery 2: decode per-op profile (HIPDNN_EP_PERF) at p128 ---
$r = Invoke-Battery "decode-profile p128" "set HIPDNN_EP_PERF=1&& $mb -i `"$Model`" -l 128 -g 64 -r 1 -w 1" "decode_perf_p128.log"
$r.type = "decode_profile"; $r.prompt = 128
$manifest.runs += $r

# --- Battery 3: launch critical-path A/B (p128 decode) ---
$r = Invoke-Battery "launch normal" "$mb -i `"$Model`" -l 128 -g 64 -r 3 -w 1" "launch_normal_p128.log"
$r.type = "launch_normal"; $r.prompt = 128
$manifest.runs += $r
$r = Invoke-Battery "launch blocking" "set HIP_LAUNCH_BLOCKING=1&& $mb -i `"$Model`" -l 128 -g 64 -r 3 -w 1" "launch_blocking_p128.log"
$r.type = "launch_blocking"; $r.prompt = 128
$manifest.runs += $r

# --- Battery 4: per-shape isolated roofline via single_op seq1 graphs ---
if (Test-Path $SingleOpDir) {
  $ops = Get-ChildItem $SingleOpDir -Directory | Where-Object { $_.Name -like "MatMulNBits_*" -or $_.Name -eq "QMoE" }
  foreach ($op in $ops) {
    $g = Join-Path $op.FullName ("{0}_seq1.onnx" -f $op.Name)
    if (-not (Test-Path $g)) { continue }
    $cmd = "set HIPDNN_EP_PERF=1&& onnxruntime_perf_test.exe --plugin_ep_libs `"hipgpu|$ibin\hipgpu.dll`" --plugin_eps `"hipgpu`" --plugin_ep_options `"config_file|$cfg`" -C `"session.disable_cpu_ep_fallback|1`" -t 4 -c 1 -s -I `"$g`""
    $r = Invoke-Battery ("singleop {0}" -f $op.Name) $cmd ("singleop_{0}.log" -f $op.Name)
    $r.type = "single_op"; $r.op = $op.Name
    $manifest.runs += $r
  }
} else {
  Write-Host "[probe]   single_op dir not found: $SingleOpDir (skipping per-shape roofline)"
}

# --- Battery 5: coarse cold-start (minimal load-dominated run) ---
$r = Invoke-Battery "coldstart" "$mb -i `"$Model`" -l 8 -g 1 -r 1 -w 0" "coldstart.log" -AllowKernelErr
$r.type = "coldstart"; $r.prompt = 8
$manifest.runs += $r

$manifestPath = Join-Path $outDir "manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Out-File $manifestPath -Encoding ascii
Write-Host "[probe] done. manifest: $manifestPath"
Write-Output $outDir
