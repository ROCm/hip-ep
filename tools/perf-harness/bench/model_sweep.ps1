## Sweep a model zoo through onnxruntime_perf_test on the HIP EP, one CSV row per
## model, so an A/B over a kernel change can be judged on whole models rather
## than on the kernel alone.
##
## This is the outer guard rail. conv_microbench.py is the inner one and is
## three orders of magnitude faster, so the working loop is: change the kernel,
## run the microbenchmark, and only come here to confirm the model-level effect
## and -- more importantly -- to prove the models that were never broken did not
## become so. A convolution heuristic tuned on the regressed models can easily
## pay for itself out of the models that were fine, and nothing but a full sweep
## will show that.
##
## Inputs are generated (-I) because these model directories ship raw .bin
## inputs rather than the test_data_set_0 layout perf_test expects. Shapes are
## unaffected, which is what convolution cost depends on; models whose work is
## data-dependent (NMS/TopK/NonZero post-processing) will vary run to run and
## should be read with that in mind.
##
##   .\model_sweep.ps1 -Arm base -Out base.csv
##   # redeploy the other build
##   .\model_sweep.ps1 -Arm cand -Out cand.csv
##   python ab_summary.py base.csv cand.csv

[CmdletBinding()]
param(
  [int]$Seconds = 30,
  [string]$Out = "$PSScriptRoot\..\..\..\sweep_results.csv",
  [string]$LogDir,
  [string]$Bin = 'C:\Users\zyq\gpu-test-package\bin',
  [string]$Rocm = 'C:\Users\zyq\therock-dist\bin',
  [string]$VisionRoot = 'C:\Users\zyq\vision_models',
  [string]$ProcyonRoot = 'C:\Users\zyq\test-models\ProcyonV2-models',
  [string[]]$Only,
  # Per-op attribution. Costs 4-8%, so the QPS it reports is not throughput --
  # read the [PERF] table, never the summary line.
  [switch]$Perf,
  # Arms must not share a TEMP: the on-disk kernel autotune cache is keyed on a
  # single build timestamp, so a shared cache makes every DLL swap a cold-tune
  # run and you measure tuning rather than the kernel.
  [string]$Arm
)

$ErrorActionPreference = 'Continue'
if (-not $LogDir) { $LogDir = Join-Path (Split-Path $Out -Parent) 'sweep-logs' }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$V = $VisionRoot
$P = $ProcyonRoot
$models = [ordered]@{
  'bevformer-fp16'                    = "$V\bevformer-fp16\vit_bevformer_sole_r50backbone_batch1_1instance.onnx"
  'bevformer-tiny-temporal-sim'       = "$V\bevformer-tiny-temporal-sim\model.onnx"
  'facebook-detr-resnet-50'           = "$V\facebook-detr-resnet-50\olive\models\detr-r50-640-fp16\model.onnx"
  'google-mobilenet_v2_1.0_224'       = "$V\google-mobilenet_v2_1.0_224\olive\models\mobilenet_v2-224-fp16\model.onnx"
  'microsoft-swin-tiny-224x224'       = "$V\microsoft-swin-tiny-224x224\olive\models\swin-tiny-224x224-fp16\model.onnx"
  'microsoft-swinv2-base-2048x3072'   = "$V\microsoft-swinv2-base-2048x3072\olive\models\swinv2-base-2048x3072-fp16\model.onnx"
  'microsoft-swinv2-base-8x2048x3072' = "$V\microsoft-swinv2-base-8x2048x3072\olive\models\swinv2-base-8x2048x3072-fp16\model.onnx"
  'microsoft-swinv2-tiny-512x1024'    = "$V\microsoft-swinv2-tiny-512x1024\olive\models\swinv2-tiny-512x1024-fp16\model.onnx"
  'microsoft-swinv2-tiny-8x521x1024'  = "$V\microsoft-swinv2-tiny-8x521x1024\olive\models\swinv2-tiny-8x521x1024-fp16\model.onnx"
  'mlcommons-ssd-resnet34-1200x1200'  = "$V\mlcommons-ssd-resnet34-1200x1200\olive\models\mlcommons-ssd-resnet34-1200\model.onnx"
  'nvidia-resnet50v1.5'               = "$V\nvidia-resnet50v1.5\olive\models\resnet50v1.5-224-fp16\model.onnx"
  'qfgaohao-mb1-ssd'                  = "$V\qfgaohao-mb1-ssd\olive\models\mb1-ssd-300-fp16\model.onnx"
  'simplebev'                         = "$V\simplebev\model.onnx"
  'ultralytics-yolov5lu'              = "$V\ultralytics-yolov5lu\olive\models\yolov5lu-640-fp16\model.onnx"
  'blip/decoder'                      = "$P\blip\onnx\decoder\fp16\model.onnx"
  'blip/encoder'                      = "$P\blip\onnx\encoder\fp16\model.onnx"
  'convnext'                          = "$P\convnext\onnx\fp16\model.onnx"
  'detr'                              = "$P\detr\onnx\fp16\model.onnx"
  'esrgan'                            = "$P\esrgan\onnx\fp16\model.onnx"
  'sam2.1/decoder'                    = "$P\sam2.1\onnx\decoder\fp16\model.onnx"
  'sam2.1/encoder'                    = "$P\sam2.1\onnx\encoder\fp16\model.onnx"
}

Remove-Item Env:HIPDNN_EP_PERF, Env:HIPDNN_EP_DEBUG, Env:HIPDNN_EP_TRACE_FILE -EA SilentlyContinue
if ($Perf) { $env:HIPDNN_EP_PERF = '1' }

if ($Arm) {
  $armTemp = Join-Path ([IO.Path]::GetTempPath()) "conv-sweep-$Arm"
  New-Item -ItemType Directory -Force -Path $armTemp | Out-Null
  $env:TEMP = $armTemp; $env:TMP = $armTemp
}

$env:PATH = "$Bin;$Rocm;$env:PATH"
Push-Location $Bin

$rows = @()
foreach ($name in $models.Keys) {
  if ($Only -and ($Only -notcontains $name)) { continue }
  $path = $models[$name]
  if (-not (Test-Path $path)) { Write-Host "MISSING $name"; continue }

  $safe = $name -replace '[\\/]', '_'
  $log = Join-Path $LogDir "$safe.log"
  Write-Host ("--- {0}" -f $name) -NoNewline

  $sw = [Diagnostics.Stopwatch]::StartNew()
  & .\onnxruntime_perf_test.exe `
      --plugin_ep_libs "hipgpu|hipgpu.dll" `
      --plugin_eps "hipgpu" `
      --plugin_ep_options "config_file|..\morphizen_config.json" `
      -C "session.disable_cpu_ep_fallback|1" `
      -t $Seconds -c 1 -s -I `
      $path > $log 2>&1
  $sw.Stop()

  $txt = Get-Content $log -Raw
  function G([string]$pat) {
    $m = [regex]::Match($txt, $pat)
    if ($m.Success) { $m.Groups[1].Value } else { $null }
  }

  $qps   = G 'Number of inferences per second:\s*([\d.]+)'
  $sess  = G 'Session creation time cost:\s*([\d.]+)'
  $first = G 'First inference time cost:\s*([\d.]+)'
  $avg   = G 'Average inference time cost total:\s*([\d.]+)'
  $cpu   = G 'Avg CPU usage:\s*(\d+)'
  $mem   = G 'Peak working set size:\s*(\d+)'

  $status = if ($qps) { 'ok' } else { 'FAIL' }
  Write-Host ("  {0}  qps={1}  first={2}ms  wall={3:N0}s" -f $status, $qps, $first, $sw.Elapsed.TotalSeconds)

  $rows += [pscustomobject]@{
    Model        = $name
    Status       = $status
    QPS          = $qps
    SessionS     = $sess
    FirstInferMs = $first
    AvgInferMs   = $avg
    CpuPct       = $cpu
    PeakMemMB    = if ($mem) { [math]::Round([double]$mem / 1MB, 1) } else { $null }
    WallS        = [math]::Round($sw.Elapsed.TotalSeconds, 1)
  }
  $rows | Export-Csv -NoTypeInformation -Path $Out
}

Pop-Location
Remove-Item Env:HIPDNN_EP_PERF -EA SilentlyContinue
Write-Host "`nwrote $Out"
$rows | Format-Table -AutoSize
