##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Operator compatibility check for one ONNX model.
#
#   1) dump the graph the EP compiles, as MLIR
#   2) count operators on both that graph and the original, and compare
#   3) probe the compiler for what converts, what lowers, and which
#      attributes the converters actually read
#   4) assemble and render the report
#
# Usage:
#   .\run_ep_compatibility_check.ps1 -ModelPath "<path>\model.onnx"
#   .\run_ep_compatibility_check.ps1 -ModelPath "<path>\model.onnx" -OutputDir "$env:TEMP\my_run"
#   .\run_ep_compatibility_check.ps1 -ModelPath "<path>\model.onnx" -SkipDump -EpMlirPath "<path>\ep_input.mlir"
#
# Requires a gpu-test-package -- the tools come from there, not from a build
# tree, so the check does not need a compiler set up. Resolution order is
# -GpuTestPackageRoot, then $env:GPU_TEST_PACKAGE_ROOT; with neither the
# script prints [GPU_TEST_PACKAGE_NOT_CONFIGURED] and exits 10 so a caller
# can prompt for one.

param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [string]$OutputDir = "",
    [string]$RepoRoot = "",
    [string]$ToolsDir = "",

    [string]$GpuTestPackageRoot = "",
    [string]$DumpFileName = "ep_input.mlir",
    [string]$MorphizenConfigPath = "",

    [switch]$SkipDump,
    [switch]$ContinueOnDumpFailure,
    [string]$EpMlirPath = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ToolsDir)) { $ToolsDir = $PSScriptRoot }
$ToolsDir = (Resolve-Path -LiteralPath $ToolsDir).Path

# <repo>/.cursor/skills/model-compatibility/scripts/ -> four levels up.
if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..\..")).Path
}
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).ProviderPath

if ([string]::IsNullOrWhiteSpace($GpuTestPackageRoot) -and $env:GPU_TEST_PACKAGE_ROOT) {
    $GpuTestPackageRoot = $env:GPU_TEST_PACKAGE_ROOT
}
if ([string]::IsNullOrWhiteSpace($GpuTestPackageRoot)) {
    Write-Output '[GPU_TEST_PACKAGE_NOT_CONFIGURED] No GpuTestPackageRoot supplied. Pass -GpuTestPackageRoot <path>, set env var GPU_TEST_PACKAGE_ROOT, or rerun with -SkipDump and -EpMlirPath to reuse an existing dump.'
    exit 10
}
$GpuTestPackageRoot = (Resolve-Path -LiteralPath $GpuTestPackageRoot).ProviderPath

# Fail here rather than deep inside a step with a cryptic error.
foreach ($exe in @("hip-onnx-runner.exe", "hip-mlir-opt.exe")) {
    if (-not (Test-Path -LiteralPath (Join-Path $GpuTestPackageRoot "bin\$exe"))) {
        throw "$exe not found under $GpuTestPackageRoot\bin. Update the gpu-test-package."
    }
}

# OutputDir defaults to a name derived from the model's path, so returning to
# a directory later still shows which model it belongs to. Take the last
# three meaningful parent segments, skipping ones that carry no identity,
# then append the basename when it has any:
#   ...\blip\onnx\decoder\fp16\model.onnx -> blip_decoder_fp16_ep_compat
#   <drive>\bar\custom_v2.onnx            -> bar_custom_v2_ep_compat
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($ModelPath)
    # [Path]::GetDirectoryName, not Split-Path -LiteralPath -Parent: in
    # PS 5.1 those two flags land in different parameter sets on some hosts.
    $parentDir = [System.IO.Path]::GetDirectoryName($ModelPath)
    $genericSegments = @('onnx', 'models')

    $allSegs = $parentDir -split '[\\/]+' | Where-Object {
        $_ -and $_ -notmatch '^[A-Za-z]:$'
    }
    $meaningful = @($allSegs | Where-Object {
        $genericSegments -notcontains $_.ToLowerInvariant()
    })
    if ($meaningful.Count -gt 3) { $meaningful = $meaningful[-3..-1] }

    $baseLower = $base.ToLowerInvariant()
    if ($baseLower -and $baseLower -ne 'model' -and ($genericSegments -notcontains $baseLower)) {
        $meaningful = @($meaningful) + @($base)
    }
    $sanitized = @($meaningful | ForEach-Object {
        ($_ -replace '[^A-Za-z0-9]+', '_').ToLowerInvariant().Trim('_')
    } | Where-Object { $_ })
    if ($sanitized.Count -eq 0) { $sanitized = @($baseLower) }

    $compatRoot = if ($env:HIP_EP_COMPAT_ROOT) { $env:HIP_EP_COMPAT_ROOT }
                  elseif ($env:TEMP) { $env:TEMP }
                  else { [System.IO.Path]::GetTempPath() }
    $OutputDir = Join-Path $compatRoot (($sanitized -join '_') + '_ep_compat')
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)

$EpInputDir = Join-Path $OutputDir "ep_input"
$OrigOpsDir = Join-Path $OutputDir "step1_original"
$EpOpsDir = Join-Path $OutputDir "step1_ep"
$CompatDir = Join-Path $OutputDir "compatibility"
$ProbeDir = Join-Path $CompatDir "probe"
New-Item -ItemType Directory -Force -Path `
    $OutputDir, $EpInputDir, $OrigOpsDir, $EpOpsDir, $CompatDir, $ProbeDir | Out-Null

$SupportedOpsDoc = Join-Path $RepoRoot "docs\supported-operations.md"

function Invoke-PythonStep {
    param([string]$Label, [string[]]$PyArgv)
    Write-Host $Label -ForegroundColor Yellow
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & python @PyArgv
    if ($LASTEXITCODE -ne 0) { throw "python failed: python $($PyArgv -join ' ')" }
    $sw.Stop()
    Write-Host ("  done in {0:N1}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor DarkGray
}

function Test-UpToDate {
    param([string]$Source, [string]$Derived)
    if (-not (Test-Path -LiteralPath $Derived)) { return $false }
    return (Get-Item -LiteralPath $Derived).LastWriteTimeUtc -ge
           (Get-Item -LiteralPath $Source).LastWriteTimeUtc
}

Write-Host "=== Operator compatibility check ===" -ForegroundColor Cyan
Write-Host "Model:        $ModelPath"
Write-Host "Package:      $GpuTestPackageRoot"
Write-Host "Output:       $OutputDir"
Write-Host ""

# --- 1/5  EP input --------------------------------------------------------
$EpMlir = if ($EpMlirPath) { (Resolve-Path -LiteralPath $EpMlirPath).Path }
          else { Join-Path $EpInputDir $DumpFileName }

$dumpError = ""
if (-not $SkipDump -and (Test-Path -LiteralPath $EpMlir)) {
    Write-Host "(1/5) EP input already present, reusing: $EpMlir" -ForegroundColor DarkGray
    $SkipDump = $true
}
if (-not $SkipDump) {
    Write-Host '(1/5) Dumping EP input MLIR...' -ForegroundColor Yellow
    $dumpArgs = @{
        ModelPath          = $ModelPath
        GpuTestPackageRoot = $GpuTestPackageRoot
        DumpDirectory      = $EpInputDir
        DumpFileName       = $DumpFileName
    }
    if ($MorphizenConfigPath) { $dumpArgs['MorphizenConfigPath'] = $MorphizenConfigPath }
    try {
        & (Join-Path $ToolsDir "dump_ep_input.ps1") @dumpArgs
        if (-not (Test-Path -LiteralPath $EpMlir)) { throw "dump produced no file: $EpMlir" }
    } catch {
        $dumpError = $_.Exception.Message
        Write-Host "WARN: EP dump failed: $dumpError" -ForegroundColor Red
        if (-not $ContinueOnDumpFailure) { throw }
    }
} elseif (-not (Test-Path -LiteralPath $EpMlir)) {
    Write-Host "(1/5) No EP input available; analysis will cover the original ONNX only" -ForegroundColor Yellow
}
$haveEpMlir = Test-Path -LiteralPath $EpMlir

# --- 2/5  Operator distribution ------------------------------------------
$origOps = Join-Path $OrigOpsDir "step1_onnx_ops.json"
if (Test-UpToDate -Source $ModelPath -Derived $origOps) {
    Write-Host '(2/5) Original model distribution: up to date' -ForegroundColor DarkGray
} else {
    Invoke-PythonStep -Label '(2/5) Counting operators in the original model...' -PyArgv @(
        (Join-Path $ToolsDir "step1_onnx_parser.py"), $ModelPath, $OrigOpsDir,
        "--max-instances-per-op", "0"
    )
}

$epOps = Join-Path $EpOpsDir "ep_input_ops.json"
if ($haveEpMlir) {
    if (Test-UpToDate -Source $EpMlir -Derived $epOps) {
        Write-Host '(2/5) EP input distribution: up to date' -ForegroundColor DarkGray
    } else {
        Invoke-PythonStep -Label '(2/5) Counting operators in the EP input...' -PyArgv @(
            (Join-Path $ToolsDir "mlir_op_parser.py"), $EpMlir, $EpOpsDir
        )
    }

    # --- 3/5  Comparison --------------------------------------------------
    Invoke-PythonStep -Label '(3/5) Comparing the two distributions...' -PyArgv @(
        (Join-Path $ToolsDir "compare_op_distribution.py"), $origOps, $epOps, $OutputDir,
        "--original-model", $ModelPath, "--ep-model", $EpMlir
    )
} else {
    Write-Host '(3/5) Skipping comparison: no EP input' -ForegroundColor Yellow
}

# --- 4/5  Compiler probe --------------------------------------------------
$probeResult = Join-Path $ProbeDir "probe_result.json"
if ($haveEpMlir) {
    Invoke-PythonStep -Label '(4/5) Probing the compiler...' -PyArgv @(
        (Join-Path $ToolsDir "probe.py"), $EpMlir, $epOps, $ProbeDir,
        "--package", $GpuTestPackageRoot
    )
} else {
    Write-Host '(4/5) Skipping probe: no EP input' -ForegroundColor Yellow
    # Support cannot be established without compiling. Hand the assembler an
    # empty probe so it reports evidence level D rather than inventing one.
    '{"stage1": {"meta": {"failed": true, "error": "no EP input MLIR"}, "operators": {}}, "stage2": {"operators": {}}}' |
        Set-Content -LiteralPath $probeResult -Encoding UTF8
}

# --- 5/5  Report ----------------------------------------------------------
$opsForReport = if ($haveEpMlir) { $epOps } else { $origOps }
$reportArgs = @(
    (Join-Path $ToolsDir "build_report_input.py"), $opsForReport, $probeResult,
    $RepoRoot, $CompatDir,
    "--model-path", $ModelPath,
    "--original-model", $ModelPath,
    "--supported-ops-doc", $SupportedOpsDoc
)
if ($haveEpMlir) { $reportArgs += @("--ep-input-path", $EpMlir) }
Invoke-PythonStep -Label '(5/5) Assembling the report...' -PyArgv $reportArgs

$comparison = Join-Path $OutputDir "op_distribution_comparison.json"
if (Test-Path -LiteralPath $comparison) {
    Copy-Item -LiteralPath $comparison -Destination $CompatDir -Force
}
Invoke-PythonStep -Label '      Rendering...' -PyArgv @(
    (Join-Path $ToolsDir "generate_final_reports.py"), $CompatDir
)

# --- Status and convenience copies ---------------------------------------
$report = Join-Path $CompatDir "model_compatibility_report.md"
$details = Join-Path $CompatDir "model_compatibility_details.md"
foreach ($f in @($report, $details)) {
    if (Test-Path -LiteralPath $f) {
        Copy-Item -LiteralPath $f -Destination $OutputDir -Force
    }
}

$evidence = "unknown"
$reportInput = Join-Path $CompatDir "report_input.json"
if (Test-Path -LiteralPath $reportInput) {
    $evidence = (Get-Content -Raw -LiteralPath $reportInput |
                 ConvertFrom-Json).meta.evidence_level
}

$statusLines = @(
    "# Compatibility check status",
    "",
    "- **Model:** ``$ModelPath``",
    "- **EP input:** ``$EpMlir``",
    "- **EP input available:** $haveEpMlir",
    "- **Evidence level:** $evidence",
    "- **Analyzed:** $(if ($haveEpMlir) { 'EP input MLIR' } else { 'original ONNX only' })",
    ""
)
if ($dumpError) {
    $statusLines += @("## Dump error", "", '```', $dumpError, '```', "")
}
$statusLines | Set-Content -LiteralPath (Join-Path $OutputDir "pipeline_status.md") -Encoding UTF8

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "Evidence level:  $evidence"
if ($haveEpMlir) {
    Write-Host "EP input:        $EpMlir"
    Write-Host "Comparison:      $(Join-Path $OutputDir 'op_distribution_comparison.md')"
}
Write-Host "Report:          $(Join-Path $OutputDir 'model_compatibility_report.md')"
Write-Host "Artifacts:       $CompatDir"
