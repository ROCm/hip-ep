##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# ONNX -> HIP compatibility pipeline, driven by what the compiler does.
#
#   S0  dump the compiler-input MLIR through hip-ep (init pass only)
#   S1  operator distribution of the original ONNX and of that MLIR, compared
#   S2  run the conversion up to convert-onnx-to-hip
#   S3  scan the result: onnx ops still present are unsupported
#   S4  pair pre/post ops by location: dropped ONNX attributes are partial
#   S5  normalize into report_input.json
#   S6  render the markdown reports
#
# Usage:
#   .\run_ep_compatibility_check.ps1 -ModelPath <model.onnx>
#   .\run_ep_compatibility_check.ps1 -ModelPath <model.onnx> -OutputDir <dir>
#   .\run_ep_compatibility_check.ps1 -ModelPath <model.onnx> -SkipDump
#
# Defaults (derived):
#   -RepoRoot          = (Resolve-Path "$PSScriptRoot\..\..\..\..")
#   -ToolsDir          = $PSScriptRoot
#   -HipEpPackageRoot  = $env:HIP_EP_PACKAGE_ROOT (when unset and -SkipDump is
#                        absent, the script emits [HIP_EP_NOT_CONFIGURED] and
#                        exits 10)
#   -OutputDir         = $env:TEMP\<meaningful-path-segments>_ep_compat
#                        (or $env:HIP_EP_COMPAT_ROOT when set)
#
# -SkipDump analyzes the original ONNX only. Without the compiler-input MLIR
# there is nothing to convert, so no operator can be classified and the report
# says so rather than guessing.

param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [string]$OutputDir = "",
    [string]$RepoRoot = "",
    [string]$ToolsDir = "",

    [string]$HipEpPackageRoot = "",
    [string]$MorphizenConfigPath = "",

    [switch]$SkipDump,
    [switch]$ContinueOnDumpFailure,
    [string]$CompilerInputMlir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ToolsDir)) {
    $ToolsDir = $PSScriptRoot
}
$ToolsDir = (Resolve-Path -LiteralPath $ToolsDir).Path

# Default RepoRoot = 4 levels above this script:
#   .cursor/skills/model-compatibility/scripts/   <- $PSScriptRoot
#   <repo>/                                       <- ..  (4 levels up)
if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..\..")).Path
}
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).ProviderPath

if ([string]::IsNullOrWhiteSpace($HipEpPackageRoot) -and $env:HIP_EP_PACKAGE_ROOT) {
    $HipEpPackageRoot = $env:HIP_EP_PACKAGE_ROOT
}
if (-not $SkipDump -and [string]::IsNullOrWhiteSpace($HipEpPackageRoot)) {
    # Write-Output (not Write-Host) so the marker is captured by any caller
    # reading the success stream. Single-quoted to keep the literal env name.
    $msg = '[HIP_EP_NOT_CONFIGURED] No HipEpPackageRoot supplied. Pass -HipEpPackageRoot <dir>, set $env:HIP_EP_PACKAGE_ROOT to a hip-ep package, or rerun with -SkipDump to analyze the original ONNX without the conversion probe.'
    Write-Output $msg
    exit 10
}

# OutputDir default: human-readable name derived from the model's path.
# Take the last 3 parent path segments, skip generic ones, append the basename
# when it is non-generic, lowercase and sanitize to [a-z0-9_], suffix with
# _ep_compat. Example:
#   ...\blip\onnx\decoder\fp16\model.onnx  -> blip_decoder_fp16_ep_compat
# When auto-derivation would collide between two distinct models the caller
# should pass an explicit -OutputDir <dir>.
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($ModelPath)
    # Use [Path]::GetDirectoryName, not Split-Path -LiteralPath ... -Parent:
    # in PS 5.1 the latter trips AmbiguousParameterSet (those two flags
    # resolve into different parameter sets of Split-Path on some hosts).
    $parentDir = [System.IO.Path]::GetDirectoryName($ModelPath)

    $genericSegments = @('onnx', 'models')

    $allSegs = $parentDir -split '[\\/]+' | Where-Object {
        $_ -and $_ -notmatch '^[A-Za-z]:$'
    }

    $meaningful = @($allSegs | Where-Object {
        $genericSegments -notcontains $_.ToLowerInvariant()
    })

    if ($meaningful.Count -gt 3) {
        $meaningful = $meaningful[-3..-1]
    }

    $baseLower = $base.ToLowerInvariant()
    if ($baseLower -and $baseLower -ne 'model' -and ($genericSegments -notcontains $baseLower)) {
        $meaningful = @($meaningful) + @($base)
    }

    $sanitized = @($meaningful | ForEach-Object {
        ($_ -replace '[^A-Za-z0-9]+', '_').ToLowerInvariant().Trim('_')
    } | Where-Object { $_ })

    if ($sanitized.Count -eq 0) {
        $sanitized = @($baseLower)
    }

    $compatRoot = if ($env:HIP_EP_COMPAT_ROOT) {
        $env:HIP_EP_COMPAT_ROOT
    } elseif ($env:TEMP) {
        $env:TEMP
    } else {
        [System.IO.Path]::GetTempPath()
    }

    $OutputDir = Join-Path $compatRoot (($sanitized -join '_') + '_ep_compat')
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)

$EpInputDir = Join-Path $OutputDir "ep_input"
$Step1OriginalDir = Join-Path $OutputDir "step1_original"
$Step1EpDir = Join-Path $OutputDir "step1_ep"
$CompatDir = Join-Path $OutputDir "compatibility"

New-Item -ItemType Directory -Force -Path $OutputDir, $EpInputDir, $Step1OriginalDir, $Step1EpDir, $CompatDir | Out-Null

function Invoke-PythonStep {
    param(
        [string]$Label,
        [string[]]$PyArgv
    )
    Write-Host $Label -ForegroundColor Yellow
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & python @PyArgv
    if ($LASTEXITCODE -ne 0) {
        throw "python failed: python $($PyArgv -join ' ')"
    }
    $sw.Stop()
    Write-Host ("  done in {0:N1}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor DarkGray
}

function OutputUpToDate {
    param([string]$SourcePath, [string]$ProducedPath)
    if (-not (Test-Path -LiteralPath $ProducedPath)) { return $false }
    return (Get-Item -LiteralPath $ProducedPath).LastWriteTimeUtc -ge (Get-Item -LiteralPath $SourcePath).LastWriteTimeUtc
}

Write-Host "=== hip-ep compatibility check ===" -ForegroundColor Cyan
Write-Host "Original model:  $ModelPath"
Write-Host "Output dir:      $OutputDir"
Write-Host "Repo root:       $RepoRoot"
Write-Host ""

# --- S0: dump the compiler-input MLIR ---------------------------------------
$CompilerInput = if ($CompilerInputMlir) {
    (Resolve-Path -LiteralPath $CompilerInputMlir).ProviderPath
} else {
    Join-Path $EpInputDir "compiler_input.mlir"
}

$dumpFailed = $false
$dumpError = ""

if (-not $SkipDump -and (Test-Path -LiteralPath $CompilerInput)) {
    Write-Host "(1/5) Skip dump: compiler input already exists at $CompilerInput" -ForegroundColor DarkGray
    $SkipDump = $true
}

if (-not $SkipDump) {
    Write-Host '(1/5) Dumping compiler-input MLIR...' -ForegroundColor Yellow
    $dumpScript = Join-Path $ToolsDir "dump_compiler_input.ps1"
    try {
        $dumpArgs = @{
            ModelPath        = $ModelPath
            OutputDir        = $EpInputDir
            HipEpPackageRoot = $HipEpPackageRoot
        }
        if (-not [string]::IsNullOrWhiteSpace($MorphizenConfigPath)) {
            $dumpArgs['ConfigPath'] = $MorphizenConfigPath
        }
        & $dumpScript @dumpArgs
        if (-not (Test-Path -LiteralPath $CompilerInput)) {
            throw "compiler-input MLIR missing: $CompilerInput"
        }
    } catch {
        $dumpFailed = $true
        $dumpError = $_.Exception.Message
        Write-Host "WARN: dump failed: $dumpError" -ForegroundColor Red
        if (-not $ContinueOnDumpFailure) {
            throw
        }
    }
} elseif (Test-Path -LiteralPath $CompilerInput) {
    Write-Host "(1/5) Skip dump; using compiler input: $CompilerInput" -ForegroundColor Yellow
} elseif ($CompilerInputMlir) {
    throw "CompilerInputMlir not found: $CompilerInput"
} else {
    Write-Host '(1/5) Skip dump; no compiler input, analyzing the original ONNX' -ForegroundColor Yellow
}

$haveCompilerInput = (Test-Path -LiteralPath $CompilerInput)

# --- S1: operator distributions ---------------------------------------------
$step1OrigJson = Join-Path $Step1OriginalDir "step1_original_onnx_ops.json"
if (OutputUpToDate -SourcePath $ModelPath -ProducedPath $step1OrigJson) {
    Write-Host '(2/5) step1 original: up-to-date, skip' -ForegroundColor DarkGray
} else {
    Invoke-PythonStep -Label '(2/5) step1 - original ONNX...' -PyArgv @(
        (Join-Path $ToolsDir "step1_onnx_parser.py"),
        $ModelPath, $Step1OriginalDir,
        "--max-instances-per-op", "0"
    )
}

if ($haveCompilerInput) {
    $step1EpJson = Join-Path $Step1EpDir "step1_compiler_input_ops.json"
    if (OutputUpToDate -SourcePath $CompilerInput -ProducedPath $step1EpJson) {
        Write-Host '(2/5) step1 compiler input: up-to-date, skip' -ForegroundColor DarkGray
    } else {
        Invoke-PythonStep -Label '(2/5) step1 - compiler input (MLIR)...' -PyArgv @(
            (Join-Path $ToolsDir "step1_mlir_parser.py"),
            $CompilerInput, $Step1EpDir,
            "--max-instances-per-op", "0"
        )
    }

    Invoke-PythonStep -Label '(3/5) Op distribution comparison...' -PyArgv @(
        (Join-Path $ToolsDir "compare_op_distribution.py"),
        $step1OrigJson,
        $step1EpJson,
        $OutputDir,
        "--original-model", $ModelPath,
        "--ep-model", $CompilerInput
    )
} else {
    Write-Host '(2/5) Skip compiler-input distribution and comparison' -ForegroundColor Yellow
}

# --- S2..S4: conversion probe, leftovers, attribute transfer ----------------
Remove-Item -LiteralPath (Join-Path $CompatDir "leftover_onnx.json") -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $CompatDir "attr_transfer.json") -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $CompatDir "leftover_reasons.json") -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $CompatDir "hip_runtime_map.json") -ErrorAction SilentlyContinue

if ($haveCompilerInput) {
    Write-Host '(4/5) Conversion probe (convert-onnx-to-hip)...' -ForegroundColor Yellow
    & (Join-Path $ToolsDir "run_convert_probe.ps1") `
        -InputMlir $CompilerInput -OutputDir $EpInputDir -HipEpPackageRoot $HipEpPackageRoot

    Invoke-PythonStep -Label '  leftover + attribute analysis' -PyArgv @(
        (Join-Path $ToolsDir "analyze_conversion.py"),
        (Join-Path $EpInputDir "compiler_input_loc.mlir"),
        (Join-Path $EpInputDir "converted.mlir"),
        $CompatDir
    )

    # Why each leftover did not convert. The conversion cannot report its own
    # refusal reason in a release build, so this reads the constraints out of
    # the converter that matches the operator.
    Invoke-PythonStep -Label '  leftover explanations' -PyArgv @(
        (Join-Path $ToolsDir "explain_leftovers.py"),
        (Join-Path $CompatDir "leftover_onnx.json"),
        $RepoRoot,
        $CompatDir
    )

    # Which runtime function executes each converted op. Keyed on the hip ops
    # the conversion produced, so this is a lookup rather than a guess.
    Invoke-PythonStep -Label '  hip op to runtime map' -PyArgv @(
        (Join-Path $ToolsDir "hip_runtime_map.py"), $RepoRoot, $CompatDir
    )
} else {
    Write-Host '(4/5) Skip conversion probe (no compiler input); support will be reported as unverified' -ForegroundColor Yellow
}

# --- S5, S6: normalize and render -------------------------------------------
$analyzedGraph = if ($haveCompilerInput) { $CompilerInput } else { $ModelPath }
$step1ForCompat = if ($haveCompilerInput) { $step1EpJson } else { $step1OrigJson }

Write-Host '(5/5) Reports...' -ForegroundColor Yellow
Invoke-PythonStep -Label '  build_report_input' -PyArgv @(
    (Join-Path $ToolsDir "build_report_input.py"),
    $analyzedGraph, $step1ForCompat, $CompatDir, $RepoRoot
)
Invoke-PythonStep -Label '  generate_final_reports' -PyArgv @(
    (Join-Path $ToolsDir "generate_final_reports.py"), $CompatDir
)

$compatNote = if ($haveCompilerInput) {
    "Compatibility analysis uses the compiler-input MLIR and the convert-onnx-to-hip result."
} else {
    "WARNING: no compiler-input MLIR; the original ONNX was counted but no operator support was verified."
}

$statusPath = Join-Path $OutputDir "pipeline_status.md"
$statusLines = @(
    "# hip-ep compatibility pipeline status",
    "",
    "- **Original model:** ``$($ModelPath)``",
    "- **Compiler input:** ``$($CompilerInput)``",
    "- **Dump succeeded:** $(-not $dumpFailed -and $haveCompilerInput)",
    "- **Conversion probed:** $haveCompilerInput",
    "- **Analyzed graph:** ``$($analyzedGraph)``",
    "- **hip-ep package:** ``$($HipEpPackageRoot)``",
    "- **Note:** $compatNote",
    ""
)
if ($dumpFailed) {
    $statusLines += @("## Dump error", "", "``````", $dumpError, "``````", "")
}
$statusLines | Set-Content -LiteralPath $statusPath -Encoding UTF8

$summarySrc = Join-Path $CompatDir "model_compatibility_report.md"
if (Test-Path -LiteralPath $summarySrc) {
    Copy-Item -LiteralPath $summarySrc -Destination (Join-Path $OutputDir "model_compatibility_report.md") -Force
    Copy-Item -LiteralPath (Join-Path $CompatDir "model_compatibility_details.md") `
        -Destination (Join-Path $OutputDir "model_compatibility_details.md") -Force

    # Without the conversion probe the report describes operator counts only.
    # Badge both copies right after the H1 so the limitation cannot be missed
    # when the agent reads the report back.
    if (-not $haveCompilerInput) {
        $badge = "> **Source:** original ONNX, conversion probe skipped. No hip-ep package was configured or the dump failed, so operator support was NOT verified against the compiler."
        foreach ($mdTarget in @(
            (Join-Path $OutputDir "model_compatibility_report.md"),
            (Join-Path $OutputDir "model_compatibility_details.md"),
            $summarySrc,
            (Join-Path $CompatDir "model_compatibility_details.md")
        )) {
            if (-not (Test-Path -LiteralPath $mdTarget)) { continue }
            $content = Get-Content -Raw -LiteralPath $mdTarget
            if ($content -match '\*\*Source:\*\* original ONNX') { continue }
            $h1End = $content.IndexOf("`n")
            if ($h1End -lt 0) { continue }
            $patched = $content.Substring(0, $h1End + 1) + "`n$badge`n" + $content.Substring($h1End + 1)
            Set-Content -LiteralPath $mdTarget -Value $patched -Encoding UTF8 -NoNewline
        }
    }
}

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "Pipeline status:       $statusPath"
if ($haveCompilerInput) {
    Write-Host "Compiler input:        $CompilerInput"
    Write-Host "Converted MLIR:        $(Join-Path $EpInputDir 'converted.mlir')"
    Write-Host "Op comparison:         $(Join-Path $OutputDir 'op_distribution_comparison.md')"
} else {
    Write-Host "Compiler input:        (not produced)"
}
Write-Host "Compatibility report:  $(Join-Path $OutputDir 'model_compatibility_report.md')"
Write-Host "Full artifacts:        $CompatDir"
if (-not $haveCompilerInput) { Write-Host $compatNote -ForegroundColor Yellow }
