##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# ONNX -> HIP compatibility pipeline, driven by what the compiler does.
#
#   1  dump the EP-input MLIR through hip-ep (init pass only)
#   2  count operators in the original ONNX and in that MLIR, and diff them
#   3  run the conversion up to convert-onnx-to-hip
#   4  analyze what it produced: leftovers, why they were refused, attribute
#      transfer, and the runtime function behind each hip op
#   5  normalize into report_input.json and render the markdown
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
# -SkipDump analyzes the original ONNX only. Without the EP-input MLIR
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
    [string]$EpInputMlir = ""
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

# Everything the pipeline produces on the way to the reports lives together;
# the two markdown files and the status note stay at the top so the directory
# opens on what a person reads.
$CompatDir = Join-Path $OutputDir "compatibility"

New-Item -ItemType Directory -Force -Path $OutputDir, $CompatDir | Out-Null

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

Write-Host "=== hip-ep compatibility check ===" -ForegroundColor Cyan
Write-Host "Original model:  $ModelPath"
Write-Host "Output dir:      $OutputDir"
Write-Host "Repo root:       $RepoRoot"
Write-Host ""

# --- S0: dump the EP-input MLIR ---------------------------------------
$EpInput = if ($EpInputMlir) {
    (Resolve-Path -LiteralPath $EpInputMlir).ProviderPath
} else {
    Join-Path $CompatDir "ep_input.mlir"
}

$dumpFailed = $false
$dumpError = ""

# Re-runs reuse the output directory, so a failure recorded by an earlier run
# would be reported against this one.
$failureRecord = Join-Path $CompatDir "pipeline_failure.json"
Remove-Item -LiteralPath $failureRecord -ErrorAction SilentlyContinue

if (-not $SkipDump -and (Test-Path -LiteralPath $EpInput)) {
    Write-Host "(1/5) Skip dump: EP input already exists at $EpInput" -ForegroundColor DarkGray
    $SkipDump = $true
}

if (-not $SkipDump) {
    Write-Host '(1/5) Dumping EP-input MLIR...' -ForegroundColor Yellow
    $dumpScript = Join-Path $ToolsDir "dump_ep_input.ps1"
    try {
        $dumpArgs = @{
            ModelPath        = $ModelPath
            OutputDir        = $CompatDir
            HipEpPackageRoot = $HipEpPackageRoot
        }
        if (-not [string]::IsNullOrWhiteSpace($MorphizenConfigPath)) {
            $dumpArgs['ConfigPath'] = $MorphizenConfigPath
        }
        & $dumpScript @dumpArgs
        if (-not (Test-Path -LiteralPath $EpInput)) {
            throw "EP-input MLIR missing: $EpInput"
        }
    } catch {
        # Why the dump failed is a compatibility finding in its own right: an
        # importer that rejects the graph means the model never reaches the
        # compiler. Record it so the report can say so.
        $dumpFailed = $true
        $dumpError = $_.Exception.Message
        Write-Host "WARN: dump failed: $dumpError" -ForegroundColor Red
        [ordered]@{
            stage     = "dump"
            command   = "hip-onnx-runner --no-run"
            exit_code = $null
            message   = $dumpError
            log       = Join-Path $CompatDir "dump_stderr.txt"
        } | ConvertTo-Json | Set-Content -LiteralPath $failureRecord -Encoding UTF8
        if (-not $ContinueOnDumpFailure) {
            throw
        }
    }
} elseif (Test-Path -LiteralPath $EpInput) {
    Write-Host "(1/5) Skip dump; using EP input: $EpInput" -ForegroundColor Yellow
} elseif ($EpInputMlir) {
    throw "EpInputMlir not found: $EpInput"
} else {
    Write-Host '(1/5) Skip dump; no EP input, analyzing the original ONNX' -ForegroundColor Yellow
}

$haveEpInput = (Test-Path -LiteralPath $EpInput)

# --- Step 2: operator distributions and their difference --------------------
$step1OrigJson = Join-Path $CompatDir "step1_original_onnx_ops.json"
$step1EpJson = Join-Path $CompatDir "step1_ep_input_ops.json"

$distributionArgs = @((Join-Path $ToolsDir "op_distribution.py"), $ModelPath, $CompatDir)
if ($haveEpInput) {
    $distributionArgs += @("--ep-input", $EpInput)
}
Invoke-PythonStep -Label '(2/5) Operator distributions...' -PyArgv $distributionArgs

# --- Step 3 and 4: conversion probe, then what it produced ------------------
foreach ($stale in @('leftover_onnx.json', 'leftover_reasons.json',
        'attr_transfer.json', 'hip_runtime_map.json')) {
    Remove-Item -LiteralPath (Join-Path $CompatDir $stale) -ErrorAction SilentlyContinue
}

$converted = Join-Path $CompatDir "converted.mlir"
if ($haveEpInput) {
    Write-Host '(3/5) Conversion probe (convert-onnx-to-hip)...' -ForegroundColor Yellow
    & (Join-Path $ToolsDir "run_convert_probe.ps1") `
        -InputMlir $EpInput -OutputDir $CompatDir -HipEpPackageRoot $HipEpPackageRoot
}

if (Test-Path -LiteralPath $converted) {
    $locatedInput = Join-Path $CompatDir "ep_input_loc.mlir"
    Invoke-PythonStep -Label '(4/5) Analyzing the conversion...' -PyArgv @(
        (Join-Path $ToolsDir "analyze_conversion.py"),
        $locatedInput,
        $converted,
        $CompatDir,
        $RepoRoot
    )
    # It only existed to join the two sides by location, and it is a copy of
    # ep_input.mlir that the probe can recreate.
    Remove-Item -LiteralPath $locatedInput -ErrorAction SilentlyContinue
} elseif ($haveEpInput) {
    Write-Host '(4/5) Conversion failed; the report will name the step and the reason' -ForegroundColor Red
} else {
    Write-Host '(3/5) Skip conversion probe (no EP input); support will be reported as unverified' -ForegroundColor Yellow
}

# --- Step 5: normalize and render -------------------------------------------
$analyzedGraph = if ($haveEpInput) { $EpInput } else { $ModelPath }
$step1ForCompat = if ($haveEpInput) { $step1EpJson } else { $step1OrigJson }

Write-Host '(5/5) Reports...' -ForegroundColor Yellow
Invoke-PythonStep -Label '  build_report_input' -PyArgv @(
    (Join-Path $ToolsDir "build_report_input.py"),
    $analyzedGraph, $step1ForCompat, $CompatDir, $RepoRoot
)
Invoke-PythonStep -Label '  generate_final_reports' -PyArgv @(
    (Join-Path $ToolsDir "generate_final_reports.py"), $CompatDir, $OutputDir
)

$compatNote = if ($haveEpInput) {
    "Compatibility analysis uses the EP-input MLIR and the convert-onnx-to-hip result."
} else {
    "WARNING: no EP-input MLIR; the original ONNX was counted but no operator support was verified."
}

$statusPath = Join-Path $OutputDir "pipeline_status.md"
$statusLines = @(
    "# hip-ep compatibility pipeline status",
    "",
    "- **Original model:** ``$($ModelPath)``",
    "- **EP input:** ``$($EpInput)``",
    "- **Dump succeeded:** $(-not $dumpFailed -and $haveEpInput)",
    "- **Conversion probed:** $haveEpInput",
    "- **Analyzed graph:** ``$($analyzedGraph)``",
    "- **hip-ep package:** ``$($HipEpPackageRoot)``",
    "- **Note:** $compatNote",
    ""
)
if ($dumpFailed) {
    $statusLines += @("## Dump error", "", "``````", $dumpError, "``````", "")
}
$statusLines | Set-Content -LiteralPath $statusPath -Encoding UTF8

# Without the conversion probe the report describes operator counts only.
# Badge it right after the H1 so the limitation cannot be missed when the
# agent reads the report back. A recorded failure already carries a banner
# that says the same thing and names the cause, so do not repeat it.
if (-not $haveEpInput -and -not (Test-Path -LiteralPath $failureRecord)) {
    $badge = "> **Source:** original ONNX, conversion probe skipped. No hip-ep package was configured or the dump failed, so operator support was NOT verified against the compiler."
    foreach ($mdTarget in @(
        (Join-Path $OutputDir "model_compatibility_report.md"),
        (Join-Path $OutputDir "model_compatibility_details.md")
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

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Green
Write-Host "Pipeline status:       $statusPath"
if ($haveEpInput) {
    Write-Host "EP input:              $EpInput"
} else {
    Write-Host "EP input:              (not produced)"
}
if (Test-Path -LiteralPath $converted) {
    Write-Host "Converted MLIR:        $converted"
    Write-Host "Converted ops:         $(Join-Path $CompatDir 'attr_transfer.json')"
}
Write-Host "Compatibility report:  $(Join-Path $OutputDir 'model_compatibility_report.md')"
Write-Host "Full artifacts:        $CompatDir"
if (Test-Path -LiteralPath $failureRecord) {
    Write-Host ("WARNING: a pipeline step failed; nothing was verified. " +
        "See the 'Where it failed' section of the report.") -ForegroundColor Yellow
} elseif (-not $haveEpInput) {
    Write-Host $compatNote -ForegroundColor Yellow
}
