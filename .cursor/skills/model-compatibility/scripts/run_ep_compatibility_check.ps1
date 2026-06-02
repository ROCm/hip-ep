# EP-aware compatibility pipeline:
#   1) Dump EP input graph -> <OutputDir>/ep_input/<DumpFileName> (default onnx.onnx)
#   2) step1 on original model and EP input; compare op distributions
#   3) step2..final compatibility reports using EP input only
#
# Usage:
#   .\run_ep_compatibility_check.ps1 -ModelPath "D:\path\model.onnx"
#   .\run_ep_compatibility_check.ps1 -ModelPath "D:\path\model.onnx" -OutputDir "D:\temp\my_run"
#   .\run_ep_compatibility_check.ps1 -ModelPath "D:\path\model.onnx" -SkipDump -EpOnnxPath "D:\path\onnx.onnx"
#
# Defaults (derived):
#   -RepoRoot         = (Resolve-Path "$PSScriptRoot\..\..\..\..\")
#                       skill lives at <repo>/.cursor/skills/model-compatibility/scripts/
#   -ToolsDir         = $PSScriptRoot
#   -VoePackageRoot   = $env:VOE_PACKAGE_ROOT  (when unset and -SkipDump absent,
#                       script emits [VOE_NOT_CONFIGURED] to stdout and exits 10)
#   -OutputDir        = D:\temp\<meaningful-path-segments>_ep_compat
#                       Built from the last 3 parent segments (skipping generic
#                       ones like "onnx" / "models") + optional non-generic
#                       basename. Examples:
#                         ...\blip\onnx\decoder\fp16\model.onnx
#                           -> D:\temp\blip_decoder_fp16_ep_compat
#                         D:\bar\custom_v2.onnx
#                           -> D:\temp\bar_custom_v2_ep_compat
#                       Override with -OutputDir <dir> if the auto name collides.

param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [string]$OutputDir = "",
    [string]$RepoRoot = "",
    [string]$ToolsDir = "",

    [string]$VoePackageRoot = "",
    [string]$DumpFileName = "onnx.onnx",
    [string]$VaipConfigPath = "",
    [string]$VaipTarget = "VAIML",

    [switch]$SkipDump,
    [switch]$ContinueOnDumpFailure,
    [string]$EpOnnxPath = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ToolsDir)) {
    $ToolsDir = $PSScriptRoot
}
$ToolsDir = (Resolve-Path -LiteralPath $ToolsDir).Path

# Default RepoRoot = 4 levels above this script:
#   .cursor/skills/model-compatibility/scripts/   <- $PSScriptRoot
#   .cursor/skills/model-compatibility/           <- ..
#   .cursor/skills/                               <- ..
#   .cursor/                                      <- ..
#   <repo>/                                       <- ..  (4 levels up)
if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..\..\..")).Path
}
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).ProviderPath

# VoePackageRoot: param > env > empty.
# Empty + no -SkipDump => emit machine-readable marker and exit 10 so the
# calling agent can AskQuestion the user (VOE path vs -SkipDump).
if ([string]::IsNullOrWhiteSpace($VoePackageRoot)) {
    if ($env:VOE_PACKAGE_ROOT) { $VoePackageRoot = $env:VOE_PACKAGE_ROOT }
}
if (-not $SkipDump -and [string]::IsNullOrWhiteSpace($VoePackageRoot)) {
    # Write-Output (not Write-Host) so the marker is reliably captured by
    # any caller piping the success stream (Python subprocess, CI, agent shell).
    # Single-quoted so the literal $env:VOE_PACKAGE_ROOT is preserved without escape.
    $msg = '[VOE_NOT_CONFIGURED] No VoePackageRoot supplied. Pass -VoePackageRoot path, set env var VOE_PACKAGE_ROOT, or rerun with -SkipDump to analyze the original ONNX without EP rewrites.'
    Write-Output $msg
    exit 10
}

# OutputDir default: human-readable name derived from the model's path.
# Take the last 3 parent path segments, skip generic ones, append the basename
# when it is non-generic, lowercase and sanitize to [a-z0-9_], suffix with
# _ep_compat. Example:
#   ...\blip\onnx\decoder\fp16\model.onnx  -> blip_decoder_fp16_ep_compat
# When auto-derivation would actually collide between two distinct models the
# caller should pass an explicit -OutputDir <dir>.
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($ModelPath)
    # Use [Path]::GetDirectoryName, not Split-Path -LiteralPath ... -Parent:
    # in PS 5.1 the latter trips AmbiguousParameterSet (those two flags
    # resolve into different parameter sets of Split-Path on some hosts).
    $parentDir = [System.IO.Path]::GetDirectoryName($ModelPath)

    # Generic path segments that carry no model identity; skip them so the
    # resulting name highlights the model variant (family + role + dtype).
    $genericSegments = @('onnx', 'models')

    # Split on / and \, drop empty pieces and drive roots like "D:".
    $allSegs = $parentDir -split '[\\/]+' | Where-Object {
        $_ -and $_ -notmatch '^[A-Za-z]:$'
    }

    $meaningful = @($allSegs | Where-Object {
        $genericSegments -notcontains $_.ToLowerInvariant()
    })

    # Keep at most the last 3 meaningful segments.
    if ($meaningful.Count -gt 3) {
        $meaningful = $meaningful[-3..-1]
    }

    # Append the basename if it carries identity (not "model" or a format word).
    $baseLower = $base.ToLowerInvariant()
    if ($baseLower -and $baseLower -ne 'model' -and ($genericSegments -notcontains $baseLower)) {
        $meaningful = @($meaningful) + @($base)
    }

    # Sanitize each segment to [a-z0-9_] and drop empties.
    $sanitized = @($meaningful | ForEach-Object {
        ($_ -replace '[^A-Za-z0-9]+', '_').ToLowerInvariant().Trim('_')
    } | Where-Object { $_ })

    if ($sanitized.Count -eq 0) {
        # Nothing identifiable in the path; fall back to the basename so the
        # dir name is at least deterministic.
        $sanitized = @($baseLower)
    }

    $OutputDir = Join-Path 'D:\temp' (($sanitized -join '_') + '_ep_compat')
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$EpInputDir = Join-Path $OutputDir "ep_input"
$Step1OriginalDir = Join-Path $OutputDir "step1_original"
$Step1EpDir = Join-Path $OutputDir "step1_ep"
$CompatDir = Join-Path $OutputDir "compatibility"

New-Item -ItemType Directory -Force -Path $EpInputDir, $Step1OriginalDir, $Step1EpDir, $CompatDir | Out-Null

$HipOpsTd = Join-Path $RepoRoot "include\hip\Dialect\IR\HipOps.td"
$ConversionDir = Join-Path $RepoRoot "lib\Conversion"
$RuntimeDir = Join-Path $RepoRoot "lib\Runtime\real"

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

function Step1UpToDate {
    param([string]$OnnxPath, [string]$Step1Json)
    if (-not (Test-Path -LiteralPath $Step1Json)) { return $false }
    return (Get-Item -LiteralPath $Step1Json).LastWriteTimeUtc -ge (Get-Item -LiteralPath $OnnxPath).LastWriteTimeUtc
}

Write-Host "=== EP compatibility check ===" -ForegroundColor Cyan
Write-Host "Original model:  $ModelPath"
Write-Host "Output dir:      $OutputDir"
Write-Host "Repo root:       $RepoRoot"
Write-Host ""

# --- Step 0: Dump EP input ONNX ---
$EpOnnx = if ($EpOnnxPath) {
    (Resolve-Path -LiteralPath $EpOnnxPath).Path
} else {
    Join-Path $EpInputDir $DumpFileName
}

$dumpFailed = $false
$dumpError = ""

# If EP graph already exists, skip dump unless user forces a new dump.
if (-not $SkipDump -and (Test-Path -LiteralPath $EpOnnx)) {
    Write-Host "(1/4) Skip dump: EP ONNX already exists at $EpOnnx" -ForegroundColor DarkGray
    $SkipDump = $true
}

if (-not $SkipDump) {
    Write-Host '(1/4) Dumping EP input graph...' -ForegroundColor Yellow
    $dumpScript = Join-Path $ToolsDir "dump_ep_onnx.ps1"
    if (-not (Test-Path -LiteralPath $dumpScript)) {
        throw "dump_ep_onnx.ps1 not found: $dumpScript"
    }
    try {
        $dumpArgs = @{
            ModelPath       = $ModelPath
            VoePackageRoot  = $VoePackageRoot
            DumpDirectory   = $EpInputDir
            DumpFileName    = $DumpFileName
        }
        if (-not [string]::IsNullOrWhiteSpace($VaipConfigPath)) {
            $dumpArgs['VaipConfigPath'] = $VaipConfigPath
        }
        if (-not [string]::IsNullOrWhiteSpace($VaipTarget)) {
            $dumpArgs['VaipTarget'] = $VaipTarget
        }
        & $dumpScript @dumpArgs
        if (-not (Test-Path -LiteralPath $EpOnnx)) {
            throw "EP dump missing: $EpOnnx"
        }
    } catch {
        $dumpFailed = $true
        $dumpError = $_.Exception.Message
        Write-Host "WARN: EP dump failed: $dumpError" -ForegroundColor Red
        if (-not $ContinueOnDumpFailure) {
            throw
        }
    }
} else {
    if (Test-Path -LiteralPath $EpOnnx) {
        Write-Host "(1/4) Skip dump; using EP ONNX: $EpOnnx" -ForegroundColor Yellow
    } elseif ($EpOnnxPath) {
        # User explicitly passed -EpOnnxPath that does not exist: hard error.
        throw "EpOnnxPath not found: $EpOnnx"
    } else {
        # -SkipDump given without -EpOnnxPath and no default EP graph present:
        # fall through to analyzing the original ONNX. Downstream code branches
        # on the haveEpOnnx flag and produces a Source-original-ONNX badge.
        Write-Host "(1/4) Skip dump; no EP ONNX available, will analyze original ONNX" -ForegroundColor Yellow
    }
}

# --- Step 1: step1 on both graphs (typically ~1-2s each; NOT the slow step) ---
$step1OrigJson = Join-Path $Step1OriginalDir "step1_onnx_ops.json"
if (Step1UpToDate -OnnxPath $ModelPath -Step1Json $step1OrigJson) {
    Write-Host '(2/4) step1 original: up-to-date, skip' -ForegroundColor DarkGray
} else {
    Invoke-PythonStep -Label '(2/4) step1 - original model...' -PyArgv @(
        (Join-Path $ToolsDir "step1_onnx_parser.py"),
        $ModelPath, $Step1OriginalDir,
        "--max-instances-per-op", "0"
    )
}

$haveEpOnnx = (Test-Path -LiteralPath $EpOnnx)

if ($haveEpOnnx) {
    $step1EpJson = Join-Path $Step1EpDir "step1_onnx_ops.json"
    if (Step1UpToDate -OnnxPath $EpOnnx -Step1Json $step1EpJson) {
        Write-Host '(2/4) step1 EP: up-to-date, skip' -ForegroundColor DarkGray
    } else {
        Invoke-PythonStep -Label '(2/4) step1 - EP input (onnx.onnx)...' -PyArgv @(
            (Join-Path $ToolsDir "step1_onnx_parser.py"),
            $EpOnnx, $Step1EpDir,
            "--max-instances-per-op", "0"
        )
    }

    Invoke-PythonStep -Label '(3/4) Op distribution comparison...' -PyArgv @(
        (Join-Path $ToolsDir "compare_op_distribution.py"),
        $step1OrigJson,
        $step1EpJson,
        $OutputDir,
        "--original-model", $ModelPath,
        "--ep-model", $EpOnnx
    )
} else {
    Write-Host '(2/4) Skip step1 EP / comparison (no onnx.onnx)' -ForegroundColor Yellow
}

$compatModel = if ($haveEpOnnx) { $EpOnnx } else { $ModelPath }
$compatNote = if ($haveEpOnnx) {
    "Compatibility analysis uses EP input (onnx.onnx)."
} else {
    "WARNING: EP dump unavailable; compatibility below uses ORIGINAL model.onnx only (not EP true input)."
}

# --- Step 2 - final: compatibility on EP input (or original if dump failed) ---
Write-Host "(4/4) Compatibility pipeline ($([System.IO.Path]::GetFileName($compatModel)))..." -ForegroundColor Yellow
Invoke-PythonStep -Label '  step2_0 hip parser' -PyArgv @((Join-Path $ToolsDir "step2_0_hip_parser.py"), $HipOpsTd, $CompatDir)
Invoke-PythonStep -Label '  step2_1 onnx->hip' -PyArgv @((Join-Path $ToolsDir "step2_1_onnx_to_hip_parser.py"), $ConversionDir, $CompatDir, $HipOpsTd)
Invoke-PythonStep -Label '  step2_2 hip->llvm' -PyArgv @((Join-Path $ToolsDir "step2_2_hip_to_llvm_parser.py"), $ConversionDir, $CompatDir, $HipOpsTd)
Invoke-PythonStep -Label '  step2_3 backend' -PyArgv @(
    (Join-Path $ToolsDir "step2_3_backend_analyzer_final.py"),
    $RuntimeDir,
    (Join-Path $CompatDir "step2_2_hip_to_llvm_mappings.json"),
    (Join-Path $CompatDir "step2_1_onnx_to_hip_mappings.json"),
    $CompatDir
)
# build_report_input expects step1_onnx_ops.json in the compatibility output dir
$step1ForCompat = if ($haveEpOnnx) {
    Join-Path $Step1EpDir "step1_onnx_ops.json"
} else {
    Join-Path $Step1OriginalDir "step1_onnx_ops.json"
}
Copy-Item -LiteralPath $step1ForCompat -Destination (Join-Path $CompatDir "step1_onnx_ops.json") -Force

Invoke-PythonStep -Label '  build_report_input' -PyArgv @(
    (Join-Path $ToolsDir "build_report_input.py"),
    $compatModel, $CompatDir, $RepoRoot
)
Invoke-PythonStep -Label '  generate_final_reports' -PyArgv @((Join-Path $ToolsDir "generate_final_reports.py"), $CompatDir)

$statusPath = Join-Path $OutputDir "pipeline_status.md"
$statusLines = @(
    "# EP compatibility pipeline status",
    "",
    "- **Original model:** ``$($ModelPath)``",
    "- **EP onnx:** ``$($EpOnnx)``",
    "- **EP dump succeeded:** $(-not $dumpFailed -and $haveEpOnnx)",
    "- **Compatibility analyzed:** ``$($compatModel)``",
    "- **Note:** $compatNote",
    ""
)
if ($dumpFailed) {
    $statusLines += @("## Dump error", "", "``````", $dumpError, "``````", "")
}
$statusLines | Set-Content -LiteralPath $statusPath -Encoding UTF8

# Copy/link key artifacts to output root for convenience
$summarySrc = Join-Path $CompatDir "model_compatibility_report.md"
if (Test-Path -LiteralPath $summarySrc) {
    Copy-Item -LiteralPath $summarySrc -Destination (Join-Path $OutputDir "model_compatibility_report.md") -Force
    Copy-Item -LiteralPath (Join-Path $CompatDir "model_compatibility_details.md") `
        -Destination (Join-Path $OutputDir "model_compatibility_details.md") -Force

    # When dump did not produce an EP graph the compatibility analysis ran on
    # the ORIGINAL ONNX, not the EP-rewritten graph. Add a prominent badge to
    # the top of both report copies (after the H1) so the limitation is
    # impossible to miss when the agent reads back the report to the user.
    if (-not $haveEpOnnx) {
        $badge = "> **Source:** original ONNX (no EP rewrites). VOE was not configured or the dump failed; report reflects the model file as authored, not the graph the EP would compile."
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
if ($haveEpOnnx) {
    Write-Host "EP input:              $EpOnnx"
    Write-Host "Op comparison:         $(Join-Path $OutputDir 'op_distribution_comparison.md')"
} else {
    Write-Host "EP input:              (not produced)"
}
Write-Host "Compatibility report:  $(Join-Path $OutputDir 'model_compatibility_report.md')"
Write-Host "Full artifacts:        $CompatDir"
if (-not $haveEpOnnx) { Write-Host $compatNote -ForegroundColor Yellow }
