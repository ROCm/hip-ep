##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Dump the graph the hip-ep compiler sees, as text MLIR.
#
# Runs the model through hip-onnx-runner with a MorphiZen configuration that
# holds only the init pass: the EP imports and dumps the graph, then the run
# stops before compilation and before inference. An init-only configuration
# claims no node, so CPU fallback must be allowed for the session to be
# created at all.
#
# Usage:
#   .\dump_ep_input.ps1 -ModelPath <model.onnx> -OutputDir <dir>
#   .\dump_ep_input.ps1 -ModelPath <model.onnx> -HipEpPackageRoot <dir>

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ModelPath,

    # Directory receiving ep_input.mlir. Defaults next to the model.
    [string]$OutputDir = "",

    # hip-ep package holding the runner and the hipgpu EP library under bin.
    # Resolution order: -HipEpPackageRoot > $env:HIP_EP_PACKAGE_ROOT > error.
    [string]$HipEpPackageRoot = "",

    [string]$ConfigPath = "",
    [string]$DumpFileName = "ep_input.mlir"
)

$ErrorActionPreference = "Stop"

# ProviderPath avoids the "Microsoft.PowerShell.Core\FileSystem::" prefix,
# which breaks native executables on UNC paths.
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).ProviderPath
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = [System.IO.Path]::GetDirectoryName($ModelPath)
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).ProviderPath

if ([string]::IsNullOrWhiteSpace($HipEpPackageRoot) -and $env:HIP_EP_PACKAGE_ROOT) {
    $HipEpPackageRoot = $env:HIP_EP_PACKAGE_ROOT
}
if ([string]::IsNullOrWhiteSpace($HipEpPackageRoot)) {
    # Machine-readable marker so the caller can ask the user for a path
    # instead of surfacing a confusing "runner not found".
    # Single-quoted to keep the literal env var name.
    $msg = '[HIP_EP_NOT_CONFIGURED] No HipEpPackageRoot supplied. Pass -HipEpPackageRoot <dir>, or set $env:HIP_EP_PACKAGE_ROOT to a hip-ep package (the directory containing bin\hip-onnx-runner.exe).'
    Write-Output $msg
    exit 10
}
$HipEpPackageRoot = (Resolve-Path -LiteralPath $HipEpPackageRoot).ProviderPath

$PackageBin = Join-Path $HipEpPackageRoot "bin"

# Executable suffixes and library naming differ by platform, and a package may
# keep the EP library beside the tools or in a sibling lib directory.
function Resolve-PackageFile {
    param([string[]]$Names, [string[]]$Directories)
    foreach ($directory in $Directories) {
        foreach ($name in $Names) {
            $candidate = Join-Path $directory $name
            if (Test-Path -LiteralPath $candidate) { return $candidate }
        }
    }
    return ""
}

$RunnerExe = Resolve-PackageFile -Names @('hip-onnx-runner.exe', 'hip-onnx-runner') `
    -Directories @($PackageBin)
if (-not $RunnerExe) {
    throw "hip-onnx-runner not found under $PackageBin"
}
$EpLib = Resolve-PackageFile -Names @('hipgpu.dll', 'libhipgpu.so') `
    -Directories @($PackageBin, (Join-Path $HipEpPackageRoot "lib"))
if (-not $EpLib) {
    throw "hipgpu EP library not found under $HipEpPackageRoot"
}

# The dump needs flags added in ROCm/hip-ep#1033. Check before running so an
# older package fails with an actionable message rather than an argument error.
$runnerHelp = & $RunnerExe --help 2>&1 | Out-String
foreach ($flag in @('--no-run', '--provider-options', '--allow-cpu-fallback')) {
    if ($runnerHelp -notmatch [regex]::Escape($flag)) {
        $msg = "[HIP_EP_RUNNER_TOO_OLD] $RunnerExe does not support $flag. Use a hip-ep package built after the runner gained --no-run / --provider-options / --allow-cpu-fallback."
        Write-Output $msg
        exit 11
    }
}

if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $PSScriptRoot "only_init_config.json"
}
if (-not (Test-Path -LiteralPath $ConfigPath)) {
    throw "MorphiZen config not found: $ConfigPath"
}
$ConfigPath = (Resolve-Path -LiteralPath $ConfigPath).ProviderPath

$DumpPath = Join-Path $OutputDir $DumpFileName
$StdoutPath = Join-Path $OutputDir "dump_stdout.txt"
$StderrPath = Join-Path $OutputDir "dump_stderr.txt"
if (Test-Path -LiteralPath $DumpPath) { Remove-Item -LiteralPath $DumpPath -Force }

Write-Host "Model:           $ModelPath"
Write-Host "hip-ep package:  $HipEpPackageRoot"
Write-Host "MorphiZen config: $ConfigPath (init pass only)"
Write-Host "Expected output: $DumpPath"
Write-Host ""

$prevEpLib = $env:MORPHIZEN_EP_LIB
$prevSaveText = $env:MORPHIZEN_SAVE_MLIR_AS_TEXT
$prevPath = $env:PATH
try {
    $env:MORPHIZEN_EP_LIB = $EpLib
    # The init pass serializes the module as bytecode unless this is set; the
    # downstream parsers read text.
    $env:MORPHIZEN_SAVE_MLIR_AS_TEXT = "1"
    $env:PATH = "$PackageBin;$env:PATH"

    $runnerArgs = @(
        '-m', $ModelPath,
        '--no-run',
        '--allow-cpu-fallback',
        '--provider-options', "config_file=$ConfigPath",
        '--mlir-dump-dir', $OutputDir
    )
    # Start-Process keeps the runner's stderr out of the PowerShell error
    # stream, which would otherwise abort the script on the first native
    # warning line under $ErrorActionPreference = "Stop".
    $proc = Start-Process -FilePath $RunnerExe -ArgumentList $runnerArgs `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath
    $exitCode = $proc.ExitCode
}
finally {
    if ($null -eq $prevEpLib) {
        Remove-Item Env:MORPHIZEN_EP_LIB -ErrorAction SilentlyContinue
    } else {
        $env:MORPHIZEN_EP_LIB = $prevEpLib
    }
    if ($null -eq $prevSaveText) {
        Remove-Item Env:MORPHIZEN_SAVE_MLIR_AS_TEXT -ErrorAction SilentlyContinue
    } else {
        $env:MORPHIZEN_SAVE_MLIR_AS_TEXT = $prevSaveText
    }
    $env:PATH = $prevPath
}

Get-Content -LiteralPath $StdoutPath | Write-Host
# A clean run leaves nothing on stderr; an empty file only invites a reader to
# open it.
if ((Get-Item -LiteralPath $StderrPath).Length -eq 0) {
    Remove-Item -LiteralPath $StderrPath -ErrorAction SilentlyContinue
}

if (-not (Test-Path -LiteralPath $DumpPath)) {
    if (Test-Path -LiteralPath $StderrPath) {
        Write-Host "--- runner stderr ---" -ForegroundColor DarkGray
        Get-Content -LiteralPath $StderrPath -Tail 20 | Write-Host
    }
    throw "Dump file was not created (runner exit code $exitCode): $DumpPath"
}
if ($exitCode -ne 0) {
    throw "hip-onnx-runner failed with exit code $exitCode"
}

$size = (Get-Item -LiteralPath $DumpPath).Length
if ($size -le 0) {
    throw "Dump file is empty: $DumpPath"
}
# Guard against a bytecode dump reaching the text parsers.
$firstLine = (Get-Content -LiteralPath $DumpPath -TotalCount 1)
if ($firstLine -notmatch '^\s*(builtin\.)?module') {
    throw "Dump is not text MLIR (first line: '$firstLine'). Check MORPHIZEN_SAVE_MLIR_AS_TEXT."
}

$meta = [ordered]@{
    model_path   = $ModelPath
    package_root = $HipEpPackageRoot
    runner       = $RunnerExe
    ep_lib       = $EpLib
    config_file  = $ConfigPath
    dump_path    = $DumpPath
    bytes        = $size
    utc          = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
}
$metaPath = Join-Path $OutputDir "dump_meta.json"
$meta | ConvertTo-Json | Set-Content -LiteralPath $metaPath -Encoding UTF8

Write-Host ""
Write-Host "OK: dumped EP-input MLIR ($size bytes)"
Write-Host "    $DumpPath"
