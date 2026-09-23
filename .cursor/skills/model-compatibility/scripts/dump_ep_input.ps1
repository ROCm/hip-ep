##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Dump the graph the EP actually compiles, as MLIR text.
#
# Runs hip-onnx-runner with an init-only MorphiZen config: session creation
# drives EP initialization and the pass.init dump, and --no-run skips
# inference afterwards. MORPHIZEN_SAVE_MLIR_AS_TEXT selects text over
# bytecode.
#
# Weights are not inlined -- onnx.Constant carries location/offset/size
# pointing at the external data file -- so the dump stays small even for
# multi-GB models.
#
# Resolution order for the package root: -GpuTestPackageRoot, then
# $env:GPU_TEST_PACKAGE_ROOT. Missing both emits a machine-readable marker
# and exits 10 so the caller can prompt.

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ModelPath,

    [string]$GpuTestPackageRoot = "",
    [string]$DumpDirectory = "",
    [string]$DumpFileName = "ep_input.mlir",
    [string]$MorphizenConfigPath = ""
)
$ErrorActionPreference = "Stop"

# ProviderPath avoids the "Microsoft.PowerShell.Core\FileSystem::" prefix,
# which breaks native executables on UNC paths.
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).ProviderPath

if ([string]::IsNullOrWhiteSpace($GpuTestPackageRoot) -and $env:GPU_TEST_PACKAGE_ROOT) {
    $GpuTestPackageRoot = $env:GPU_TEST_PACKAGE_ROOT
}
if ([string]::IsNullOrWhiteSpace($GpuTestPackageRoot)) {
    Write-Output '[GPU_TEST_PACKAGE_NOT_CONFIGURED] dump_ep_input.ps1: no GpuTestPackageRoot. Pass -GpuTestPackageRoot <path> or set env var GPU_TEST_PACKAGE_ROOT to the package containing bin\hip-onnx-runner.exe.'
    exit 10
}
$GpuTestPackageRoot = (Resolve-Path -LiteralPath $GpuTestPackageRoot).ProviderPath

$PkgBin = Join-Path $GpuTestPackageRoot "bin"
$RunnerExe = Join-Path $PkgBin "hip-onnx-runner.exe"
if (-not (Test-Path -LiteralPath $RunnerExe)) {
    throw "hip-onnx-runner.exe not found: $RunnerExe"
}

# An older package silently lacks the flags this dump depends on and would
# only print its usage text. Check up front and say what to do about it.
$help = & $RunnerExe --help 2>&1 | Out-String
$missing = @('--no-run', '--allow-cpu-fallback', '--provider-options') |
    Where-Object { $help -notmatch [regex]::Escape($_) }
if ($missing.Count -gt 0) {
    throw ("hip-onnx-runner.exe is missing required option(s): {0}. " -f ($missing -join ', ')) +
          "Update the gpu-test-package at $GpuTestPackageRoot."
}

if ([string]::IsNullOrWhiteSpace($MorphizenConfigPath)) {
    $MorphizenConfigPath = Join-Path $PSScriptRoot "morphizen_init_config.json"
}
if (-not (Test-Path -LiteralPath $MorphizenConfigPath)) {
    throw "MorphiZen config not found: $MorphizenConfigPath (default ships at scripts/morphizen_init_config.json; pass -MorphizenConfigPath <file> to override)"
}
$MorphizenConfigPath = (Resolve-Path -LiteralPath $MorphizenConfigPath).ProviderPath

if ([string]::IsNullOrWhiteSpace($DumpDirectory)) {
    $DumpDirectory = [System.IO.Path]::GetDirectoryName($ModelPath)
}
New-Item -ItemType Directory -Force -Path $DumpDirectory | Out-Null
$DumpDirectory = (Resolve-Path -LiteralPath $DumpDirectory).ProviderPath
$DumpPath = Join-Path $DumpDirectory $DumpFileName

# pass.init.directory is per-run, so it is passed here rather than baked into
# the shared config file. The runner merges these over the config's own
# provider_options. Forward slashes keep the value clear of escaping issues.
$DumpDirForOption = $DumpDirectory -replace '\\', '/'

Write-Host "Model:            $ModelPath"
Write-Host "Package bin:      $PkgBin"
Write-Host "MorphiZen config: $MorphizenConfigPath"
Write-Host "Expected output:  $DumpPath"
Write-Host ""

$prevSaveAsText = $env:MORPHIZEN_SAVE_MLIR_AS_TEXT
Push-Location $PkgBin
try {
    $env:MORPHIZEN_SAVE_MLIR_AS_TEXT = "1"

    # glog always writes "WARNING: Logging before InitGoogleLogging()" to
    # stderr at startup. Under $ErrorActionPreference = "Stop" PowerShell
    # turns that first native-stderr line into a terminating
    # NativeCommandError, killing the runner before pass.init can dump.
    # Relax it just around the call; the checks below still catch real
    # failures.
    $savedErrPref = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        # Captured, not streamed: when the runner fails it usually says why
        # on stderr, and that line is what the report needs to show.
        $runnerOutput = & $RunnerExe `
            -m $ModelPath `
            --no-run `
            --allow-cpu-fallback `
            --provider-options "config_file=$MorphizenConfigPath" `
            --provider-options "pass.init.directory=$DumpDirForOption" 2>&1 |
            ForEach-Object { $_.ToString() }
        $exitCode = $LASTEXITCODE
        $runnerOutput | ForEach-Object { Write-Host $_ }
    } finally {
        $ErrorActionPreference = $savedErrPref
    }
} finally {
    Pop-Location
    if ($null -eq $prevSaveAsText) {
        Remove-Item Env:MORPHIZEN_SAVE_MLIR_AS_TEXT -ErrorAction SilentlyContinue
    } else {
        $env:MORPHIZEN_SAVE_MLIR_AS_TEXT = $prevSaveAsText
    }
}

if (-not (Test-Path -LiteralPath $DumpPath)) {
    # Surface the runner's own diagnosis rather than just the exit code:
    # "Sequence type conversion is not implemented yet" tells the reader
    # what to do, 0xC0000409 does not.
    $why = @($runnerOutput | Where-Object {
        $_ -match 'Error in ORT API|error:|Exception|not implemented|failed'
    } | Select-Object -Last 1)
    $reason = if ($why.Count) { ([string]$why[0]).Trim() } else { "" }
    if ($exitCode -ne 0) {
        $msg = "hip-onnx-runner.exe failed with exit code $exitCode and no dump was created"
        if ($reason) { $msg += ": $reason" }
        throw $msg
    }
    throw "Dump file was not created: $DumpPath"
}
$size = (Get-Item -LiteralPath $DumpPath).Length
if ($size -le 0) {
    throw "Dump file is empty: $DumpPath"
}

Write-Host ""
if ($exitCode -ne 0) {
    # The dump completes during session creation, so a later non-zero exit
    # does not invalidate it.
    Write-Host "WARN: hip-onnx-runner.exe exited with code $exitCode, but the dump exists." -ForegroundColor Yellow
}
Write-Host "OK: dumped EP input MLIR ($size bytes)"
Write-Host "    $DumpPath"
