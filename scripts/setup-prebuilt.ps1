# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# PowerShell port of setup-prebuilt.sh that uses Invoke-WebRequest +
# Expand-Archive instead of `gh release download` + `unzip` so it works
# without GitHub CLI authentication on Windows hosts.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot    = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$PrebuiltDir = (Resolve-Path (Join-Path $RepoRoot "..")).Path
$PrebuiltDir = Join-Path $PrebuiltDir "prebuilt-local"
New-Item -ItemType Directory -Force -Path $PrebuiltDir | Out-Null

$Repo = "wcy123/llvm-mlir-prebuilt"

$assets = @(
    @{ tag = "llvm-22.1.0-release";        asset = "llvm-22.1.0-release-windows-x64.zip" },
    @{ tag = "protobuf-34.0-release";      asset = "protobuf-34.0-release-windows-x64.zip" },
    @{ tag = "flatbuffers-25.12.19-release"; asset = "flatbuffers-25.12.19-release-windows-x64.zip" }
)

foreach ($a in $assets) {
    $tag      = $a.tag
    $asset    = $a.asset
    $zipPath  = Join-Path $PrebuiltDir $asset
    $sentinel = Join-Path $PrebuiltDir ".extracted-$asset"
    $url      = "https://github.com/$Repo/releases/download/$tag/$asset"

    if (-not (Test-Path $zipPath)) {
        Write-Host "Downloading $asset ..."
        Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
        if (Test-Path $sentinel) { Remove-Item $sentinel -Force }
    } else {
        Write-Host "Already downloaded: $asset"
    }

    if (-not (Test-Path $sentinel)) {
        Write-Host "Extracting $asset ..."
        Expand-Archive -LiteralPath $zipPath -DestinationPath $PrebuiltDir -Force
        New-Item -ItemType File -Path $sentinel -Force | Out-Null
    } else {
        Write-Host "Already extracted: $asset"
    }
}

Write-Host ""
Write-Host "Pre-built binaries ready at: $PrebuiltDir"
Write-Host "CMake configs:"
Get-ChildItem (Join-Path $PrebuiltDir "lib\cmake") -ErrorAction SilentlyContinue | Select-Object Name | ForEach-Object { Write-Host "  $($_.Name)" }
