##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Full environment setup + CMake configure for onnx-hipdnn-ep.
# Run this after a reboot or new terminal session. Only needed once per session;
# after this, use build_and_install.ps1 for incremental builds.
#
# Usage:
#   cd <path-to>\onnx-hipdnn-ep
#   .\.venv\Scripts\Activate.ps1
#   . .\setup_and_configure.ps1
#
# Path defaults are derived from $PSScriptRoot (the location of this script),
# so the workspace layout documented in docs/quick_start.md works out of the
# box. To override any path, set the matching environment variable below
# before dot-sourcing the script:
#   $env:HIPDNN_WORKSPACE_ROOT  # parent of <onnx-hipdnn-ep>; defaults to ..
#   $env:HIPDNN_BUILD_DIR       # cmake build directory
#   $env:HIPDNN_PREBUILT_DIR    # cmake install prefix (prebuilt-local)
#   $env:THEROCK_DIST_OVERRIDE  # TheRock SDK directory
$ErrorActionPreference = "Stop"

# Repo root = directory this script lives in.
$repoDir = $PSScriptRoot

# Workspace root = parent of the repo. Default layout has onnx-hipdnn-ep,
# build, prebuilt-local, and therock-* as siblings under this directory.
$workspaceRoot = if ($env:HIPDNN_WORKSPACE_ROOT) {
    $env:HIPDNN_WORKSPACE_ROOT
} else {
    Split-Path -Parent $repoDir
}

$buildDir = if ($env:HIPDNN_BUILD_DIR) {
    $env:HIPDNN_BUILD_DIR
} else {
    Join-Path $workspaceRoot "build\onnx-hipdnn-ep"
}
$prebuilt = if ($env:HIPDNN_PREBUILT_DIR) {
    $env:HIPDNN_PREBUILT_DIR
} else {
    Join-Path $workspaceRoot "prebuilt-local"
}
$therock = if ($env:THEROCK_DIST_OVERRIDE) {
    $env:THEROCK_DIST_OVERRIDE
} else {
    Join-Path $workspaceRoot "therock-7.11.0-clean"
}

Set-Location $repoDir

# --- 1. Activate venv (if not already active) ---
if (-not $env:VIRTUAL_ENV) {
    Write-Host "=== Activating Python venv ===" -ForegroundColor Yellow
    & "$repoDir\.venv\Scripts\Activate.ps1"
}

# --- 2. Load MSVC environment (if not already loaded) ---
if (-not $env:INCLUDE) {
    # Locate vcvarsall.bat. Prefer vswhere (the official VS discovery tool);
    # fall back to the C:\msvsn2022 junction described in
    # docs/quick_start.md (Troubleshooting -> "vcvarsall.bat not found").
    $vswhere   = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vcvarsall = $null
    if (Test-Path $vswhere) {
        $vsInstall = & $vswhere -latest -property installationPath
        if ($vsInstall) {
            $vcvarsall = Join-Path $vsInstall "VC\Auxiliary\Build\vcvarsall.bat"
        }
    }
    if (-not $vcvarsall -or -not (Test-Path $vcvarsall)) {
        $vcvarsall = "C:\msvsn2022\VC\Auxiliary\Build\vcvarsall.bat"
    }
    if (-not (Test-Path $vcvarsall)) {
        Write-Host "ERROR: vcvarsall.bat not found at $vcvarsall" -ForegroundColor Red
        Write-Host "       Install Visual Studio 2022 with the 'Desktop development" -ForegroundColor Red
        Write-Host "       with C++' workload, or create the C:\msvsn2022 junction" -ForegroundColor Red
        Write-Host "       described in docs/quick_start.md." -ForegroundColor Red
        exit 1
    }

    Write-Host "=== Loading MSVC environment ===" -ForegroundColor Yellow
    Write-Host "    vcvarsall: $vcvarsall" -ForegroundColor DarkGray
    cmd /c "`"$vcvarsall`" x64 && set" |
        ForEach-Object {
            $s = $_.ToString(); $eq = $s.IndexOf('=')
            if ($eq -gt 0) {
                $k = $s.Substring(0, $eq); $v = $s.Substring($eq + 1)
                [Environment]::SetEnvironmentVariable($k, $v, 'Process')
            }
        }
    # vcvarsall may fail to discover the Windows 10 SDK; patch manually if needed
    $sdkRoot = "C:\Program Files (x86)\Windows Kits\10"
    $sdkVer  = "10.0.26100.0"
    if ($env:INCLUDE -notlike "*Windows Kits*") {
        Write-Host "  -> Patching Windows SDK into environment ($sdkVer)" -ForegroundColor Yellow
        $env:INCLUDE += ";$sdkRoot\Include\$sdkVer\ucrt;$sdkRoot\Include\$sdkVer\um;$sdkRoot\Include\$sdkVer\shared;$sdkRoot\Include\$sdkVer\winrt;$sdkRoot\Include\$sdkVer\cppwinrt"
        $env:LIB     += ";$sdkRoot\Lib\$sdkVer\ucrt\x64;$sdkRoot\Lib\$sdkVer\um\x64"
        $env:PATH     = "$sdkRoot\bin\$sdkVer\x64;$env:PATH"
    }
}

# --- 3. Set HIP/TheRock env vars ---
$env:HIP_PATH = $therock
$env:THEROCK_DIST = $therock
if ($env:PATH -notlike "*$therock\bin*") {
    $env:PATH = "$therock\bin;$env:PATH"
}

# --- 4. Delete stale build dir and reconfigure ---
if (Test-Path $buildDir) {
    Write-Host "=== Removing stale build directory ===" -ForegroundColor Yellow
    Remove-Item -Recurse -Force $buildDir
}

Write-Host "=== Configuring CMake ===" -ForegroundColor Cyan
Write-Host "    repo:     $repoDir" -ForegroundColor DarkGray
Write-Host "    build:    $buildDir" -ForegroundColor DarkGray
Write-Host "    prebuilt: $prebuilt" -ForegroundColor DarkGray
Write-Host "    therock:  $therock" -ForegroundColor DarkGray
cmake -S . -B $buildDir -G Ninja `
    -DCMAKE_C_COMPILER=cl `
    -DCMAKE_CXX_COMPILER=cl `
    -DBUILD_SHARED_LIBS=OFF `
    -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_PREFIX_PATH=$prebuilt" `
    "-DCMAKE_INSTALL_PREFIX=$prebuilt" `
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON `
    "-DTHEROCK_DIST=$therock" `
    -DHIP_PLATFORM=amd `
    -DHIP_ARCHITECTURES=gfx1151 `
    -DBUILD_EP=ON `
    -DBUILD_HIP_TOOLS=ON `
    -DBUILD_MOCK_RUNTIME=OFF `
    -DBUILD_HIPDNN_GRAPH=OFF `
    "-DCMAKE_CXX_FLAGS=/wd4706" `
    -DCMAKE_FIND_PACKAGE_PREFER_CONFIG=ON `
    -Dprotobuf_MODULE_COMPATIBLE=ON

if ($LASTEXITCODE -ne 0) { Write-Host "CONFIGURE FAILED" -ForegroundColor Red; exit 1 }

Write-Host "=== Configure done. Now run: . .\build_and_install.ps1 ===" -ForegroundColor Green
