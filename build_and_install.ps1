##
## Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# Build, install, and switch to prebuilt-local\bin for benchmarking.
# Ensures MSVC environment, HIP_PATH, and THEROCK_DIST are set
# (safe to re-run after reboot).
#
# Path defaults are derived from $PSScriptRoot (the location of this script).
# Override any of the following environment variables before invoking the
# script if your layout differs:
#   $env:HIPDNN_WORKSPACE_ROOT  # parent of <onnx-hipdnn-ep>; defaults to ..
#   $env:HIPDNN_BUILD_DIR       # cmake build directory
#   $env:HIPDNN_BIN_DIR         # install bin directory (prebuilt-local\bin)
#   $env:THEROCK_DIST_OVERRIDE  # TheRock SDK directory
#   $env:ORT_PERF_TEST_PATH     # full path to onnxruntime_perf_test.exe
$ErrorActionPreference = "Stop"

$repoDir = $PSScriptRoot
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
$binDir = if ($env:HIPDNN_BIN_DIR) {
    $env:HIPDNN_BIN_DIR
} else {
    Join-Path $workspaceRoot "prebuilt-local\bin"
}
$therock = if ($env:THEROCK_DIST_OVERRIDE) {
    $env:THEROCK_DIST_OVERRIDE
} else {
    Join-Path $workspaceRoot "therock-7.11.0-clean"
}

# --- Environment setup (idempotent) ---

# 1. MSVC toolchain: import vcvarsall if INCLUDE is not already set.
#    Prefer vswhere (the official VS discovery tool); fall back to the
#    C:\msvsn2022 junction described in docs/quick_start.md.
if (-not $env:INCLUDE) {
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
    $sdkRoot = "C:\Program Files (x86)\Windows Kits\10"
    $sdkVer  = "10.0.26100.0"
    if ($env:INCLUDE -notlike "*Windows Kits*") {
        Write-Host "  -> Patching Windows SDK into environment ($sdkVer)" -ForegroundColor Yellow
        $env:INCLUDE += ";$sdkRoot\Include\$sdkVer\ucrt;$sdkRoot\Include\$sdkVer\um;$sdkRoot\Include\$sdkVer\shared;$sdkRoot\Include\$sdkVer\winrt;$sdkRoot\Include\$sdkVer\cppwinrt"
        $env:LIB     += ";$sdkRoot\Lib\$sdkVer\ucrt\x64;$sdkRoot\Lib\$sdkVer\um\x64"
        $env:PATH     = "$sdkRoot\bin\$sdkVer\x64;$env:PATH"
    }
}

# 2. HIP_PATH: point hipcc to TheRock (avoids space-in-path with system ROCm)
$env:HIP_PATH = $therock

# 3. THEROCK_DIST: needed by hip_runtime_static for HIP headers and by
#    hip-compiler.dll at runtime to find amdhip64.lib/MIOpen.lib/hipblaslt
$env:THEROCK_DIST = $therock

# 4. PATH: ensure TheRock binaries (amdhip64.dll, hipcc, etc.) are findable
if ($env:PATH -notlike "*$therock\bin*") {
    $env:PATH = "$therock\bin;$env:PATH"
}

# 5. Suppress noisy library logging that hurts benchmark accuracy
$env:MIOPEN_LOG_LEVEL = "1"
$env:MIOPEN_ENABLE_LOGGING = "0"
$env:MIOPEN_ENABLE_LOGGING_CMD = "0"
$env:HIPDNN_EP_DEBUG = "0"

Set-Location $repoDir

Write-Host "=== Building ===" -ForegroundColor Cyan
cmake --build $buildDir --config Release
if ($LASTEXITCODE -ne 0) { Write-Host "BUILD FAILED" -ForegroundColor Red; exit 1 }

Write-Host "=== Installing ===" -ForegroundColor Cyan
cmake --install $buildDir --config Release
if ($LASTEXITCODE -ne 0) { Write-Host "INSTALL FAILED" -ForegroundColor Red; exit 1 }

# 6. Sync TheRock SDK runtime DLLs and engine plugins into prebuilt-local/bin.
#    Always force-copy when sizes differ: previous "newer than" predicate silently
#    skipped SDK *downgrades* (e.g. 7.13 -> 7.11, where SDK files are older than
#    prior-run copies), leaving stale DLLs that shadow the real SDK via Windows
#    DLL search order and break JIT loads with subtle ABI mismatches.
$therockDlls = @("hipdnn_backend.dll", "amdhip64_7.dll")
foreach ($dll in $therockDlls) {
    $src = "$therock\bin\$dll"
    $dst = "$binDir\$dll"
    if (Test-Path $src) {
        $srcLen = (Get-Item $src).Length
        $dstLen = if (Test-Path $dst) { (Get-Item $dst).Length } else { -1 }
        if ($srcLen -ne $dstLen) {
            Write-Host "=== Refreshing $dll from TheRock SDK ($dstLen -> $srcLen bytes) ===" -ForegroundColor Yellow
            Copy-Item $src $dst -Force
        }
    }
}
if (Test-Path "$therock\bin\hipdnn_plugins") {
    xcopy /E /I /Y "$therock\bin\hipdnn_plugins" "$binDir\hipdnn_plugins" | Out-Null
}

# 7. Copy onnxruntime_perf_test.exe (not installed by ORT's cmake --install).
#    Search well-known locations: first the doc's layout (OnnxRuntime under
#    OnnxHipDNN), then one level up (OnnxRuntime as a sibling of the
#    workspace root). Set $env:ORT_PERF_TEST_PATH to override.
$ortPerfTest = if ($env:ORT_PERF_TEST_PATH) {
    $env:ORT_PERF_TEST_PATH
} else {
    $candidates = @(
        (Join-Path $workspaceRoot "OnnxRuntime\onnxruntime\build\Windows\Release\Release\onnxruntime_perf_test.exe"),
        (Join-Path (Split-Path -Parent $workspaceRoot) "OnnxRuntime\onnxruntime\build\Windows\Release\Release\onnxruntime_perf_test.exe")
    )
    $found = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($found) { $found } else { $candidates[0] }
}
$destPerfTest = Join-Path $binDir "onnxruntime_perf_test.exe"
if ((Test-Path $ortPerfTest) -and -not (Test-Path $destPerfTest)) {
    Write-Host "=== Copying onnxruntime_perf_test.exe ===" -ForegroundColor Yellow
    Copy-Item $ortPerfTest $destPerfTest
} elseif (-not (Test-Path $ortPerfTest)) {
    Write-Host "WARNING: onnxruntime_perf_test.exe not found at $ortPerfTest -- skipping copy" -ForegroundColor DarkYellow
    Write-Host "         (Set `$env:ORT_PERF_TEST_PATH if your ORT build is elsewhere.)" -ForegroundColor DarkYellow
}

Write-Host "=== Done. Switching to bin dir ===" -ForegroundColor Green
Set-Location $binDir
