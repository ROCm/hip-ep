# hipDNNEP Installation Script
# 
# NOTE: This is a ONE-TIME SETUP SCRIPT for initial installation.
# If hipDNNEP is already installed and built, you do not need to run this script again.
# This script was used during initial project setup and is kept for reference.
#
# Run this script AFTER downloading and extracting TheRock SDK to C:\Develop\TheRock

param(
    [string]$TherockPath = "C:\Develop\TheRock",
    [string]$OnnxRuntimePath = "C:/Develop/m/local",
    [string]$SourcePath = "C:\Develop\m\source\hipDNNEP",
    [string]$BuildPath = "C:/Develop/m/build/hipDNNEP/RelWithDebInfo"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "hipDNNEP Installation Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Verify TheRock SDK
Write-Host "Step 1: Verifying TheRock SDK installation..." -ForegroundColor Green
if (-not (Test-Path "$TherockPath\bin\hipconfig.exe")) {
    Write-Host "[ERROR] TheRock SDK not found at $TherockPath" -ForegroundColor Red
    Write-Host "Please download and extract TheRock SDK first." -ForegroundColor Yellow
    Write-Host "Visit: https://therock-nightly-tarball.s3.amazonaws.com/index.html" -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] TheRock SDK found at $TherockPath" -ForegroundColor Green

# Verify required components
$requiredPaths = @(
    "$TherockPath\bin\iree-compile.exe",
    "$TherockPath\lib\cmake\hipdnn_frontend",
    "$TherockPath\lib\cmake\hipdnn_backend"
)

foreach ($path in $requiredPaths) {
    if (-not (Test-Path $path)) {
        Write-Host "[ERROR] Required component not found: $path" -ForegroundColor Red
        exit 1
    }
}
Write-Host "[OK] All required TheRock components found" -ForegroundColor Green

# Step 2: Verify ONNXRuntime
Write-Host ""
Write-Host "Step 2: Verifying ONNXRuntime installation..." -ForegroundColor Green
if (-not (Test-Path "$OnnxRuntimePath\include\onnxruntime")) {
    Write-Host "[ERROR] ONNXRuntime not found at $OnnxRuntimePath" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] ONNXRuntime found at $OnnxRuntimePath" -ForegroundColor Green

# Step 3: Verify hipDNNEP source
Write-Host ""
Write-Host "Step 3: Verifying hipDNNEP source..." -ForegroundColor Green
if (-not (Test-Path "$SourcePath\CMakeLists.txt")) {
    Write-Host "[ERROR] hipDNNEP source not found at $SourcePath" -ForegroundColor Red
    Write-Host "Please clone the repository first:" -ForegroundColor Yellow
    Write-Host "  git clone https://github.com/zpye/hipDNNEP.git C:\Develop\m\source\hipDNNEP" -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] hipDNNEP source found at $SourcePath" -ForegroundColor Green

# Step 4: Set environment variables
Write-Host ""
Write-Host "Step 4: Setting environment variables..." -ForegroundColor Green
$env:THEROCK_DIST = $TherockPath
$env:HIP_PLATFORM = "amd"
$env:PATH = "$TherockPath\bin;" + $env:PATH
Write-Host "[OK] Environment variables set" -ForegroundColor Green
Write-Host "  THEROCK_DIST=$env:THEROCK_DIST" -ForegroundColor Cyan
Write-Host "  HIP_PLATFORM=$env:HIP_PLATFORM" -ForegroundColor Cyan

# Step 5: Configure CMake
Write-Host ""
Write-Host "Step 5: Configuring CMake build with Clang..." -ForegroundColor Green
Push-Location $SourcePath

# Clang is required - MSVC cannot compile TheRock SDK headers
$clangPath = "c:/LLVM20/bin/clang++.exe"
if (-not (Test-Path $clangPath)) {
    Write-Host "[ERROR] Clang not found at $clangPath" -ForegroundColor Red
    Write-Host "Clang 20.x is REQUIRED to build hipDNNEP (MSVC is incompatible)" -ForegroundColor Yellow
    Pop-Location
    exit 1
}

try {
    cmake -G Ninja `
        -B $BuildPath `
        -S . `
        -DCMAKE_BUILD_TYPE=RelWithDebInfo `
        -DCMAKE_CXX_COMPILER=$clangPath `
        -DTHEROCK_DIST=$TherockPath `
        -DONNXRUNTIME_ROOT=$OnnxRuntimePath `
        -DHIPDNN_EP_BUILD_TESTS=OFF `
        -DCMAKE_PREFIX_PATH="$OnnxRuntimePath;$TherockPath"

    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed"
    }
    Write-Host "[OK] CMake configuration successful (using Clang)" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] CMake configuration failed: $_" -ForegroundColor Red
    Pop-Location
    exit 1
}

# Step 6: Build
Write-Host ""
Write-Host "Step 6: Building hipDNNEP..." -ForegroundColor Green
try {
    cmake --build $BuildPath

    if ($LASTEXITCODE -ne 0) {
        throw "Build failed"
    }
    Write-Host "[OK] Build successful" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Build failed: $_" -ForegroundColor Red
    Pop-Location
    exit 1
}

Pop-Location

# Step 7: Prepare test environment
Write-Host ""
Write-Host "Step 7: Preparing test environment..." -ForegroundColor Green
try {
    $testDir = Join-Path $BuildPath "test"
    
    # Copy hipdnn_ep.dll
    $hipdnnEpDll = Join-Path $BuildPath "hipdnn_ep.dll"
    if (Test-Path $hipdnnEpDll) {
        Copy-Item $hipdnnEpDll $testDir -Force
        Write-Host "[OK] Copied hipdnn_ep.dll to test directory" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] hipdnn_ep.dll not found at $hipdnnEpDll" -ForegroundColor Yellow
    }
    
    # Copy onnxruntime.dll
    $onnxRuntimeDll = Join-Path $OnnxRuntimePath "bin\onnxruntime.dll"
    if (Test-Path $onnxRuntimeDll) {
        Copy-Item $onnxRuntimeDll $testDir -Force
        Write-Host "[OK] Copied onnxruntime.dll to test directory" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] onnxruntime.dll not found at $onnxRuntimeDll" -ForegroundColor Yellow
    }
} catch {
    Write-Host "[WARNING] Failed to prepare test environment: $_" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Build outputs:" -ForegroundColor White
Write-Host "  hipdnn_ep.dll: $BuildPath\hipdnn_ep.dll" -ForegroundColor Cyan
Write-Host "  Tests: $BuildPath\test\hipdnn_ep_tests.exe" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run tests (may fail without GPU):" -ForegroundColor Yellow
Write-Host "  cd $BuildPath\test" -ForegroundColor Cyan
Write-Host "  .\hipdnn_ep_tests.exe" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: Tests require AMD GPU hardware to run successfully." -ForegroundColor Yellow
Write-Host "      Build artifacts can be used for development even without GPU." -ForegroundColor Yellow
