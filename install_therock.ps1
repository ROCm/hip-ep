# TheRock and hipDNN Installation Script for Windows
#
# NOTE: This is a ONE-TIME SETUP SCRIPT for initial installation.
# If TheRock SDK is already installed, you do not need to run this script again.
# This script was used during initial project setup and is kept for reference.
#
# Run this script in Administrative PowerShell

param(
    [string]$GpuArch = "",
    [string]$TargetDir = "C:\dist"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "TheRock and hipDNN Installation Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Create target directory
if (-not (Test-Path $TargetDir)) {
    Write-Host "Creating directory: $TargetDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
}

# Step 1: Check/Install Clang
Write-Host "Step 1: Checking Clang 20.x installation..." -ForegroundColor Green
$clangPath = "$TargetDir\clang"
if (Test-Path "$clangPath\bin\amdgpu-arch.exe") {
    Write-Host "[OK] Clang already installed at $clangPath" -ForegroundColor Green
} else {
    Write-Host "[REQUIRED] Clang not found. Please download manually:" -ForegroundColor Yellow
    Write-Host "  1. Visit: https://github.com/llvm/llvm-project/releases" -ForegroundColor Yellow
    Write-Host "  2. Download: LLVM-20.x.x-win64.exe or tar.xz" -ForegroundColor Yellow
    Write-Host "  3. Extract to: $clangPath" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter after installing Clang to continue"
    
    if (-not (Test-Path "$clangPath\bin\amdgpu-arch.exe")) {
        Write-Host "[ERROR] Clang still not found at $clangPath" -ForegroundColor Red
        exit 1
    }
}

# Step 2: Determine GPU architecture
Write-Host ""
Write-Host "Step 2: Determining GPU architecture..." -ForegroundColor Green
if ($GpuArch -eq "") {
    try {
        $GpuArch = & "$clangPath\bin\amdgpu-arch.exe" 2>$null
        Write-Host "[OK] Detected GPU architecture: $GpuArch" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Could not detect GPU. Please specify with -GpuArch parameter" -ForegroundColor Red
        Write-Host "  Common values: gfx1103, gfx1100, gfx90a" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "[OK] Using specified GPU architecture: $GpuArch" -ForegroundColor Green
}

# Determine GFX family
$gfxFamily = switch -Regex ($GpuArch) {
    '^gfx90[0-9a-c]$' { "gfx90X-all" }
    '^gfx94[0-2]$' { "gfx94X-all" }
    '^gfx103[0-6]$' { "gfx103X-all" }
    '^gfx110[0-3]$' { "gfx110X-all" }
    '^gfx115[0-2]$' { "gfx115X-all" }
    '^gfx120[0-1]$' { "gfx120X-all" }
    default { 
        Write-Host "[ERROR] Unknown GPU architecture: $GpuArch" -ForegroundColor Red
        exit 1
    }
}
Write-Host "  GFX Family: $gfxFamily" -ForegroundColor Cyan

# Step 3: Download TheRock
Write-Host ""
Write-Host "Step 3: TheRock ROCm SDK installation..." -ForegroundColor Green
$therockPath = "$TargetDir\therock"

if (Test-Path "$therockPath\bin\hipconfig.exe") {
    Write-Host "[OK] TheRock already installed at $therockPath" -ForegroundColor Green
} else {
    Write-Host "[REQUIRED] Please download TheRock manually:" -ForegroundColor Yellow
    Write-Host "  1. Visit: https://therock-nightly-tarball.s3.amazonaws.com/index.html" -ForegroundColor Yellow
    Write-Host "  2. Find latest: therock-dist-windows-$gfxFamily-*.tar.gz" -ForegroundColor Yellow
    Write-Host "  3. Extract to: $therockPath" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You can use 7-Zip or tar command to extract:" -ForegroundColor Cyan
    Write-Host "  tar -xzf therock-dist-windows-$gfxFamily-*.tar.gz -C $TargetDir" -ForegroundColor Cyan
    Write-Host ""
    Read-Host "Press Enter after extracting TheRock to continue"
    
    if (-not (Test-Path "$therockPath\bin\hipconfig.exe")) {
        Write-Host "[ERROR] TheRock still not found at $therockPath" -ForegroundColor Red
        exit 1
    }
}

# Step 4: Verify TheRock
Write-Host ""
Write-Host "Step 4: Verifying TheRock installation..." -ForegroundColor Green
$env:PATH = "$therockPath\bin;" + $env:PATH
$env:HIP_PLATFORM = "amd"

try {
    $hipconfig = & "$therockPath\bin\hipconfig.exe" -rocmpath -n --hipclangpath 2>&1
    Write-Host "[OK] TheRock verification successful:" -ForegroundColor Green
    Write-Host $hipconfig
} catch {
    Write-Host "[ERROR] TheRock verification failed" -ForegroundColor Red
    exit 1
}

# Step 5: Set environment variables permanently
Write-Host ""
Write-Host "Step 5: Setting environment variables..." -ForegroundColor Green
Write-Host "Setting system environment variables (requires admin)..." -ForegroundColor Yellow

# Add to system PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
if ($currentPath -notlike "*$therockPath\bin*") {
    $newPath = "$therockPath\bin;" + $currentPath
    [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
    Write-Host "[OK] Added TheRock to system PATH" -ForegroundColor Green
} else {
    Write-Host "[OK] TheRock already in system PATH" -ForegroundColor Green
}

# Set HIP_PLATFORM
[Environment]::SetEnvironmentVariable("HIP_PLATFORM", "amd", "Machine")
Write-Host "[OK] Set HIP_PLATFORM=amd" -ForegroundColor Green

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "TheRock Installation Summary" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Clang Path:    $clangPath" -ForegroundColor White
Write-Host "TheRock Path:  $therockPath" -ForegroundColor White
Write-Host "GPU Arch:      $GpuArch" -ForegroundColor White
Write-Host "GFX Family:    $gfxFamily" -ForegroundColor White
Write-Host ""
Write-Host "[OK] TheRock ROCm SDK installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Restart your PowerShell session to use new environment variables" -ForegroundColor Yellow
Write-Host "2. Follow the guide to build hipDNN (Section 5 in doc/HIPDNN_WINDOWS_SETUP.md)" -ForegroundColor Yellow
Write-Host "3. Build morphizen-hipdnn (Section 6)" -ForegroundColor Yellow
