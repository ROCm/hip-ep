##
## Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
## Licensed under the MIT License.
##

# MorphiZen Developer Environment Setup Script
# This script installs pre-commit hooks for automatic code formatting

Write-Host "=== MorphiZen Developer Environment Setup ===" -ForegroundColor Cyan

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    Write-Host "  Install via: winget install Python.Python.3.11" -ForegroundColor Yellow
    exit 1
}

# Install pre-commit
Write-Host "`nInstalling pre-commit..." -ForegroundColor Cyan
pip install pre-commit

# Install git hooks
Write-Host "`nInstalling pre-commit git hooks..." -ForegroundColor Cyan
pre-commit install

Write-Host "`n=== Setup Complete ===" -ForegroundColor Green
Write-Host "Pre-commit hooks are now installed." -ForegroundColor Green
Write-Host "`nIMPORTANT:" -ForegroundColor Yellow
Write-Host "  - Pre-commit manages all formatting tools (clang-format, lintrunner, etc.)" -ForegroundColor White
Write-Host "  - Do NOT install or run clang-format manually" -ForegroundColor White
Write-Host "  - Always use pre-commit for code formatting" -ForegroundColor White
Write-Host "`nUsage:" -ForegroundColor Cyan
Write-Host "  - Hooks run automatically on 'git commit'" -ForegroundColor White
Write-Host "  - Manual run: pre-commit run --all-files" -ForegroundColor White
Write-Host "  - Update hooks: pre-commit autoupdate" -ForegroundColor White
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Run 'pre-commit run --all-files' to check existing code" -ForegroundColor White
Write-Host "  2. See docs/developer-guide.md for more information" -ForegroundColor White
