@echo off
REM Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
REM Licensed under the MIT License.
REM
REM Run ROCm tests with TheRock environment

setlocal enabledelayedexpansion

REM Configuration
set THEROCK_DIST=C:\Develop\m\dist\therock
set BUILD_DIR=C:\Develop\m\build\morphizen-rocm

REM Check TheRock installation
if not exist "%THEROCK_DIST%\bin" (
    echo ERROR: TheRock not found at %THEROCK_DIST%
    echo Please set THEROCK_DIST to your TheRock installation path
    exit /b 1
)

echo === Setting up TheRock environment ===
set PATH=%THEROCK_DIST%\bin;%PATH%

REM Run Conv test
echo.
echo === Running Conv Test (MIOpen) ===
if exist "%BUILD_DIR%\test\rocm_conv_test.exe" (
    "%BUILD_DIR%\test\rocm_conv_test.exe"
) else (
    echo Conv test not built yet. Run build.bat first.
)

REM Run Gemm test  
echo.
echo === Running Gemm Test (hipBLASLt) ===
if exist "%BUILD_DIR%\test\rocm_gemm_test.exe" (
    "%BUILD_DIR%\test\rocm_gemm_test.exe"
) else (
    echo Gemm test not built yet. Run build.bat first.
)

echo.
echo === Tests completed ===
