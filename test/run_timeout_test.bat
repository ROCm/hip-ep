@echo off
REM Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
REM Licensed under the MIT License.

REM Script to run the GPU timeout test with proper environment setup

echo ============================================
echo   Running GPU Timeout Test
echo ============================================
echo.

REM Set TheRock environment
set THEROCK_DIST=C:\Develop\m\dist\therock
if not exist "%THEROCK_DIST%" (
    echo ERROR: TheRock not found at %THEROCK_DIST%
    echo Please install TheRock SDK or update THEROCK_DIST
    exit /b 1
)

REM Add TheRock to PATH for DLLs
set PATH=%THEROCK_DIST%\bin;%PATH%

REM Set GPU timeout (5 seconds default)
if not defined MORPHIZEN_GPU_TIMEOUT_MS (
    set MORPHIZEN_GPU_TIMEOUT_MS=5000
)

echo TheRock: %THEROCK_DIST%
echo GPU Timeout: %MORPHIZEN_GPU_TIMEOUT_MS%ms
echo.

REM Find the test executable
set BUILD_DIR=C:\Develop\m\build\morphizen-rocm
set TEST_EXE=%BUILD_DIR%\test\rocm_timeout_test.exe

if not exist "%TEST_EXE%" (
    echo ERROR: Test executable not found at %TEST_EXE%
    echo Please build the project first with build.bat
    exit /b 1
)

echo Running: %TEST_EXE%
echo.

REM Run the test
"%TEST_EXE%"

set TEST_EXIT_CODE=%ERRORLEVEL%

echo.
echo ============================================
if %TEST_EXIT_CODE% equ 0 (
    echo   Test PASSED
) else (
    echo   Test FAILED with exit code %TEST_EXIT_CODE%
)
echo ============================================

exit /b %TEST_EXIT_CODE%
