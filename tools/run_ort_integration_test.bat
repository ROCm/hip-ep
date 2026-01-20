@echo off
REM Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
REM Licensed under the MIT License.

echo ============================================================
echo Running ORT Integration Test for VitisAI MLIR EP
echo ============================================================
echo.

REM Detect workspace drive from current script location
set SCRIPT_DIR=%~dp0
set WORKSPACE_DRIVE=%SCRIPT_DIR:~0,2%
set WORKSPACE_ROOT=%WORKSPACE_DRIVE%/Develop/m
echo Detected workspace: %WORKSPACE_ROOT%
echo.

REM Set environment variable for MLIR backend
echo Setting environment variable...
set MORPHIZEN_ORT_BRIDGE_UNITTEST_BACKEND=mlir-backend

echo Environment variable set:
echo   MORPHIZEN_ORT_BRIDGE_UNITTEST_BACKEND=%MORPHIZEN_ORT_BRIDGE_UNITTEST_BACKEND%
echo.

REM Check if test executable exists
set TEST_EXE=%WORKSPACE_ROOT%/build/morphizen-mlir/bin/ort_integration_test.exe
if not exist "%TEST_EXE%" (
    echo ERROR: Test executable not found at: %TEST_EXE%
    echo Please build the project first using build.bat
    exit /b 1
)

REM Check if test model exists
set TEST_DIR=%WORKSPACE_ROOT%/morphizen-mlir/test
set TEST_MODEL=%TEST_DIR%/conv_model.onnx
if not exist "%TEST_MODEL%" (
    echo Test model not found, generating it...
    cd /d "%TEST_DIR%"
    python gen_conv_model.py
    if errorlevel 1 (
        echo ERROR: Failed to generate test model
        exit /b 1
    )
    echo.
)

REM Run the test
echo Running test from directory: %TEST_DIR%
cd /d "%TEST_DIR%"
echo.
echo ============================================================
echo Executing: %TEST_EXE%
echo ============================================================
echo.

"%TEST_EXE%"
set TEST_EXITCODE=%errorlevel%

echo.
echo ============================================================
if %TEST_EXITCODE% equ 0 (
    echo Test completed successfully!
) else (
    echo Test failed with exit code %TEST_EXITCODE%
)
echo ============================================================

exit /b %TEST_EXITCODE%
