::
:: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
:: Licensed under the MIT License.
::
@echo off
REM === End-to-End Mul Pipeline ===
REM Compiles mul.hip.mlir to DLL and links with main driver

call "%~dp0env.bat"

cd /d "%SRC_DIR%\build"

echo.
echo ============================================================
echo  MLIR Mul Pipeline (hip-compiler)
echo ============================================================

echo.
echo [1/2] Compiling MLIR to DLL
"%HIP_OPT_BIN%\hip-compiler.exe" ..\examples\mul.hip.mlir -o mul.dll
if errorlevel 1 (echo FAILED at step 1 && exit /b 1)
echo      OK: mul.dll

echo.
echo [2/2] Compile and link driver
cl.exe /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\examples\main_mul.cpp mul.lib amdhip64.lib /Fe:mul_test.exe
if errorlevel 1 (echo FAILED at step 2 && exit /b 1)
echo      OK: mul_test.exe

echo.
echo ============================================================
echo  Build completed! Run with: build\mul_test.exe
echo ============================================================
echo.
echo  NOTE: Ensure these DLLs are in PATH at runtime:
echo    - amdhip64_7.dll   (from %THEROCK_DIST%\bin)
echo    - MIOpen.dll       (from %THEROCK_DIST%\bin)
echo    - mul.dll          (generated)
echo ============================================================
