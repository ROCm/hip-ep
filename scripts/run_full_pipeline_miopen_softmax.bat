::
:: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
:: Licensed under the MIT License.
::
@echo off
REM === End-to-End Softmax Pipeline ===
REM Compiles softmax.hip.mlir to DLL and links with main driver

call "%~dp0env.bat"

cd /d "%SRC_DIR%\build"

echo.
echo ============================================================
echo  MLIR Softmax Pipeline (hip-compiler)
echo ============================================================

echo.
echo [1/2] Compiling MLIR to DLL
"%HIP_OPT_BIN%\hip-compiler.exe" ..\examples\softmax.hip.mlir -o softmax.dll
if errorlevel 1 (echo FAILED at step 1 && exit /b 1)
echo      OK: softmax.dll

echo.
echo [2/2] Compile and link driver
cl.exe /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\examples\main_softmax.cpp softmax.lib amdhip64.lib /Fe:softmax_test.exe
if errorlevel 1 (echo FAILED at step 2 && exit /b 1)
echo      OK: softmax_test.exe

echo.
echo ============================================================
echo  Build completed! Run with: build\softmax_test.exe
echo ============================================================
echo.
echo  NOTE: Ensure these DLLs are in PATH at runtime:
echo    - amdhip64_7.dll   (from %THEROCK_DIST%\bin)
echo    - MIOpen.dll       (from %THEROCK_DIST%\bin)
echo    - softmax.dll      (generated)
echo ============================================================
