::
:: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
:: Licensed under the MIT License.
::
@echo off
REM === End-to-End Matmul Pipeline ===
REM Compiles gemm.hip.mlir to DLL and links with main driver

call "%~dp0env.bat"

cd /d "%SRC_DIR%\build"

echo.
echo ============================================================
echo  MLIR Matmul Pipeline (hip-compiler)
echo ============================================================

echo.
echo [1/2] Compiling MLIR to DLL
"%HIP_OPT_BIN%\hip-compiler.exe" ..\examples\gemm.hip.mlir -o gemm.dll
if errorlevel 1 (echo FAILED at step 1 && exit /b 1)
echo      OK: gemm.dll

echo.
echo [2/2] Compile and link driver
cl.exe /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\examples\main_gemm.cpp gemm.lib amdhip64.lib /Fe:matmul_test.exe
if errorlevel 1 (echo FAILED at step 2 && exit /b 1)
echo      OK: matmul_test.exe

echo.
echo ============================================================
echo  Build completed! Run with: build\matmul_test.exe
echo ============================================================
echo.
echo  NOTE: Ensure these DLLs are in PATH at runtime:
echo    - amdhip64_7.dll     (from %THEROCK_DIST%\bin)
echo    - libhipblaslt.dll   (from %THEROCK_DIST%\bin)
echo    - gemm.dll           (generated)
echo ============================================================
