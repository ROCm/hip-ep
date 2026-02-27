::
:: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
:: Licensed under the MIT License.
::
@echo off
REM === End-to-End Attention Pipeline (pre-bufferized .hip.mlir format) ===
REM
REM Compiles attention.hip.mlir (already in memref/pool form) to DLL,
REM builds the driver that compares GPU output against ORT CPU reference,
REM and runs the test.
REM
REM Expects attention.onnx in examples/ alongside the .hip.mlir.

call "%~dp0env.bat"

cd /d "%SRC_DIR%\build"

echo.
echo ============================================================
echo  Attention Pipeline (pre-bufferized .hip.mlir)
echo ============================================================

echo.
echo [1/3] Compiling attention.hip.mlir to DLL
"%HIP_OPT_BIN%\hip-compiler.exe" ..\examples\attention.hip.mlir -o attention.dll
if errorlevel 1 (echo FAILED at step 1 && exit /b 1)
echo      OK: attention.dll

echo.
echo [2/3] Compile and link driver (with ORT + HIP)
cl.exe /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ ^
  /I"%THEROCK_DIST%\include" ^
  /I"%ORT_HOME%\include" ^
  /Fe:attention_test.exe ^
  ..\examples\main_attention.cpp ^
  attention.lib ^
  amdhip64.lib ^
  /link /LIBPATH:"%ORT_HOME%\lib" onnxruntime.lib
if errorlevel 1 (echo FAILED at step 2 && exit /b 1)
echo      OK: attention_test.exe

echo.
echo [3/3] Running test
attention_test.exe ..\examples\attention.onnx
if errorlevel 1 (echo TEST FAILED && exit /b 1)

echo.
echo ============================================================
echo  All steps completed successfully.
echo ============================================================
echo.
echo  NOTE: Ensure these DLLs are in PATH at runtime:
echo    - amdhip64_7.dll     (from %THEROCK_DIST%\bin)
echo    - libhipblaslt.dll   (from %THEROCK_DIST%\bin)
echo    - MIOpen.dll         (from %THEROCK_DIST%\bin)
echo    - onnxruntime.dll    (from %ORT_HOME%\lib)
echo    - attention.dll      (generated)
echo ============================================================
