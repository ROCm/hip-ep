::
:: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
:: Licensed under the MIT License.
::
@echo off
REM === End-to-End Attention Pipeline (externalized constants) ===
REM
REM Demonstrates the full flow from ONNX MLIR to GPU execution with
REM externalized constants:
REM
REM   1. hip-mlir-opt: ONNX->HIP lowering + bufferization + constant
REM      externalization (produces .mlir + model.constants.bin)
REM   2. hip-compiler: MLIR -> DLL
REM   3. cl.exe: build driver (links DLL + ORT + HIP)
REM   4. run test (compares GPU output against ORT CPU reference)
REM
REM Expects attention.onnx.mlir and attention.onnx in examples/.

call "%~dp0env.bat"

cd /d "%SRC_DIR%\build"

echo.
echo ============================================================
echo  Attention Pipeline (externalized constants)
echo ============================================================

echo.
echo [1/4] Lowering attention.onnx.mlir with externalized constants
"%HIP_OPT_BIN%\hip-mlir-opt.exe" ^
  --convert-onnx-to-hip="externalize-min-num-elements=1 externalize-output-dir=." ^
  --one-shot-bufferize="bufferize-function-boundaries" ^
  --buffer-results-to-out-params="hoist-static-allocs hoist-dynamic-allocs add-result-attr modify-public-functions" ^
  --buffer-deallocation-pipeline ^
  --cse --canonicalize ^
  --hip-optimize-memrefs ^
  --hip-pool-allocs ^
  --convert-bufferization-to-memref ^
  --cse --canonicalize ^
  --hip-lower-allocs ^
  --hip-resolve-extern-constants ^
  ..\examples\attention.onnx.mlir ^
  -o attention_ext.mlir
if errorlevel 1 (echo FAILED at step 1 && exit /b 1)
echo      OK: attention_ext.mlir + model.constants.bin

echo.
echo [2/4] Compiling attention_ext.mlir to DLL
"%HIP_OPT_BIN%\hip-compiler.exe" attention_ext.mlir -o attention_ext.dll
if errorlevel 1 (echo FAILED at step 2 && exit /b 1)
echo      OK: attention_ext.dll

echo.
echo [3/4] Compile and link driver (with ORT + HIP)
cl.exe /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ ^
  /I"%THEROCK_DIST%\include" ^
  /I"%ORT_HOME%\include\onnxruntime" ^
  /Fe:attention_ext_test.exe ^
  ..\examples\main_attention_ext.cpp ^
  attention_ext.lib ^
  amdhip64.lib ^
  /link /LIBPATH:"%ORT_HOME%\lib" onnxruntime.lib
if errorlevel 1 (echo FAILED at step 3 && exit /b 1)
echo      OK: attention_ext_test.exe

echo.
echo [4/4] Running test
attention_ext_test.exe ..\examples\attention.onnx model.constants.bin
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
echo    - attention_ext.dll  (generated)
echo ============================================================
