::
:: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
:: Licensed under the MIT License.
::
@echo off
REM ============================================================
REM  Shared environment setup for all pipeline scripts.
REM  >>> EDIT THESE PATHS TO MATCH YOUR LOCAL MACHINE <<<
REM ============================================================

REM --- Visual Studio (vcvarsall.bat path) ---
if not defined VSINSTALLDIR (
  call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)

REM --- Conda environment name ---
if not defined CONDA_PREFIX (
  call C:\Users\chiz\anaconda3\condabin\conda.bat activate llvm
)

REM --- LLVM/MLIR build output (contains mlir-translate, llc) ---
set LLVM_BIN=C:\Users\chiz\work\gpu\llvm-project\build\Debug\bin

REM --- TheRock ROCm dist (contains amdhip64, hipblaslt, MIOpen) ---
set THEROCK_DIST=C:\Users\chiz\work\gpu\TheRock\build\dist\rocm

REM --- MLIR tools build output ---
set HIP_OPT_BIN=C:\Users\chiz\work\gpu\onnx-hipdnn-ep\build\bin\Debug

REM --- project source root ---
set SRC_DIR=C:\Users\chiz\work\gpu\onnx-hipdnn-ep

REM --- ONNX Runtime (onnxruntime_c_api.h, onnxruntime.lib) ---
if not defined ORT_HOME set ORT_HOME=C:\Users\chiz\work\onnxruntime

REM --- Generate import libraries from TheRock DLLs (one-time, cached in build dir) ---
if not exist "%SRC_DIR%\build" mkdir "%SRC_DIR%\build"
pushd "%SRC_DIR%\build"
if not exist hipblaslt.lib (
  echo Generating hipblaslt.lib from libhipblaslt.dll...
  dumpbin /EXPORTS "%THEROCK_DIST%\bin\libhipblaslt.dll" | findstr /R "^  *[0-9]" > _exports_raw.txt
  echo LIBRARY libhipblaslt.dll > hipblaslt.def
  echo EXPORTS >> hipblaslt.def
  for /f "tokens=4" %%a in (_exports_raw.txt) do echo   %%a >> hipblaslt.def
  lib /def:hipblaslt.def /out:hipblaslt.lib /machine:x64 >nul 2>&1
  del _exports_raw.txt hipblaslt.def hipblaslt.exp 2>nul
)
if not exist amdhip64.lib (
  echo Generating amdhip64.lib from amdhip64_7.dll...
  dumpbin /EXPORTS "%THEROCK_DIST%\bin\amdhip64_7.dll" | findstr /R "^  *[0-9]" > _exports_raw.txt
  echo LIBRARY amdhip64_7.dll > amdhip64.def
  echo EXPORTS >> amdhip64.def
  for /f "tokens=4" %%a in (_exports_raw.txt) do echo   %%a >> amdhip64.def
  lib /def:amdhip64.def /out:amdhip64.lib /machine:x64 >nul 2>&1
  del _exports_raw.txt amdhip64.def amdhip64.exp 2>nul
)
if not exist MIOpen.lib (
  echo Generating MIOpen.lib from MIOpen.dll...
  dumpbin /EXPORTS "%THEROCK_DIST%\bin\MIOpen.dll" | findstr /R "^  *[0-9]" > _exports_raw.txt
  echo LIBRARY MIOpen.dll > MIOpen.def
  echo EXPORTS >> MIOpen.def
  for /f "tokens=4" %%a in (_exports_raw.txt) do echo   %%a >> MIOpen.def
  lib /def:MIOpen.def /out:MIOpen.lib /machine:x64 >nul 2>&1
  del _exports_raw.txt MIOpen.def MIOpen.exp 2>nul
)
popd
