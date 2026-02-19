@echo off
REM === End-to-End Softmax Pipeline (MIOpen) ===
REM Compiles test_softmax.mlir (two chained hip.miopen.softmax ops in DPS)

if not defined VSINSTALLDIR (
  call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)
if not defined CONDA_PREFIX (
  call C:\Users\chiz\anaconda3\condabin\conda.bat activate llvm
)

set LLVM_BIN=C:\Users\chiz\work\gpu\llvm-project\build\Debug\bin
set THEROCK_DIST=C:\Users\chiz\work\gpu\TheRock\build\dist\rocm
set HIP_OPT_BIN=C:\Users\chiz\work\gpu\onnx-hipdnn-ep\tools\hip-opt\build\Debug
set SRC_DIR=C:\Users\chiz\work\gpu\onnx-hipdnn-ep\tools\hip-opt

cd /d "%SRC_DIR%\build"

echo.
echo ============================================================
echo  MLIR Softmax Pipeline (MIOpen, DPS)
echo ============================================================

echo.
echo [1/8] MLIR -^> LLVM Dialect (hip-opt)
"%HIP_OPT_BIN%\hip-opt.exe" ..\examples\test_softmax.mlir ^
  --convert-hip-to-llvm ^
  --finalize-memref-to-llvm ^
  --convert-arith-to-llvm ^
  --convert-func-to-llvm ^
  --reconcile-unrealized-casts ^
  -o softmax_lowered.mlir
if errorlevel 1 (echo FAILED at step 1 && exit /b 1)
echo      OK: softmax_lowered.mlir

echo.
echo [2/8] LLVM Dialect -^> LLVM IR (mlir-translate)
"%LLVM_BIN%\mlir-translate.exe" softmax_lowered.mlir --mlir-to-llvmir -o softmax.ll
if errorlevel 1 (echo FAILED at step 2 && exit /b 1)
echo      OK: softmax.ll

echo.
echo [3/8] LLVM IR -^> Object File (llc)
"%LLVM_BIN%\llc.exe" softmax.ll -filetype=obj -o softmax.obj
if errorlevel 1 (echo FAILED at step 3 && exit /b 1)
echo      OK: softmax.obj

echo.
echo [4/8] Generate import libraries (if needed)
if not exist amdhip64.lib (
  dumpbin /EXPORTS "%THEROCK_DIST%\bin\amdhip64_7.dll" | findstr /R "^  *[0-9]" > _hip_exports.txt
  echo LIBRARY amdhip64_7.dll > amdhip64.def
  echo EXPORTS >> amdhip64.def
  for /f "tokens=4" %%a in (_hip_exports.txt) do echo   %%a >> amdhip64.def
  lib /def:amdhip64.def /out:amdhip64.lib /machine:x64 >nul 2>&1
  del _hip_exports.txt
)
if not exist MIOpen.lib (
  dumpbin /EXPORTS "%THEROCK_DIST%\bin\MIOpen.dll" | findstr /R "^  *[0-9]" > _miopen_exports.txt
  echo LIBRARY MIOpen.dll > MIOpen.def
  echo EXPORTS >> MIOpen.def
  for /f "tokens=4" %%a in (_miopen_exports.txt) do echo   %%a >> MIOpen.def
  lib /def:MIOpen.def /out:MIOpen.lib /machine:x64 >nul 2>&1
  del _miopen_exports.txt
)
echo      OK: amdhip64.lib, MIOpen.lib

echo.
echo [5/8] Compile hip runtime
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\ops_runtime\hip_runtime.cpp /Fo:hip_runtime.obj
if errorlevel 1 (echo FAILED at step 5 && exit /b 1)
echo      OK: hip_runtime.obj

echo.
echo [6/8] Compile miopen softmax runtime
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\ops_runtime\miopen_softmax.cpp /Fo:miopen_softmax.obj
if errorlevel 1 (echo FAILED at step 6 && exit /b 1)
echo      OK: miopen_softmax.obj

echo.
echo [7/8] Compile main driver
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\examples\main_softmax.cpp /Fo:main_softmax.obj
if errorlevel 1 (echo FAILED at step 7 && exit /b 1)
echo      OK: main_softmax.obj

echo.
echo [8/8] Link executable
link.exe softmax.obj hip_runtime.obj miopen_softmax.obj main_softmax.obj /LIBPATH:. /LIBPATH:"%THEROCK_DIST%\lib" amdhip64.lib MIOpen.lib /out:softmax_test.exe
if errorlevel 1 (echo FAILED at step 8 && exit /b 1)
echo      OK: softmax_test.exe

echo.
echo ============================================================
echo  Build completed! Run with: build\softmax_test.exe
echo ============================================================
echo.
echo  NOTE: Ensure these DLLs are in PATH at runtime:
echo    - amdhip64_7.dll   (from %THEROCK_DIST%\bin)
echo    - MIOpen.dll       (from %THEROCK_DIST%\bin)
echo ============================================================
