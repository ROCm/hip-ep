@echo off
REM === End-to-End Attention Pipeline ===
REM Compiles test_attention.mlir (single-head attention from composed ops)
REM Uses: hipBLASLt (matmul), MIOpen (mul, softmax), custom (transpose)

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
echo  MLIR Attention Pipeline (hipBLASLt + MIOpen + custom)
echo ============================================================

echo.
echo [1/10] MLIR -^> LLVM Dialect (hip-opt)
"%HIP_OPT_BIN%\hip-opt.exe" ..\examples\test_attention.mlir ^
  --convert-hip-to-llvm ^
  --finalize-memref-to-llvm ^
  --convert-arith-to-llvm ^
  --convert-func-to-llvm ^
  --reconcile-unrealized-casts ^
  -o attention_lowered.mlir
if errorlevel 1 (echo FAILED at step 1 && exit /b 1)
echo      OK: attention_lowered.mlir

echo.
echo [2/10] LLVM Dialect -^> LLVM IR (mlir-translate)
"%LLVM_BIN%\mlir-translate.exe" attention_lowered.mlir --mlir-to-llvmir -o attention.ll
if errorlevel 1 (echo FAILED at step 2 && exit /b 1)
echo      OK: attention.ll

echo.
echo [3/10] LLVM IR -^> Object File (llc)
"%LLVM_BIN%\llc.exe" attention.ll -filetype=obj -o attention.obj
if errorlevel 1 (echo FAILED at step 3 && exit /b 1)
echo      OK: attention.obj

echo.
echo [4/10] Generate import libraries (if needed)
if not exist hipblaslt.lib (
  dumpbin /EXPORTS "%THEROCK_DIST%\bin\libhipblaslt.dll" | findstr /R "^  *[0-9]" > _exports_raw.txt
  echo LIBRARY libhipblaslt.dll > hipblaslt.def
  echo EXPORTS >> hipblaslt.def
  for /f "tokens=4" %%a in (_exports_raw.txt) do echo   %%a >> hipblaslt.def
  lib /def:hipblaslt.def /out:hipblaslt.lib /machine:x64 >nul 2>&1
  del _exports_raw.txt
)
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
echo      OK: hipblaslt.lib, amdhip64.lib, MIOpen.lib

echo.
echo [5/10] Compile hip runtime
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\ops_runtime\hip_runtime.cpp /Fo:hip_runtime.obj
if errorlevel 1 (echo FAILED at step 5 && exit /b 1)
echo      OK: hip_runtime.obj

echo.
echo [6/10] Compile hipblaslt matmul runtime
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\ops_runtime\hipblaslt_matmul.cpp /Fo:hipblaslt_matmul.obj
if errorlevel 1 (echo FAILED at step 6 && exit /b 1)
echo      OK: hipblaslt_matmul.obj

echo.
echo [7/10] Compile miopen ops (mul + softmax)
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\ops_runtime\miopen_mul.cpp /Fo:miopen_mul.obj
if errorlevel 1 (echo FAILED at step 7a && exit /b 1)
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\ops_runtime\miopen_softmax.cpp /Fo:miopen_softmax.obj
if errorlevel 1 (echo FAILED at step 7b && exit /b 1)
echo      OK: miopen_mul.obj, miopen_softmax.obj

echo.
echo [8/10] Compile transpose runtime
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\ops_runtime\transpose.cpp /Fo:transpose.obj
if errorlevel 1 (echo FAILED at step 8 && exit /b 1)
echo      OK: transpose.obj

echo.
echo [9/10] Compile main driver
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\examples\main_attention.cpp /Fo:main_attention.obj
if errorlevel 1 (echo FAILED at step 9 && exit /b 1)
echo      OK: main_attention.obj

echo.
echo [10/10] Link executable
link.exe attention.obj hip_runtime.obj hipblaslt_matmul.obj miopen_mul.obj miopen_softmax.obj transpose.obj main_attention.obj ^
  /LIBPATH:. /LIBPATH:"%THEROCK_DIST%\lib" amdhip64.lib hipblaslt.lib MIOpen.lib /out:attention_test.exe
if errorlevel 1 (echo FAILED at step 10 && exit /b 1)
echo      OK: attention_test.exe

echo.
echo ============================================================
echo  Build completed! Run with: build\attention_test.exe
echo ============================================================
echo.
echo  NOTE: Ensure these DLLs are in PATH at runtime:
echo    - amdhip64_7.dll     (from %THEROCK_DIST%\bin)
echo    - libhipblaslt.dll   (from %THEROCK_DIST%\bin)
echo    - MIOpen.dll         (from %THEROCK_DIST%\bin)
echo ============================================================
