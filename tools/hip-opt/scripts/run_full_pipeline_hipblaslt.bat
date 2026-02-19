@echo off
REM === End-to-End Matmul Pipeline (hipBLASLt) ===
REM Compiles test_gemm.mlir (two chained hip.hipblaslt.matmul ops in DPS)
REM through the full pipeline to produce matmul_test.exe.

call "%~dp0env.bat"

cd /d "%SRC_DIR%\build"

echo.
echo ============================================================
echo  MLIR Matmul Pipeline (hipBLASLt, DPS)
echo ============================================================

echo.
echo [1/7] MLIR -^> LLVM Dialect (hip-opt)
"%HIP_OPT_BIN%\hip-opt.exe" ..\examples\test_gemm.mlir ^
  --convert-hip-to-llvm ^
  --finalize-memref-to-llvm ^
  --convert-arith-to-llvm ^
  --convert-func-to-llvm ^
  --reconcile-unrealized-casts ^
  -o gemm_lowered.mlir
if errorlevel 1 (echo FAILED at step 1 && exit /b 1)
echo      OK: gemm_lowered.mlir

echo.
echo [2/7] LLVM Dialect -^> LLVM IR (mlir-translate)
"%LLVM_BIN%\mlir-translate.exe" gemm_lowered.mlir --mlir-to-llvmir -o gemm.ll
if errorlevel 1 (echo FAILED at step 2 && exit /b 1)
echo      OK: gemm.ll

echo.
echo [3/7] LLVM IR -^> Object File (llc)
"%LLVM_BIN%\llc.exe" gemm.ll -filetype=obj -o gemm.obj
if errorlevel 1 (echo FAILED at step 3 && exit /b 1)
echo      OK: gemm.obj

echo.
echo [4/7] Generate import libraries (if needed)
if not exist hipblaslt.lib (
  dumpbin /EXPORTS "%THEROCK_DIST%\bin\libhipblaslt.dll" | findstr /R "^  *[0-9]" > _exports_raw.txt
  echo LIBRARY libhipblaslt.dll > hipblaslt.def
  echo EXPORTS >> hipblaslt.def
  for /f "tokens=4" %%a in (_exports_raw.txt) do echo   %%a >> hipblaslt.def
  lib /def:hipblaslt.def /out:hipblaslt.lib /machine:x64 >nul 2>&1
  del _exports_raw.txt
  if errorlevel 1 (echo FAILED generating hipblaslt.lib && exit /b 1)
)
if not exist amdhip64.lib (
  dumpbin /EXPORTS "%THEROCK_DIST%\bin\amdhip64_7.dll" | findstr /R "^  *[0-9]" > _hip_exports.txt
  echo LIBRARY amdhip64_7.dll > amdhip64.def
  echo EXPORTS >> amdhip64.def
  for /f "tokens=4" %%a in (_hip_exports.txt) do echo   %%a >> amdhip64.def
  lib /def:amdhip64.def /out:amdhip64.lib /machine:x64 >nul 2>&1
  del _hip_exports.txt
)
echo      OK: hipblaslt.lib, amdhip64.lib

echo.
echo [5/8] Compile hip runtime (handle lifecycle + memory)
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\ops_runtime\hip_runtime.cpp /Fo:hip_runtime.obj
if errorlevel 1 (echo FAILED at step 5 && exit /b 1)
echo      OK: hip_runtime.obj

echo.
echo [6/8] Compile hipblaslt matmul runtime
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\ops_runtime\hipblaslt_matmul.cpp /Fo:hipblaslt_matmul.obj
if errorlevel 1 (echo FAILED at step 6 && exit /b 1)
echo      OK: hipblaslt_matmul.obj

echo.
echo [7/8] Compile main driver
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\examples\main_gemm.cpp /Fo:main.obj
if errorlevel 1 (echo FAILED at step 7 && exit /b 1)
echo      OK: main.obj

echo.
echo [8/8] Link executable
link.exe gemm.obj hip_runtime.obj hipblaslt_matmul.obj main.obj /LIBPATH:. /LIBPATH:"%THEROCK_DIST%\lib" amdhip64.lib hipblaslt.lib /out:matmul_test.exe
if errorlevel 1 (echo FAILED at step 8 && exit /b 1)
echo      OK: matmul_test.exe

echo.
echo ============================================================
echo  Build completed! Run with: build\matmul_test.exe
echo ============================================================
echo.
echo  NOTE: Ensure these DLLs are in PATH at runtime:
echo    - amdhip64_7.dll     (from %THEROCK_DIST%\bin)
echo    - libhipblaslt.dll   (from %THEROCK_DIST%\bin)
echo ============================================================
