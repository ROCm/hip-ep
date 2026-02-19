@echo off
REM === End-to-End Mul Pipeline (MIOpen) ===
REM Compiles test_mul.mlir (two chained hip.miopen.mul ops in DPS)

call "%~dp0env.bat"

cd /d "%SRC_DIR%\build"

echo.
echo ============================================================
echo  MLIR Mul Pipeline (MIOpen, DPS)
echo ============================================================

echo.
echo [1/8] MLIR -^> LLVM Dialect (hip-opt)
"%HIP_OPT_BIN%\hip-opt.exe" ..\examples\test_mul.mlir ^
  --convert-hip-to-llvm ^
  --finalize-memref-to-llvm ^
  --convert-arith-to-llvm ^
  --convert-func-to-llvm ^
  --reconcile-unrealized-casts ^
  -o mul_lowered.mlir
if errorlevel 1 (echo FAILED at step 1 && exit /b 1)
echo      OK: mul_lowered.mlir

echo.
echo [2/8] LLVM Dialect -^> LLVM IR (mlir-translate)
"%LLVM_BIN%\mlir-translate.exe" mul_lowered.mlir --mlir-to-llvmir -o mul.ll
if errorlevel 1 (echo FAILED at step 2 && exit /b 1)
echo      OK: mul.ll

echo.
echo [3/8] LLVM IR -^> Object File (llc)
"%LLVM_BIN%\llc.exe" mul.ll -filetype=obj -o mul.obj
if errorlevel 1 (echo FAILED at step 3 && exit /b 1)
echo      OK: mul.obj

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
echo [6/8] Compile miopen mul runtime
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\ops_runtime\miopen_mul.cpp /Fo:miopen_mul.obj
if errorlevel 1 (echo FAILED at step 6 && exit /b 1)
echo      OK: miopen_mul.obj

echo.
echo [7/8] Compile main driver
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\examples\main_mul.cpp /Fo:main_mul.obj
if errorlevel 1 (echo FAILED at step 7 && exit /b 1)
echo      OK: main_mul.obj

echo.
echo [8/8] Link executable
link.exe mul.obj hip_runtime.obj miopen_mul.obj main_mul.obj /LIBPATH:. /LIBPATH:"%THEROCK_DIST%\lib" amdhip64.lib MIOpen.lib /out:mul_test.exe
if errorlevel 1 (echo FAILED at step 8 && exit /b 1)
echo      OK: mul_test.exe

echo.
echo ============================================================
echo  Build completed! Run with: build\mul_test.exe
echo ============================================================
echo.
echo  NOTE: Ensure these DLLs are in PATH at runtime:
echo    - amdhip64_7.dll   (from %THEROCK_DIST%\bin)
echo    - MIOpen.dll       (from %THEROCK_DIST%\bin)
echo ============================================================
