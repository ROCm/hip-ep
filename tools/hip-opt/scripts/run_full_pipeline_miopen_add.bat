@echo off
REM === End-to-End Add Pipeline (MIOpen) ===
REM Compiles test_add.mlir (two chained hip.miopen.add ops in DPS)
REM through the full pipeline to produce add_test.exe.

call "%~dp0env.bat"

cd /d "%SRC_DIR%\build"

echo.
echo ============================================================
echo  MLIR Add Pipeline (MIOpen, DPS)
echo ============================================================

echo.
echo [1/8] MLIR -^> LLVM Dialect (hip-opt)
"%HIP_OPT_BIN%\hip-opt.exe" ..\examples\test_add.mlir ^
  --convert-hip-to-llvm ^
  --finalize-memref-to-llvm ^
  --convert-arith-to-llvm ^
  --convert-func-to-llvm ^
  --reconcile-unrealized-casts ^
  -o add_lowered.mlir
if errorlevel 1 (echo FAILED at step 1 && exit /b 1)
echo      OK: add_lowered.mlir

echo.
echo [2/8] LLVM Dialect -^> LLVM IR (mlir-translate)
"%LLVM_BIN%\mlir-translate.exe" add_lowered.mlir --mlir-to-llvmir -o add.ll
if errorlevel 1 (echo FAILED at step 2 && exit /b 1)
echo      OK: add.ll

echo.
echo [3/8] LLVM IR -^> Object File (llc)
"%LLVM_BIN%\llc.exe" add.ll -filetype=obj -o add.obj
if errorlevel 1 (echo FAILED at step 3 && exit /b 1)
echo      OK: add.obj

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
echo [5/8] Compile hip runtime (handle lifecycle + memory)
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\ops_runtime\hip_runtime.cpp /Fo:hip_runtime.obj
if errorlevel 1 (echo FAILED at step 5 && exit /b 1)
echo      OK: hip_runtime.obj

echo.
echo [6/8] Compile miopen add runtime
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\ops_runtime\miopen_add.cpp /Fo:miopen_add.obj
if errorlevel 1 (echo FAILED at step 6 && exit /b 1)
echo      OK: miopen_add.obj

echo.
echo [7/8] Compile main driver
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\examples\main_add.cpp /Fo:main_add.obj
if errorlevel 1 (echo FAILED at step 7 && exit /b 1)
echo      OK: main_add.obj

echo.
echo [8/8] Link executable
link.exe add.obj hip_runtime.obj miopen_add.obj main_add.obj /LIBPATH:. /LIBPATH:"%THEROCK_DIST%\lib" amdhip64.lib MIOpen.lib /out:add_test.exe
if errorlevel 1 (echo FAILED at step 8 && exit /b 1)
echo      OK: add_test.exe

echo.
echo ============================================================
echo  Build completed! Run with: build\add_test.exe
echo ============================================================
echo.
echo  NOTE: Ensure these DLLs are in PATH at runtime:
echo    - amdhip64_7.dll   (from %THEROCK_DIST%\bin)
echo    - MIOpen.dll       (from %THEROCK_DIST%\bin)
echo ============================================================
