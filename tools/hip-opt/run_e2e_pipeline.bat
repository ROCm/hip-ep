@echo off
REM === End-to-End Transformer Pipeline ===
REM Compiles test_e2e.mlir through the full pipeline to produce test_e2e.exe

REM --- Activate VS tools only if not already active ---
if not defined VSINSTALLDIR (
  call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)

REM --- Activate conda env only if not already active ---
if not defined CONDA_PREFIX (
  call C:\Users\chiz\anaconda3\condabin\conda.bat activate llvm
)

REM --- Tool paths ---
set LLVM_BIN=C:\Users\chiz\work\gpu\llvm-project\build\Debug\bin
set THEROCK_DIST=C:\Users\chiz\work\gpu\TheRock\build\dist\rocm
set HIP_OPT_BIN=C:\Users\chiz\work\gpu\onnx-hipdnn-ep\tools\hip-opt\build\Debug
set SRC_DIR=C:\Users\chiz\work\gpu\onnx-hipdnn-ep\tools\hip-opt

cd /d "%SRC_DIR%\build"

echo.
echo ============================================================
echo  HIP Dialect E2E Transformer Pipeline
echo ============================================================

echo.
echo [1/7] MLIR -^> LLVM Dialect (hip-opt)
"%HIP_OPT_BIN%\hip-opt.exe" ..\test_e2e.mlir ^
  --convert-hip-to-llvm ^
  --convert-scf-to-cf ^
  --convert-func-to-llvm ^
  --convert-cf-to-llvm ^
  --reconcile-unrealized-casts ^
  -o e2e_lowered.mlir
if errorlevel 1 (echo FAILED at step 1 && exit /b 1)
echo      OK: e2e_lowered.mlir

echo.
echo [2/7] LLVM Dialect -^> LLVM IR (mlir-translate)
"%LLVM_BIN%\mlir-translate.exe" e2e_lowered.mlir --mlir-to-llvmir -o e2e.ll
if errorlevel 1 (echo FAILED at step 2 && exit /b 1)
echo      OK: e2e.ll

echo.
echo [3/7] LLVM IR -^> Object File (llc)
"%LLVM_BIN%\llc.exe" e2e.ll -filetype=obj -o e2e.obj
if errorlevel 1 (echo FAILED at step 3 && exit /b 1)
echo      OK: e2e.obj

echo.
echo [4/7] Generate import libraries (if needed)
if not exist amdhip64.lib (
  dumpbin /EXPORTS "%THEROCK_DIST%\bin\amdhip64_7.dll" | findstr /R "^  *[0-9]" > _hip_exports.txt
  echo LIBRARY amdhip64_7.dll > amdhip64.def
  echo EXPORTS >> amdhip64.def
  for /f "tokens=4" %%a in (_hip_exports.txt) do echo   %%a >> amdhip64.def
  lib /def:amdhip64.def /out:amdhip64.lib /machine:x64 >nul 2>&1
  del _hip_exports.txt
)
echo      OK: amdhip64.lib

echo.
echo [5/7] Compile runtime
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\ops_runtime\all_runtime.cpp /Fo:runtime.obj
if errorlevel 1 (echo FAILED at step 5 && exit /b 1)
echo      OK: runtime.obj

echo.
echo [6/7] Compile main driver
cl.exe /c /EHsc /std:c++17 ..\main_e2e.cpp /Fo:main_e2e.obj
if errorlevel 1 (echo FAILED at step 6 && exit /b 1)
echo      OK: main_e2e.obj

echo.
echo [7/7] Link executable
link.exe e2e.obj runtime.obj main_e2e.obj /LIBPATH:. amdhip64.lib /out:test_e2e.exe
if errorlevel 1 (echo FAILED at step 7 && exit /b 1)
echo      OK: test_e2e.exe

echo.
echo ============================================================
echo  Build completed!
echo ============================================================
echo.
echo  To run:
echo    cd build
echo    set PATH=%THEROCK_DIST%\bin;%%PATH%%
echo    test_e2e.exe
echo.
echo  To enable hipBLASLt (real matmul):
echo    cl.exe ... /DHAS_HIPBLASLT ...
echo    link.exe ... hipblaslt.lib ...
echo.
echo  To enable MIOpen (real norm/activation):
echo    Build MIOpen from TheRock: cmake --build build --target miopen
echo    cl.exe ... /DHAS_MIOPEN /I%%MIOPEN_INCLUDE%% ...
echo    link.exe ... MIOpen.lib ...
echo ============================================================
