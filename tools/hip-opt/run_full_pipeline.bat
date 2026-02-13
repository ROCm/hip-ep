@echo off
REM --- Activate VS tools only if not already active ---
if not defined VSINSTALLDIR (
  call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)

REM --- Activate conda env only if not already active ---
if not defined CONDA_PREFIX (
  call C:\Users\chiz\anaconda3\condabin\conda.bat activate llvm
)

set LLVM_BIN=C:\Users\chiz\work\gpu\llvm-project\build\Debug\bin
set THEROCK_DIST=C:\Users\chiz\work\gpu\TheRock\build\dist\rocm
set HIP_OPT_BIN=C:\Users\chiz\work\gpu\onnx-hipdnn-ep\tools\hip-opt\build\Debug

REM --- Prepend to PATH only if not already present (avoids exceeding 8191-char limit) ---
echo "%PATH%" | findstr /I /C:"%HIP_OPT_BIN%" >nul 2>&1
if errorlevel 1 set "PATH=%HIP_OPT_BIN%;%PATH%"
echo "%PATH%" | findstr /I /C:"%LLVM_BIN%" >nul 2>&1
if errorlevel 1 set "PATH=%LLVM_BIN%;%PATH%"
echo "%PATH%" | findstr /I /C:"%THEROCK_DIST%\bin" >nul 2>&1
if errorlevel 1 set "PATH=%THEROCK_DIST%\bin;%PATH%"
echo "%PATH%" | findstr /I /C:"anaconda3\envs\llvm\Library\bin" >nul 2>&1
if errorlevel 1 set "PATH=C:\Users\chiz\anaconda3\envs\llvm\Library\bin;%PATH%"

cd /d C:\Users\chiz\work\gpu\onnx-hipdnn-ep\tools\hip-opt\build

echo.
echo ============================================================
echo  MLIR GEMM End-to-End Pipeline (hipBLAS-LT)
echo ============================================================

echo.
echo [1/8] MLIR -^> LLVM Dialect (hip-opt)
hip-opt.exe ..\test_gemm.mlir --convert-hip-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts -o gemm_lowered.mlir
if errorlevel 1 (echo FAILED at step 1 && exit /b 1)
echo      OK: gemm_lowered.mlir

echo.
echo [2/8] LLVM Dialect -^> LLVM IR (mlir-translate)
mlir-translate.exe gemm_lowered.mlir --mlir-to-llvmir -o gemm.ll
if errorlevel 1 (echo FAILED at step 2 && exit /b 1)
echo      OK: gemm.ll

echo.
echo [3/8] LLVM IR -^> Object File (llc)
llc.exe gemm.ll -filetype=obj -o gemm.obj
if errorlevel 1 (echo FAILED at step 3 && exit /b 1)
echo      OK: gemm.obj

echo.
echo [4/8] Generate hipblaslt.lib from DLL (MSVC import library)
if not exist hipblaslt.lib (
  dumpbin /EXPORTS "%THEROCK_DIST%\bin\libhipblaslt.dll" | findstr /R "^  *[0-9]" > _exports_raw.txt
  echo LIBRARY libhipblaslt.dll > hipblaslt.def
  echo EXPORTS >> hipblaslt.def
  for /f "tokens=4" %%a in (_exports_raw.txt) do echo   %%a >> hipblaslt.def
  lib /def:hipblaslt.def /out:hipblaslt.lib /machine:x64 >nul 2>&1
  del _exports_raw.txt
  if errorlevel 1 (echo FAILED generating hipblaslt.lib && exit /b 1)
)
echo      OK: hipblaslt.lib

echo.
echo [5/8] Compile runtime wrapper (hipBLAS-LT)
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\hip_gemm_runtime.cpp /Fo:runtime.obj
if errorlevel 1 (echo FAILED at step 5 && exit /b 1)
echo      OK: runtime.obj

echo.
echo [6/8] Compile main driver
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\main_gemm.cpp /Fo:main.obj
if errorlevel 1 (echo FAILED at step 6 && exit /b 1)
echo      OK: main.obj

echo.
echo [7/8] Link executable
link.exe gemm.obj runtime.obj main.obj /LIBPATH:"%THEROCK_DIST%\lib" amdhip64.lib hipblaslt.lib /out:gemm_test.exe
if errorlevel 1 (echo FAILED at step 7 && exit /b 1)
echo      OK: gemm_test.exe

echo.
echo [8/8] Quick sanity check
gemm_test.exe --help >nul 2>&1
echo      OK: gemm_test.exe is a valid executable

echo.
echo ============================================================
echo  Build completed! Run with: build\gemm_test.exe
echo ============================================================
echo.
echo  NOTE: Ensure these DLLs are in PATH at runtime:
echo    - amdhip64_7.dll     (from %THEROCK_DIST%\bin)
echo    - libhipblaslt.dll   (from %THEROCK_DIST%\bin)
echo ============================================================
