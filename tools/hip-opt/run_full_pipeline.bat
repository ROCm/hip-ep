@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
call C:\Users\chiz\anaconda3\condabin\conda.bat activate llvm

set LLVM_BIN=C:\Users\chiz\work\gpu\llvm-project\build\Debug\bin
set ROCM_ROOT=C:\Users\chiz\anaconda3\envs\llvm\Lib\site-packages\_rocm_sdk_devel
set ROCM_LIBS_BIN=C:\Users\chiz\anaconda3\envs\llvm\Lib\site-packages\_rocm_sdk_libraries_gfx1151\bin
set PATH=%LLVM_BIN%;%ROCM_ROOT%\bin;%ROCM_LIBS_BIN%;C:\Users\chiz\anaconda3\envs\llvm\Library\bin;%PATH%

cd /d C:\Users\chiz\work\gpu\onnx-hipdnn-ep\tools\hip-opt\build

echo.
echo ============================================================
echo  MLIR GEMM End-to-End Pipeline (hipBLAS-LT)
echo ============================================================

echo.
echo [1/6] MLIR -^> LLVM Dialect (hip-opt)
hip-opt.exe ..\test_gemm.mlir --convert-hip-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts -o gemm_lowered.mlir
if errorlevel 1 (echo FAILED at step 1 && exit /b 1)
echo      OK: gemm_lowered.mlir

echo.
echo [2/6] LLVM Dialect -^> LLVM IR (mlir-translate)
mlir-translate.exe gemm_lowered.mlir --mlir-to-llvmir -o gemm.ll
if errorlevel 1 (echo FAILED at step 2 && exit /b 1)
echo      OK: gemm.ll

echo.
echo [3/6] LLVM IR -^> Object File (llc)
llc.exe gemm.ll -filetype=obj -o gemm.obj
if errorlevel 1 (echo FAILED at step 3 && exit /b 1)
echo      OK: gemm.obj

echo.
echo [4/8] Generate hipblaslt.lib from DLL (MSVC import library)
if not exist hipblaslt.lib (
  dumpbin /EXPORTS "%ROCM_LIBS_BIN%\libhipblaslt.dll" | findstr /R "^  *[0-9]" > _exports_raw.txt
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
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%ROCM_ROOT%\include" ..\hip_gemm_runtime.cpp /Fo:runtime.obj
if errorlevel 1 (echo FAILED at step 4 && exit /b 1)
echo      OK: runtime.obj

echo.
echo [6/8] Compile main driver
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%ROCM_ROOT%\include" ..\main_gemm.cpp /Fo:main.obj
if errorlevel 1 (echo FAILED at step 5 && exit /b 1)
echo      OK: main.obj

echo.
echo [7/8] Link executable
link.exe gemm.obj runtime.obj main.obj /LIBPATH:"%ROCM_ROOT%\lib" amdhip64.lib hipblaslt.lib /out:gemm_test.exe
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
echo    - amdhip64.dll  (from %ROCM_ROOT%\bin)
echo    - libhipblaslt.dll (from %ROCM_LIBS_BIN%)
echo ============================================================
