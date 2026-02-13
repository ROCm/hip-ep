@echo off
REM --- Activate VS tools only if not already active ---
if not defined VSINSTALLDIR (
  call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)

REM --- Activate conda env only if not already active ---
if not defined CONDA_PREFIX (
  call C:\Users\chiz\anaconda3\condabin\conda.bat activate llvm
)

REM --- Tool paths (no PATH modification) ---
set LLVM_BIN=C:\Users\chiz\work\gpu\llvm-project\build\Debug\bin
set THEROCK_DIST=C:\Users\chiz\work\gpu\TheRock\build\dist\rocm
set THEROCK_CLANG=C:\Users\chiz\work\gpu\TheRock\build\compiler\amd-llvm\dist\lib\llvm\bin
set HIP_OPT_BIN=C:\Users\chiz\work\gpu\onnx-hipdnn-ep\tools\hip-opt\build\Debug

cd /d C:\Users\chiz\work\gpu\onnx-hipdnn-ep\tools\hip-opt\build

echo.
echo ============================================================
echo  MLIR GEMM End-to-End Pipeline (hipDNN graph API)
echo ============================================================

echo.
echo [1/7] MLIR -^> LLVM Dialect (hip-opt)
"%HIP_OPT_BIN%\hip-opt.exe" ..\test_gemm.mlir --convert-hip-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts -o gemm_lowered.mlir
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
echo [4/7] Compile runtime wrapper (hipDNN graph API, using clang++)
set HIPDNN_INC=%THEROCK_DIST%\include
"%THEROCK_CLANG%\clang++.exe" -c -std=c++17 -fms-extensions -fms-compatibility ^
  -D__HIP_PLATFORM_AMD__ -D__HIPCC__ -DSPDLOG_FMT_EXTERNAL -DFMT_HEADER_ONLY ^
  -I"%HIPDNN_INC%" ^
  -I"%HIPDNN_INC%\hipdnn\frontend" ^
  -I"%HIPDNN_INC%\hipdnn\backend" ^
  -I"%HIPDNN_INC%\hipdnn\data_sdk" ^
  -Wno-unused-value -Wno-c++11-narrowing ^
  ..\hip_gemm_runtime_hipdnn.cpp -o runtime.obj
if errorlevel 1 (echo FAILED at step 4 && exit /b 1)
echo      OK: runtime.obj

echo.
echo [5/7] Compile main driver
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /I"%THEROCK_DIST%\include" ..\main_gemm.cpp /Fo:main.obj
if errorlevel 1 (echo FAILED at step 5 && exit /b 1)
echo      OK: main.obj

echo.
echo [6/7] Link executable
link.exe gemm.obj runtime.obj main.obj /LIBPATH:"%THEROCK_DIST%\lib" amdhip64.lib hipdnn_backend.lib /out:gemm_test_hipdnn.exe
if errorlevel 1 (echo FAILED at step 6 && exit /b 1)
echo      OK: gemm_test_hipdnn.exe

echo.
echo [7/7] Quick sanity check
gemm_test_hipdnn.exe --help >nul 2>&1
echo      OK: gemm_test_hipdnn.exe is a valid executable

echo.
echo ============================================================
echo  Build completed! Run with: build\gemm_test_hipdnn.exe
echo ============================================================
echo.
echo  NOTE: Ensure these DLLs are in PATH at runtime:
echo    - amdhip64_7.dll       (from %THEROCK_DIST%\bin)
echo    - hipdnn_backend.dll   (from %THEROCK_DIST%\bin)
echo ============================================================
