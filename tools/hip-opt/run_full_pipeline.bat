@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
call C:\Users\chiz\anaconda3\condabin\conda.bat activate llvm

set PATH=C:\Users\chiz\work\gpu\llvm-project\build\Debug\bin;C:\Users\chiz\anaconda3\envs\llvm\Library\bin;%PATH%

cd /d C:\Users\chiz\work\gpu\onnx-hipdnn-ep\tools\hip-opt\build

echo.
echo ============================================================
echo  MLIR GEMM End-to-End Pipeline
echo ============================================================

echo.
echo [1/3] MLIR -^> LLVM Dialect (hip-opt)
hip-opt.exe ..\test_gemm.mlir --convert-hip-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts -o gemm_lowered.mlir
if errorlevel 1 (echo FAILED at step 1 && exit /b 1)
echo      OK: gemm_lowered.mlir

echo.
echo [2/3] LLVM Dialect -^> LLVM IR (mlir-translate)
mlir-translate.exe gemm_lowered.mlir --mlir-to-llvmir -o gemm.ll
if errorlevel 1 (echo FAILED at step 2 && exit /b 1)
echo      OK: gemm.ll

echo.
echo [3/3] LLVM IR -^> Object File (llc)
llc.exe gemm.ll -filetype=obj -o gemm.obj
if errorlevel 1 (echo FAILED at step 3 && exit /b 1)
echo      OK: gemm.obj

echo.
echo ============================================================
echo  MLIR compilation pipeline completed!
echo ============================================================
echo.
echo  Generated files:
echo    gemm_lowered.mlir  - Fully lowered MLIR (LLVM dialect)
echo    gemm.ll            - LLVM IR (human-readable)
echo    gemm.obj           - Object file (ready to link)
echo.
echo  To build the final executable (requires THEROCK_DIST):
echo    cl /c /EHsc /I%%THEROCK_DIST%%\include ..\hip_gemm_runtime.cpp /Fo:runtime.obj
echo    cl /c /EHsc /I%%THEROCK_DIST%%\include ..\main_gemm.cpp /Fo:main.obj
echo    link gemm.obj runtime.obj main.obj /LIBPATH:%%THEROCK_DIST%%\lib hipdnn.lib amdhip64.lib /out:gemm_test.exe
