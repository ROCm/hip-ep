@echo off
REM ============================================================
REM  Shared environment setup for all pipeline scripts.
REM  >>> EDIT THESE PATHS TO MATCH YOUR LOCAL MACHINE <<<
REM ============================================================

REM --- Visual Studio (vcvarsall.bat path) ---
if not defined VSINSTALLDIR (
  call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
)

REM --- Conda environment name ---
if not defined CONDA_PREFIX (
  call C:\Users\chiz\anaconda3\condabin\conda.bat activate llvm
)

REM --- LLVM/MLIR build output (contains mlir-translate, llc) ---
set LLVM_BIN=C:\Users\chiz\work\gpu\llvm-project\build\Debug\bin

REM --- TheRock ROCm dist (contains amdhip64, hipblaslt, MIOpen) ---
set THEROCK_DIST=C:\Users\chiz\work\gpu\TheRock\build\dist\rocm

REM --- hip-opt build output ---
set HIP_OPT_BIN=C:\Users\chiz\work\gpu\onnx-hipdnn-ep\tools\hip-opt\build\Debug

REM --- hip-opt source root ---
set SRC_DIR=C:\Users\chiz\work\gpu\onnx-hipdnn-ep\tools\hip-opt
