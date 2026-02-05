@echo off
echo ============================================================
echo Building onnx-hipdnn-ep with MLIR backend
echo ============================================================
echo.

cd /d "%~dp0"

REM First, build LLVM/MLIR if not already built
echo Step 1: Checking LLVM/MLIR installation...
if not exist "%~dp0..\local\lib\cmake\mlir" (
    echo LLVM/MLIR not found. Building LLVM/MLIR first...
    echo This will take several hours. Please be patient.
    echo.
    call build_llvm.bat
    if errorlevel 1 (
        echo ERROR: LLVM/MLIR build failed
        exit /b 1
    )
    echo.
) else (
    echo LLVM/MLIR already installed.
    echo.
)

REM Build onnx-hipdnn-ep with MLIR backend enabled
echo Step 2: Building onnx-hipdnn-ep with MLIR backend...
set WITH_MLIR_BACKEND=true
call build.bat
