@echo off
echo Performing clean build with MLIR backend enabled...
echo.

REM Change to the onnx-hipdnn-ep directory
cd /d "%~dp0"

REM Clean the build directory
echo Cleaning build directory...
if exist "D:\Develop\m\build\onnx-hipdnn-ep" (
    rmdir /s /q "D:\Develop\m\build\onnx-hipdnn-ep"
    echo Build directory cleaned.
) else (
    echo Build directory does not exist, skipping clean.
)
echo.

REM Set MLIR backend flag
set WITH_MLIR_BACKEND=true

REM Run the build
echo Starting build with MLIR backend enabled...
call build.bat
