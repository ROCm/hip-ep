::
:: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
:: Licensed under the MIT License.
::
@echo off
echo ============================================================
echo Building onnx-hipdnn-ep with Visual Studio 2022
echo ============================================================
echo.

REM Set TheRock environment - check common locations
echo Setting TheRock environment...
if exist "C:\Develop\m\dist\therock" (
    set THEROCK_DIST=C:\Develop\m\dist\therock
) else if exist "C:\dist\therock" (
    set THEROCK_DIST=C:\dist\therock
) else if exist "C:\Develop\TheRock" (
    set THEROCK_DIST=C:\Develop\TheRock
) else (
    echo ERROR: TheRock ROCm SDK not found!
    echo.
    echo Please install TheRock ROCm SDK first:
    echo   1. Download from: https://therock-nightly-tarball.s3.amazonaws.com/index.html
    echo   2. Extract to: C:\dist\therock
    echo   3. See doc\HIPDNN_WINDOWS_SETUP.md for full instructions
    echo.
    exit /b 1
)
set HIP_PLATFORM=amd
echo THEROCK_DIST=%THEROCK_DIST%
echo.

REM Set up Visual Studio 2022 environment
echo Setting up MSVC environment...
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
    echo ERROR: Failed to set up MSVC environment
    exit /b 1
)
echo.

REM Configure with CMake using Ninja generator
echo Configuring project with CMake using Ninja...
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" -B C:/Develop/m/build/onnx-hipdnn-ep -S . -DCMAKE_INSTALL_PREFIX=C:/Develop/m/local -DTHEROCK_DIST=%THEROCK_DIST%
if errorlevel 1 (
    echo ERROR: CMake configuration failed
    exit /b 1
)
echo.

REM Build the project
echo Building project...
cmake --build C:/Develop/m/build/onnx-hipdnn-ep
if errorlevel 1 (
    echo ERROR: Build failed
    exit /b 1
)
echo.

echo ============================================================
echo Build completed successfully!
echo ============================================================
