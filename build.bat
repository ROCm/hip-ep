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

REM Check for hipDNN installation
set HIPDNN_PREFIX_PATH=
if exist "C:\Develop\m\local\hipdnn\lib\cmake\hipdnn_frontend" (
    echo Found hipDNN at C:\Develop\m\local\hipdnn
    set HIPDNN_PREFIX_PATH=-DCMAKE_PREFIX_PATH=C:/Develop/m/local/hipdnn
) else (
    echo WARNING: hipDNN not found at C:\Develop\m\local\hipdnn
    echo          Build may fail if level-1-pass-hipdnn is enabled.
    echo          See doc\HIPDNN_WINDOWS_SETUP.md for installation instructions.
    echo.
)

REM Set GSL include path for ORT VitisAI headers
set GSL_INCLUDE_DIR=C:\Develop\m\build\onnx-hipdnn-ep\_deps\microsoft.gsl-src\include
set INCLUDE=%GSL_INCLUDE_DIR%;%INCLUDE%
echo Added GSL include path: %GSL_INCLUDE_DIR%
echo.

REM Create missing natvis file for nlohmann_json (TheRock packaging issue)
if not exist "%THEROCK_DIST%\nlohmann_json.natvis" (
    echo Creating missing nlohmann_json.natvis file...
    echo ^<?xml version="1.0" encoding="utf-8"?^>^<AutoVisualizer xmlns="http://schemas.microsoft.com/vstudio/debugger/natvis/2010"^>^</AutoVisualizer^> > "%THEROCK_DIST%\nlohmann_json.natvis"
)

REM Configure with CMake using Ninja generator
echo Configuring project with CMake using Ninja...
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -B C:/Develop/m/build/onnx-hipdnn-ep -S . -DCMAKE_INSTALL_PREFIX=C:/Develop/m/local -DTHEROCK_DIST=%THEROCK_DIST% %HIPDNN_PREFIX_PATH%
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
