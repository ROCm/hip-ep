@echo off
echo ============================================================
echo Building morphizen-rocm with Visual Studio 2022
echo ============================================================
echo.

REM Detect workspace drive from current script location
set SCRIPT_DIR=%~dp0
set WORKSPACE_DRIVE=%SCRIPT_DIR:~0,2%
set WORKSPACE_ROOT=%WORKSPACE_DRIVE%/Develop/m
echo Detected workspace: %WORKSPACE_ROOT%
echo.

REM Set TheRock environment - check common locations
echo Setting TheRock environment...
if exist "%WORKSPACE_DRIVE%\Develop\m\dist\therock" (
    set THEROCK_DIST=%WORKSPACE_DRIVE%\Develop\m\dist\therock
) else if exist "C:\Develop\m\dist\therock" (
    set THEROCK_DIST=C:\Develop\m\dist\therock
) else if exist "D:\Develop\m\dist\therock" (
    set THEROCK_DIST=D:\Develop\m\dist\therock
) else if exist "C:\dist\therock" (
    set THEROCK_DIST=C:\dist\therock
) else if exist "C:\Develop\TheRock" (
    set THEROCK_DIST=C:\Develop\TheRock
) else (
    echo ERROR: TheRock ROCm SDK not found!
    echo.
    echo Please install TheRock ROCm SDK first:
    echo   1. Download from: https://therock-nightly-tarball.s3.amazonaws.com/index.html
    echo   2. Extract to: C:\Develop\m\dist\therock or D:\Develop\m\dist\therock
    echo   3. See doc\01_DESIGN.md for full instructions
    echo.
    exit /b 1
)
set HIP_PLATFORM=amd
echo THEROCK_DIST=%THEROCK_DIST%
echo.

REM Set up Visual Studio 2022 environment - check both old and new paths
echo Setting up MSVC environment...
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
) else if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
) else (
    echo ERROR: Visual Studio 2022 not found!
    echo Checked paths:
    echo   - C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat
    echo   - C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat
    exit /b 1
)
if errorlevel 1 (
    echo ERROR: Failed to set up MSVC environment
    exit /b 1
)
echo.

REM Create missing natvis file for nlohmann_json (TheRock packaging issue)
if not exist "%THEROCK_DIST%\nlohmann_json.natvis" (
    echo Creating missing nlohmann_json.natvis file...
    echo ^<?xml version="1.0" encoding="utf-8"?^>^<AutoVisualizer xmlns="http://schemas.microsoft.com/vstudio/debugger/natvis/2010"^>^</AutoVisualizer^> > "%THEROCK_DIST%\nlohmann_json.natvis"
)

REM Configure with CMake using Ninja generator (skip if build.ninja exists for incremental build)
REM Using dynamic runtime (MD) to match TheRock's protobuf library
REM Enable ort-bridge for the new ORT API 2.0 support
if exist "%WORKSPACE_ROOT%\build\morphizen-rocm\build.ninja" (
    echo Skipping CMake configuration - build.ninja already exists
    echo To force reconfigure, delete %WORKSPACE_ROOT%\build\morphizen-rocm\build.ninja
    echo.
) else (
    echo Configuring project with CMake using Ninja...
    cmake -G "Ninja" -DCMAKE_CXX_FLAGS="/EHsc" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -B %WORKSPACE_ROOT%/build/morphizen-rocm -S . -DCMAKE_INSTALL_PREFIX=%WORKSPACE_ROOT%/local -DCMAKE_PREFIX_PATH=%WORKSPACE_ROOT%/local -DTHEROCK_DIST=%THEROCK_DIST% -Dmorphizen_ENABLE_ORT_BRIDGE=ON
    if errorlevel 1 (
        echo ERROR: CMake configuration failed
        exit /b 1
    )
    echo.
)

REM Build the project
echo Building project...
cmake --build %WORKSPACE_ROOT%/build/morphizen-rocm
set BUILD_EXITCODE=%errorlevel%
if %BUILD_EXITCODE% neq 0 (
    echo.
    echo ============================================================
    echo ERROR: Build failed with exit code %BUILD_EXITCODE%
    echo ============================================================
    exit /b %BUILD_EXITCODE%
)
echo.

REM Install the project
echo Installing project...
cmake --install %WORKSPACE_ROOT%/build/morphizen-rocm
set INSTALL_EXITCODE=%errorlevel%
if %INSTALL_EXITCODE% neq 0 (
    echo.
    echo ============================================================
    echo ERROR: Install failed with exit code %INSTALL_EXITCODE%
    echo ============================================================
    exit /b %INSTALL_EXITCODE%
)
echo.

echo ============================================================
echo Build and install completed successfully!
echo Installation directory: %WORKSPACE_ROOT%/local
echo ============================================================
