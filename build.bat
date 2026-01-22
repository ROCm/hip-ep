@echo off
setlocal enabledelayedexpansion
echo ============================================================
echo Building morphizen-mlir with Visual Studio 2022
echo ============================================================
echo.

REM Detect workspace drive from current script location
set SCRIPT_DIR=%~dp0
set WORKSPACE_DRIVE=%SCRIPT_DIR:~0,2%
set WORKSPACE_ROOT=%WORKSPACE_DRIVE%/Develop/m
echo Detected workspace: %WORKSPACE_ROOT%
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

REM Check for required dependencies
echo Checking dependencies...
if not exist "%WORKSPACE_ROOT%\local\lib\cmake\onnxruntime" (
    echo WARNING: ONNXRuntime not found in %WORKSPACE_ROOT%\local
    echo Please build ONNXRuntime first. See README.md for instructions.
    echo.
)

REM Configure with CMake using Ninja generator (skip if build.ninja exists for incremental build)
if exist "%WORKSPACE_ROOT%\build\morphizen-mlir\build.ninja" (
    echo Skipping CMake configuration - build.ninja already exists
    echo To force reconfigure, delete %WORKSPACE_ROOT%\build\morphizen-mlir\build.ninja
    echo.
) else (
    echo Configuring project with CMake using Ninja...
    echo.
    echo [DEBUG] CMake command:
    echo cmake -G "Ninja" -DCMAKE_CXX_FLAGS="/EHsc /wd4996 /D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -B %WORKSPACE_ROOT%/build/morphizen-mlir -S . -DCMAKE_INSTALL_PREFIX=%WORKSPACE_ROOT%/local -DCMAKE_PREFIX_PATH=%WORKSPACE_ROOT%/local -Dmorphizen_ENABLE_UNIT_TEST=ON
    echo.
    cmake -G "Ninja" -DCMAKE_CXX_FLAGS="/EHsc /wd4996 /D_SILENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -B %WORKSPACE_ROOT%/build/morphizen-mlir -S . -DCMAKE_INSTALL_PREFIX=%WORKSPACE_ROOT%/local -DCMAKE_PREFIX_PATH=%WORKSPACE_ROOT%/local -Dmorphizen_ENABLE_UNIT_TEST=ON
    if errorlevel 1 (
        echo ERROR: CMake configuration failed
        exit /b 1
    )
    echo.
)

REM Build the project
echo Building project...
cmake --build %WORKSPACE_ROOT%/build/morphizen-mlir
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
cmake --install %WORKSPACE_ROOT%/build/morphizen-mlir
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
