@echo off
echo ============================================================
echo Building morphizen-hipdnn with Visual Studio 2022
echo ============================================================
echo.

REM Set TheRock environment
echo Setting TheRock environment...
set THEROCK_DIST=C:\Develop\TheRock
set HIP_PLATFORM=amd
echo THEROCK_DIST=%THEROCK_DIST%
echo.

REM Set up Visual Studio 2022 environment
echo Setting up MSVC environment...
call "C:\msvsn2022\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
    echo ERROR: Failed to set up MSVC environment
    exit /b 1
)
echo.

REM Configure with CMake using Ninja generator
echo Configuring project with CMake using Ninja...
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF -B C:/Develop/m/build/morphizen-hipdnn -S . -DCMAKE_INSTALL_PREFIX=C:/Develop/m/local -DTHEROCK_DIST=C:/Develop/TheRock
if errorlevel 1 (
    echo ERROR: CMake configuration failed
    exit /b 1
)
echo.

REM Build the project
echo Building project...
cmake --build C:/Develop/m/build/morphizen-hipdnn
if errorlevel 1 (
    echo ERROR: Build failed
    exit /b 1
)
echo.

echo ============================================================
echo Build completed successfully!
echo ============================================================
