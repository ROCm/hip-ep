@echo off
setlocal enabledelayedexpansion
echo ============================================================
echo Building LLVM with MLIR for morphizen-rocm
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

REM Clone LLVM project if not already cloned
if not exist "%WORKSPACE_ROOT%\llvm" (
    echo Cloning LLVM project...
    echo This may take several minutes...
    git clone https://github.com/llvm/llvm-project.git "%WORKSPACE_ROOT%\llvm"
    if errorlevel 1 (
        echo ERROR: Failed to clone LLVM project
        exit /b 1
    )
    echo.
) else (
    echo LLVM project already cloned at %WORKSPACE_ROOT%\llvm
    echo.
)

REM Checkout specific commit
echo Checking out LLVM commit f8cb7987c64dcffb72414a40560055cb717dbf74...
cd /d "%WORKSPACE_ROOT%\llvm"
git checkout f8cb7987c64dcffb72414a40560055cb717dbf74
if errorlevel 1 (
    echo ERROR: Failed to checkout LLVM commit
    exit /b 1
)
echo.

REM Return to morphizen-rocm directory
cd /d "%SCRIPT_DIR%"

REM Configure LLVM with CMake (skip if build.ninja exists for incremental build)
if exist "%WORKSPACE_ROOT%\build\llvm\build.ninja" (
    echo Skipping CMake configuration - build.ninja already exists
    echo To force reconfigure, delete %WORKSPACE_ROOT%\build\llvm\build.ninja
    echo.
) else (
    echo Configuring LLVM with CMake using Ninja...
    echo.
    echo [DEBUG] CMake command:
    echo cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -B %WORKSPACE_ROOT%/build/llvm -S %WORKSPACE_ROOT%/llvm/llvm -DCMAKE_INSTALL_PREFIX=%WORKSPACE_ROOT%/local -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_LIBEDIT=OFF -DLLVM_BUILD_TOOLS=ON -DLLVM_INSTALL_UTILS=ON -DLLVM_INCLUDE_TESTS=ON -DZLIB_USE_STATIC_LIBS=ON -DLLVM_ENABLE_ZSTD=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON
    echo.
    cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -B %WORKSPACE_ROOT%/build/llvm -S %WORKSPACE_ROOT%/llvm/llvm -DCMAKE_INSTALL_PREFIX=%WORKSPACE_ROOT%/local -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_LIBEDIT=OFF -DLLVM_BUILD_TOOLS=ON -DLLVM_INSTALL_UTILS=ON -DLLVM_INCLUDE_TESTS=ON -DZLIB_USE_STATIC_LIBS=ON -DLLVM_ENABLE_ZSTD=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON
    if errorlevel 1 (
        echo ERROR: CMake configuration failed
        exit /b 1
    )
    echo.
)

REM Build LLVM/MLIR
echo Building LLVM/MLIR...
echo WARNING: This will take a LONG time (several hours). Please be patient.
echo.
cmake --build %WORKSPACE_ROOT%/build/llvm
set BUILD_EXITCODE=%errorlevel%
if %BUILD_EXITCODE% neq 0 (
    echo.
    echo ============================================================
    echo ERROR: Build failed with exit code %BUILD_EXITCODE%
    echo ============================================================
    exit /b %BUILD_EXITCODE%
)
echo.

REM Install LLVM/MLIR
echo Installing LLVM/MLIR...
cmake --install %WORKSPACE_ROOT%/build/llvm
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
echo LLVM/MLIR build and install completed successfully!
echo Installation directory: %WORKSPACE_ROOT%/local
echo.
echo LLVM installed to: %WORKSPACE_ROOT%/local/lib/cmake/llvm
echo MLIR installed to: %WORKSPACE_ROOT%/local/lib/cmake/mlir
echo.
echo You can now build morphizen-rocm with MLIR backend enabled:
echo   set WITH_MLIR_BACKEND=true
echo   build.bat
echo ============================================================
