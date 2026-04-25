@echo off
:: Build onnx-hipdnn-ep against the venv-installed TheRock SDK on this
:: machine.  No sccache, no ORT (BUILD_EP=OFF) so we can iterate on the
:: MLIR compiler / LIT tests without waiting for an onnxruntime build.
::
:: Usage:
::   call scripts\build_local.cmd          (configure + build)
::   call scripts\build_local.cmd configure (configure only)
::   call scripts\build_local.cmd build     (build only)
::   call scripts\build_local.cmd test      (run LIT tests)
::   call scripts\build_local.cmd fresh     (delete build dir and reconfigure)

setlocal EnableExtensions EnableDelayedExpansion

set REPO_ROOT=%~dp0..
pushd "%REPO_ROOT%"
set REPO_ROOT=%CD%
popd

set BUILD_DIR=%REPO_ROOT%\..\build\onnx-hipdnn-ep
set PREBUILT_DIR=%REPO_ROOT%\..\prebuilt-local

set VENV=%REPO_ROOT%\..\demos\.venv
if not exist "%VENV%\Scripts\python.exe" (
    echo ERROR: venv not found at %VENV%
    exit /b 1
)
call "%VENV%\Scripts\activate.bat"

for /f "delims=" %%i in ('rocm-sdk path --root') do set THEROCK_DIST=%%i
for /f "delims=" %%i in ('rocm-sdk targets') do set ROCM_TARGETS=%%i
echo THEROCK_DIST=%THEROCK_DIST%
echo ROCM_TARGETS=%ROCM_TARGETS%

if "%HIP_ARCHITECTURES%"=="" set HIP_ARCHITECTURES=gfx1201

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo ERROR: vcvars64.bat failed
    exit /b 1
)

set CMD=%~1
if "%CMD%"=="" set CMD=both

if /i "%CMD%"=="fresh" (
    if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
    set CMD=both
)

if /i "%CMD%"=="configure" goto configure
if /i "%CMD%"=="both"      goto configure
if /i "%CMD%"=="build"     goto build
if /i "%CMD%"=="test"      goto test
echo ERROR: unknown command: %CMD%
exit /b 2

:configure
echo === configure ===
cmake -S "%REPO_ROOT%" -B "%BUILD_DIR%" ^
  -G Ninja ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_PREFIX_PATH="%PREBUILT_DIR%" ^
  -DCMAKE_INSTALL_PREFIX="%PREBUILT_DIR%" ^
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ^
  -DTHEROCK_DIST="%THEROCK_DIST%" ^
  -DHIP_PLATFORM=amd ^
  -DHIP_ARCHITECTURES=%HIP_ARCHITECTURES% ^
  -DBUILD_EP=ON ^
  -DBUILD_MOCK_RUNTIME=OFF ^
  -DBUILD_HIP_TOOLS=ON ^
  -DBUILD_HIP_UNIT_TESTS=ON
if errorlevel 1 exit /b 1
if /i "%CMD%"=="configure" goto :eof

:build
echo === build ===
cmake --build "%BUILD_DIR%" --config Release --parallel
if errorlevel 1 exit /b 1
if /i "%CMD%"=="build" goto :eof

:install
echo === install ===
cmake --install "%BUILD_DIR%" --config Release
if errorlevel 1 exit /b 1
goto :eof

:test
echo === test ===
:: ROCm DLLs (amdhip64, MIOpen, hipBLASLt, ...) must be on PATH for the
:: generated model DLLs to load at execute-test time.  Pull the bin dir
:: from the venv's rocm-sdk install.
for /f "delims=" %%i in ('rocm-sdk path --bin') do set ROCM_BIN=%%i
set PATH=%ROCM_BIN%;%PATH%
ctest --test-dir "%BUILD_DIR%" --output-on-failure -C Release
exit /b %errorlevel%
