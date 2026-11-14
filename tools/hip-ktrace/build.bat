::
:: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
:: Licensed under the MIT License.
::
@echo off
REM Configure and build ktrace plus the combined RDP launcher.
setlocal EnableExtensions EnableDelayedExpansion

set "SRC=%~dp0"
if not defined BUILD_DIR set "BUILD_DIR=%SRC%build"
if not defined BUILD_TYPE set "BUILD_TYPE=Release"
if not defined KTRACE_BUILD_CAPTURE set "KTRACE_BUILD_CAPTURE=ON"
if not defined KTRACE_BUILD_RDP set "KTRACE_BUILD_RDP=ON"
if not defined KTRACE_BUILD_POSTPROCESS set "KTRACE_BUILD_POSTPROCESS=ON"
if not defined KTRACE_BUILD_WORKLOAD set "KTRACE_BUILD_WORKLOAD=ON"

if defined VCVARS (
    if not exist "%VCVARS%" ( echo ERROR: VCVARS not found: %VCVARS% & exit /b 1 )
    call "%VCVARS%" >nul 2>&1
    if errorlevel 1 ( echo ERROR: vcvars64 failed & exit /b 1 )
) else (
    where cl >nul 2>&1
    if errorlevel 1 (
        echo ERROR: MSVC environment is not initialized.
        echo Set VCVARS to vcvars64.bat or run from an x64 Native Tools prompt.
        exit /b 1
    )
)

if /I not "%KTRACE_BUILD_CAPTURE%"=="OFF" (
    if not defined HIP_HEADERS (
        echo ERROR: HIP_HEADERS is not set.
        echo It must match the HIP runtime loaded by the traced process.
        exit /b 1
    )
    if not exist "%HIP_HEADERS%\hip\amd_detail\hip_prof_str.h" (
        echo ERROR: invalid HIP_HEADERS: %HIP_HEADERS%
        exit /b 1
    )
)
if /I not "%KTRACE_BUILD_RDP%"=="OFF" (
    if /I "%KTRACE_BUILD_CAPTURE%"=="OFF" (
        echo ERROR: KTRACE_BUILD_RDP requires KTRACE_BUILD_CAPTURE=ON.
        exit /b 1
    )
    call :prepare_rdp_runtime
    if errorlevel 1 exit /b 1
    if exist "%SRC%bin\ktrace_launch.exe" del /q "%SRC%bin\ktrace_launch.exe"
    if exist "%SRC%bin\ktrace_rdp.exe" del /q "%SRC%bin\ktrace_rdp.exe"
)
if /I not "%KTRACE_BUILD_WORKLOAD%"=="OFF" (
    if not defined HIP (
        echo ERROR: HIP is not set. Point it to the HIP/ROCm installation.
        exit /b 1
    )
    if not exist "%HIP%\bin\hipcc.exe" if not exist "%HIP%\bin\hipcc.bat" (
        echo ERROR: hipcc was not found under HIP: %HIP%
        exit /b 1
    )
    if not defined GPU_ARCH (
        echo ERROR: GPU_ARCH is not set.
        echo Use one architecture or a comma-separated list, for example:
        echo   gfx1100
        echo   gfx1100,gfx1103,gfx1150,gfx1151,gfx1152,gfx1200,gfx1201
        exit /b 1
    )
)

echo.
echo [configure] %BUILD_DIR%
cmake -S "%SRC%." -B "%BUILD_DIR%" -G Ninja ^
    "-DCMAKE_BUILD_TYPE=%BUILD_TYPE%" ^
    "-DKTRACE_BUILD_CAPTURE=%KTRACE_BUILD_CAPTURE%" ^
    "-DKTRACE_BUILD_RDP=%KTRACE_BUILD_RDP%" ^
    "-DKTRACE_BUILD_POSTPROCESS=%KTRACE_BUILD_POSTPROCESS%" ^
    "-DKTRACE_BUILD_WORKLOAD=%KTRACE_BUILD_WORKLOAD%" ^
    "-DKTRACE_HIP_ROOT=%HIP%" ^
    "-DKTRACE_HIP_HEADERS=%HIP_HEADERS%" ^
    "-DKTRACE_GPU_ARCH=%GPU_ARCH%"
if errorlevel 1 ( echo ERROR: CMake configure failed & exit /b 1 )

echo.
echo [build]
cmake --build "%BUILD_DIR%" --parallel
if errorlevel 1 ( echo ERROR: build failed & exit /b 1 )

echo.
echo BUILD OK -^> %SRC%bin
endlocal & exit /b 0

:prepare_rdp_runtime
set "RDP_API=%SRC%bin\RdpCaptureApi.dll"
set "RDP_COUNTERS=%SRC%bin\GPUPerfAPICounters-x64.dll"
set "RDP_COMGR=%SRC%bin\amd_comgr_3.dll"
set "RDP_DISASM=%SRC%bin\amdgpu_dis.dll"
if exist "!RDP_API!" if exist "!RDP_COUNTERS!" if exist "!RDP_COMGR!" if exist "!RDP_DISASM!" (
    echo [rdp] runtime DLLs already available
    exit /b 0
)

where curl.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: curl.exe was not found; cannot download RDTS runtime.
    exit /b 1
)
where powershell.exe >nul 2>nul
if errorlevel 1 (
    echo ERROR: powershell.exe was not found; cannot extract RDTS runtime.
    exit /b 1
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if errorlevel 1 exit /b 1
set "RDTS_ARCHIVE=%BUILD_DIR%\rdts-windows.zip"
set "RDTS_EXTRACT=%BUILD_DIR%\rdts-runtime"
if exist "!RDTS_EXTRACT!" rmdir /s /q "!RDTS_EXTRACT!"

if not exist "!RDTS_ARCHIVE!" (
    echo [rdp] downloading Radeon Developer Tool Suite runtime...
    curl.exe --fail --location --retry 3 --retry-delay 2 ^
        --output "!RDTS_ARCHIVE!" "https://gpuopen.com/rdts-windows/"
    if errorlevel 1 (
        echo ERROR: RDTS download failed.
        exit /b 1
    )
)
if exist "!RDTS_ARCHIVE!" (
    echo [rdp] using !RDTS_ARCHIVE!
)

echo [rdp] extracting runtime DLLs...
set "RDTS_ARCHIVE_ENV=!RDTS_ARCHIVE!"
set "RDTS_DEST_ENV=!RDTS_EXTRACT!"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop'; Expand-Archive -LiteralPath $env:RDTS_ARCHIVE_ENV -DestinationPath $env:RDTS_DEST_ENV -Force"
if errorlevel 1 (
    echo ERROR: RDTS extraction failed.
    exit /b 1
)

if not exist "%SRC%bin" mkdir "%SRC%bin"
set "RDP_API_ENV=!RDP_API!"
set "RDP_COUNTERS_ENV=!RDP_COUNTERS!"
set "RDP_COMGR_ENV=!RDP_COMGR!"
set "RDP_DISASM_ENV=!RDP_DISASM!"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "$api=Get-ChildItem -LiteralPath $env:RDTS_DEST_ENV -Recurse -Filter RdpCaptureApi.dll | Select-Object -First 1;" ^
    "$counters=Get-ChildItem -LiteralPath $env:RDTS_DEST_ENV -Recurse -Filter GPUPerfAPICounters-x64.dll | Select-Object -First 1;" ^
    "$comgr=Get-ChildItem -LiteralPath $env:RDTS_DEST_ENV -Recurse -Filter amd_comgr_3.dll | Select-Object -First 1;" ^
    "$disasm=Get-ChildItem -LiteralPath $env:RDTS_DEST_ENV -Recurse -Filter amdgpu_dis.dll | Select-Object -First 1;" ^
    "if(-not $api){throw 'RdpCaptureApi.dll was not found in the RDTS archive'};" ^
    "if(-not $counters){throw 'GPUPerfAPICounters-x64.dll was not found in the RDTS archive'};" ^
    "if(-not $comgr){throw 'amd_comgr_3.dll was not found in the RDTS archive'};" ^
    "if(-not $disasm){throw 'amdgpu_dis.dll was not found in the RDTS archive'};" ^
    "Copy-Item -LiteralPath $api.FullName -Destination $env:RDP_API_ENV -Force;" ^
    "Copy-Item -LiteralPath $counters.FullName -Destination $env:RDP_COUNTERS_ENV -Force;" ^
    "Copy-Item -LiteralPath $comgr.FullName -Destination $env:RDP_COMGR_ENV -Force;" ^
    "Copy-Item -LiteralPath $disasm.FullName -Destination $env:RDP_DISASM_ENV -Force"
if errorlevel 1 (
    echo ERROR: failed to locate or copy RDTS runtime DLLs.
    exit /b 1
)
del /q "!RDTS_ARCHIVE!" >nul 2>nul
echo [rdp] runtime DLLs copied to %SRC%bin
exit /b 0
