::
:: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
:: Licensed under the MIT License.
::
@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT=%~dp0"
set "BIN=%ROOT%bin"
set "TRACE_TAG=%RANDOM%"
for /f %%I in ('%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss" 2^>nul') do (
    set "TRACE_TAG=%%I"
)

REM Fixed run configuration. These values intentionally override the caller.
set "KTRACE_OUTPUT_ROOT=%ROOT%out\trace-!TRACE_TAG!"
set "KTRACE_OUTPUT_DIR=!KTRACE_OUTPUT_ROOT!"
set "KTRACE_MAX=0"
set "KTRACE_API_MAX=0"
set "KTRACE_OUTPUT_FILE=trace"
set "KTRACE_VRAM_MS=100"
set "KTRACE_NO_API="
set "KTRACE_API_ARGS=1"
set "KTRACE_NULL="
set "KTRACE_CSV="

if "%~1"=="" (
    call :print_usage
    exit /b 2
)
set "MODE_KTRACE=0"
set "MODE_RGP=0"
set "CAPTURE_ONLY=0"
set "CHROME_JSON=0"
set "CHROME_PFTRACE=0"
set "CHROME_FLOWS=0"
set "CHROME_FORMAT_SELECTED=0"
set "HAS_AUTO_CAPTURE=0"
set "SAW_SEPARATOR=0"
set "HAS_TARGET=0"
set "FORWARDED_ANY=0"
set "COMMAND_ARGS="

REM Take the target command line as raw text. cmd splits %1 on commas as well as
REM spaces, so rebuilding it token by token would forward "-npp 256,512" as
REM "-npp 256 512". Peel space-delimited tokens until "--" is found, then keep
REM the rest of the line as written: splitting on spaces alone leaves commas
REM intact, and "--" never appears inside a quoted argument.
set "TARGET_RAW="
set "SCAN=%*"
:find_separator
if not defined SCAN goto :separator_done
set "HEAD="
set "REST="
for /f "tokens=1* delims= " %%A in ("!SCAN!") do (
    set "HEAD=%%A"
    set "REST=%%B"
)
if "!HEAD!"=="--" (
    set "TARGET_RAW=!REST!"
    goto :separator_done
)
set "SCAN=!REST!"
goto :find_separator
:separator_done

REM Mode tokens may appear in any order and each may be followed by its own
REM options: run.bat system [args] kernel [args] -- application.exe [args]
REM Everything that is not a mode token or a run.bat option is forwarded to
REM ktrace.exe unchanged, so ktrace.exe itself decides what each option means.
:parse_args
if "%~1"=="" goto :parse_done
REM Only honour -h/-v before any forwarded argument, so that a target option of
REM the same name cannot hijack them when "--" is missing.
if "!FORWARDED_ANY!"=="0" (
    if /I "%~1"=="-h" goto :show_help
    if /I "%~1"=="--help" goto :show_help
    if /I "%~1"=="-v" goto :show_version
    if /I "%~1"=="--version" goto :show_version
)
if /I "%~1"=="runtime" (
    echo [ktrace] runtime mode was renamed to system
    exit /b 2
)
if /I "%~1"=="all" (
    echo [ktrace] all mode was replaced by combining system and kernel
    echo [ktrace]   run.bat system kernel -a dispatch:1000:16 -- application.exe
    exit /b 2
)
if /I "%~1"=="system" (
    set "MODE_KTRACE=1"
    shift
    goto :parse_args
)
if /I "%~1"=="kernel" (
    set "MODE_RGP=1"
    shift
    goto :parse_args
)
if /I "%~1"=="--capture-only" (
    set "CAPTURE_ONLY=1"
    shift
    goto :parse_args
)
if /I "%~1"=="--chrome-json" (
    set "CHROME_JSON=1"
    set "CHROME_FORMAT_SELECTED=1"
    shift
    goto :parse_args
)
if /I "%~1"=="--chrome-pftrace" (
    set "CHROME_PFTRACE=1"
    set "CHROME_FORMAT_SELECTED=1"
    shift
    goto :parse_args
)
if /I "%~1"=="--flows" (
    set "CHROME_FLOWS=1"
    shift
    goto :parse_args
)
if "%~1"=="--" (
    set "SAW_SEPARATOR=1"
    if defined TARGET_RAW set "HAS_TARGET=1"
    set "COMMAND_ARGS=!COMMAND_ARGS! -- !TARGET_RAW!"
    goto :parse_done
)
if /I "%~1"=="-a" set "HAS_AUTO_CAPTURE=1"
if /I "%~1"=="--rgp-auto-capture" set "HAS_AUTO_CAPTURE=1"
set "CURRENT_ARG=%~1"
if /I "!CURRENT_ARG:~0,18!"=="--rgp-auto-capture" set "HAS_AUTO_CAPTURE=1"
set "COMMAND_ARGS=!COMMAND_ARGS! %1"
set "FORWARDED_ANY=1"
shift
goto :parse_args

:parse_done
if "!MODE_KTRACE!!MODE_RGP!"=="00" set "MODE_KTRACE=1"
if "!CHROME_FORMAT_SELECTED!"=="0" set "CHROME_JSON=1"

if "!SAW_SEPARATOR!"=="0" (
    echo [ktrace] missing "--" before the target command
    echo.
    call :print_usage
    exit /b 2
)
if "!HAS_TARGET!"=="0" (
    echo [ktrace] missing target command after "--"
    exit /b 2
)
REM kernel names the RGP feature area rather than the capture itself, so it does
REM nothing until an action is requested. Capture is the only action so far.
if "!MODE_RGP!"=="1" if "!HAS_AUTO_CAPTURE!"=="0" (
    echo [ktrace] kernel mode was given without an action, so nothing would run
    echo [ktrace]   to capture: add -a/--rgp-auto-capture, for example
    echo [ktrace]     run.bat kernel -a dispatch:1000:16 -- application.exe
    echo [ktrace]   to analyse an existing .rgp: not implemented yet
    exit /b 2
)
if "!MODE_RGP!"=="0" if "!HAS_AUTO_CAPTURE!"=="1" (
    echo [ktrace] -a/--rgp-auto-capture requires kernel mode
    echo [ktrace]   run.bat system kernel -a dispatch:1000:16 -- application.exe
    exit /b 2
)

if "!CAPTURE_ONLY!"=="1" set "KTRACE_OUTPUT_DIR=!KTRACE_OUTPUT_ROOT!\raw"

if not exist "%KTRACE_OUTPUT_DIR%" mkdir "%KTRACE_OUTPUT_DIR%"
if errorlevel 1 (
    echo [ktrace] failed to create output directory: %KTRACE_OUTPUT_DIR%
    exit /b 1
)

echo [ktrace] output: !KTRACE_OUTPUT_DIR!
if "!CAPTURE_ONLY!"=="1" echo [ktrace] postprocess: disabled
if "!CHROME_JSON!"=="1" echo [ktrace] export: Chrome JSON
if "!CHROME_PFTRACE!"=="1" echo [ktrace] export: native pftrace
if "!CHROME_FLOWS!"=="1" echo [ktrace] export: all flows
if "!MODE_KTRACE!!MODE_RGP!"=="10" echo [ktrace] mode: system ^(ktrace API + GPU commands^)
if "!MODE_KTRACE!!MODE_RGP!"=="01" echo [ktrace] mode: kernel ^(RGP only^)
if "!MODE_KTRACE!!MODE_RGP!"=="11" echo [ktrace] mode: system + kernel ^(ktrace + RGP^)
echo [ktrace] command:!COMMAND_ARGS!
echo.

if "!MODE_KTRACE!"=="0" (
    "%BIN%\ktrace.exe" --rdp-only !COMMAND_ARGS!
) else (
    "%BIN%\ktrace.exe" !COMMAND_ARGS!
)
set "TARGET_RC=!ERRORLEVEL!"

echo.
if "!CAPTURE_ONLY!"=="0" if "!MODE_KTRACE!"=="1" (
    set "RAW_ARCHIVE=!KTRACE_OUTPUT_ROOT!\raw"
    if not exist "!RAW_ARCHIVE!" mkdir "!RAW_ARCHIVE!"
    if errorlevel 1 (
        echo [ktrace] failed to create raw archive: !RAW_ARCHIVE!
        exit /b 1
    )
    for %%F in ("!KTRACE_OUTPUT_DIR!\.*_ops.csv*") do (
        if exist "%%~fF" copy /B /Y "%%~fF" "!RAW_ARCHIVE!\" >nul
    )
    echo [ktrace] raw spool archived: !RAW_ARCHIVE!
)

if "!CAPTURE_ONLY!"=="1" (
    echo [ktrace] capture-only complete; private spool files were preserved
    set "POSTPROCESS_RC=0"
) else (
    if "!MODE_KTRACE!"=="0" (
        echo [ktrace] RGP capture complete; no ktrace post-processing required
        set "POSTPROCESS_RC=0"
    ) else (
        echo [ktrace] finalize + analyze + full Perfetto ...
        set "EXPORT_FLAGS="
        if "!CHROME_JSON!"=="1" set "EXPORT_FLAGS=!EXPORT_FLAGS! --chrome-json"
        if "!CHROME_PFTRACE!"=="1" set "EXPORT_FLAGS=!EXPORT_FLAGS! --chrome-pftrace"
        if "!CHROME_FLOWS!"=="1" set "EXPORT_FLAGS=!EXPORT_FLAGS! --flows"
        "%BIN%\ktrace_postprocess.exe" all "!KTRACE_OUTPUT_DIR!" !EXPORT_FLAGS!
        set "POSTPROCESS_RC=!ERRORLEVEL!"
    )
)

echo.
if not "!POSTPROCESS_RC!"=="0" (
    echo [ktrace] post-processing failed with !POSTPROCESS_RC!
    exit /b !POSTPROCESS_RC!
)
if not "!TARGET_RC!"=="0" (
    echo [ktrace] target exited with !TARGET_RC!
    exit /b !TARGET_RC!
)

REM Analyse the .rgp with Radeon GPU Profiler from RDTS for Windows.
if "!MODE_RGP!"=="1" (
    set "RGP_FILE=!KTRACE_OUTPUT_DIR!\trace_capture.rgp"
    if exist "!RGP_FILE!" (
        echo [ktrace] analyse the capture with Radeon GPU Profiler from RDTS:
        echo [ktrace]   RadeonGPUProfiler.exe "!RGP_FILE!"
    )
)

echo [ktrace] complete: !KTRACE_OUTPUT_DIR!
endlocal
exit /b 0

:show_help
call :print_usage
echo.
"%BIN%\ktrace.exe" --help
exit /b %ERRORLEVEL%

:show_version
"%BIN%\ktrace.exe" --version
exit /b %ERRORLEVEL%

:print_usage
echo Usage:
echo   run.bat ^<mode^> [options] -- ^<application.exe^> [args...]
echo.
echo Modes ^(combine both for dual capture; default is system^):
echo   system                ktrace HIP API and GPU command tracing
echo   kernel                RGP/SQTT capture, requires -a
echo.
echo Each mode may be followed by its own options:
echo   run.bat system [args] kernel [args] -- ^<application.exe^> [args...]
echo.
echo "--" is required before the target command.
echo.
echo Examples:
echo   run.bat system -- application.exe args...
echo   run.bat kernel -a dispatch:1000:16 -- application.exe args...
echo   run.bat system kernel -a dispatch:1000:16 -- application.exe args...
echo   run.bat system --chrome-pftrace --flows -- application.exe args...
echo   run.bat system --capture-only -- application.exe args...
echo   run.bat kernel --capture-only -a dispatch:1000:16 -- application.exe args...
echo.
echo run.bat options: --capture-only --chrome-json --chrome-pftrace --flows
exit /b 0
