@echo off
REM Credential helper for Bazel to access private GitHub registries on Windows
REM Bridges gh auth token to Bazel's credential helper protocol

gh auth token >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: gh auth token failed. Run 'gh auth login' first. >&2
    exit /b 1
)

for /f "delims=" %%i in ('gh auth token') do set TOKEN=%%i
echo headers=Authorization=Bearer %TOKEN%
