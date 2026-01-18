@echo off
echo ============================================================
echo Incremental Build (uses existing CMake cache)
echo ============================================================
cmake --build C:/Develop/m/build/morphizen-rocm
if %errorlevel% neq 0 (
    echo Build FAILED!
    exit /b %errorlevel%
)
echo Build completed successfully!
