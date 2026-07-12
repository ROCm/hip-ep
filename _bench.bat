@echo off
REM NATIVE artifact needs MSVC LIB env for lld-link.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo VCVARS FAILED & exit /b 1)
set "WS=C:\Users\Administrator\Desktop\vakulkar"
set "THEROCK_DIST=%WS%\build\hip-ep\_therock"
set "PATH=%WS%\install\bin;%THEROCK_DIST%\bin;%PATH%"
call "C:\ProgramData\miniforge3\condabin\conda.bat" activate hipdnn-ep
if errorlevel 1 (echo CONDA ACTIVATE FAILED & exit /b 1)
cd /d "%WS%\hip-ep"
python tools\bench_orca.py "%WS%\orca2bit-latest" "%WS%\install\bin\hipgpu.dll"
echo === BENCH EXIT CODE: %errorlevel% ===
