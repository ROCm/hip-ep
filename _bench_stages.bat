@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "WS=C:\Users\Administrator\Desktop\vakulkar"
set "THEROCK_DIST=%WS%\build\hip-ep\_therock"
set "PATH=%WS%\install\bin;%THEROCK_DIST%\bin;%PATH%"
call "C:\ProgramData\miniforge3\condabin\conda.bat" activate hipdnn-ep
cd /d "%WS%\hip-ep"
python tools\bench_orca_stages.py "%WS%\orca2bit-latest" "%WS%\install\bin\hipgpu.dll"
echo === EXIT: %errorlevel% ===
