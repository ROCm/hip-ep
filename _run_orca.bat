@echo off
REM Usage: _run_orca.bat <max_tokens> [extra python args...]
REM Env vars HIPDNN_EP_DEBUG / HIPDNN_EP_PERF / HIPDNN_EP_STRICT are inherited.
REM NATIVE artifact format needs the MSVC LIB env (lld-link), so load vcvars64 first.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo VCVARS FAILED & exit /b 1)
set "WS=C:\Users\Administrator\Desktop\vakulkar"
set "THEROCK_DIST=%WS%\build\hip-ep\_therock"
REM Prepend our bins AFTER vcvars so the matching onnxruntime.dll wins the DLL search.
set "PATH=%WS%\install\bin;%THEROCK_DIST%\bin;%PATH%"
call "C:\ProgramData\miniforge3\condabin\conda.bat" activate hipdnn-ep
if errorlevel 1 (echo CONDA ACTIVATE FAILED & exit /b 1)
cd /d "%WS%\hip-ep"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "MAXTOK=%1"
set "MODEL=%WS%\orca2bit-latest"
set "EP=%WS%\install\bin\hipgpu.dll"
python tools\run_orca_2bit.py --model_dir "%MODEL%" --ep_dll "%EP%" --max_tokens %MAXTOK% %2 %3 %4 %5 %6 %7 %8 %9
echo === RUN EXIT CODE: %errorlevel% ===
