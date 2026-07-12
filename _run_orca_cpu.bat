@echo off
call "C:\ProgramData\miniforge3\condabin\conda.bat" activate hipdnn-ep
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "C:\Users\Administrator\Desktop\vakulkar\hip-ep"
python tools\run_orca_2bit.py --model_dir "C:\Users\Administrator\Desktop\vakulkar\orca2bit-latest" --max_tokens %1
echo === CPU RUN EXIT: %errorlevel% ===
