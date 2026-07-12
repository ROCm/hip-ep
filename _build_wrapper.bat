@echo off
REM MSVC x64 env + conda hipdnn-ep + build.py
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (echo VCVARS FAILED & exit /b 1)
call "C:\ProgramData\miniforge3\condabin\conda.bat" activate hipdnn-ep
if errorlevel 1 (echo CONDA ACTIVATE FAILED & exit /b 1)
cd /d "C:\Users\Administrator\Desktop\vakulkar\hip-ep"
echo === where cl === & where cl
echo === where cmake === & where cmake
echo === python === & python --version
echo === STARTING build.py ===
python build.py --cmake_generator Ninja --skip_tests
echo === BUILD EXIT CODE: %errorlevel% ===
