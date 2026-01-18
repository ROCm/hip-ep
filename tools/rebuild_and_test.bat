@echo off
call "%~dp0rebuild.bat"
if %errorlevel% neq 0 exit /b %errorlevel%
echo.
call "%~dp0test_vitisai.bat"
