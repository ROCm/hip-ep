@echo off
setlocal
set SCRIPT=%~dp0github-credential-helper.sh
set SCRIPT=%SCRIPT:\=/%
C:\PROGRA~1\Git\bin\bash.exe "%SCRIPT%" %*
