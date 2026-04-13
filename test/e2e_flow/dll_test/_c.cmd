@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
set THEROCK_DIST=C:\Users\tsiddaga\Documents\code\therock
set PATH=C:\Users\tsiddaga\Documents\code\therock\bin;%PATH%
"C:\Users\tsiddaga\Documents\code\build\onnx-hipdnn-ep\bin\hip-compiler.exe" "C:\Users\tsiddaga\Documents\code\onnx-hipdnn-ep\scripts\..\test\e2e_flow\dll_test\model.mlir" -o "C:\Users\tsiddaga\Documents\code\onnx-hipdnn-ep\scripts\..\test\e2e_flow\dll_test\model.dll"
