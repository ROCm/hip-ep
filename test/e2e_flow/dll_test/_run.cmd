@echo off
set PATH=C:\Users\tsiddaga\Documents\code\therock\bin;%PATH%
"C:\Users\tsiddaga\Documents\code\build\onnx-hipdnn-ep\bin\hip-test-dll.exe" "C:\Users\tsiddaga\Documents\code\onnx-hipdnn-ep\test\e2e_flow\dll_test\model.dll" --verbose --validate
