@echo off
echo Setting up MSVC environment from C:\msvsn2022...
call "C:\msvsn2022\VC\Auxiliary\Build\vcvarsall.bat" x64
echo.
echo MSVC environment configured successfully!
echo.
echo You can now run CMake commands:
echo   cmake -DBUILD_SHARED_LIBS=OFF -B C:/Develop/m/build/onnx-hipdnn-ep -S . -DCMAKE_INSTALL_PREFIX=C:/Develop/m/local
echo   cmake --build C:/Develop/m/build/onnx-hipdnn-ep --config Debug
echo.
cmd /k
