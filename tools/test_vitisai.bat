@echo off
echo ============================================================
echo Running VitisAI EP Integration Test
echo ============================================================

REM Set up TheRock environment
set PATH=C:\Develop\m\dist\therock\bin;%PATH%

REM Enable all debug logging
set MORPHIZEN_DEBUG_ROCM=2
set GLOG_logtostderr=1
set GLOG_minloglevel=0
set MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=1

echo Environment:
echo   MORPHIZEN_DEBUG_ROCM=%MORPHIZEN_DEBUG_ROCM%
echo   GLOG_logtostderr=%GLOG_logtostderr%
echo   MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=%MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE%
echo.

REM Copy test model if needed
if not exist "C:\Develop\m\build\morphizen-rocm\bin\conv_model.onnx" (
    echo Copying test model...
    copy "test\conv_model.onnx" "C:\Develop\m\build\morphizen-rocm\bin\" >nul 2>&1
)

cd C:\Develop\m\build\morphizen-rocm\bin
ort_integration_test.exe --gtest_filter=OrtIntegrationTest.VitisAIProviderInference
