@echo off
REM Script to generate ONNX model, run test with environment variables, and capture errors

setlocal enabledelayedexpansion

echo ========================================
echo ONNX Model Test Runner
echo ========================================
echo.

REM Step 1: Generate the ONNX model
echo [Step 1/3] Generating ONNX test model...
python d:\Develop\m\onnx-hipdnn-ep\test\gen_conv_gemm_model.py --output test_conv_gemm.onnx
if errorlevel 1 (
    echo ERROR: Failed to generate ONNX model
    exit /b 1
)
echo Model generated successfully: test_conv_gemm.onnx
echo.

REM Step 2: Set environment variables
echo [Step 2/3] Setting environment variables...
set DEBUG_VAIP_PASS=1
set XLNX_ENABLE_CACHE=0
set DEBUG_DPU_CUSTOM_OP=1
set XLNX_ONNX_EP_VERBOSE=2
set DEBUG_LOG_LEVEL=info
set DEBUG_EP_CONTEXT=1
set XLNX_ENABLE_CACHE_CONTEXT=0
set CACHE_CONTEXT_EMBEDED_MODE=1
set ENABLE_CACHE_FILE_IO_IN_MEM=0
set VITISAI_EP_JSON_CONFIG=D:\Develop\m\local\bin\morphizen_config.json
set PATH=D:\Develop\m\dist\therock\bin;D:\Develop\m\local\bin;%PATH%
set USE_ORT_API_2_0=1
set MORPHIZEN_VITISAI_EP=D:\Develop\m\local\bin\onnxruntime_vitisai_ep.dll
set MORPHIZEN_DEBUG_VITISAI_EP=1
set MORPHIZEN_DEBUG_HIPDNN=1
set MORPHIZEN_DEBUG_PLUGIN=1
set MORPHIZEN_DEBUG_DEINITIALIZE=1
set MORPHIZEN_MAX_FUSED_SUBGRAPH_NUM=1
set XLNX_ENABLE_DUMP_ONNX_MODEL=1
set MORPHIZEN_DEBUG_ROCM=2
set ENABLE_SAVE_ONNX_MODEL=1
set ENABLE_SAVE_GRAPH_TXT=1
echo Environment variables set successfully
echo.

REM Step 3: Run the test and capture output
echo [Step 3/3] Running test and capturing output...
set LOG_FILE=test_output_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG_FILE=%LOG_FILE: =0%

echo Running: D:\Develop\m\local\bin\test_onnx_runner.exe test_conv_gemm.onnx
echo Output will be saved to: %LOG_FILE%
echo.

D:\Develop\m\local\bin\test_onnx_runner.exe test_conv_gemm.onnx > %LOG_FILE% 2>&1

echo.
echo ========================================
echo Test execution completed
echo ========================================
echo.

REM Parse the log file for errors
echo Analyzing log file for errors...
echo.
echo ======== ONNX RUNTIME ERROR CHECK ========
findstr /c:"[E:onnxruntime:" %LOG_FILE%
if errorlevel 1 (
    echo No [E:onnxruntime: errors found
) else (
    echo.
    echo CRITICAL: [E:onnxruntime: errors detected above
)
echo.

echo ======== GENERAL ERROR ANALYSIS ========
findstr /i /c:"error" /c:"fail" /c:"exception" /c:"fatal" /c:"abort" %LOG_FILE%
if errorlevel 1 (
    echo No obvious errors found in log file
) else (
    echo.
    echo Errors detected above
)
echo.

echo ======== WARNING ANALYSIS ========
findstr /i /c:"warning" /c:"warn" %LOG_FILE%
if errorlevel 1 (
    echo No warnings found in log file
) else (
    echo.
    echo Warnings detected above
)
echo.

echo ========================================
echo Full log saved to: %LOG_FILE%
echo ========================================
echo.
echo To view full log, run: type %LOG_FILE%
echo.

endlocal
