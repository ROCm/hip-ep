# Test VitisAI EP Integration

Run the failing VitisAI EP integration test with full debug logging enabled.

## Environment Setup

Set these environment variables:
- `MORPHIZEN_DEBUG_ROCM=2` (verbose logging)
- `GLOG_logtostderr=1` (output to stderr)
- `GLOG_minloglevel=0` (all log levels)
- `MORPHIZEN_VITISAI_EP_ENABLE_CPU_DEVICE=1` (enable device detection)
- `PATH=C:\Develop\m\dist\therock\bin;%PATH%` (ROCm SDK)

## Test Command

```batch
cd C:\Develop\m\build\morphizen-rocm\bin
ort_integration_test.exe --gtest_filter=OrtIntegrationTest.VitisAIProviderInference
```

## Expected Behavior

- Test should run without SEH exceptions
- MY_LOG output should show weight loading and MIOpen execution
- GPU output should match CPU reference within 1e-4 tolerance

Use the batch script at `tools/test_vitisai.bat` for direct execution.
