# Rebuild and Test

Complete workflow: rebuild the project and run the VitisAI EP integration test.

## Steps

1. Run incremental build using `/rebuild` command or `tools/rebuild.bat`
2. If build succeeds, run VitisAI EP test using `/test-vitisai-ep` command or `tools/test_vitisai.bat`
3. Analyze test output for MY_LOG messages and GPU execution results

Use the batch script at `tools/rebuild_and_test.bat` for direct execution.
