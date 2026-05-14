<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# E2E Integration Tests

End-to-end tests for the HipDNN MLIR compiler. These tests previously
verified `.mlir` -> per-model `.dll` -> execution via `hip-test-dll`.

> **Status:** disabled. The DLL path has been retired -- the compiler now
> emits LLVM bitcode that is JITted in-process by the EP DLL (see
> `backend-mlir-compiler/custom-op-mlir/src/BitcodeJIT.h`). The test
> harness needs to be re-pointed at the BitcodeJIT loader before these
> tests can run again; the input `.mlir` fixtures in `test/lit/e2e/` are
> preserved for that future work.

For the active end-to-end coverage, run ORT integration tests against
`onnxruntime_perf_test.exe` -- see the Performance Testing section below.

## Test Files

Test `.mlir` files are located in `test/lit/e2e/` and serve dual purposes:
- **LIT tests**: Verify MLIR pass transformations (via `// RUN:` directives)
- **E2E tests**: (disabled) Verify full compilation and execution

## Performance Testing

Use `onnxruntime_perf_test` to benchmark inference latency. The examples below
compare MorphiZen EP (AMD GPU via HIP) against DML EP.

### Setup

`onnxruntime_perf_test.exe` and `morphizen_config.json` are not installed by
default. Copy them into `../prebuilt-local/bin` before running:

```bash
cp ../build/onnxruntime/Release/Release/onnxruntime_perf_test.exe $PREBUILT_DIR/bin/
cp etc/morphizen_config.json $PREBUILT_DIR/bin/

export THEROCK_DIST=~/workspace/therock
export PATH="$THEROCK_DIST/bin:$PATH"
cd $PREBUILT_DIR/bin
```

### MorphiZen EP

```bash
./onnxruntime_perf_test.exe \
  --plugin_ep_libs "MorphiZenExecutionProvider|onnxruntime_morphizen_ep.dll" \
  --plugin_eps "MorphiZenExecutionProvider" \
  --plugin_ep_options "config_file|morphizen_config.json" \
  -t 60 -c 1 -s -I \
  /path/to/models/full_model_seq1.onnx
```

### DML EP

```bash
./onnxruntime_perf_test.exe \
  -e dml \
  -C "ep.dml.disable_graph_fusion|1" \
  -t 60 -c 1 -s -I \
  /path/to/models/full_model_seq1.onnx
```

### Key flags

| Flag | Description |
|------|-------------|
| `-t 60` | Run for 60 seconds |
| `-c 1` | 1 concurrent thread |
| `-s` | Show per-iteration latency statistics |
| `-I` | Use sequential inputs (do not randomize) |
