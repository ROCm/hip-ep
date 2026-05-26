<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Bisecting MorphiZen EP compile failures

When an ONNX model fails to compile inside the MorphiZen EP (typical
symptoms: SSA dominance error, `op was not bufferized`, an unsupported
type/dialect error from a specific pass) the fastest path to a fix is to
narrow the failure down to a single MLIR pass and a single op location.
This page documents the supported workflow.

## 1. Capture the failure point

With Phase 0 in place the EP throws at session creation when MLIR
compilation fails. The Python-visible error looks like:

```text
RuntimeError: ... MorphiZen EP MLIR compilation failed:
  hip-compiler plugin failed (code=3):
  loc("-":42:5): error: operand #0 does not dominate this use
```

Note the pass and source location — if the message already pinpoints the
op, you can often skip straight to Step 4.

## 2. Dump the imported MLIR bytecode

The bytecode the EP feeds to hip-compiler is what we want to bisect, not
the original ONNX. Use the helper script:

```bash
python tools/dump_imported_mlir.py \
    --ep-dll install/dist/bin/onnxruntime_morphizen_ep.dll \
    --ep-config install/dist/bin/morphizen_config.json \
    --model path/to/embedding.onnx \
    --out dump.mlirbc
```

The script sets `HIPDNN_EP_BYTECODE_DUMP_PATH=dump.mlirbc` and
`HIPDNN_EP_ALLOW_CPU_FALLBACK=1` so the level-1 pass writes the bytecode
to a known location even if compilation later fails.

The same env var (`HIPDNN_EP_BYTECODE_DUMP_PATH=<file>`) works when
invoking the EP from any other client (`model_benchmark.exe`, an OGA
harness, ...) — the helper script is just a no-input convenience
wrapper.

Note: `HIPDNN_EP_IR_DUMP_PATH` is a separate, complementary env var
consumed by `lib/Compiler/CompilerDriver.cpp`. It enables MLIR
"print-IR-after-each-pass" dumping inside the pass pipeline. Useful
once you have the bytecode and want to localise the failing pass:
set both vars and the bytecode dump will land at
`HIPDNN_EP_BYTECODE_DUMP_PATH` while the per-pass IR dump lands at
`HIPDNN_EP_IR_DUMP_PATH`.

## 3. Convert to readable MLIR

```bash
install/dist/bin/hip-mlir-opt.exe dump.mlirbc -o dump.mlir
```

`hip-mlir-opt` with no passes is a round-trip parser; it accepts the
bytecode and prints textual MLIR. Diff this against a known-good model's
dump to spot structural differences without yet running any pass.

## 4. Bisect failing passes

Run individual pipeline stages until the failure reproduces. The full
EP pipeline (lib/Dialect/Transforms/Pipelines.cpp) is roughly:

```
--hip-add-context-arg
--convert-onnx-to-hip
--one-shot-bufferize
--buffer-deallocation
--hip-optimize-memrefs
--hip-pool-allocs
--hip-lower-allocs
--lower-affine            # since the gpt-oss-20b MoE fix
--convert-hip-to-llvm
--generate-interface
```

The most common failure point for dynamic-input models is
`--convert-onnx-to-hip` — every conversion pattern under
`lib/Conversion/OnnxToHip/*` has to materialize runtime dims before
their consumer, or SSA dominance fails. Reproduce with:

```bash
install/dist/bin/hip-mlir-opt.exe dump.mlir \
    --hip-add-context-arg \
    --convert-onnx-to-hip \
    --mlir-print-ir-after-failure \
    --mlir-print-stacktrace-on-diagnostic \
    -o /dev/null
```

`--mlir-print-ir-after-failure` dumps the partially-converted IR after
the failing pass, giving you the exact pre-failure state. The
`loc(...)` markers in the error message map back to the op in
`dump.mlir`.

To narrow further (verify a specific *op* triggers the failure rather
than its neighbours), use `hip-mlir-opt --convert-onnx-to-hip
--mlir-disable-threading dump.mlir` and check whether the diagnostic
location matches the op you suspect.

## 5. Fix and add a LIT regression

Most fixes follow the existing idiom in `lib/Conversion/OnnxToHip/*`:
materialize runtime dims via `tensor.dim(input, axis)` BEFORE the
`tensor.empty(ValueRange{dims})` consumer, so SSA dominance is preserved.
See `EqualConversion.cpp` (Phase 2a) for the canonical example.

Add a LIT test for the fix under
`test/lit/Conversion/onnx-to-hip/test_<op>_dyn.mlir` that pipes a
`tensor<?x...>` form through the same passes and FileCheck-asserts the
conversion succeeds.

## Related env vars

| Env var | Purpose |
|---|---|
| `HIPDNN_EP_BYTECODE_DUMP_PATH=<file>` | Force level-1 pass to dump imported MLIR bytecode to this file (Phase 1 entry point). Fires even without `MORPHIZEN_DEBUG_MLIR_BACKEND`. |
| `HIPDNN_EP_IR_DUMP_PATH=<file>` | Enable `pm.enableIRPrinting` inside hip-compiler — prints IR after each pass to the given file. Use to pinpoint the failing pass once you have the bytecode. |
| `HIPDNN_EP_ALLOW_CPU_FALLBACK=1` | Allow the EP to silently return on compile failure (legacy behavior). The bytecode dump still fires before the compile attempt. |
| `MORPHIZEN_DEBUG_MLIR_BACKEND=2` | Dump bytecode under `<dump_dir>/mlir_bytecode_dump.mlir` (default `<temp>/morphizen_dumps/<cache_key>`). |
| `MORPHIZEN_DEBUG_MLIR_BACKEND=3` | All of the above plus verbose level-1/CustomOp logging. |
