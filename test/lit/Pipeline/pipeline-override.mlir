// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline test: the HIPDNN_EP_PIPELINE override hook in CompilerDriver.
//
// When HIPDNN_EP_PIPELINE is set, the driver skips the built-in
// ONNX->HIP->LLVM pipeline and instead parses the env value as a textual pass
// pipeline, resolving names from MLIR's global registry (populated by
// registerAllPasses()). The three RUN lines below cover the behaviours a
// downstream relies on:
//
//   1. parse failure   -- an unknown pass name is rejected up front with a
//      "failed to parse pipeline" diagnostic (no partial compile).
//   2. missing entry    -- a pipeline that parses and runs but drops
//      generate-interface leaves the module without the C-ABI
//      `inference_compute` symbol the rest of the EP consumes; the driver
//      hard-fails here rather than emitting an unusable .bc.
//   3. full registry    -- a production pass outside the historical minimal
//      register set resolves (rather than failing to parse), proving the EP
//      path registers the complete pass set, not just the pipeline-builder
//      subset.
//
// All three fail before any LLVM codegen / linking, so this test needs no GPU
// or ROCm toolchain. See lib/Compiler/CompilerDriver.cpp::runMLIRPasses,
// include/hip/InitAllPasses.h, and docs/pipeline_pass_menu.md.
//===----------------------------------------------------------------------===//

// An unknown pass name in the override is rejected with a parse diagnostic.
// RUN: env HIPDNN_EP_PIPELINE="builtin.module(func.func(this-pass-does-not-exist))" \
// RUN:   not hip-compiler %s -o %t.bc 2>&1 \
// RUN:   | FileCheck --check-prefix=PARSE %s

// PARSE: HIPDNN_EP_PIPELINE: failed to parse pipeline

// A pipeline that parses and runs but never emits the C-ABI entry point is
// caught by the post-run guardrail (here: a no-op canonicalize over the
// function, which leaves no inference_compute symbol).
// RUN: env HIPDNN_EP_PIPELINE="builtin.module(func.func(canonicalize))" \
// RUN:   not hip-compiler %s -o %t.bc 2>&1 \
// RUN:   | FileCheck --check-prefix=GUARD %s

// GUARD: did not produce an 'inference_compute'

// A production HIP pass that is NOT in the historical minimal register set
// still resolves in the override -- the EP/hip-compiler driver registers the
// full production pass set (see InitAllPasses.h::registerAllPasses). It runs
// (no-op on this trivial module) and reaches the same entry-point guardrail
// rather than failing to parse. Regression guard: if the driver ever narrows
// the registered set, this case flips from the guardrail diagnostic to a
// "failed to parse pipeline" diagnostic.
// RUN: env HIPDNN_EP_PIPELINE="builtin.module(hip-infer-shapes)" \
// RUN:   not hip-compiler %s -o %t.bc 2>&1 \
// RUN:   | FileCheck --check-prefix=REGISTERED %s

// REGISTERED-NOT: failed to parse pipeline
// REGISTERED: did not produce an 'inference_compute'

func.func @f(%arg0: i32) -> i32 {
  return %arg0 : i32
}
