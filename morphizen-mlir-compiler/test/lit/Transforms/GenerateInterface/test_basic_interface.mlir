// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify GenerateInterface pass creates ONNX Runtime integration functions
// (inference_init, inference_compute, inference_cleanup).
//
// This pass runs at the END of compilation after:
// 1. ONNX → HIP conversion
// 2. Memory pooling analysis
// 3. HIP → LLVM lowering
// 4. Metadata generation
//
// It generates three C-ABI functions for ONNX Runtime:
// - int inference_init(void** out_state)
// - int inference_compute(void* state, span_t* inputs, span_t* outputs)
// - int inference_cleanup(void* state)
//
// Prerequisites:
// - llvm.func @main_graph(ptr, ptr, ptr) -> i32 (lowered main function)
// - llvm.func @get_constant_registry() -> ptr
// - Module attributes: hipdnn.input_count, hipdnn.input_ranks, etc.
//
// This is a complex end-to-end pass that's better tested via integration
// tests (test/e2e/) rather than isolated LIT tests. This test verifies
// the pass loads correctly and checks prerequisites.
// ============================================================================

// RUN: not hip-opt %s --generate-interface 2>&1 | FileCheck %s

module {
  // Minimal module that will fail prerequisite checks
  // This verifies the pass runs and checks for required symbols

  func.func @dummy() {
    return
  }

  // CHECK: GenerateInterface
  // CHECK: @main_graph (llvm.func) not found
}
