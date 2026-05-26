// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the `--onnx-loop-outline` pass rejects unsupported onnx.Loop
// constructs at compile time with clean, actionable diagnostics instead of
// producing IR that would later fail at codegen / runtime.
//
// This file uses `--verify-diagnostics` (not FileCheck) -- the pass is
// expected to fail and emit specific diagnostics.  Sister file
// `test_loop_outline.mlir` covers the happy paths via FileCheck.
//
// This test validates:
// - Nested onnx.Loop (an inner onnx.Loop inside another onnx.Loop's body
//   region) is rejected with a two-location diagnostic: error on the
//   outer Loop, note pointing at the inner Loop.  The HIP runtime drivers
//   share one iter/cond buffer pair per RuntimeState (see
//   runtime_state_internal.h), so nesting would race -- the outliner
//   catches this before outlining/codegen rather than letting it manifest
//   as a runtime fault.
// ============================================================================

// RUN: hip-mlir-opt --hip-add-context-arg --onnx-loop-outline --split-input-file --verify-diagnostics %s

// -----

// Nested onnx.Loop: outer body contains an inner onnx.Loop.  Both errors
// fire on a single section of the input.

module {
  func.func @nested_loop(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %M_outer = "onnx.Constant"() {value = dense<4> : tensor<i64>} : () -> tensor<i64>
    %c_outer = "onnx.Constant"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
    // expected-error @+1 {{'onnx.Loop' op nested onnx.Loop is not supported by the MorphiZen EP}}
    %v_outer = "onnx.Loop"(%M_outer, %c_outer, %arg0) ({
    ^bb0(%iter_o: tensor<i64>, %cond_in_o: tensor<i1>, %v_in_o: tensor<16xf32>):
      %M_inner = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
      %c_inner = "onnx.Constant"() {value = dense<1> : tensor<i1>} : () -> tensor<i1>
      // expected-note @+1 {{inner onnx.Loop is here}}
      %v_inner = "onnx.Loop"(%M_inner, %c_inner, %v_in_o) ({
      ^bb1(%iter_i: tensor<i64>, %cond_in_i: tensor<i1>, %v_in_i: tensor<16xf32>):
        %v_out_i = "onnx.Add"(%v_in_i, %v_in_i) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
        "onnx.Yield"(%cond_in_i, %v_out_i) : (tensor<i1>, tensor<16xf32>) -> ()
      }) : (tensor<i64>, tensor<i1>, tensor<16xf32>) -> tensor<16xf32>
      "onnx.Yield"(%cond_in_o, %v_inner) : (tensor<i1>, tensor<16xf32>) -> ()
    }) : (tensor<i64>, tensor<i1>, tensor<16xf32>) -> tensor<16xf32>
    return %v_outer : tensor<16xf32>
  }
}
