// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-infer-shapes --split-input-file --verify-diagnostics %s
// RUN: hip-mlir-opt --hip-infer-shapes --split-input-file --verify-diagnostics %s | FileCheck %s

// The tool-only hip.test_malformed_reify fixture deliberately returns
// successful but contract-invalid shape lists. These cases must fail through
// explicit checks in Release builds; assertions are not part of the oracle.

func.func @wrong_result_count() {
  // expected-error @+1 {{'hip.test_malformed_reify' op --hip-infer-shapes: successful reification returned 0 shape vector(s) for 1 result(s); expected exactly one vector per result}}
  %0 = "hip.test_malformed_reify"() {kind = "result_count"} : () -> tensor<?xf32>
  return
}

// -----

func.func @wrong_rank() {
  // expected-error @+1 {{'hip.test_malformed_reify' op --hip-infer-shapes: successful reification returned 1 dimension(s) for result #0 of rank 2}}
  %0 = "hip.test_malformed_reify"() {kind = "rank"} : () -> tensor<?x?xf32>
  return
}

// -----

func.func @static_contradiction() {
  // expected-error @+1 {{'hip.test_malformed_reify' op --hip-infer-shapes: successful reification returned extent 3 for result #0, dimension #0, contradicting existing static extent 2}}
  %0 = "hip.test_malformed_reify"() {kind = "static_contradiction"} : () -> tensor<2x?xf32>
  return
}

// -----

func.func @negative_extent() {
  // expected-error @+1 {{'hip.test_malformed_reify' op --hip-infer-shapes: successful reification returned negative extent -2 for result #0, dimension #0}}
  %0 = "hip.test_malformed_reify"() {kind = "negative_extent"} : () -> tensor<?xf32>
  return
}

// -----

// CHECK-LABEL: func.func @mixed_non_shaped_results
// CHECK: "hip.test_malformed_reify"() {kind = "mixed_results"} : () -> (tensor<2x4xf32>, i32, !hip.loop_frame)
func.func @mixed_non_shaped_results() {
  %tensor, %status, %frame = "hip.test_malformed_reify"()
      {kind = "mixed_results"} : () -> (tensor<?x?xf32>, i32, !hip.loop_frame)
  return
}
