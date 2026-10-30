// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --hip-infer-shapes --split-input-file --verify-diagnostics %s | FileCheck %s

// --hip-infer-shapes refines operations of a dialect that registers
// ShapeInferenceDialectInterface, and only those that are destination-style
// (the pass rebuilds the destination together with the result). The
// `test_shape_*` dialects are tool-only fixtures: `<dialect>.like(%x, %init)`
// is destination-style and reifies its result to `%x`'s shape.

// A non-HIP dialect that opts in is refined like a HIP operation: the `?`
// becomes static, the tensor.empty destination is rebuilt, and the original
// type is restored for the existing user through a cast.
// CHECK-LABEL: func.func @opted_in_dialect_is_refined
// CHECK:         %[[E:.*]] = tensor.empty() : tensor<4x8xf32>
// CHECK:         %[[Y:.*]] = "test_shape_opt_in.like"(%{{.*}}, %[[E]]) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK:         %[[C:.*]] = tensor.cast %[[Y]] : tensor<4x8xf32> to tensor<?x8xf32>
// CHECK:         return %[[C]]
func.func @opted_in_dialect_is_refined(%x: tensor<4x8xf32>, %n: index)
    -> tensor<?x8xf32> {
  %e = tensor.empty(%n) : tensor<?x8xf32>
  %y = "test_shape_opt_in.like"(%x, %e)
      : (tensor<4x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>
  return %y : tensor<?x8xf32>
}

// -----

// The same operation in a dialect that does not opt in is left alone.
// CHECK-LABEL: func.func @unmarked_dialect_is_skipped
// CHECK:         %[[E:.*]] = tensor.empty(%{{.*}}) : tensor<?x8xf32>
// CHECK:         %[[Y:.*]] = "test_shape_unmarked.like"(%{{.*}}, %[[E]]) : (tensor<4x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>
// CHECK-NOT:     tensor.cast
// CHECK:         return %[[Y]]
func.func @unmarked_dialect_is_skipped(%x: tensor<4x8xf32>, %n: index)
    -> tensor<?x8xf32> {
  %e = tensor.empty(%n) : tensor<?x8xf32>
  %y = "test_shape_unmarked.like"(%x, %e)
      : (tensor<4x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>
  return %y : tensor<?x8xf32>
}

// -----

// An operation of an opted-in dialect that reifies but is not
// destination-style is skipped. hip.test_malformed_reify reports a malformed
// shape whenever it is reified, so the absence of a diagnostic shows that the
// pass never visited it.
// CHECK-LABEL: func.func @non_destination_style_is_skipped
// CHECK:         "hip.test_malformed_reify"() {kind = "rank"} : () -> tensor<?x?xf32>
func.func @non_destination_style_is_skipped() -> tensor<?x?xf32> {
  %0 = "hip.test_malformed_reify"() {kind = "rank"} : () -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
