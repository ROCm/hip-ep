// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify foldGatherShapeBeforeLowering (lib/Conversion/OnnxToHip/
// GatherShapeFold.cpp) recognises the
//
//   onnx.Gather(onnx.Shape(x), onnx.Constant{value=dense<[k]>})
//
// idiom and rewrites it to
//
//   tensor.dim %x, %k -> arith.index_cast -> tensor.from_elements
//
// while leaving non-matching variants for the general Gather lowering.
//
// CHECK patterns are deliberately structural (no SSA-name captures) because
// canonicalization runs inline inside --convert-onnx-to-hip and reorders /
// renames constants. The key assertions are:
//   - matched cases: `onnx.Gather` / `onnx.Shape` are gone, tensor.dim or
//     a static i64 constant feeds a tensor.from_elements;
//   - unmatched cases: a `hip.gather` lowered from the general pattern
//     still references the runtime index (i.e. the fold did NOT fire).
//
// Coverage:
//   1. Scalar Gather, positive k                 -> folds (dynamic dim)
//   2. 1-D Gather (single elem 1xi64), pos k     -> folds (dynamic dim)
//   3. Negative k normalized via SUB-RANGE len   -> folds (static dim)
//   4. start/end on Shape with positive k        -> folds (static dim)
//   5. Non-constant Gather index                 -> NO fold
//   6. Multi-element Gather index                -> NO fold
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  // Placeholder @main_graph -- required by the metadata-generation step
  // inside --convert-onnx-to-hip.
  func.func @main_graph(%arg0: tensor<?x16xf32>) -> tensor<?x16xf32> {
    return %arg0 : tensor<?x16xf32>
  }

  // --- Case 1: scalar Gather, positive k, dynamic source dim ---
  // CHECK-LABEL: func.func @gather_shape_scalar_pos
  // CHECK-NOT:   onnx.Gather
  // CHECK-NOT:   onnx.Shape
  // CHECK-NOT:   hip.gather
  // CHECK:       tensor.dim {{.*}} : tensor<?x16x32xf32>
  // CHECK:       arith.index_cast {{.*}} : index to i64
  // CHECK:       tensor.from_elements {{.*}} : tensor<i64>
  func.func @gather_shape_scalar_pos(%x: tensor<?x16x32xf32>) -> tensor<i64> {
    %shape = "onnx.Shape"(%x) : (tensor<?x16x32xf32>) -> tensor<3xi64>
    %idx = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %r = "onnx.Gather"(%shape, %idx) {axis = 0 : si64}
       : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    return %r : tensor<i64>
  }

  // --- Case 2: 1-D Gather, single-element index (1xi64 -> 1xi64) ---
  // CHECK-LABEL: func.func @gather_shape_1d_pos
  // CHECK-NOT:   onnx.Gather
  // CHECK-NOT:   onnx.Shape
  // CHECK-NOT:   hip.gather
  // CHECK:       tensor.dim {{.*}} : tensor<?x16xf32>
  // CHECK:       arith.index_cast {{.*}} : index to i64
  // CHECK:       tensor.from_elements {{.*}} : tensor<1xi64>
  func.func @gather_shape_1d_pos(%x: tensor<?x16xf32>) -> tensor<1xi64> {
    %shape = "onnx.Shape"(%x) : (tensor<?x16xf32>) -> tensor<2xi64>
    %idx = "onnx.Constant"() {value = dense<[0]> : tensor<1xi64>}
         : () -> tensor<1xi64>
    %r = "onnx.Gather"(%shape, %idx) {axis = 0 : si64}
       : (tensor<2xi64>, tensor<1xi64>) -> tensor<1xi64>
    return %r : tensor<1xi64>
  }

  // --- Case 3: negative k normalized via SUB-RANGE length ---
  // full range (start=0, end=rank=3), L = 3, k = -1 -> k = 2 -> dim[2] = 32 (static).
  // The fold emits `from_elements(arith.constant 32 : i64)` which the
  // upstream from-elements-of-constants canonicalization further folds
  // into a single `arith.constant dense<32> : tensor<i64>`.
  // CHECK-LABEL: func.func @gather_shape_negative_idx
  // CHECK-NOT:   onnx.Gather
  // CHECK-NOT:   onnx.Shape
  // CHECK-NOT:   hip.gather
  // CHECK:       arith.constant dense<32> : tensor<i64>
  func.func @gather_shape_negative_idx(%x: tensor<?x16x32xf32>) -> tensor<i64> {
    %shape = "onnx.Shape"(%x) : (tensor<?x16x32xf32>) -> tensor<3xi64>
    %idx = "onnx.Constant"() {value = dense<-1> : tensor<i64>}
         : () -> tensor<i64>
    %r = "onnx.Gather"(%shape, %idx) {axis = 0 : si64}
       : (tensor<3xi64>, tensor<i64>) -> tensor<i64>
    return %r : tensor<i64>
  }

  // --- Case 4: Shape with start/end, k references the sub-range ---
  // start=1, end=3 on rank-4 input picks dims (3, 32); k=1 -> absDim = start+1 = 2 -> 32.
  // Same canonicalization collapse as Case 3 -> single dense<32> constant.
  // CHECK-LABEL: func.func @gather_shape_with_start_end
  // CHECK-NOT:   onnx.Gather
  // CHECK-NOT:   onnx.Shape
  // CHECK-NOT:   hip.gather
  // CHECK:       arith.constant dense<32> : tensor<i64>
  func.func @gather_shape_with_start_end(%x: tensor<2x3x32x?xf32>) -> tensor<i64> {
    %shape = "onnx.Shape"(%x) {start = 1 : si64, end = 3 : si64}
           : (tensor<2x3x32x?xf32>) -> tensor<2xi64>
    %idx = "onnx.Constant"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %r = "onnx.Gather"(%shape, %idx) {axis = 0 : si64}
       : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
    return %r : tensor<i64>
  }

  // --- Case 5: non-constant index -> NO fold ---
  // The fold requires an onnx.Constant index. With a runtime index, the
  // Gather falls through to the general lowering and becomes hip.gather.
  // CHECK-LABEL: func.func @gather_shape_non_const_idx
  // CHECK:       hip.gather
  func.func @gather_shape_non_const_idx(%x: tensor<?x16xf32>, %idx: tensor<i64>) -> tensor<i64> {
    %shape = "onnx.Shape"(%x) : (tensor<?x16xf32>) -> tensor<2xi64>
    %r = "onnx.Gather"(%shape, %idx) {axis = 0 : si64}
       : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
    return %r : tensor<i64>
  }

  // --- Case 6: multi-element Gather index -> NO fold ---
  // Result is 2xi64 (not 0-D or 1xi64), so the fold's shape check rejects.
  // The general Gather pattern handles it.
  // CHECK-LABEL: func.func @gather_shape_multi_elem_idx
  // CHECK:       hip.gather
  func.func @gather_shape_multi_elem_idx(%x: tensor<?x16x32xf32>) -> tensor<2xi64> {
    %shape = "onnx.Shape"(%x) : (tensor<?x16x32xf32>) -> tensor<3xi64>
    %idx = "onnx.Constant"() {value = dense<[0, 2]> : tensor<2xi64>}
         : () -> tensor<2xi64>
    %r = "onnx.Gather"(%shape, %idx) {axis = 0 : si64}
       : (tensor<3xi64>, tensor<2xi64>) -> tensor<2xi64>
    return %r : tensor<2xi64>
  }
}
