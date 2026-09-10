// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the pre-lowering Gather(Shape(x), const_idx) fold runs BEFORE
// constants are externalized and short-circuits the Shape -> Gather chain
// directly to tensor.dim. This is the fix for the gfx1151 dynseqlen
// regression where the full memref<Nxi64> shape vector landed in pool
// memory and was written by host stores.
//
// Coverage:
//   - dynamic-dim Gather(Shape, [k]) on a 1-element index (1xi64 result)
//   - dynamic-dim Gather(Shape, scalar k) on a 0-D index (i64 result)
//   - static-dim Gather(Shape, [k]) folds to arith.constant (no tensor.dim)
//   - non-constant index is left alone (becomes generic Gather lowering)
//   - non-Gather-of-Shape pattern is left alone (becomes generic Gather)
//   - Shape with start != 0 + Gather with negative k folds correctly
//     (ONNX Gather negative-index normalization uses the sub-range length,
//     not the input rank — see GatherShapeFold.cpp comment)
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<?x?xi64>) -> tensor<?x?xi64> {
    return %arg0 : tensor<?x?xi64>
  }

  // --- Dynamic input, scalar Gather index k=1 -> tensor.dim of dim 1 ---
  func.func @test_fold_1xi64_dynamic(%data: tensor<?x?xi64>) -> tensor<1xi64> {
    %shape = "onnx.Shape"(%data) : (tensor<?x?xi64>) -> tensor<2xi64>
    %idx = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %gathered = "onnx.Gather"(%shape, %idx) {axis = 0 : si64}
        : (tensor<2xi64>, tensor<1xi64>) -> tensor<1xi64>
    return %gathered : tensor<1xi64>
  }

  // --- Dynamic input, 0-D Gather index k=1 -> tensor.dim of dim 1 ---
  func.func @test_fold_scalar_dynamic(%data: tensor<?x?xi64>) -> tensor<i64> {
    %shape = "onnx.Shape"(%data) : (tensor<?x?xi64>) -> tensor<2xi64>
    %idx = "onnx.Constant"() {value = dense<1> : tensor<i64>}
        : () -> tensor<i64>
    %gathered = "onnx.Gather"(%shape, %idx) {axis = 0 : si64}
        : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
    return %gathered : tensor<i64>
  }

  // --- Static input: should fold to arith.constant via the static branch ---
  func.func @test_fold_static(%data: tensor<2x128xi64>) -> tensor<1xi64> {
    %shape = "onnx.Shape"(%data) : (tensor<2x128xi64>) -> tensor<2xi64>
    %idx = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %gathered = "onnx.Gather"(%shape, %idx) {axis = 0 : si64}
        : (tensor<2xi64>, tensor<1xi64>) -> tensor<1xi64>
    return %gathered : tensor<1xi64>
  }

  // --- Negative case: non-constant index. The fold requires an
  //     onnx.Constant operand (getInlineScalarIndex returns nullopt
  //     otherwise), so the pattern survives unchanged and the generic
  //     Gather lowering takes over. ---
  func.func @test_no_fold_dynamic_idx(%data: tensor<?x?xi64>,
                                      %idx: tensor<1xi64>) -> tensor<1xi64> {
    %shape = "onnx.Shape"(%data) : (tensor<?x?xi64>) -> tensor<2xi64>
    %gathered = "onnx.Gather"(%shape, %idx) {axis = 0 : si64}
        : (tensor<2xi64>, tensor<1xi64>) -> tensor<1xi64>
    return %gathered : tensor<1xi64>
  }

  // --- Negative case: Gather over a non-Shape source. The fold matches
  //     ONLY when Gather's source is onnx.Shape; here the source is the
  //     function argument directly, so the fold doesn't fire and generic
  //     Gather lowering handles it. ---
  func.func @test_no_fold_non_shape_source(%data: tensor<8xi64>) -> tensor<1xi64> {
    %idx = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %gathered = "onnx.Gather"(%data, %idx) {axis = 0 : si64}
        : (tensor<8xi64>, tensor<1xi64>) -> tensor<1xi64>
    return %gathered : tensor<1xi64>
  }

  // --- Shape with start != 0 + Gather with negative k.  The fold's
  //     negative-index normalization must use the SUB-RANGE length
  //     (end - start), not the input rank, per ONNX Gather semantics on
  //     a 1-D tensor of length L (k += L if k < 0).  For rank-4 input
  //     with start=2 (default end = rank = 4), range length is 2; k=-1
  //     maps to range index 1, i.e. absolute input dim 3.
  //
  //     Pre-fix this case did NOT fold (k normalized against rank=4
  //     gave absDim=5, falling out of bounds and rejecting the fold).
  //     Generic Gather lowering produced the correct result, so the
  //     bug was a missed-optimization rather than a correctness bug,
  //     but locking it down via LIT prevents regression of the fix.
  func.func @test_fold_shape_start_negative_k(
      %data: tensor<?x?x?x?xi64>) -> tensor<1xi64> {
    %shape = "onnx.Shape"(%data) {start = 2 : si64}
        : (tensor<?x?x?x?xi64>) -> tensor<2xi64>
    %idx = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64>}
        : () -> tensor<1xi64>
    %gathered = "onnx.Gather"(%shape, %idx) {axis = 0 : si64}
        : (tensor<2xi64>, tensor<1xi64>) -> tensor<1xi64>
    return %gathered : tensor<1xi64>
  }
}

// CHECK-LABEL: func.func @test_fold_1xi64_dynamic
// CHECK-NOT: onnx.Shape
// CHECK-NOT: onnx.Gather
// CHECK-NOT: hip.gather
// CHECK: %[[D:.+]] = tensor.dim %{{.+}}, %{{.+}} : tensor<?x?xi64>
// CHECK: %[[C:.+]] = arith.index_cast %[[D]] : index to i64
// CHECK: tensor.from_elements %[[C]] : tensor<1xi64>

// CHECK-LABEL: func.func @test_fold_scalar_dynamic
// CHECK-NOT: onnx.Shape
// CHECK-NOT: onnx.Gather
// CHECK-NOT: hip.gather
// CHECK: %[[D:.+]] = tensor.dim %{{.+}}, %{{.+}} : tensor<?x?xi64>
// CHECK: %[[C:.+]] = arith.index_cast %[[D]] : index to i64
// CHECK: tensor.from_elements %[[C]] : tensor<i64>

// CHECK-LABEL: func.func @test_fold_static
// CHECK-NOT: onnx.Shape
// CHECK-NOT: onnx.Gather
// CHECK-NOT: hip.gather
// CHECK-NOT: tensor.dim
// Canonicalization folds tensor.from_elements over a constant scalar to a
// dense tensor constant; we just verify no shape/gather residue.
// CHECK: arith.constant dense<128>

// CHECK-LABEL: func.func @test_no_fold_dynamic_idx
// CHECK-NOT: onnx.Gather
// The Gather-of-Shape fold required a constant index; with a runtime index
// the fold is rejected and the generic Gather lowering (hip.gather) takes
// over. The fold-specific signature -- single tensor.dim of %data feeding
// a 1-element tensor.from_elements -- must NOT appear.
// CHECK: hip.gather

// CHECK-LABEL: func.func @test_no_fold_non_shape_source
// CHECK-NOT: onnx.Gather
// GatherShapeFold does not fire (source is not onnx.Shape), but axis-0
// Gather with a compile-time len-1 index still lowers to extract_slice.
// CHECK: tensor.extract_slice {{.*}}[1] [1] [1]
// CHECK-NOT: hip.gather

// CHECK-LABEL: func.func @test_fold_shape_start_negative_k
// CHECK-NOT: onnx.Shape
// CHECK-NOT: onnx.Gather
// CHECK-NOT: hip.gather
// Range length = end_default(=rank=4) - start(=2) = 2; k=-1 normalized
// against range length 2 gives k=1; absolute dim = start+k = 3.  The
// fold replaces the chain with tensor.dim of dim 3 wrapped by
// tensor.from_elements, with no residual Shape/Gather op.
// CHECK: %[[D:.+]] = tensor.dim %{{.+}}, %{{.+}} : tensor<?x?x?x?xi64>
// CHECK: %[[C:.+]] = arith.index_cast %[[D]] : index to i64
// CHECK: tensor.from_elements %[[C]] : tensor<1xi64>
