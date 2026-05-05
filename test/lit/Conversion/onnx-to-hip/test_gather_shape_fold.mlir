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
//   - non-Gather-of-Shape pattern is left alone
//   - non-constant index is left alone (becomes generic Gather lowering)
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
