// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Converts onnx.MatMul to hipsr.matmul. The shape region is left empty (zero
// blocks); a later dedicated pass fills in the shape computation (and the K /
// batch shape checks), so this stage does not populate it.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// 2-D x 2-D.
// CHECK-LABEL: func.func @matmul_2d
func.func @matmul_2d(%a: tensor<?x4096xf16>, %b: tensor<4096x1024xf16>)
    -> tensor<?x1024xf16> {
  // The DPS init is a hipsr.placeholder that mirrors the result type, so no
  // output-shape computation (tensor.empty / tensor.dim) is emitted here.
  // CHECK: %[[INIT:.+]] = hipsr.placeholder : tensor<?x1024xf16>
  // The shape region is left empty (zero blocks) here; a later pass populates
  // it. The optional region group prints nothing when the region is empty.
  // CHECK: hipsr.matmul ins(%[[A:.+]], %[[B:.+]] : tensor<?x4096xf16>, tensor<4096x1024xf16>) outs(%[[INIT]] : tensor<?x1024xf16>) -> tensor<?x1024xf16>
  // CHECK-NOT: tensor.empty
  // CHECK-NOT: tensor.dim
  // CHECK-NOT: shape_region
  %0 = "onnx.MatMul"(%a, %b) : (tensor<?x4096xf16>, tensor<4096x1024xf16>)
      -> tensor<?x1024xf16>
  return %0 : tensor<?x1024xf16>
}

// -----

// Batched 3-D x 2-D (broadcast weight): batch + M from A, N from B's last dim.
// CHECK-LABEL: func.func @matmul_batched
func.func @matmul_batched(%a: tensor<2x?x4096xf16>, %b: tensor<4096x1024xf16>)
    -> tensor<2x?x1024xf16> {
  // CHECK: %[[INIT:.+]] = hipsr.placeholder : tensor<2x?x1024xf16>
  // CHECK: hipsr.matmul ins(%[[A:.+]], %[[B:.+]] : tensor<2x?x4096xf16>, tensor<4096x1024xf16>) outs(%[[INIT]] : tensor<2x?x1024xf16>) -> tensor<2x?x1024xf16>
  // CHECK-NOT: shape_region
  %0 = "onnx.MatMul"(%a, %b) : (tensor<2x?x4096xf16>, tensor<4096x1024xf16>)
      -> tensor<2x?x1024xf16>
  return %0 : tensor<2x?x1024xf16>
}

// -----

// 4-D x 4-D equally-batched: the conversion just mirrors the result type into
// the placeholder init and leaves the region empty, so rank does not matter.
// CHECK-LABEL: func.func @matmul_4d
func.func @matmul_4d(%a: tensor<2x3x?x4096xf16>, %b: tensor<2x3x4096x1024xf16>)
    -> tensor<2x3x?x1024xf16> {
  // CHECK: %[[INIT:.+]] = hipsr.placeholder : tensor<2x3x?x1024xf16>
  // CHECK: hipsr.matmul ins(%[[A:.+]], %[[B:.+]] : tensor<2x3x?x4096xf16>, tensor<2x3x4096x1024xf16>) outs(%[[INIT]] : tensor<2x3x?x1024xf16>) -> tensor<2x3x?x1024xf16>
  // CHECK-NOT: tensor.empty
  // CHECK-NOT: tensor.dim
  // CHECK-NOT: shape_region
  %0 = "onnx.MatMul"(%a, %b) : (tensor<2x3x?x4096xf16>, tensor<2x3x4096x1024xf16>)
      -> tensor<2x3x?x1024xf16>
  return %0 : tensor<2x3x?x1024xf16>
}

// -----

// 4-D x 2-D broadcast weight: leading batch dims + M from A, N from B's last
// dim. Fully static A/B shapes -> fully static result placeholder.
// CHECK-LABEL: func.func @matmul_4d_by_2d
func.func @matmul_4d_by_2d(%a: tensor<2x3x64x4096xf16>, %b: tensor<4096x1024xf16>)
    -> tensor<2x3x64x1024xf16> {
  // CHECK: %[[INIT:.+]] = hipsr.placeholder : tensor<2x3x64x1024xf16>
  // CHECK: hipsr.matmul ins(%[[A:.+]], %[[B:.+]] : tensor<2x3x64x4096xf16>, tensor<4096x1024xf16>) outs(%[[INIT]] : tensor<2x3x64x1024xf16>) -> tensor<2x3x64x1024xf16>
  // CHECK-NOT: shape_region
  %0 = "onnx.MatMul"(%a, %b) : (tensor<2x3x64x4096xf16>, tensor<4096x1024xf16>)
      -> tensor<2x3x64x1024xf16>
  return %0 : tensor<2x3x64x1024xf16>
}

// -----

// 1-D B (ONNX/NumPy): (M,K) @ (K) -> (M). The conversion mirrors the 1-D ONNX
// result type into the placeholder; populateShapeRegion later handles the
// promote-and-strip.
// CHECK-LABEL: func.func @matmul_1d_rhs
func.func @matmul_1d_rhs(%a: tensor<?x4096xf16>, %b: tensor<4096xf16>)
    -> tensor<?xf16> {
  // CHECK: %[[INIT:.+]] = hipsr.placeholder : tensor<?xf16>
  // CHECK: hipsr.matmul ins(%[[A:.+]], %[[B:.+]] : tensor<?x4096xf16>, tensor<4096xf16>) outs(%[[INIT]] : tensor<?xf16>) -> tensor<?xf16>
  // CHECK-NOT: shape_region
  %0 = "onnx.MatMul"(%a, %b) : (tensor<?x4096xf16>, tensor<4096xf16>)
      -> tensor<?xf16>
  return %0 : tensor<?xf16>
}

// -----

// NumPy batch broadcast (A batch dim 1, B batch dim 8 -> 8). The conversion is
// shape-agnostic; the broadcast is resolved in the result type it mirrors.
// CHECK-LABEL: func.func @matmul_broadcast_batch
func.func @matmul_broadcast_batch(%a: tensor<1x64x4096xf16>,
                                  %b: tensor<8x4096x1024xf16>)
    -> tensor<8x64x1024xf16> {
  // CHECK: %[[INIT:.+]] = hipsr.placeholder : tensor<8x64x1024xf16>
  // CHECK: hipsr.matmul ins(%[[A:.+]], %[[B:.+]] : tensor<1x64x4096xf16>, tensor<8x4096x1024xf16>) outs(%[[INIT]] : tensor<8x64x1024xf16>) -> tensor<8x64x1024xf16>
  // CHECK-NOT: shape_region
  %0 = "onnx.MatMul"(%a, %b) : (tensor<1x64x4096xf16>, tensor<8x4096x1024xf16>)
      -> tensor<8x64x1024xf16>
  return %0 : tensor<8x64x1024xf16>
}
