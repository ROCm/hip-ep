// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Converts onnx.MatMul to hipsr.matmul. The shape region is left empty (zero
// blocks); a later dedicated pass fills in the shape computation, so this stage
// does not populate it.
//
// The conversion is shape-agnostic: it mirrors the result type into a
// hipsr.placeholder init and leaves the region empty, with no rank/shape
// branching. So a few representative shapes suffice: a 2-D case, a 1-D-operand
// (rank-reducing) case, and a batched case.
//
// The positive CHECK lines were seeded with mlir/utils/generate-test-checks.py
// (full signature + named operand captures), then refined by hand: the
// CHECK-NOT intent checks (no tensor.empty/tensor.dim/shape_region) were
// re-added and the return/terminator boilerplate dropped.
//
// Each case is one module (split by `// -----`); all CHECK directives are
// grouped in the header above the function, isolated by CHECK-LABEL.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// -----

// 2-D x 2-D (dynamic M): the DPS init is a hipsr.placeholder mirroring the
// result type, so no output-shape computation (tensor.empty / tensor.dim) is
// emitted, and the shape region is left empty (prints nothing).
// CHECK-LABEL: func.func @matmul_2d(
// CHECK-SAME:    %[[A:.*]]: tensor<?x4096xf16>,
// CHECK-SAME:    %[[B:.*]]: tensor<4096x1024xf16>) -> tensor<?x1024xf16> {
// CHECK:         %[[INIT:.*]] = hipsr.placeholder : tensor<?x1024xf16>
// CHECK:         hipsr.matmul ins(%[[A]], %[[B]] : tensor<?x4096xf16>, tensor<4096x1024xf16>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<?x1024xf16>) -> tensor<?x1024xf16>
// CHECK-NOT:     tensor.empty
// CHECK-NOT:     tensor.dim
// CHECK-NOT:     shape_region
func.func @matmul_2d(%a: tensor<?x4096xf16>, %b: tensor<4096x1024xf16>)
    -> tensor<?x1024xf16> {
  %0 = "onnx.MatMul"(%a, %b) : (tensor<?x4096xf16>, tensor<4096x1024xf16>)
      -> tensor<?x1024xf16>
  return %0 : tensor<?x1024xf16>
}

// -----

// 1-D B (ONNX/NumPy): (M,K) @ (K) -> (M), a rank-reducing result. The
// conversion still just mirrors the result type into the placeholder.
// CHECK-LABEL: func.func @matmul_1d_rhs(
// CHECK-SAME:    %[[A:.*]]: tensor<?x4096xf16>,
// CHECK-SAME:    %[[B:.*]]: tensor<4096xf16>) -> tensor<?xf16> {
// CHECK:         %[[INIT:.*]] = hipsr.placeholder : tensor<?xf16>
// CHECK:         hipsr.matmul ins(%[[A]], %[[B]] : tensor<?x4096xf16>, tensor<4096xf16>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<?xf16>) -> tensor<?xf16>
// CHECK-NOT:     shape_region
func.func @matmul_1d_rhs(%a: tensor<?x4096xf16>, %b: tensor<4096xf16>)
    -> tensor<?xf16> {
  %0 = "onnx.MatMul"(%a, %b) : (tensor<?x4096xf16>, tensor<4096xf16>)
      -> tensor<?xf16>
  return %0 : tensor<?xf16>
}

// -----

// Batched with NumPy batch broadcast (A batch dim 1, B batch dim 8 -> 8). The
// broadcast is resolved in the result type the conversion mirrors.
// CHECK-LABEL: func.func @matmul_broadcast_batch(
// CHECK-SAME:    %[[A:.*]]: tensor<1x64x4096xf16>,
// CHECK-SAME:    %[[B:.*]]: tensor<8x4096x1024xf16>) -> tensor<8x64x1024xf16> {
// CHECK:         %[[INIT:.*]] = hipsr.placeholder : tensor<8x64x1024xf16>
// CHECK:         hipsr.matmul ins(%[[A]], %[[B]] : tensor<1x64x4096xf16>, tensor<8x4096x1024xf16>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<8x64x1024xf16>) -> tensor<8x64x1024xf16>
// CHECK-NOT:     shape_region
func.func @matmul_broadcast_batch(%a: tensor<1x64x4096xf16>,
                                  %b: tensor<8x4096x1024xf16>)
    -> tensor<8x64x1024xf16> {
  %0 = "onnx.MatMul"(%a, %b) : (tensor<1x64x4096xf16>, tensor<8x4096x1024xf16>)
      -> tensor<8x64x1024xf16>
  return %0 : tensor<8x64x1024xf16>
}
