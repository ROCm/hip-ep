// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Converts onnx.Cast to hipsr.cast. The shape region is left empty (zero
// blocks); a later dedicated pass fills in the shape computation, so this stage
// does not populate it.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s -allow-unregistered-dialect -convert-onnx-to-hipsr | FileCheck %s

// CHECK-LABEL: func.func @cast
func.func @cast(%input: tensor<?x8xf32>) -> tensor<?x8xf16> {
  // CHECK: %[[D0:.+]] = tensor.dim %[[IN:.+]], %{{.+}} : tensor<?x8xf32>
  // CHECK: %[[INIT:.+]] = tensor.empty(%[[D0]]) : tensor<?x8xf16>
  // The shape region is left empty (zero blocks) here; a later pass populates
  // it. The optional region group prints nothing when the region is empty.
  // CHECK: hipsr.cast ins(%[[IN]] : tensor<?x8xf32>) outs(%[[INIT]] : tensor<?x8xf16>) -> tensor<?x8xf16>
  // CHECK-NOT: shape_region
  %0 = "onnx.Cast"(%input) : (tensor<?x8xf32>) -> tensor<?x8xf16>
  return %0 : tensor<?x8xf16>
}
