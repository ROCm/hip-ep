// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Round-trips hipsr.cast: a Regular shaped DPS op whose shape region copies
// the input shape (only the element type changes).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s | hip-mlir-opt | FileCheck %s

// CHECK-LABEL: func.func @cast
func.func @cast(%input: tensor<?x8xf32>, %init: tensor<?x8xf16>) -> tensor<?x8xf16> {
  // CHECK: hipsr.cast ins(%{{.+}} : tensor<?x8xf32>) outs(%{{.+}} : tensor<?x8xf16>) -> tensor<?x8xf16> shape_region {
  %0 = hipsr.cast ins(%input : tensor<?x8xf32>)
                  outs(%init : tensor<?x8xf16>) -> tensor<?x8xf16>
                  shape_region {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %d0 = tensor.dim %input, %c0 : tensor<?x8xf32>
    // CHECK: hipsr.shape_yield %{{.+}}, %{{.+}}
    hipsr.shape_yield %d0, %c8
  }
  return %0 : tensor<?x8xf16>
}
