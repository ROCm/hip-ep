// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Round-trips hipsr.matmul: a Regular shaped DPS op carrying a shape region
// that computes its output shape [M, N] from the input shapes.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s | hip-mlir-opt | FileCheck %s

// CHECK-LABEL: func.func @matmul
func.func @matmul(%lhs: tensor<?x256xf16>, %rhs: tensor<256x512xf16>,
                  %init: tensor<?x512xf16>) -> tensor<?x512xf16> {
  // CHECK: hipsr.matmul ins(%{{.+}}, %{{.+}} : tensor<?x256xf16>, tensor<256x512xf16>) outs(%{{.+}} : tensor<?x512xf16>) -> tensor<?x512xf16> shape_region {
  %0 = hipsr.matmul ins(%lhs, %rhs : tensor<?x256xf16>, tensor<256x512xf16>)
                    outs(%init : tensor<?x512xf16>) -> tensor<?x512xf16>
                    shape_region {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = tensor.dim %lhs, %c0 : tensor<?x256xf16>
    %n = tensor.dim %rhs, %c1 : tensor<256x512xf16>
    // CHECK: hipsr.shape_yield %{{.+}}, %{{.+}}
    hipsr.shape_yield %m, %n
  }
  return %0 : tensor<?x512xf16>
}
