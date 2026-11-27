// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// `--hip-resolve-if-result-dims` reads a Reshape target held by a file-backed
// `hip.constant`: an external initializer keeps its file range rather than an
// inline `value`, and is still a constant shape, not a data-dependent one.

// RUN: split-file %s %t
// The i64 shape [6, 8, 64], little-endian.
// RUN: printf '\006\000\000\000\000\000\000\000\010\000\000\000\000\000\000\000\100\000\000\000\000\000\000\000' > %t/shape.bin
// RUN: cd %t && hip-mlir-opt file-shape.mlir --hip-resolve-if-result-dims --cse | FileCheck %s

// CHECK-LABEL: func.func @file_shape(
// CHECK-SAME: %{{.*}}: !hip.context, %[[COND:.*]]: i1, %{{.*}}: tensor<?x?xf32>)
// CHECK-DAG: %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[SIZE:.*]] = arith.select %[[COND]], %[[C6]], %[[C4]] : index
// CHECK: tensor.empty(%[[SIZE]]) : tensor<?x8x64xf32>

//--- file-shape.mlir
module {
  func.func @file_shape(%ctx: !hip.context, %c: i1, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %d0 = hip.if_result_dim(%c) captures(%x : tensor<?x?xf32>)
        then @file_then else @file_else {result_index = 0 : i32, dim = 0 : i64}
    %init = tensor.empty(%d0) : tensor<?x8x64xf32>
    %r = hip.if(%ctx, %c) outs(%init : tensor<?x8x64xf32>)
        captures(%x : tensor<?x?xf32>) -> (tensor<?x8x64xf32>)
        then @file_then else @file_else {num_outputs = 1 : i32}
    return %r : tensor<?x8x64xf32>
  }

  func.func private @file_then(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %shape = hip.constant {location = "shape.bin", offset = 0 : i64,
                           size = 24 : i64} : tensor<3xi64>
    %y = tensor.reshape %x(%shape)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }

  func.func private @file_else(%ctx: !hip.context, %x: tensor<?x?xf32>)
      -> tensor<?x8x64xf32> {
    %shape = arith.constant dense<[4, 8, 64]> : tensor<3xi64>
    %y = tensor.reshape %x(%shape)
        : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x8x64xf32>
    return %y : tensor<?x8x64xf32>
  }
}
