// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.reduce_mean is the same rock.reduce kernel as a sum. HipToTosa expands
// it to a multiply by 1/N in front of tosa.reduce_sum.

// RUN: hip-mlir-opt --hip-rocmlir-prepare-quant \
// RUN:   --hip-fuse-rocmlir-elementwise --hip-fuse-rocmlir-anchor %s \
// RUN:   | hip-mlir-opt --convert-hip-to-tosa | FileCheck %s

// CHECK-LABEL: func.func @mlir_reduce_mean_0
// CHECK-SAME: attributes {rock.kernel, rock.reduce}
// CHECK: tosa.mul
// CHECK: tosa.reduce_sum
// CHECK-LABEL: func.func @main_graph
// CHECK: hip.rocmlir
// CHECK-NOT: hip.reduce_mean

func.func @main_graph(%ctx: !hip.context, %data: tensor<8x16xf32>) -> tensor<8x1xf32> {
  %axes = hip.constant {value = dense<1> : tensor<i64>} : tensor<i64>
  %empty = tensor.empty() : tensor<8x1xf32>
  %mean = hip.reduce_mean(%ctx) ins(%data, %axes : tensor<8x16xf32>, tensor<i64>)
      outs(%empty : tensor<8x1xf32>)
      {keepdims = 1 : i64, noop_with_empty_axes = 0 : i64, reduced_axes = array<i64: 1>}
      : tensor<8x1xf32>
  return %mean : tensor<8x1xf32>
}
