// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --convert-hip-to-tosa --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @cumsum_inclusive
// CHECK-COUNT-4: tosa.reduce_sum
// CHECK: tosa.concat
// CHECK-NOT: hip.cumsum
func.func @cumsum_inclusive(%ctx: !hip.context, %x: tensor<2x4xf32>,
                            %init: tensor<2x4xf32>) -> tensor<2x4xf32>
    attributes {rock.kernel} {
  %axis = arith.constant dense<1> : tensor<i64>
  %r = hip.cumsum(%ctx)
      ins(%x, %axis : tensor<2x4xf32>, tensor<i64>)
      outs(%init : tensor<2x4xf32>) : tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: func.func @cumsum_exclusive_reverse
// CHECK: tosa.reverse
// CHECK: tosa.const
// CHECK-COUNT-3: tosa.reduce_sum
// CHECK: tosa.concat
// CHECK: tosa.reverse
// CHECK-NOT: hip.cumsum
func.func @cumsum_exclusive_reverse(
    %ctx: !hip.context, %x: tensor<2x4xf32>,
    %init: tensor<2x4xf32>) -> tensor<2x4xf32>
    attributes {rock.kernel} {
  %axis = arith.constant dense<-1> : tensor<i64>
  %r = hip.cumsum(%ctx)
      ins(%x, %axis : tensor<2x4xf32>, tensor<i64>)
      outs(%init : tensor<2x4xf32>)
      {exclusive = 1 : i64, reverse = 1 : i64} : tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// -----

// A dynamic axis remains legal and uses the custom HIP path.
// CHECK-LABEL: func.func @cumsum_dynamic_axis
// CHECK: hip.cumsum
func.func @cumsum_dynamic_axis(
    %ctx: !hip.context, %x: tensor<2x4xf32>, %axis: tensor<i64>,
    %init: tensor<2x4xf32>) -> tensor<2x4xf32>
    attributes {rock.kernel} {
  %r = hip.cumsum(%ctx)
      ins(%x, %axis : tensor<2x4xf32>, tensor<i64>)
      outs(%init : tensor<2x4xf32>) : tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}
