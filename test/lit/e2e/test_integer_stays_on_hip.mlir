// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Wide i32 matmul and convolution stay in main_graph. The rocMLIR outline
// pass leaves them there, so they lower through the exact HIP kernels
// instead of a non-quantized integer tosa op.

// RUN: hip-mlir-opt %s --rocmlir-pipeline | FileCheck %s

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.matmul
// CHECK: hip.conv
// CHECK-NOT: tosa.matmul
// CHECK-NOT: tosa.conv2d
// CHECK-NOT: rock.kernel
module {
  func.func @main_graph(
      %ctx: !hip.context,
      %a: tensor<1x259xi32>,
      %b: tensor<259x1xi32>,
      %mm_init: tensor<1x1xi32>,
      %x: tensor<1x259x1x1xi32>,
      %w: tensor<1x259x1x1xi32>,
      %conv_init: tensor<1x1x1x1xi32>) -> (tensor<1x1xi32>, tensor<1x1x1x1xi32>) {
    %mm = hip.matmul(%ctx) ins(%a, %b : tensor<1x259xi32>, tensor<259x1xi32>)
        outs(%mm_init : tensor<1x1xi32>)
        {transA = 0 : i64, transB = 0 : i64} : tensor<1x1xi32>
    %conv = hip.conv(%ctx) ins(
        %x, %w : tensor<1x259x1x1xi32>, tensor<1x259x1x1xi32>)
        outs(%conv_init : tensor<1x1x1x1xi32>)
        {dilations = [1, 1], group = 1 : i64, kernel_shape = [1, 1],
         pads = [0, 0, 0, 0], strides = [1, 1]} : tensor<1x1x1x1xi32>
    return %mm, %conv : tensor<1x1xi32>, tensor<1x1x1x1xi32>
  }
}
