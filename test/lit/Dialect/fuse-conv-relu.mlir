// RUN: hip-mlir-opt --hip-fuse-conv-relu6 %s | FileCheck %s

module {
  func.func @main_graph(%ctx: !hip.context, %arg0: tensor<1x3x4x4xf16>, %arg1: tensor<8x3x3x3xf16>, %arg2: tensor<8xf16>) -> tensor<1x8x4x4xf16> {
    %c0 = arith.constant dense<0.0> : tensor<f16>
    %empty_conv = tensor.empty() : tensor<1x8x4x4xf16>
    %conv = hip.conv(%ctx) ins(%arg0, %arg1, %arg2 : tensor<1x3x4x4xf16>, tensor<8x3x3x3xf16>, tensor<8xf16>) outs(%empty_conv : tensor<1x8x4x4xf16>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x8x4x4xf16>
    %empty_out = tensor.empty() : tensor<1x8x4x4xf16>
    %max = hip.max(%ctx) ins(%conv, %c0 : tensor<1x8x4x4xf16>, tensor<f16>) outs(%empty_out : tensor<1x8x4x4xf16>) : tensor<1x8x4x4xf16>
    return %max : tensor<1x8x4x4xf16>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK-NOT: hip.max
// CHECK: hip.conv{{.*}}fused_activation = true
// CHECK: return
