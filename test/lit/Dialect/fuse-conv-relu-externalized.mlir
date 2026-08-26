// RUN: hip-mlir-opt --hip-fuse-conv-relu6 %s | FileCheck %s

// ReLU clip constants are externalized before fuse-conv-relu6 runs in the EP
// pipeline. The pass must read splat metadata, not just inline hip.constant.

module attributes {
  hipdnn.constant_source_kinds = array<i32: 1, 1>,
  hipdnn.constant_splat_elem_values = array<i64: 0, 17920>,
  hipdnn.constant_splat_elem_sizes = array<i64: 2, 2>
} {
  memref.global "private" @hip_ext_constant_0 : memref<f16> {hip.external_data = {index = 0 : i64, offset = 0 : i64, size = 2 : i64}}
  memref.global "private" @hip_ext_constant_1 : memref<f16> {hip.external_data = {index = 1 : i64, offset = 64 : i64, size = 2 : i64}}

  func.func @relu(%ctx: !hip.context, %arg0: tensor<1x3x4x4xf16>, %arg1: tensor<8x3x3x3xf16>, %arg2: tensor<8xf16>) -> tensor<1x8x4x4xf16> {
    %c0_mem = memref.get_global @hip_ext_constant_0 : memref<f16>
    %c0 = bufferization.to_tensor %c0_mem restrict : memref<f16> to tensor<f16>
    %empty_conv = tensor.empty() : tensor<1x8x4x4xf16>
    %conv = hip.conv(%ctx) ins(%arg0, %arg1, %arg2 : tensor<1x3x4x4xf16>, tensor<8x3x3x3xf16>, tensor<8xf16>) outs(%empty_conv : tensor<1x8x4x4xf16>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x8x4x4xf16>
    %empty_out = tensor.empty() : tensor<1x8x4x4xf16>
    %max = hip.max(%ctx) ins(%conv, %c0 : tensor<1x8x4x4xf16>, tensor<f16>) outs(%empty_out : tensor<1x8x4x4xf16>) : tensor<1x8x4x4xf16>
    return %max : tensor<1x8x4x4xf16>
  }

  func.func @relu6(%ctx: !hip.context, %arg0: tensor<1x3x4x4xf16>, %arg1: tensor<8x3x3x3xf16>, %arg2: tensor<8xf16>) -> tensor<1x8x4x4xf16> {
    %c0_mem = memref.get_global @hip_ext_constant_0 : memref<f16>
    %c0 = bufferization.to_tensor %c0_mem restrict : memref<f16> to tensor<f16>
    %c6_mem = memref.get_global @hip_ext_constant_1 : memref<f16>
    %c6 = bufferization.to_tensor %c6_mem restrict : memref<f16> to tensor<f16>
    %empty_conv = tensor.empty() : tensor<1x8x4x4xf16>
    %conv = hip.conv(%ctx) ins(%arg0, %arg1, %arg2 : tensor<1x3x4x4xf16>, tensor<8x3x3x3xf16>, tensor<8xf16>) outs(%empty_conv : tensor<1x8x4x4xf16>) {dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : tensor<1x8x4x4xf16>
    %empty_max = tensor.empty() : tensor<1x8x4x4xf16>
    %max = hip.max(%ctx) ins(%conv, %c0 : tensor<1x8x4x4xf16>, tensor<f16>) outs(%empty_max : tensor<1x8x4x4xf16>) : tensor<1x8x4x4xf16>
    %empty_out = tensor.empty() : tensor<1x8x4x4xf16>
    %min = hip.min(%ctx) ins(%max, %c6 : tensor<1x8x4x4xf16>, tensor<f16>) outs(%empty_out : tensor<1x8x4x4xf16>) : tensor<1x8x4x4xf16>
    return %min : tensor<1x8x4x4xf16>
  }
}

// CHECK-LABEL: func.func @relu
// CHECK-NOT: hip.max
// CHECK: hip.conv{{.*}}fused_activation = true

// CHECK-LABEL: func.func @relu6
// CHECK-NOT: hip.max
// CHECK-NOT: hip.min
// CHECK: hip.conv{{.*}}activation_clip_hi = 6.000000e+00{{.*}}fused_activation = true
