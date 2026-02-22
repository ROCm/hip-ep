// RUN: hip-opt %s -canonicalize | FileCheck %s

// Test self-copy elimination
func.func @test_self_copy(%ctx: !hip.context, %buf: memref<1x64xf32, 1>) {
  // CHECK-LABEL: @test_self_copy
  // CHECK-NOT: hip.copy
  hip.copy(%ctx, %buf, %buf) : (!hip.context, memref<1x64xf32, 1>, memref<1x64xf32, 1>)
  return
}

// Test single-use buffer copy elimination with hip.relu
func.func @test_single_use_relu(%ctx: !hip.context, %input: memref<1x3xf32, 1>, %output: memref<1x64xf32, 1>) {
  // CHECK-LABEL: @test_single_use_relu
  // CHECK: hip.alloc
  // CHECK-NEXT: hip.relu
  // CHECK-NOT: hip.copy
  %temp = hip.alloc(%ctx) : memref<1x64xf32, 1>
  hip.relu(%ctx, %input, %temp) : (!hip.context, memref<1x3xf32, 1>, memref<1x64xf32, 1>)
  hip.copy(%ctx, %temp, %output) : (!hip.context, memref<1x64xf32, 1>, memref<1x64xf32, 1>)
  return
}

// Test single-use buffer copy elimination with hip.conv
func.func @test_single_use_conv(%ctx: !hip.context,
                                 %input: memref<1x3x224x224xf32, 1>,
                                 %weights: memref<64x3x3x3xf32, 1>,
                                 %bias: memref<64xf32, 1>,
                                 %output: memref<1x64x224x224xf32, 1>) {
  // CHECK-LABEL: @test_single_use_conv
  // CHECK: hip.alloc
  // CHECK-NEXT: hip.conv
  // CHECK-NOT: hip.copy
  %temp = hip.alloc(%ctx) : memref<1x64x224x224xf32, 1>
  hip.conv(%ctx, %input, %weights, %bias, %temp) {
    kernel_shape = [3, 3], strides = [1, 1],
    pads = [1, 1, 1, 1], dilations = [1, 1], group = 1
  } : (!hip.context, memref<1x3x224x224xf32, 1>, memref<64x3x3x3xf32, 1>,
       memref<64xf32, 1>, memref<1x64x224x224xf32, 1>)
  hip.copy(%ctx, %temp, %output) : (!hip.context, memref<1x64x224x224xf32, 1>, memref<1x64x224x224xf32, 1>)
  return
}

// Test copy is NOT eliminated when buffer has multiple uses
func.func @test_multiple_uses(%ctx: !hip.context,
                               %input: memref<1x3xf32, 1>,
                               %output1: memref<1x64xf32, 1>,
                               %output2: memref<1x64xf32, 1>) {
  // CHECK-LABEL: @test_multiple_uses
  // CHECK: hip.alloc
  // CHECK-NEXT: hip.relu
  // CHECK-NEXT: hip.copy
  // CHECK-NEXT: hip.copy
  %temp = hip.alloc(%ctx) : memref<1x64xf32, 1>
  hip.relu(%ctx, %input, %temp) : (!hip.context, memref<1x3xf32, 1>, memref<1x64xf32, 1>)
  hip.copy(%ctx, %temp, %output1) : (!hip.context, memref<1x64xf32, 1>, memref<1x64xf32, 1>)
  hip.copy(%ctx, %temp, %output2) : (!hip.context, memref<1x64xf32, 1>, memref<1x64xf32, 1>)
  return
}
