// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify ONNX Split is correctly lowered to tensor.extract_slice
// (zero-cost standard MLIR slice ops).
//
// Split is a zero-cost operation that creates views into the input tensor
// without moving data. No custom HIP dialect op or kernel is needed.
//
// This test validates:
// - Equal split, static shape:   Split along last dimension
// - Equal split, different axis:  Split along axis 0
// - Custom split, static:         Split with explicit lengths
// - Dynamic shape:                Split with runtime dimension
// - Single output:                Identity operation (no-op)
// - Multi-dimensional:            Split in middle dimension
// - Different data types:         f32, f16, bf16
//
// Model: Multi-head attention head splitting
// ============================================================================

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<1x128x4096xf16>) -> tensor<1x128x4096xf16> {
    return %arg0 : tensor<1x128x4096xf16>
  }

  // --- Equal split, static shape: split last dim into 2 outputs ---
  func.func @test_split_equal_static(%data: tensor<1x128x4096xf16>) -> (tensor<1x128x2048xf16>, tensor<1x128x2048xf16>) {
    %out0, %out1 = "onnx.Split"(%data) {axis = -1 : si64} : (tensor<1x128x4096xf16>) -> (tensor<1x128x2048xf16>, tensor<1x128x2048xf16>)
    return %out0, %out1 : tensor<1x128x2048xf16>, tensor<1x128x2048xf16>
  }

  // --- Equal split, axis=0: split first dim into 3 outputs ---
  func.func @test_split_axis0(%data: tensor<6x32x64xf32>) -> (tensor<2x32x64xf32>, tensor<2x32x64xf32>, tensor<2x32x64xf32>) {
    %out0, %out1, %out2 = "onnx.Split"(%data) {axis = 0 : si64} : (tensor<6x32x64xf32>) -> (tensor<2x32x64xf32>, tensor<2x32x64xf32>, tensor<2x32x64xf32>)
    return %out0, %out1, %out2 : tensor<2x32x64xf32>, tensor<2x32x64xf32>, tensor<2x32x64xf32>
  }

  // --- Custom split, static: split axis=0 with explicit lengths [3, 3, 2] ---
  func.func @test_split_custom(%data: tensor<8x256xf16>) -> (tensor<3x256xf16>, tensor<3x256xf16>, tensor<2x256xf16>) {
    %split_lengths = "onnx.Constant"() {value = dense<[3, 3, 2]> : tensor<3xi64>} : () -> tensor<3xi64>
    %out0, %out1, %out2 = "onnx.Split"(%data, %split_lengths) {axis = 0 : si64} : (tensor<8x256xf16>, tensor<3xi64>) -> (tensor<3x256xf16>, tensor<3x256xf16>, tensor<2x256xf16>)
    return %out0, %out1, %out2 : tensor<3x256xf16>, tensor<3x256xf16>, tensor<2x256xf16>
  }

  // --- Dynamic shape: split dynamic dimension into 2 outputs ---
  func.func @test_split_dynamic(%data: tensor<?x128xf32>) -> (tensor<?x128xf32>, tensor<?x128xf32>) {
    %out0, %out1 = "onnx.Split"(%data) {axis = 0 : si64} : (tensor<?x128xf32>) -> (tensor<?x128xf32>, tensor<?x128xf32>)
    return %out0, %out1 : tensor<?x128xf32>, tensor<?x128xf32>
  }

  // --- Single output: identity operation (no-op) ---
  func.func @test_split_single_output(%data: tensor<4x16xf16>) -> tensor<4x16xf16> {
    %out = "onnx.Split"(%data) {axis = 0 : si64} : (tensor<4x16xf16>) -> tensor<4x16xf16>
    return %out : tensor<4x16xf16>
  }

  // --- Multi-dimensional: split middle dimension ---
  func.func @test_split_middle_dim(%data: tensor<2x4x8x16xf32>) -> (tensor<2x4x4x16xf32>, tensor<2x4x4x16xf32>) {
    %out0, %out1 = "onnx.Split"(%data) {axis = 2 : si64} : (tensor<2x4x8x16xf32>) -> (tensor<2x4x4x16xf32>, tensor<2x4x4x16xf32>)
    return %out0, %out1 : tensor<2x4x4x16xf32>, tensor<2x4x4x16xf32>
  }

  // --- bf16 data type ---
  func.func @test_split_bf16(%data: tensor<1x256x512xbf16>) -> (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>) {
    %out0, %out1 = "onnx.Split"(%data) {axis = -1 : si64} : (tensor<1x256x512xbf16>) -> (tensor<1x256x256xbf16>, tensor<1x256x256xbf16>)
    return %out0, %out1 : tensor<1x256x256xbf16>, tensor<1x256x256xbf16>
  }

  // --- f32 data type with custom split ---
  func.func @test_split_f32_custom(%data: tensor<10x64xf32>) -> (tensor<2x64xf32>, tensor<5x64xf32>, tensor<3x64xf32>) {
    %split_lengths = "onnx.Constant"() {value = dense<[2, 5, 3]> : tensor<3xi64>} : () -> tensor<3xi64>
    %out0, %out1, %out2 = "onnx.Split"(%data, %split_lengths) {axis = 0 : si64} : (tensor<10x64xf32>, tensor<3xi64>) -> (tensor<2x64xf32>, tensor<5x64xf32>, tensor<3x64xf32>)
    return %out0, %out1, %out2 : tensor<2x64xf32>, tensor<5x64xf32>, tensor<3x64xf32>
  }

  // --- Equal split with onnx.NoValue (representing none/optional) ---
  func.func @test_split_novalue(%data: tensor<1x128x6144xf16>) -> (tensor<1x128x2048xf16>, tensor<1x128x2048xf16>, tensor<1x128x2048xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %out0, %out1, %out2 = "onnx.Split"(%data, %none) {axis = -1 : si64} : (tensor<1x128x6144xf16>, none) -> (tensor<1x128x2048xf16>, tensor<1x128x2048xf16>, tensor<1x128x2048xf16>)
    return %out0, %out1, %out2 : tensor<1x128x2048xf16>, tensor<1x128x2048xf16>, tensor<1x128x2048xf16>
  }

  // --- Non-divisible equal split (last chunk smaller) ---
  func.func @test_split_nondivisible(%data: tensor<1x10xf32>) -> (tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x1xf32>) {
    %out0, %out1, %out2, %out3 = "onnx.Split"(%data) {axis = 1 : si64} : (tensor<1x10xf32>) -> (tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x1xf32>)
    return %out0, %out1, %out2, %out3 : tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x1xf32>
  }

  // --- Non-divisible with dynamic output (last chunk computed via subtraction) ---
  func.func @test_split_nondivisible_dynamic(%data: tensor<?x10xf32>) -> (tensor<?x10xf32>, tensor<?x10xf32>, tensor<?x10xf32>) {
    %out0, %out1, %out2 = "onnx.Split"(%data) {axis = 0 : si64} : (tensor<?x10xf32>) -> (tensor<?x10xf32>, tensor<?x10xf32>, tensor<?x10xf32>)
    return %out0, %out1, %out2 : tensor<?x10xf32>, tensor<?x10xf32>, tensor<?x10xf32>
  }

  // --- Custom split with i32 type (verifies APInt support for different integer widths) ---
  func.func @test_split_i32_type(%data: tensor<10x64xf32>) -> (tensor<2x64xf32>, tensor<5x64xf32>, tensor<3x64xf32>) {
    %split_lengths = "onnx.Constant"() {value = dense<[2, 5, 3]> : tensor<3xi32>} : () -> tensor<3xi32>
    %out0, %out1, %out2 = "onnx.Split"(%data, %split_lengths) {axis = 0 : si64} : (tensor<10x64xf32>, tensor<3xi32>) -> (tensor<2x64xf32>, tensor<5x64xf32>, tensor<3x64xf32>)
    return %out0, %out1, %out2 : tensor<2x64xf32>, tensor<5x64xf32>, tensor<3x64xf32>
  }

  // Note: the malformed-input negative case (custom split lengths that
  // don't sum to the axis dimension — the converter must refuse to match,
  // leaving onnx.Split in the IR) is now its own file
  // (test_split_invalid.mlir). convert-onnx-to-hip's surviving-op
  // diagnostic catches such cases as a hard error, so it can no longer
  // live next to the successful conversions in a single FileCheck run.
}

// CHECK-LABEL: func.func @test_split_equal_static
// CHECK-NOT: onnx.Split
// CHECK: tensor.extract_slice{{.*}}[0, 0, 0] [1, 128, 2048] [1, 1, 1]
// CHECK: tensor.extract_slice{{.*}}[{{.*}}] [1, 128, 2048] [1, 1, 1]

// CHECK-LABEL: func.func @test_split_axis0
// CHECK-NOT: onnx.Split
// CHECK: tensor.extract_slice{{.*}}[0, 0, 0] [2, 32, 64] [1, 1, 1]
// CHECK: tensor.extract_slice{{.*}}[{{.*}}] [2, 32, 64] [1, 1, 1]
// CHECK: tensor.extract_slice{{.*}}[{{.*}}] [2, 32, 64] [1, 1, 1]

// CHECK-LABEL: func.func @test_split_custom
// CHECK-NOT: onnx.Split
// CHECK: tensor.extract_slice{{.*}}[0, 0] [3, 256] [1, 1]
// CHECK: tensor.extract_slice{{.*}}[{{.*}}] [3, 256] [1, 1]
// CHECK: tensor.extract_slice{{.*}}[{{.*}}] [2, 256] [1, 1]

// CHECK-LABEL: func.func @test_split_dynamic
// CHECK-NOT: onnx.Split
// CHECK: %[[DIM:.*]] = tensor.dim
// CHECK: %[[CHUNK:.*]] = arith.divui %[[DIM]]
// CHECK: tensor.extract_slice{{.*}}[0, 0] [%{{.*}}, 128] [1, 1]
// CHECK: tensor.extract_slice{{.*}}[{{.*}}] [%{{.*}}, 128] [1, 1]

// CHECK-LABEL: func.func @test_split_single_output
// CHECK-NOT: onnx.Split
// CHECK-NOT: tensor.extract_slice
// CHECK: return %arg{{[0-9]+}}

// CHECK-LABEL: func.func @test_split_middle_dim
// CHECK-NOT: onnx.Split
// CHECK: tensor.extract_slice{{.*}}[0, 0, 0, 0] [2, 4, 4, 16] [1, 1, 1, 1]
// CHECK: tensor.extract_slice{{.*}}[{{.*}}] [2, 4, 4, 16] [1, 1, 1, 1]

// CHECK-LABEL: func.func @test_split_bf16
// CHECK-NOT: onnx.Split
// CHECK: tensor.extract_slice
// CHECK: tensor.extract_slice

// CHECK-LABEL: func.func @test_split_f32_custom
// CHECK-NOT: onnx.Split
// CHECK: tensor.extract_slice{{.*}}[0, 0] [2, 64] [1, 1]
// CHECK: tensor.extract_slice{{.*}}[{{.*}}] [5, 64] [1, 1]
// CHECK: tensor.extract_slice{{.*}}[{{.*}}] [3, 64] [1, 1]

// CHECK-LABEL: func.func @test_split_novalue
// CHECK-NOT: onnx.Split
// CHECK-NOT: onnx.NoValue
// CHECK: tensor.extract_slice{{.*}}[0, 0, 0] [1, 128, 2048] [1, 1, 1]
// CHECK: tensor.extract_slice
// CHECK: tensor.extract_slice

// CHECK-LABEL: func.func @test_split_nondivisible
// CHECK-NOT: onnx.Split
// With static output types, we use static sizes directly (no tensor.dim needed)
// CHECK: tensor.extract_slice{{.*}}[0, 0] [1, 3] [1, 1]
// CHECK: tensor.extract_slice{{.*}}[0, {{.*}}] [1, 3] [1, 1]
// CHECK: tensor.extract_slice{{.*}}[0, {{.*}}] [1, 3] [1, 1]
// CHECK: tensor.extract_slice{{.*}}[0, {{.*}}] [1, 1] [1, 1]

// CHECK-LABEL: func.func @test_split_nondivisible_dynamic
// CHECK-NOT: onnx.Split
// CHECK: %[[DIM:.*]] = tensor.dim
// CHECK: %[[CHUNK:.*]] = arith.divui %[[DIM]]
// CHECK: %[[MUL:.*]] = arith.muli %[[CHUNK]]
// CHECK: %[[LAST:.*]] = arith.subi %[[DIM]], %[[MUL]]
// CHECK: tensor.extract_slice{{.*}}[0, 0] [%[[CHUNK]], 10] [1, 1]
// CHECK: tensor.extract_slice{{.*}}[{{.*}}] [%[[CHUNK]], 10] [1, 1]
// CHECK: tensor.extract_slice{{.*}}[{{.*}}] [%[[LAST]], 10] [1, 1]

// CHECK-LABEL: func.func @test_split_i32_type
// CHECK-NOT: onnx.Split
// Verify i32 split lengths work (APInt handles different integer widths)
// CHECK: tensor.extract_slice{{.*}}[0, 0] [2, 64] [1, 1]
// CHECK: tensor.extract_slice{{.*}}[{{.*}}] [5, 64] [1, 1]
// CHECK: tensor.extract_slice{{.*}}[{{.*}}] [3, 64] [1, 1]
