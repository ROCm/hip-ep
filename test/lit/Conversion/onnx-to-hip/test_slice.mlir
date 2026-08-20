// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify the two ONNX Slice lowering paths:
//
//   1. SliceDecompose (preferred) — all slice params are compile-time
//      constants with positive unit stride, so onnx.Slice is rewritten to
//      a zero-cost tensor.extract_slice.
//   2. SliceToHip (fallback) — non-constant indices or negative steps fall
//      through to a native hip.slice op whose runtime is a stub today.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: classic prefix slice (axis 0, [1:3], stride 1) — decomposes to
  // tensor.extract_slice.
  func.func @test_slice_decompose_simple(%input: tensor<4x6xf32>) -> tensor<2x6xf32> {
    // CHECK-LABEL: func.func @test_slice_decompose_simple
    %starts = arith.constant dense<[1]> : tensor<1xi64>
    %ends   = arith.constant dense<[3]> : tensor<1xi64>
    %axes   = arith.constant dense<[0]> : tensor<1xi64>
    %steps  = arith.constant dense<[1]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<4x6xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<2x6xf32>

    // CHECK-NOT: onnx.Slice
    // CHECK-NOT: hip.slice
    // CHECK: tensor.extract_slice {{.*}}[1, 0] [2, 6] [1, 1]

    return %r : tensor<2x6xf32>
  }

  // Test 2: per-axis slice with stride > 1 — still decomposes (extract_slice
  // supports strides).
  func.func @test_slice_decompose_stride(%input: tensor<2x4xf32>) -> tensor<1x2xf32> {
    // CHECK-LABEL: func.func @test_slice_decompose_stride
    %starts = arith.constant dense<[1, 0]> : tensor<2xi64>
    %ends   = arith.constant dense<[2, 3]> : tensor<2xi64>
    %axes   = arith.constant dense<[0, 1]> : tensor<2xi64>
    %steps  = arith.constant dense<[1, 2]> : tensor<2xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>,
           tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>

    // CHECK-NOT: onnx.Slice
    // CHECK: tensor.extract_slice {{.*}}[1, 0] [1, 2] [1, 2]

    return %r : tensor<1x2xf32>
  }

  // Test 3: omitted axes / steps — default to all axes, unit stride.
  func.func @test_slice_decompose_default_axes(%input: tensor<4x6xf32>) -> tensor<2x3xf32> {
    // CHECK-LABEL: func.func @test_slice_decompose_default_axes
    %starts = arith.constant dense<[0, 0]> : tensor<2xi64>
    %ends   = arith.constant dense<[2, 3]> : tensor<2xi64>
    %r = "onnx.Slice"(%input, %starts, %ends)
        : (tensor<4x6xf32>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x3xf32>

    // CHECK-NOT: onnx.Slice
    // CHECK: tensor.extract_slice {{.*}}[0, 0] [2, 3] [1, 1]

    return %r : tensor<2x3xf32>
  }

  // Test 4: negative step forces the native fallback (hip.slice).  The
  // runtime is a stub today, but the conversion + bufferization pipeline
  // must still produce valid IR.
  func.func @test_slice_native_negative_step(%input: tensor<6xf32>) -> tensor<3xf32> {
    // CHECK-LABEL: func.func @test_slice_native_negative_step
    %starts = arith.constant dense<[5]> : tensor<1xi64>
    %ends   = arith.constant dense<[-1]> : tensor<1xi64>
    %axes   = arith.constant dense<[0]> : tensor<1xi64>
    %steps  = arith.constant dense<[-2]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<6xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>

    // CHECK-NOT: onnx.Slice
    // CHECK: tensor.empty() : tensor<3xf32>
    // CHECK: hip.slice({{.*}}) ins({{.*}}, {{.*}}, {{.*}} : tensor<6xf32>, tensor<1xi64>, tensor<1xi64>)

    return %r : tensor<3xf32>
  }

  // Test 5: non-constant starts (block argument) also falls back to the
  // native op — the decompose pattern needs to read the values.
  func.func @test_slice_native_dynamic_starts(
      %input: tensor<8xf32>, %starts: tensor<1xi64>) -> tensor<4xf32> {
    // CHECK-LABEL: func.func @test_slice_native_dynamic_starts
    %ends   = arith.constant dense<[7]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends)
        : (tensor<8xf32>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xf32>

    // CHECK-NOT: onnx.Slice
    // CHECK: hip.slice({{.*}}) ins(
    return %r : tensor<4xf32>
  }

  // Test 6: SliceDecompose on a tensor with a dynamic non-sliced axis.
  // axis 0 is sliced (input dim is static = 4), axis 1 is left alone
  // (input dim is ? -> the corresponding extract_slice size is a
  // tensor.dim Value, not a constant).
  func.func @test_slice_decompose_dyn_untouched(%input: tensor<4x?xf32>) -> tensor<2x?xf32> {
    // CHECK-LABEL: func.func @test_slice_decompose_dyn_untouched
    %starts = arith.constant dense<[1]> : tensor<1xi64>
    %ends   = arith.constant dense<[3]> : tensor<1xi64>
    %axes   = arith.constant dense<[0]> : tensor<1xi64>
    %steps  = arith.constant dense<[1]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<4x?xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<2x?xf32>

    // CHECK-NOT: onnx.Slice
    // CHECK-NOT: hip.slice
    // CHECK-DAG: %[[A1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[DIM:.*]] = tensor.dim %{{.*}}, %[[A1]] : tensor<4x?xf32>
    // Shared normalization materializes exact offsets and strides before
    // constructing the view; the untouched dimension keeps its runtime size.
    // CHECK: tensor.extract_slice %{{.*}}[{{.*}}] [{{.*}}] [{{.*}}] : tensor<4x?xf32> to tensor<2x?xf32>
    return %r : tensor<2x?xf32>
  }

  // Test 7: shared normalization emits clamp SSA for a dynamic sliced axis,
  // allowing the positive-step case to remain an exact extract_slice.
  func.func @test_slice_native_dyn_axis(%input: tensor<?xf32>) -> tensor<?xf32> {
    // CHECK-LABEL: func.func @test_slice_native_dyn_axis
    %starts = arith.constant dense<[1]> : tensor<1xi64>
    %ends   = arith.constant dense<[3]> : tensor<1xi64>
    %axes   = arith.constant dense<[0]> : tensor<1xi64>
    %steps  = arith.constant dense<[1]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<?xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>

    // CHECK-DAG: %[[A0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[DIM:.*]] = tensor.dim %{{.*}}, %[[A0]] : tensor<?xf32>
    // CHECK-NOT: hip.slice
    // CHECK: tensor.extract_slice %{{.*}}[{{.*}}] [{{.*}}] [{{.*}}] : tensor<?xf32> to tensor<?xf32>
    return %r : tensor<?xf32>
  }
}
