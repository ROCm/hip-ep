// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Verify ONNX Slice lowering: all paths lower to hip.slice (SliceToHip).
//
// The SliceDecompose pattern (tensor.extract_slice for fully-const, all-static
// slices) was intentionally dropped because strided subview descriptors are
// incompatible with downstream HIP kernels that assume contiguous memrefs.
// All slices now go through the hip.slice runtime copy path.

// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }

  // Test 1: classic prefix slice (axis 0, [1:3], stride 1) — goes through
  // hip.slice (the SliceDecompose/tensor.extract_slice path was removed because
  // strided subviews are incompatible with downstream HIP kernels).
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
    // CHECK-NOT: tensor.extract_slice
    // CHECK: hip.slice({{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<4x6xf32>, tensor<1xi64>, tensor<1xi64>)

    return %r : tensor<2x6xf32>
  }

  // Test 2: per-axis slice with stride > 1 — also goes through hip.slice.
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
    // CHECK-NOT: tensor.extract_slice
    // CHECK: hip.slice({{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>)

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
    // CHECK-NOT: tensor.extract_slice
    // CHECK: hip.slice({{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<4x6xf32>, tensor<2xi64>, tensor<2xi64>)

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

  // Test 6: Slice on a tensor with a dynamic non-sliced axis.
  // axis 0 is sliced (input dim is static = 4), axis 1 is left alone
  // (input dim is ? -> hip.slice output sized via tensor.dim).
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
    // CHECK-NOT: tensor.extract_slice
    // CHECK-DAG: %[[A1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[DIM:.*]] = tensor.dim %{{.*}}, %[[A1]] : tensor<4x?xf32>
    // The non-sliced axis uses the runtime dim value for the output empty.
    // CHECK: tensor.empty(%[[DIM]]) : tensor<2x?xf32>
    // CHECK: hip.slice({{.*}}) ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<4x?xf32>, tensor<1xi64>, tensor<1xi64>)
    return %r : tensor<2x?xf32>
  }

  // Test 7: SliceDecompose bails when a sliced axis has a dynamic
  // input dim (ONNX clamping rules need the static dim size); falls
  // through to hip.slice. The data dim is forwarded as an upper-bound
  // tensor.dim for the dynamic output dim.
  func.func @test_slice_native_dyn_axis(%input: tensor<?xf32>) -> tensor<?xf32> {
    // CHECK-LABEL: func.func @test_slice_native_dyn_axis
    %starts = arith.constant dense<[1]> : tensor<1xi64>
    %ends   = arith.constant dense<[3]> : tensor<1xi64>
    %axes   = arith.constant dense<[0]> : tensor<1xi64>
    %steps  = arith.constant dense<[1]> : tensor<1xi64>
    %r = "onnx.Slice"(%input, %starts, %ends, %axes, %steps)
        : (tensor<?xf32>, tensor<1xi64>, tensor<1xi64>,
           tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>

    // CHECK-NOT: tensor.extract_slice
    // CHECK-DAG: %[[A0:.*]] = arith.constant 0 : index
    // CHECK-DAG: tensor.dim %{{.*}}, %[[A0]] : tensor<?xf32>
    // Output size is the clamped slice extent (not the raw dim).
    // CHECK: tensor.empty(%{{.*}}) : tensor<?xf32>
    // CHECK: hip.slice({{.*}}) ins({{.*}}, {{.*}}, {{.*}} : tensor<?xf32>, tensor<1xi64>, tensor<1xi64>)
    return %r : tensor<?xf32>
  }
}
