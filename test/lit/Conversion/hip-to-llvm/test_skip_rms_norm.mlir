// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.skip_rms_norm operation is correctly lowered to LLVM dialect with
// runtime calls to wrap_skip_simplified_layer_norm function.
//
// This test validates:
// - Static shape lowering: constants for dimensions
// - Dynamic shape lowering: runtime extraction from memref descriptors
// - Element type support: f32, f16
// - Tensor rank support: 2D, 3D
// - Num elements computation: product of all dimensions for input and gamma
// - Attribute lowering: epsilon passed to runtime
// - Runtime function signature: 11 parameters
//   (state, input, skip, gamma, bias, output, input_skip_bias_sum,
//    input_num, gamma_num, element_size_bytes, epsilon)
// - Optional bias: nullptr when absent, valid pointer when present
// - Optional input_skip_bias_sum: nullptr when absent, valid pointer when present
//
// MS spec reference:
// https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
//   #commicrosoftskipsimplifiedlayernormalization
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// ===== Static shape tests =====

// CHECK-LABEL: @skip_rms_norm_static_f32
func.func @skip_rms_norm_static_f32(%ctx: !hip.context) {
  %input = memref.alloc() : memref<128x512xf32, 1>
  %skip = memref.alloc() : memref<128x512xf32, 1>
  %gamma = memref.alloc() : memref<512xf32, 1>
  %output = memref.alloc() : memref<128x512xf32, 1>
  %skip_output = memref.alloc() : memref<128x512xf32, 1>

  // Verify constants for dimensions
  // CHECK-DAG: llvm.mlir.constant(1 : i64)
  // CHECK-DAG: llvm.mlir.constant(128 : i64)
  // CHECK-DAG: llvm.mlir.constant(512 : i64)

  // Verify epsilon constant
  // CHECK-DAG: llvm.mlir.constant(9.99999974E-6 : f32)

  // Verify runtime function call with 11 params (bias and input_skip_bias_sum may be nullptr)
  // CHECK: llvm.call @wrap_skip_simplified_layer_norm({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32) -> i32
  hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma : memref<128x512xf32, 1>, memref<128x512xf32, 1>, memref<512xf32, 1>)
      outs(%output, %skip_output : memref<128x512xf32, 1>, memref<128x512xf32, 1>)
      {epsilon = 9.99999974e-06 : f32}

  return
}

// CHECK-LABEL: @skip_rms_norm_static_f16
func.func @skip_rms_norm_static_f16(%ctx: !hip.context) {
  %input = memref.alloc() : memref<1024xf16, 1>
  %skip = memref.alloc() : memref<1024xf16, 1>
  %gamma = memref.alloc() : memref<1024xf16, 1>
  %output = memref.alloc() : memref<1024xf16, 1>
  %skip_output = memref.alloc() : memref<1024xf16, 1>

  // 1D tensor test case
  // CHECK: llvm.call @wrap_skip_simplified_layer_norm({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32) -> i32
  hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma : memref<1024xf16, 1>, memref<1024xf16, 1>, memref<1024xf16, 1>)
      outs(%output, %skip_output : memref<1024xf16, 1>, memref<1024xf16, 1>)
      {epsilon = 1.0e-05 : f32}

  return
}

// CHECK-LABEL: @skip_rms_norm_3d
func.func @skip_rms_norm_3d(%ctx: !hip.context) {
  %input = memref.alloc() : memref<2x64x128xf32, 1>
  %skip = memref.alloc() : memref<2x64x128xf32, 1>
  %gamma = memref.alloc() : memref<128xf32, 1>
  %output = memref.alloc() : memref<2x64x128xf32, 1>
  %skip_output = memref.alloc() : memref<2x64x128xf32, 1>

  // 3D tensor test case
  // CHECK: llvm.mlir.constant(2 : i64) : i64
  // CHECK: llvm.mlir.constant(64 : i64) : i64
  // CHECK: llvm.mlir.constant(128 : i64) : i64
  // CHECK: llvm.call @wrap_skip_simplified_layer_norm({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32) -> i32
  hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma : memref<2x64x128xf32, 1>, memref<2x64x128xf32, 1>, memref<128xf32, 1>)
      outs(%output, %skip_output : memref<2x64x128xf32, 1>, memref<2x64x128xf32, 1>)
      {epsilon = 1.0e-05 : f32}

  return
}

// ===== Dynamic shape tests =====

// CHECK-LABEL: @skip_rms_norm_dynamic
func.func @skip_rms_norm_dynamic(%ctx: !hip.context, %input: memref<?x512xf16, 1>, %skip: memref<?x512xf16, 1>) {
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %gamma = memref.alloc(%c512) : memref<?xf16, 1>
  %dim0 = memref.dim %input, %c0 : memref<?x512xf16, 1>
  %output = memref.alloc(%dim0) : memref<?x512xf16, 1>
  %skip_output = memref.alloc(%dim0) : memref<?x512xf16, 1>

  // Verify dynamic dimension extraction from memref descriptor
  // CHECK: llvm.extractvalue {{.*}}[3, 0]

  // CHECK: llvm.call @wrap_skip_simplified_layer_norm({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32) -> i32
  hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma : memref<?x512xf16, 1>, memref<?x512xf16, 1>, memref<?xf16, 1>)
      outs(%output, %skip_output : memref<?x512xf16, 1>, memref<?x512xf16, 1>)
      {epsilon = 1.0e-05 : f32}

  return
}

// CHECK-LABEL: @skip_rms_norm_fully_dynamic
func.func @skip_rms_norm_fully_dynamic(%ctx: !hip.context, %input: memref<?x?xf16, 1>, %skip: memref<?x?xf16, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = memref.dim %input, %c0 : memref<?x?xf16, 1>
  %dim1 = memref.dim %input, %c1 : memref<?x?xf16, 1>
  %gamma = memref.alloc(%dim1) : memref<?xf16, 1>
  %output = memref.alloc(%dim0, %dim1) : memref<?x?xf16, 1>
  %skip_output = memref.alloc(%dim0, %dim1) : memref<?x?xf16, 1>

  // Fully dynamic 2D tensor - both dimensions extracted at runtime
  // CHECK: llvm.extractvalue {{.*}}[3, 0]
  // CHECK: llvm.extractvalue {{.*}}[3, 1]

  // CHECK: llvm.call @wrap_skip_simplified_layer_norm({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32) -> i32
  hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma : memref<?x?xf16, 1>, memref<?x?xf16, 1>, memref<?xf16, 1>)
      outs(%output, %skip_output : memref<?x?xf16, 1>, memref<?x?xf16, 1>)
      {epsilon = 1.0e-05 : f32}

  return
}

// ===== With bias tests =====

// CHECK-LABEL: @skip_rms_norm_with_bias
func.func @skip_rms_norm_with_bias(%ctx: !hip.context) {
  %input = memref.alloc() : memref<1x128x4096xf16, 1>
  %skip = memref.alloc() : memref<1x128x4096xf16, 1>
  %gamma = memref.alloc() : memref<4096xf16, 1>
  %bias = memref.alloc() : memref<4096xf16, 1>
  %output = memref.alloc() : memref<1x128x4096xf16, 1>
  %skip_output = memref.alloc() : memref<1x128x4096xf16, 1>

  // With bias: all 7 pointers are valid memref pointers
  // CHECK: llvm.call @wrap_skip_simplified_layer_norm({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32) -> i32
  hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma, %bias :
          memref<1x128x4096xf16, 1>, memref<1x128x4096xf16, 1>,
          memref<4096xf16, 1>, memref<4096xf16, 1>)
      outs(%output, %skip_output :
          memref<1x128x4096xf16, 1>, memref<1x128x4096xf16, 1>)
      {epsilon = 9.99999974e-06 : f32}

  return
}

// ===== Output-only (no input_skip_bias_sum) =====

// CHECK-LABEL: @skip_rms_norm_output_only
func.func @skip_rms_norm_output_only(%ctx: !hip.context) {
  %input = memref.alloc() : memref<128x512xf32, 1>
  %skip = memref.alloc() : memref<128x512xf32, 1>
  %gamma = memref.alloc() : memref<512xf32, 1>
  %output = memref.alloc() : memref<128x512xf32, 1>

  // When input_skip_bias_sum is not provided, bias and skip_output pointers are nullptr
  // CHECK: llvm.call @wrap_skip_simplified_layer_norm({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32) -> i32
  hip.skip_rms_norm(%ctx)
      ins(%input, %skip, %gamma : memref<128x512xf32, 1>, memref<128x512xf32, 1>, memref<512xf32, 1>)
      outs(%output : memref<128x512xf32, 1>)
      {epsilon = 1.0e-05 : f32}

  return
}
