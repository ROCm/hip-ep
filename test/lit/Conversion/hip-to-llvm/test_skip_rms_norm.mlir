// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.skip_rms_norm operation is correctly lowered to LLVM dialect with
// runtime calls to wrap_miopenAddT5LayerNormForward function.
//
// This test validates:
// - Static shape lowering: constants for dimensions
// - Dynamic shape lowering: runtime extraction from memref descriptors
// - Element type support: f32, f16
// - Tensor rank support: 2D, 3D
// - Num elements computation: product of all dimensions for x, skip, and scale
// - Attribute lowering: axis, epsilon, stash_type passed to runtime
// - Runtime function signature: 12 parameters
//   (context, x, skip, scale, output, residual,
//    x_num, skip_num, scale_num, axis, epsilon, stash_type)
// - Fused operation: residual = x + skip, output = RMSNorm(residual) * scale
//
// Model: Llama-3.1-8B skip RMS layer normalization
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// ===== Static shape tests =====

// CHECK-LABEL: @skip_rms_norm_static_f32
func.func @skip_rms_norm_static_f32(%ctx: !hip.context) {
  %x = memref.alloc() : memref<128x512xf32, 1>
  %skip = memref.alloc() : memref<128x512xf32, 1>
  %scale = memref.alloc() : memref<512xf32, 1>
  %output = memref.alloc() : memref<128x512xf32, 1>
  %residual = memref.alloc() : memref<128x512xf32, 1>

  // Verify constants for dimensions
  // CHECK-DAG: llvm.mlir.constant(1 : i64)
  // CHECK-DAG: llvm.mlir.constant(128 : i64)
  // CHECK-DAG: llvm.mlir.constant(512 : i64)

  // Verify num_elements computation for x, skip, and scale
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64

  // Verify attribute constants
  // CHECK-DAG: llvm.mlir.constant(-1 : i64)
  // CHECK-DAG: llvm.mlir.constant(9.99999974E-6 : f32)

  // Verify runtime function call
  // CHECK: llvm.call @wrap_miopenAddT5LayerNormForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, i64) -> i32
  hip.skip_rms_norm(%ctx)
      ins(%x, %skip, %scale : memref<128x512xf32, 1>, memref<128x512xf32, 1>, memref<512xf32, 1>)
      outs(%output, %residual : memref<128x512xf32, 1>, memref<128x512xf32, 1>)
      {axis = -1 : i64, epsilon = 9.99999974e-06 : f32, stash_type = 1 : i64}

  return
}

// CHECK-LABEL: @skip_rms_norm_static_f16
func.func @skip_rms_norm_static_f16(%ctx: !hip.context) {
  %x = memref.alloc() : memref<1024xf16, 1>
  %skip = memref.alloc() : memref<1024xf16, 1>
  %scale = memref.alloc() : memref<1024xf16, 1>
  %output = memref.alloc() : memref<1024xf16, 1>
  %residual = memref.alloc() : memref<1024xf16, 1>

  // 1D tensor test case
  // CHECK: llvm.call @wrap_miopenAddT5LayerNormForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, i64) -> i32
  hip.skip_rms_norm(%ctx)
      ins(%x, %skip, %scale : memref<1024xf16, 1>, memref<1024xf16, 1>, memref<1024xf16, 1>)
      outs(%output, %residual : memref<1024xf16, 1>, memref<1024xf16, 1>)
      {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}

  return
}

// CHECK-LABEL: @skip_rms_norm_3d
func.func @skip_rms_norm_3d(%ctx: !hip.context) {
  %x = memref.alloc() : memref<2x64x128xf32, 1>
  %skip = memref.alloc() : memref<2x64x128xf32, 1>
  %scale = memref.alloc() : memref<128xf32, 1>
  %output = memref.alloc() : memref<2x64x128xf32, 1>
  %residual = memref.alloc() : memref<2x64x128xf32, 1>

  // 3D tensor test case
  // CHECK: llvm.mlir.constant(2 : i64) : i64
  // CHECK: llvm.mlir.constant(64 : i64) : i64
  // CHECK: llvm.mlir.constant(128 : i64) : i64
  // CHECK: llvm.call @wrap_miopenAddT5LayerNormForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, i64) -> i32
  hip.skip_rms_norm(%ctx)
      ins(%x, %skip, %scale : memref<2x64x128xf32, 1>, memref<2x64x128xf32, 1>, memref<128xf32, 1>)
      outs(%output, %residual : memref<2x64x128xf32, 1>, memref<2x64x128xf32, 1>)
      {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}

  return
}

// ===== Dynamic shape tests =====

// CHECK-LABEL: @skip_rms_norm_dynamic
func.func @skip_rms_norm_dynamic(%ctx: !hip.context, %x: memref<?x512xf16, 1>, %skip: memref<?x512xf16, 1>) {
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %scale = memref.alloc(%c512) : memref<?xf16, 1>
  %dim0 = memref.dim %x, %c0 : memref<?x512xf16, 1>
  %output = memref.alloc(%dim0) : memref<?x512xf16, 1>
  %residual = memref.alloc(%dim0) : memref<?x512xf16, 1>

  // Verify dynamic dimension extraction from memref descriptor
  // CHECK: llvm.extractvalue {{.*}}[3, 0]

  // Verify runtime computation of num_elements
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64

  // CHECK: llvm.call @wrap_miopenAddT5LayerNormForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, i64) -> i32
  hip.skip_rms_norm(%ctx)
      ins(%x, %skip, %scale : memref<?x512xf16, 1>, memref<?x512xf16, 1>, memref<?xf16, 1>)
      outs(%output, %residual : memref<?x512xf16, 1>, memref<?x512xf16, 1>)
      {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}

  return
}

// CHECK-LABEL: @skip_rms_norm_fully_dynamic
func.func @skip_rms_norm_fully_dynamic(%ctx: !hip.context, %x: memref<?x?xf16, 1>, %skip: memref<?x?xf16, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = memref.dim %x, %c0 : memref<?x?xf16, 1>
  %dim1 = memref.dim %x, %c1 : memref<?x?xf16, 1>
  %scale = memref.alloc(%dim1) : memref<?xf16, 1>
  %output = memref.alloc(%dim0, %dim1) : memref<?x?xf16, 1>
  %residual = memref.alloc(%dim0, %dim1) : memref<?x?xf16, 1>

  // Fully dynamic 2D tensor - both dimensions extracted at runtime
  // CHECK: llvm.extractvalue {{.*}}[3, 0]
  // CHECK: llvm.extractvalue {{.*}}[3, 1]

  // Compute num_elements for x, skip, and scale
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64

  // CHECK: llvm.call @wrap_miopenAddT5LayerNormForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, i64) -> i32
  hip.skip_rms_norm(%ctx)
      ins(%x, %skip, %scale : memref<?x?xf16, 1>, memref<?x?xf16, 1>, memref<?xf16, 1>)
      outs(%output, %residual : memref<?x?xf16, 1>, memref<?x?xf16, 1>)
      {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}

  return
}
