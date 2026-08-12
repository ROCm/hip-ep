// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify hip.rms_norm operation is correctly lowered to LLVM dialect with
// runtime calls to wrap_miopenT5LayerNormForward function.
//
// This test validates:
// - Static shape lowering: constants for dimensions
// - Dynamic shape lowering: runtime extraction from memref descriptors
// - Element type support: f32, f16
// - Tensor rank support: 1D, 2D, 3D
// - Num elements computation: product of all dimensions
// - Attribute lowering: axis, epsilon, stash_type passed to runtime
// - Runtime function signature: 10 parameters
//   (context, input, scale, output, input_num, scale_num, element_size_bytes, axis, epsilon, stash_type)
//
// Model: Llama-3.1-8B RMS layer normalization
// ============================================================================

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// ===== Static shape tests =====

// CHECK-LABEL: @rms_norm_static_f32
func.func @rms_norm_static_f32(%ctx: !hip.context) {
  %input = memref.alloc() : memref<128x512xf32, 1>
  %scale = memref.alloc() : memref<512xf32, 1>
  %output = memref.alloc() : memref<128x512xf32, 1>

  // Verify constants for dimensions
  // CHECK-DAG: llvm.mlir.constant(1 : i64)
  // CHECK-DAG: llvm.mlir.constant(128 : i64)
  // CHECK-DAG: llvm.mlir.constant(512 : i64)

  // Verify num_elements computation
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64

  // Verify attribute constants
  // CHECK-DAG: llvm.mlir.constant(-1 : i64)
  // CHECK-DAG: llvm.mlir.constant(9.99999974E-6 : f32)

  // Verify runtime function call (10 params with element_size_bytes)
  // CHECK: llvm.call @wrap_miopenT5LayerNormForward({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, i64) -> i32
  hip.rms_norm(%ctx)
      ins(%input, %scale : memref<128x512xf32, 1>, memref<512xf32, 1>)
      outs(%output : memref<128x512xf32, 1>)
      {axis = -1 : i64, epsilon = 9.99999974e-06 : f32, stash_type = 1 : i64}

  return
}

// CHECK-LABEL: @rms_norm_static_f16
func.func @rms_norm_static_f16(%ctx: !hip.context) {
  %input = memref.alloc() : memref<1024xf16, 1>
  %scale = memref.alloc() : memref<1024xf16, 1>
  %output = memref.alloc() : memref<1024xf16, 1>

  // 1D tensor test case
  // CHECK: llvm.call @wrap_miopenT5LayerNormForward({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, i64) -> i32
  hip.rms_norm(%ctx)
      ins(%input, %scale : memref<1024xf16, 1>, memref<1024xf16, 1>)
      outs(%output : memref<1024xf16, 1>)
      {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}

  return
}

// CHECK-LABEL: @rms_norm_3d
func.func @rms_norm_3d(%ctx: !hip.context) {
  %input = memref.alloc() : memref<2x64x128xf32, 1>
  %scale = memref.alloc() : memref<128xf32, 1>
  %output = memref.alloc() : memref<2x64x128xf32, 1>

  // 3D tensor: input_num_elements = 2 * 64 * 128 = 16384
  // CHECK: llvm.mlir.constant(2 : i64) : i64
  // CHECK: llvm.mlir.constant(64 : i64) : i64
  // CHECK: llvm.mlir.constant(128 : i64) : i64
  // CHECK: llvm.call @wrap_miopenT5LayerNormForward({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, i64) -> i32
  hip.rms_norm(%ctx)
      ins(%input, %scale : memref<2x64x128xf32, 1>, memref<128xf32, 1>)
      outs(%output : memref<2x64x128xf32, 1>)
      {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}

  return
}

// ===== Dynamic shape tests =====

// CHECK-LABEL: @rms_norm_dynamic
func.func @rms_norm_dynamic(%ctx: !hip.context, %input: memref<?x512xf16, 1>) {
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %scale = memref.alloc(%c512) : memref<?xf16, 1>
  %dim0 = memref.dim %input, %c0 : memref<?x512xf16, 1>
  %output = memref.alloc(%dim0) : memref<?x512xf16, 1>

  // Verify dynamic dimension extraction from memref descriptor
  // CHECK: llvm.extractvalue {{.*}}[3, 0]

  // Verify runtime computation of num_elements
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64

  // CHECK: llvm.call @wrap_miopenT5LayerNormForward({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, i64) -> i32
  hip.rms_norm(%ctx)
      ins(%input, %scale : memref<?x512xf16, 1>, memref<?xf16, 1>)
      outs(%output : memref<?x512xf16, 1>)
      {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}

  return
}

// Rank-3 [batch, seq, hidden] with both leading extents dynamic: the shape a
// decoder layer's input norm actually has. input_num_elements must be the
// product of THREE distinct descriptor reads -- if dim 0 and dim 1 resolve to
// the same value the norm is dispatched over seq^2 rows.
// CHECK-LABEL: @rms_norm_dynamic_batch_seq_3d
func.func @rms_norm_dynamic_batch_seq_3d(%ctx: !hip.context,
                                         %input: memref<?x?x2816xf16, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2816 = arith.constant 2816 : index
  %dim0 = memref.dim %input, %c0 : memref<?x?x2816xf16, 1>
  %dim1 = memref.dim %input, %c1 : memref<?x?x2816xf16, 1>
  %scale = memref.alloc(%c2816) : memref<?xf16, 1>
  %output = memref.alloc(%dim0, %dim1) : memref<?x?x2816xf16, 1>

  // Batch and sequence come from separate descriptor slots.
  // CHECK: llvm.extractvalue {{.*}}[3, 0]
  // CHECK: llvm.extractvalue {{.*}}[3, 1]

  // input_num_elements = dim0 * dim1 * 2816
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64

  // CHECK: llvm.call @wrap_miopenT5LayerNormForward({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, i64) -> i32
  hip.rms_norm(%ctx)
      ins(%input, %scale : memref<?x?x2816xf16, 1>, memref<?xf16, 1>)
      outs(%output : memref<?x?x2816xf16, 1>)
      {axis = -1 : i64, epsilon = 9.99999997e-07 : f32, stash_type = 1 : i64}

  return
}

// CHECK-LABEL: @rms_norm_fully_dynamic
func.func @rms_norm_fully_dynamic(%ctx: !hip.context, %input: memref<?x?xf16, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = memref.dim %input, %c0 : memref<?x?xf16, 1>
  %dim1 = memref.dim %input, %c1 : memref<?x?xf16, 1>
  %scale = memref.alloc(%dim1) : memref<?xf16, 1>
  %output = memref.alloc(%dim0, %dim1) : memref<?x?xf16, 1>

  // Fully dynamic 2D tensor - both dimensions extracted at runtime
  // CHECK: llvm.extractvalue {{.*}}[3, 0]
  // CHECK: llvm.extractvalue {{.*}}[3, 1]

  // Compute input_num_elements = dim0 * dim1
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64
  // CHECK: llvm.mul {{.*}}, {{.*}} : i64

  // CHECK: llvm.call @wrap_miopenT5LayerNormForward({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, i64) -> i32
  hip.rms_norm(%ctx)
      ins(%input, %scale : memref<?x?xf16, 1>, memref<?xf16, 1>)
      outs(%output : memref<?x?xf16, 1>)
      {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}

  return
}
