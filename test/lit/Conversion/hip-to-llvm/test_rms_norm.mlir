// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// ===== Static shape tests =====

// CHECK-LABEL: @rms_norm_static_f32
func.func @rms_norm_static_f32(%ctx: !hip.context) {
  %input = memref.alloc() : memref<128x512xf32, 1>
  %scale = memref.alloc() : memref<512xf32, 1>
  %output = memref.alloc() : memref<128x512xf32, 1>

  // Verify constants are generated for static shapes
  // CHECK-DAG: llvm.mlir.constant(1 : i64) : i64
  // CHECK-DAG: llvm.mlir.constant(128 : i64) : i64
  // CHECK-DAG: llvm.mlir.constant(512 : i64) : i64
  // CHECK: [[INPUT_ELEMS:%.*]] = llvm.mul {{.*}}, {{.*}} : i64
  // CHECK: [[INPUT_TOTAL:%.*]] = llvm.mul [[INPUT_ELEMS]], {{.*}} : i64
  // CHECK: [[SCALE_ELEMS:%.*]] = llvm.mul {{.*}}, {{.*}} : i64

  // Verify attributes are passed as constants
  // CHECK: llvm.mlir.constant(-1 : i64) : i64
  // CHECK: llvm.mlir.constant(9.99999974E-6 : f32) : f32
  // CHECK: llvm.mlir.constant(1 : i64) : i64

  // Verify runtime call with 9 parameters
  // CHECK: llvm.call @wrap_miopenT5LayerNormForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, i64) -> i32
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
  // CHECK: llvm.call @wrap_miopenT5LayerNormForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, i64) -> i32
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
  // CHECK: llvm.call @wrap_miopenT5LayerNormForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, i64) -> i32
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

  // CHECK: llvm.call @wrap_miopenT5LayerNormForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, i64) -> i32
  hip.rms_norm(%ctx)
      ins(%input, %scale : memref<?x512xf16, 1>, memref<?xf16, 1>)
      outs(%output : memref<?x512xf16, 1>)
      {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}

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

  // CHECK: llvm.call @wrap_miopenT5LayerNormForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, i64) -> i32
  hip.rms_norm(%ctx)
      ins(%input, %scale : memref<?x?xf16, 1>, memref<?xf16, 1>)
      outs(%output : memref<?x?xf16, 1>)
      {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}

  return
}
