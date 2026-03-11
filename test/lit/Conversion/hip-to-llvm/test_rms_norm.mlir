// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// CHECK-LABEL: @rms_norm_static_f32
func.func @rms_norm_static_f32(%ctx: !hip.context) {
  %input = memref.alloc() : memref<128x512xf32, 1>
  %scale = memref.alloc() : memref<512xf32, 1>
  %output = memref.alloc() : memref<128x512xf32, 1>

  // CHECK: llvm.mlir.constant(65536 : {{i64|index}})
  // CHECK: llvm.mlir.constant(512 : {{i64|index}})
  // CHECK: llvm.mlir.constant(-1 : {{i64|index}})
  // CHECK: llvm.mlir.constant(9.99999974E-6 : f32)
  // CHECK: llvm.mlir.constant(1 : {{i64|index}})
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

  // CHECK: llvm.call @wrap_miopenT5LayerNormForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, i64) -> i32
  hip.rms_norm(%ctx)
      ins(%input, %scale : memref<1024xf16, 1>, memref<1024xf16, 1>)
      outs(%output : memref<1024xf16, 1>)
      {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}

  return
}

// CHECK-LABEL: @rms_norm_dynamic
func.func @rms_norm_dynamic(%ctx: !hip.context, %input: memref<?x512xf16, 1>) {
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %scale = memref.alloc(%c512) : memref<?xf16, 1>
  %dim0 = memref.dim %input, %c0 : memref<?x512xf16, 1>
  %output = memref.alloc(%dim0) : memref<?x512xf16, 1>

  // CHECK: llvm.extractvalue {{.*}}[3, 0]
  // CHECK: llvm.mul
  // CHECK: llvm.mul
  // CHECK: llvm.call @wrap_miopenT5LayerNormForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, i64) -> i32
  hip.rms_norm(%ctx)
      ins(%input, %scale : memref<?x512xf16, 1>, memref<?xf16, 1>)
      outs(%output : memref<?x512xf16, 1>)
      {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}

  return
}

// CHECK-LABEL: @rms_norm_3d
func.func @rms_norm_3d(%ctx: !hip.context) {
  %input = memref.alloc() : memref<2x64x128xf32, 1>
  %scale = memref.alloc() : memref<128xf32, 1>
  %output = memref.alloc() : memref<2x64x128xf32, 1>

  // CHECK: llvm.mlir.constant(16384 : {{i64|index}})
  // CHECK: llvm.call @wrap_miopenT5LayerNormForward({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, f32, i64) -> i32
  hip.rms_norm(%ctx)
      ins(%input, %scale : memref<2x64x128xf32, 1>, memref<128xf32, 1>)
      outs(%output : memref<2x64x128xf32, 1>)
      {axis = -1 : i64, epsilon = 1.0e-05 : f32, stash_type = 1 : i64}

  return
}
