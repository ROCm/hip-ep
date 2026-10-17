// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.gated_rms_norm lowers to wrap_gated_rms_norm(state, input, gate,
// weight, output, q, scale, input_num_elements, gate_num_elements,
// row_size, group_size, epsilon, quant_block_size). q and scale are null
// without quantized outputs, and quant_block_size is then 0.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// CHECK-LABEL: @gated_rms_norm_static
// CHECK: llvm.call @malloc
// CHECK: llvm.call @malloc
// CHECK: llvm.call @malloc
// CHECK: llvm.call @malloc
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NOT: llvm.call
// CHECK: %[[GROUP:.*]] = llvm.mlir.constant(128 : i64) : i64
// CHECK-NEXT: %[[EPS:.*]] = llvm.mlir.constant(9.99999997E-7 : f32) : f32
// CHECK-NEXT: %[[NOQ:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_gated_rms_norm(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[NULL]], %[[NULL]], %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[GROUP]], %[[EPS]], %[[NOQ]])
// CHECK-SAME: : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, i64) -> i32
func.func @gated_rms_norm_static(%ctx: !hip.context) {
  %x = memref.alloc() : memref<1x1x4096xf16, 1>
  %z = memref.alloc() : memref<1x1x4096xf16, 1>
  %w = memref.alloc() : memref<128xf16, 1>
  %y = memref.alloc() : memref<1x1x4096xf16, 1>
  hip.gated_rms_norm(%ctx) ins(%x, %z, %w : memref<1x1x4096xf16, 1>, memref<1x1x4096xf16, 1>, memref<128xf16, 1>)
      outs(%y : memref<1x1x4096xf16, 1>) {epsilon = 9.99999997E-7 : f32}
  return
}

// CHECK-LABEL: @gated_rms_norm_quant_dynamic
// CHECK: %[[Q:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[QP:.*]] = llvm.addrspacecast %[[Q]]
// CHECK-NEXT: %[[S:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[SP:.*]] = llvm.addrspacecast %[[S]]
// CHECK: llvm.extractvalue %{{.*}}[3, 1]
// CHECK: llvm.mlir.constant(9.99999997E-7 : f32) : f32
// CHECK-NEXT: %[[QB:.*]] = llvm.mlir.constant(32 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_gated_rms_norm(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[QP]], %[[SP]], {{.*}}, %[[QB]])
func.func @gated_rms_norm_quant_dynamic(%ctx: !hip.context, %t: index) {
  %x = memref.alloc(%t) : memref<1x?x64xf16, 1>
  %z = memref.alloc(%t) : memref<1x?x64xf16, 1>
  %w = memref.alloc() : memref<32xf16, 1>
  %y = memref.alloc(%t) : memref<1x?x64xf16, 1>
  %q = memref.alloc(%t) : memref<1x?x64xi8, 1>
  %s = memref.alloc(%t) : memref<1x?x2xf32, 1>
  hip.gated_rms_norm(%ctx) ins(%x, %z, %w : memref<1x?x64xf16, 1>, memref<1x?x64xf16, 1>, memref<32xf16, 1>)
      outs(%y, %q, %s : memref<1x?x64xf16, 1>, memref<1x?x64xi8, 1>, memref<1x?x2xf32, 1>)
      {epsilon = 9.99999997E-7 : f32, quant_block_size = 32 : i64}
  return
}
