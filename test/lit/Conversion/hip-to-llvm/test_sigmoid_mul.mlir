// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.sigmoid_mul lowers to wrap_sigmoid_mul(state, input, gate, output, q,
// scale, num_elements, row_size, quant_block_size). q and scale are null
// without quantized outputs, and quant_block_size is then 0.

// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

// CHECK-LABEL: @sigmoid_mul_static
// CHECK: llvm.call @malloc
// CHECK: llvm.call @malloc
// CHECK: llvm.call @malloc
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NOT: llvm.call
// CHECK: %[[NOQ:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_sigmoid_mul(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[NULL]], %[[NULL]], %{{[^,]*}}, %{{[^,]*}}, %[[NOQ]])
// CHECK-SAME: : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> i32
func.func @sigmoid_mul_static(%ctx: !hip.context) {
  %x = memref.alloc() : memref<1x1x4096xf16, 1>
  %g = memref.alloc() : memref<1x1x4096xf16, 1>
  %y = memref.alloc() : memref<1x1x4096xf16, 1>
  hip.sigmoid_mul(%ctx) ins(%x, %g : memref<1x1x4096xf16, 1>, memref<1x1x4096xf16, 1>)
      outs(%y : memref<1x1x4096xf16, 1>)
  return
}

// CHECK-LABEL: @sigmoid_mul_quant_dynamic
// CHECK: %[[Q:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[QP:.*]] = llvm.addrspacecast %[[Q]]
// CHECK-NEXT: %[[S:.*]] = llvm.extractvalue %{{.*}}[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-NEXT: %[[SP:.*]] = llvm.addrspacecast %[[S]]
// CHECK: llvm.extractvalue %{{.*}}[3, 1]
// CHECK-NEXT: llvm.mul
// CHECK-NEXT: llvm.mlir.constant(64 : i64) : i64
// CHECK-NEXT: %[[NUM:.*]] = llvm.mul
// CHECK-NEXT: %[[ROW:.*]] = llvm.mlir.constant(64 : i64) : i64
// CHECK-NEXT: %[[QB:.*]] = llvm.mlir.constant(32 : i64) : i64
// CHECK-NEXT: llvm.call @wrap_sigmoid_mul(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[QP]], %[[SP]], %[[NUM]], %[[ROW]], %[[QB]])
func.func @sigmoid_mul_quant_dynamic(%ctx: !hip.context, %t: index) {
  %x = memref.alloc(%t) : memref<1x?x64xf16, 1>
  %g = memref.alloc(%t) : memref<1x?x64xf16, 1>
  %y = memref.alloc(%t) : memref<1x?x64xf16, 1>
  %q = memref.alloc(%t) : memref<1x?x64xi8, 1>
  %s = memref.alloc(%t) : memref<1x?x2xf32, 1>
  hip.sigmoid_mul(%ctx) ins(%x, %g : memref<1x?x64xf16, 1>, memref<1x?x64xf16, 1>)
      outs(%y, %q, %s : memref<1x?x64xf16, 1>, memref<1x?x64xi8, 1>, memref<1x?x2xf32, 1>)
      {quant_block_size = 32 : i64}
  return
}
