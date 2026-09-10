// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Pipeline regression: reuse late strided-promotion temporaries, then move the
// surviving slot's dealloc after its final reused lifetime.
//
// The production pipeline runs this pair after hip-promote-strided-operands.
// The IR below represents two non-overlapping identity-layout temporaries that
// promotion introduced for consumers with contiguous bare-pointer ABIs.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-optimize-memrefs --optimize-allocation-liveness %s | FileCheck %s

// CHECK-LABEL: func.func @reuse_late_promotion_temporaries
// CHECK:         %[[SLOT:.*]] = memref.alloc() : memref<4xf16>
// CHECK:         memref.copy %arg1, %[[SLOT]]
// CHECK:         hip.cast{{.*}}ins(%[[SLOT]] : memref<4xf16>)
// CHECK-NOT:     memref.dealloc
// CHECK-NOT:     memref.alloc
// CHECK:         memref.copy %arg2, %[[SLOT]]
// CHECK:         hip.cast{{.*}}ins(%[[SLOT]] : memref<4xf16>)
// CHECK-NEXT:    memref.dealloc %[[SLOT]]
// CHECK-NEXT:    return
func.func @reuse_late_promotion_temporaries(
    %ctx: !hip.context,
    %src0: memref<4xf16, strided<[2]>>,
    %src1: memref<4xf16, strided<[2]>>,
    %out0: memref<4xf32>,
    %out1: memref<4xf32>) {
  %tmp0 = memref.alloc() : memref<4xf16>
  memref.copy %src0, %tmp0
      : memref<4xf16, strided<[2]>> to memref<4xf16>
  hip.cast(%ctx)
      ins(%tmp0 : memref<4xf16>)
      outs(%out0 : memref<4xf32>)
      {to = 1 : i64}
  memref.dealloc %tmp0 : memref<4xf16>

  %tmp1 = memref.alloc() : memref<4xf16>
  memref.copy %src1, %tmp1
      : memref<4xf16, strided<[2]>> to memref<4xf16>
  hip.cast(%ctx)
      ins(%tmp1 : memref<4xf16>)
      outs(%out1 : memref<4xf32>)
      {to = 1 : i64}
  memref.dealloc %tmp1 : memref<4xf16>
  return
}
