// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s

// A window narrows the data, so the result cannot share the data's buffer and
// comes from the destination. A window operand is read-only, so a constant one
// becomes a host global the op reads in place, while the attributes carry
// straight over.
// CHECK:       memref.global "private" constant @[[ENDS:.+]] : memref<1xi64, #hipsr.mem<host>> = dense<7>
// CHECK-LABEL: func.func @slice(
// CHECK-SAME:    %[[CTX:[^:]+]]: !hipsr.context,
// CHECK-SAME:    %[[DATA:[^:]+]]: memref<8x4xf16, #hipsr.mem<device>>) -> memref<3x4xf16, #hipsr.mem<device>> {
// CHECK-NEXT:    %[[ENDS_BUF:.+]] = memref.get_global @[[ENDS]] : memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:    %[[OUT:.+]] = memref.alloc() {{.*}}: memref<3x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:    hipsr.slice(%[[CTX]]) ins(%[[DATA]] : memref<8x4xf16, #hipsr.mem<device>>) ends(%[[ENDS_BUF]] : memref<1xi64, #hipsr.mem<host>>) outs(%[[OUT]] : memref<3x4xf16, #hipsr.mem<device>>) {axes_attr = array<i64: 0>, starts_attr = array<i64: 1>, steps_attr = array<i64: 1>}
// CHECK-NEXT:    return %[[OUT]] : memref<3x4xf16, #hipsr.mem<device>>
// CHECK-NEXT:  }
func.func @slice(%ctx: !hipsr.context,
                 %data: tensor<8x4xf16, #hipsr.mem<device>>)
    -> tensor<3x4xf16, #hipsr.mem<device>> {
  %ends = arith.constant dense<7> : tensor<1xi64, #hipsr.mem<host>>
  %init = tensor.empty() : tensor<3x4xf16, #hipsr.mem<device>>
  %0 = hipsr.slice(%ctx)
      ins(%data : tensor<8x4xf16, #hipsr.mem<device>>)
      ends(%ends : tensor<1xi64, #hipsr.mem<host>>)
      outs(%init : tensor<3x4xf16, #hipsr.mem<device>>)
      {starts_attr = array<i64: 1>, axes_attr = array<i64: 0>,
       steps_attr = array<i64: 1>}
      : tensor<3x4xf16, #hipsr.mem<device>>
  return %0 : tensor<3x4xf16, #hipsr.mem<device>>
}
