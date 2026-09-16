// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map use-encoding-for-memory-space" %s | FileCheck %s

// A graph input is always device memory, so the host destination is built
// inside the function. Each operand keeps its memory space on the memref, so
// the copy still crosses spaces after bufferization. The op drops its result in
// favour of the destination buffer, and `{{$}}` pins the absence of a trailing
// result type.
// CHECK-LABEL: func.func @copy_d2h(
// CHECK-SAME:    %[[CTX:[^:]+]]: !hipsr.context,
// CHECK-SAME:    %[[SRC:[^:]+]]: memref<1xi64, #hipsr.mem<device>>) {
// CHECK-NEXT:    %[[DST:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1xi64, #hipsr.mem<host>>
// CHECK-NEXT:    hipsr.copy_d2h(%[[CTX]]) ins(%[[SRC]] : memref<1xi64, #hipsr.mem<device>>) outs(%[[DST]] : memref<1xi64, #hipsr.mem<host>>){{$}}
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func @copy_d2h(%ctx: !hipsr.context,
                    %src: tensor<1xi64, #hipsr.mem<device>>) {
  %init = tensor.empty() : tensor<1xi64, #hipsr.mem<host>>
  %0 = hipsr.copy_d2h(%ctx)
      ins(%src : tensor<1xi64, #hipsr.mem<device>>)
      outs(%init : tensor<1xi64, #hipsr.mem<host>>)
      : tensor<1xi64, #hipsr.mem<host>>
  return
}
