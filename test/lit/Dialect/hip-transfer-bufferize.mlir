// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Bufferization of hip.transfer_to_host (HipTransferToHostBufferizableModel):
// a tensor-phase device->host transfer becomes a STACK memref.alloca lifted to
// #hip.mem<host> via memory_space_cast, then hip.memcpy_d2h_async +
// hip.stream_sync (the trailing sync is required -- the host reads dst next).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   | FileCheck %s

// The result is consumed by a host-side tensor.extract (not returned), so no
// allocation escapes the boundary -- keeps the focus on the transfer artifacts.

// CHECK-LABEL: func.func @transfer_d2h_bufferize
// CHECK:         %[[SLOT:.*]] = memref.alloca() : memref<8xi64>
// CHECK:         %[[DST:.*]] = memref.memory_space_cast %[[SLOT]] : memref<8xi64> to memref<8xi64, #hip.mem<host>>
// CHECK:         hip.memcpy_d2h_async(%{{.*}}, %[[DST]], %{{.*}} : memref<8xi64, #hip.mem<host>>,
// CHECK:         hip.stream_sync(%{{.*}})
// CHECK:         memref.load %[[DST]]
func.func @transfer_d2h_bufferize(%ctx: !hip.context, %src: tensor<8xi64>) -> i64 {
  %h = hip.transfer_to_host(%ctx, %src : tensor<8xi64>) -> tensor<8xi64>
  %c0 = arith.constant 0 : index
  %e = tensor.extract %h[%c0] : tensor<8xi64>
  return %e : i64
}
