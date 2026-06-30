// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Bufferization round-trip for hip.transfer (HipTransferBufferizableModel).
//
// hip-transfer-ops.mlir only parse/print round-trips the op; the pipeline
// test (hip-memory-space-pipeline.mlir) drives a full onnx.Pad through
// bufferize but asserts only that hip.pad reaches memref form. This test
// pins the device->host lowering ARTIFACTS directly: a tensor-phase
// hip.transfer to #hip.mem<host> must become a STACK alloca lifted to the host
// space via memory_space_cast, plus hip.memcpy_d2h_async and hip.stream_sync:
//   memref.alloca (default space)
//     + memref.memory_space_cast -> #hip.mem<host>
//     + hip.memcpy_d2h_async  +  hip.stream_sync
// (the trailing stream_sync is required because the host reads `dst` next).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --one-shot-bufferize="bufferize-function-boundaries" \
// RUN:   | FileCheck %s

// The result is consumed by a host-side tensor.extract (a CPU read of the
// transferred buffer) rather than returned, so no allocation escapes the
// function boundary -- keeps the test focused on the transfer artifacts.

// CHECK-LABEL: func.func @transfer_d2h_bufferize
// CHECK:         %[[SLOT:.*]] = memref.alloca() : memref<8xi64>
// CHECK:         %[[DST:.*]] = memref.memory_space_cast %[[SLOT]] : memref<8xi64> to memref<8xi64, #hip.mem<host>>
// CHECK:         hip.memcpy_d2h_async(%{{.*}}, %[[DST]], %{{.*}} : memref<8xi64, #hip.mem<host>>,
// CHECK:         hip.stream_sync(%{{.*}})
// CHECK:         memref.load %[[DST]]
func.func @transfer_d2h_bufferize(%ctx: !hip.context, %src: tensor<8xi64>) -> i64 {
  %h = hip.transfer(%ctx, %src : tensor<8xi64>) to #hip.mem<host> -> tensor<8xi64>
  %c0 = arith.constant 0 : index
  %e = tensor.extract %h[%c0] : tensor<8xi64>
  return %e : i64
}
