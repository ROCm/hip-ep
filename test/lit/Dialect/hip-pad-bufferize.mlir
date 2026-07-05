// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Device-space OUTPUT for hip.pad: bufferization + pooling artifacts.
//
// hip-memory-space-pipeline.mlir drives onnx.Pad through convert + bufferize and
// checks the device-space output at the op boundary. This test starts from the
// tensor-phase hip dialect (the form convert-onnx-to-hip produces) with the pad
// result CONSUMED (so it is not a bare function return) and pins:
//
//  BUF  (one-shot-bufferize): the DPS init (a device-space
//       bufferization.alloc_tensor) becomes an IDENTITY #hip.mem<device>
//       memref.alloc, and hip.pad's `outs` is device-typed.
//
//  POOL (+ hip-pool-allocs): hip-pool-allocs folds that device alloc into the
//       space-less GPU pool. memref.view requires the view and pool base to
//       share a space, so the view is created SPACE-LESS over memref<?xi8> and
//       relabeled back to #hip.mem<device> with a memory_space_cast (lowers to
//       an addrspacecast AS 0 -> AS 1 that the host JIT flattens to a no-op).
//       hip.pad's `outs` stays device-typed.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --one-shot-bufferize \
// RUN:   | FileCheck %s --check-prefix=BUF
// RUN: hip-mlir-opt %s --one-shot-bufferize --hip-pool-allocs \
// RUN:   | FileCheck %s --check-prefix=POOL

// The pad result is transferred to host + read so it is a real (non-returned)
// consumer, keeping the test focused on the device-output bufferization
// artifacts rather than function-boundary handling.

// BUF-LABEL: func.func @pad_device_out
// BUF:         %[[ALLOC:.*]] = memref.alloc() {{.*}} : memref<5x6xf32, #hip.mem<device>>
// BUF:         hip.pad(%{{.*}}) ins({{.*}}) outs(%[[ALLOC]] : memref<5x6xf32, #hip.mem<device>>)

// POOL-LABEL: func.func @pad_device_out
// POOL:         %[[POOL:.*]] = hip.get_pool(%{{.*}}, %{{.*}}) : memref<?xi8>
// POOL:         %[[VIEW:.*]] = memref.view %[[POOL]]{{.*}} : memref<?xi8> to memref<5x6xf32>
// POOL:         %[[DEV:.*]] = memref.memory_space_cast %[[VIEW]] : memref<5x6xf32> to memref<5x6xf32, #hip.mem<device>>
// POOL:         hip.pad(%{{.*}}) ins({{.*}}) outs(%[[DEV]] : memref<5x6xf32, #hip.mem<device>>)
func.func @pad_device_out(%ctx: !hip.context, %data: tensor<3x4xf32>,
                          %pads: tensor<4xi64>) -> f32 {
  %ph = hip.transfer(%ctx, %pads : tensor<4xi64>) to #hip.mem<host> -> tensor<4xi64>
  %init = bufferization.alloc_tensor() {memory_space = #hip.mem<device>}
    : tensor<5x6xf32>
  %r = hip.pad(%ctx) ins(%data, %ph : tensor<3x4xf32>, tensor<4xi64>)
                     outs(%init : tensor<5x6xf32>) : tensor<5x6xf32>
  // Host read of the device result: forces a non-returned consumer.
  %rh = hip.transfer(%ctx, %r : tensor<5x6xf32>) to #hip.mem<host> -> tensor<5x6xf32>
  %c0 = arith.constant 0 : index
  %e = tensor.extract %rh[%c0, %c0] : tensor<5x6xf32>
  return %e : f32
}
