// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Device-space OUTPUT for hip.pad: bufferization + pooling artifacts. Starts
// from the tensor-phase hip dialect with the pad result CONSUMED (not a bare
// return) and pins:
//   BUF  (one-shot-bufferize): the device-space DPS init becomes an identity
//        #hip.mem<device> memref.alloc; hip.pad's outs is device-typed.
//   POOL (+ hip-pool-allocs): the device alloc folds into the space-less GPU
//        pool as a space-less memref.view relabeled back to #hip.mem<device>
//        with a memory_space_cast (an AS0->AS1 addrspacecast, no-op on host JIT).
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt %s --one-shot-bufferize \
// RUN:   | FileCheck %s --check-prefix=BUF
// RUN: hip-mlir-opt %s --one-shot-bufferize --hip-pool-allocs \
// RUN:   | FileCheck %s --check-prefix=POOL

// The pad result is transferred to host + read (a real, non-returned consumer)
// to keep the focus on the device-output artifacts, not boundary handling.

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
  %ph = hip.transfer_to_host(%ctx, %pads : tensor<4xi64>) -> tensor<4xi64>
  %init = bufferization.alloc_tensor() {memory_space = #hip.mem<device>}
    : tensor<5x6xf32>
  %r = hip.pad(%ctx) ins(%data, %ph : tensor<3x4xf32>, tensor<4xi64>)
                     outs(%init : tensor<5x6xf32>) : tensor<5x6xf32>
  // Host read of the device result: a non-returned consumer.
  %rh = hip.transfer_to_host(%ctx, %r : tensor<5x6xf32>) -> tensor<5x6xf32>
  %c0 = arith.constant 0 : index
  %e = tensor.extract %rh[%c0, %c0] : tensor<5x6xf32>
  return %e : f32
}
