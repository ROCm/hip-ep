// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// FileCheck tests for --hip-pool-host-transfers.
//
// Verifies that #hip.mem<host> memref.alloc ops (the D2H destinations of
// bufferized hip.transfer) are packed into one per-function hip.get_host_mem
// pool at distinct aligned offsets, while device-space allocs are left alone.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --hip-pool-host-transfers %s 2>&1 | FileCheck %s

// --- Two host allocs share one hip.get_host_mem at distinct offsets; the
//     original deallocs are erased. ---
// CHECK-LABEL: func.func @two_host_allocs
// CHECK-SAME:    (%[[CTX:.*]]: !hip.context
// CHECK-NOT:     memref.alloc()
// CHECK:         %[[POOL:.*]] = hip.get_host_mem(%[[CTX]], %{{.*}}) : memref<?xi8, #hip.mem<host>>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8, #hip.mem<host>> to memref<8xi64, #hip.mem<host>>
// CHECK:         memref.view %[[POOL]]{{.*}} : memref<?xi8, #hip.mem<host>> to memref<4xi64, #hip.mem<host>>
// CHECK-NOT:     memref.dealloc
func.func @two_host_allocs(%ctx: !hip.context,
                           %src0: memref<8xi64, #hip.mem<device>>,
                           %src1: memref<4xi64, #hip.mem<device>>) {
  %a = memref.alloc() : memref<8xi64, #hip.mem<host>>
  %b = memref.alloc() : memref<4xi64, #hip.mem<host>>
  hip.memcpy_d2h_async(%ctx, %a, %src0
      : memref<8xi64, #hip.mem<host>>, memref<8xi64, #hip.mem<device>>)
  hip.memcpy_d2h_async(%ctx, %b, %src1
      : memref<4xi64, #hip.mem<host>>, memref<4xi64, #hip.mem<device>>)
  hip.stream_sync(%ctx)
  memref.dealloc %a : memref<8xi64, #hip.mem<host>>
  memref.dealloc %b : memref<4xi64, #hip.mem<host>>
  return
}

// --- A device-space alloc is NOT touched (belongs to the GPU pool). ---
// CHECK-LABEL: func.func @device_alloc_untouched
// CHECK-NOT:     hip.get_host_mem
// CHECK:         memref.alloc() : memref<16xf16, #hip.mem<device>>
func.func @device_alloc_untouched(%ctx: !hip.context) {
  %d = memref.alloc() : memref<16xf16, #hip.mem<device>>
  memref.dealloc %d : memref<16xf16, #hip.mem<device>>
  return
}
