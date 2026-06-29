// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the memref-phase transfer op family lowers to the EXISTING runtime
// wrappers (no new memcpy/sync runtime symbols):
//   hip.memcpy_d2h_async -> wrap_hipMemcpyD2H
//   hip.memcpy_h2d_async -> wrap_hipMemcpyH2D
//   hip.stream_sync      -> wrap_hipStreamSynchronize (plain sync, NOT
//                           hipdnn_ep_stream_sync)
//   hip.get_host_mem     -> hipdnn_ep_get_host_mem_base (the one new symbol)
// Every memcpy/sync first resolves the stream via hipdnn_ep_state_get_stream.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-llvm --split-input-file %s | FileCheck %s

// -----

// D2H: stream lookup, then wrap_hipMemcpyD2H; stream_sync -> wrap_hipStreamSynchronize.
// CHECK-LABEL: llvm.func @d2h_then_sync
// CHECK:         llvm.call @hipdnn_ep_state_get_stream({{.*}})
// CHECK:         llvm.call @wrap_hipMemcpyD2H({{.*}})
// CHECK:         llvm.call @wrap_hipStreamSynchronize({{.*}})
// CHECK-NOT:     llvm.call @hipdnn_ep_stream_sync
func.func @d2h_then_sync(%ctx: !hip.context,
                         %dst: memref<8xi64, #hip.mem<host>>,
                         %src: memref<8xi64, #hip.mem<device>>) {
  hip.memcpy_d2h_async(%ctx, %dst, %src
      : memref<8xi64, #hip.mem<host>>, memref<8xi64, #hip.mem<device>>)
  hip.stream_sync(%ctx)
  return
}

// -----

// H2D symmetry path.
// CHECK-LABEL: llvm.func @h2d
// CHECK:         llvm.call @hipdnn_ep_state_get_stream({{.*}})
// CHECK:         llvm.call @wrap_hipMemcpyH2D({{.*}})
func.func @h2d(%ctx: !hip.context,
               %dst: memref<4xi32, #hip.mem<device>>,
               %src: memref<4xi32, #hip.mem<host>>) {
  hip.memcpy_h2d_async(%ctx, %dst, %src
      : memref<4xi32, #hip.mem<device>>, memref<4xi32, #hip.mem<host>>)
  return
}

// -----

// hip.get_host_mem -> hipdnn_ep_get_host_mem_base + a memref descriptor.
// CHECK-LABEL: llvm.func @host_mem
// CHECK:         llvm.call @hipdnn_ep_get_host_mem_base({{.*}})
func.func @host_mem(%ctx: !hip.context, %sz: index) {
  %m = hip.get_host_mem(%ctx, %sz) : memref<?xi8, #hip.mem<host>>
  return
}
