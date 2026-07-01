// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// ============================================================================
// TEST PURPOSE:
// Verify the memref-phase transfer op family lowers to the runtime wrappers:
//   hip.memcpy_d2h_async -> wrap_hipMemcpyD2HAsync (state-based)
//   hip.memcpy_h2d_async -> wrap_hipMemcpyH2DAsync (state-based)
//   hip.stream_sync      -> wrap_hipStreamSynchronize (plain sync, NOT
//                           hipdnn_ep_stream_sync)
// All three wrappers take the RuntimeState* (ctx) and resolve the stream
// internally, so none of their lowerings emit a hipdnn_ep_state_get_stream
// call.
// ============================================================================

// RUN: hip-mlir-opt --convert-hip-to-llvm --split-input-file %s | FileCheck %s

// -----

// D2H: state-based wrap_hipMemcpyD2HAsync, then state-based stream_sync ->
// wrap_hipStreamSynchronize. Neither does its own stream lookup.
// CHECK-LABEL: llvm.func @d2h_then_sync
// CHECK-NOT:     llvm.call @hipdnn_ep_state_get_stream
// CHECK:         llvm.call @wrap_hipMemcpyD2HAsync({{.*}})
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

// H2D symmetry path: state-based wrap_hipMemcpyH2DAsync, no stream lookup.
// CHECK-LABEL: llvm.func @h2d
// CHECK-NOT:     llvm.call @hipdnn_ep_state_get_stream
// CHECK:         llvm.call @wrap_hipMemcpyH2DAsync({{.*}})
func.func @h2d(%ctx: !hip.context,
               %dst: memref<4xi32, #hip.mem<device>>,
               %src: memref<4xi32, #hip.mem<host>>) {
  hip.memcpy_h2d_async(%ctx, %dst, %src
      : memref<4xi32, #hip.mem<device>>, memref<4xi32, #hip.mem<host>>)
  return
}
