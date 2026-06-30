// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Verifier diagnostics for the memref-phase async memcpy ops: dst/src must
// agree on element type and (when both static) element count.
//===----------------------------------------------------------------------===//

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s

func.func @memcpy_elem_type_mismatch(%ctx: !hip.context,
                                     %dst: memref<8xi32, #hip.mem<host>>,
                                     %src: memref<8xi64, #hip.mem<device>>) {
  // expected-error @+1 {{dst/src element type mismatch}}
  hip.memcpy_d2h_async(%ctx, %dst, %src
      : memref<8xi32, #hip.mem<host>>, memref<8xi64, #hip.mem<device>>)
  return
}

// -----

func.func @memcpy_elem_count_mismatch(%ctx: !hip.context,
                                      %dst: memref<8xi64, #hip.mem<host>>,
                                      %src: memref<4xi64, #hip.mem<device>>) {
  // expected-error @+1 {{dst/src element count mismatch}}
  hip.memcpy_d2h_async(%ctx, %dst, %src
      : memref<8xi64, #hip.mem<host>>, memref<4xi64, #hip.mem<device>>)
  return
}

// -----

// Wrong EXPLICIT space is rejected even under the transitional
// kAcceptUnspecifiedMemorySpace toggle (which only waives a *missing* space):
// a d2h `src` must be device, so an explicit #hip.mem<host> src fails the
// Hip_DeviceMemRef operand constraint.
func.func @memcpy_d2h_src_not_device(%ctx: !hip.context,
                                     %dst: memref<8xi64, #hip.mem<host>>,
                                     %src: memref<8xi64, #hip.mem<host>>) {
  // expected-error @+1 {{must be device memref}}
  hip.memcpy_d2h_async(%ctx, %dst, %src
      : memref<8xi64, #hip.mem<host>>, memref<8xi64, #hip.mem<host>>)
  return
}

// -----

// A d2h `dst` must be host or pinned, so an explicit #hip.mem<device> dst is
// rejected.
func.func @memcpy_d2h_dst_not_host(%ctx: !hip.context,
                                   %dst: memref<8xi64, #hip.mem<device>>,
                                   %src: memref<8xi64, #hip.mem<device>>) {
  // expected-error @+1 {{must be host or pinned memref}}
  hip.memcpy_d2h_async(%ctx, %dst, %src
      : memref<8xi64, #hip.mem<device>>, memref<8xi64, #hip.mem<device>>)
  return
}

// -----

// An h2d `dst` must be device, so an explicit #hip.mem<managed> dst is rejected
// (managed is a distinct space, not device).
func.func @memcpy_h2d_dst_not_device(%ctx: !hip.context,
                                     %dst: memref<4xi32, #hip.mem<managed>>,
                                     %src: memref<4xi32, #hip.mem<host>>) {
  // expected-error @+1 {{must be device memref}}
  hip.memcpy_h2d_async(%ctx, %dst, %src
      : memref<4xi32, #hip.mem<managed>>, memref<4xi32, #hip.mem<host>>)
  return
}
