// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
// Verifier diagnostics for the memory-space transfer ops:
//   - memref-phase async memcpy: dst/src must agree on element type and (when
//     both static) element count, and the dst/src operand memory spaces must
//     match the copy direction.
//   - tensor-phase hip.transfer: value-preserving, so src/result element type
//     and shape must match (only the target memory space may differ).
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

// -----

// hip.transfer is value-preserving: src/result element type must match.
func.func @transfer_elem_type_mismatch(%ctx: !hip.context,
                                       %src: tensor<8xi64>) {
  // expected-error @+1 {{src/result element type mismatch}}
  %h = hip.transfer(%ctx, %src : tensor<8xi64>) to #hip.mem<host>
         -> tensor<8xi32>
  return
}

// -----

// hip.transfer is value-preserving: src/result shape must match (only the
// memory space may differ).
func.func @transfer_shape_mismatch(%ctx: !hip.context,
                                   %src: tensor<8xi64>) {
  // expected-error @+1 {{src/result shape mismatch}}
  %h = hip.transfer(%ctx, %src : tensor<8xi64>) to #hip.mem<host>
         -> tensor<4xi64>
  return
}
