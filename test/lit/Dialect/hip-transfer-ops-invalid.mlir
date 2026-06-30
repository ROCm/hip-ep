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
