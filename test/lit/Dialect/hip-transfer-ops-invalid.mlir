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
