// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Pins the memory-space operand constraints on the pilot op:
//   * hip.pad     : data / output device; pads / axes host; constant_value is
//                   a by-value scalar (no memory space at all).
// Covers the positive (correctly-spaced) form, the negative (wrong-space)
// diagnostics, the by-value-scalar contract on constant_value, and the
// TRANSITIONAL acceptance of memrefs that carry no hip memory space (plain, or
// a legacy integer space) — the form the current pipeline still emits.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// -----

// Aspect D (positive): hip.pad with device payload (data/output), host
// shape-control tensors (pads/axes), and a by-value scalar constant_value
// (plain f32, no memory space) verifies.
// CHECK-LABEL: func.func @pad_spaces_ok
// CHECK:         hip.pad
func.func @pad_spaces_ok(%ctx: !hip.context,
                         %data: memref<3x4xf32, #hip.mem<device>>,
                         %pads: memref<4xi64, #hip.mem<host>>,
                         %cval: f32,
                         %axes: memref<2xi64, #hip.mem<host>>,
                         %out: memref<5x6xf32, #hip.mem<device>>) {
  hip.pad(%ctx) ins(%data, %pads : memref<3x4xf32, #hip.mem<device>>, memref<4xi64, #hip.mem<host>>)
                cval(%cval : f32)
                axes(%axes : memref<2xi64, #hip.mem<host>>)
                outs(%out : memref<5x6xf32, #hip.mem<device>>) {mode = "constant"}
  return
}

// -----

// Aspect D (negative, device slot): a host `data` is rejected (device required).
func.func @pad_data_wrong_space(%ctx: !hip.context,
                                %data: memref<3x4xf32, #hip.mem<host>>,
                                %pads: memref<4xi64, #hip.mem<host>>,
                                %out: memref<5x6xf32, #hip.mem<device>>) {
  // expected-error @+1 {{must be ranked tensor or device memref}}
  hip.pad(%ctx) ins(%data, %pads : memref<3x4xf32, #hip.mem<host>>, memref<4xi64, #hip.mem<host>>)
                outs(%out : memref<5x6xf32, #hip.mem<device>>) {mode = "constant"}
  return
}

// -----

// Aspect D (negative, host slot): a device `pads` is rejected (host required).
func.func @pad_pads_wrong_space(%ctx: !hip.context,
                                %data: memref<3x4xf32, #hip.mem<device>>,
                                %pads: memref<4xi64, #hip.mem<device>>,
                                %out: memref<5x6xf32, #hip.mem<device>>) {
  // expected-error @+1 {{must be ranked tensor or host memref}}
  hip.pad(%ctx) ins(%data, %pads : memref<3x4xf32, #hip.mem<device>>, memref<4xi64, #hip.mem<device>>)
                outs(%out : memref<5x6xf32, #hip.mem<device>>) {mode = "constant"}
  return
}

// -----

// Aspect D (negative, scalar slot): constant_value is a by-value scalar, so a
// buffer (memref) in that slot is rejected — even a 0-D device memref (the old
// form). The value must be a plain float/integer SSA scalar.
func.func @pad_cval_must_be_scalar(%ctx: !hip.context,
                                   %data: memref<3x4xf32, #hip.mem<device>>,
                                   %pads: memref<4xi64, #hip.mem<host>>,
                                   %cval: memref<f32, #hip.mem<device>>,
                                   %out: memref<5x6xf32, #hip.mem<device>>) {
  // expected-error @+1 {{scalar float or integer}}
  hip.pad(%ctx) ins(%data, %pads : memref<3x4xf32, #hip.mem<device>>, memref<4xi64, #hip.mem<host>>)
                cval(%cval : memref<f32, #hip.mem<device>>)
                outs(%out : memref<5x6xf32, #hip.mem<device>>) {mode = "constant"}
  return
}

// -----

// Aspect F (transitional): hip.pad on a legacy integer memory space (the
// `, 1` form the current pipeline emits) still verifies.
// CHECK-LABEL: func.func @pad_intspace_ok
// CHECK:         hip.pad
func.func @pad_intspace_ok(%ctx: !hip.context,
                           %data: memref<3x4xf32, 1>,
                           %pads: memref<4xi64, 1>,
                           %out: memref<5x6xf32, 1>) {
  hip.pad(%ctx) ins(%data, %pads : memref<3x4xf32, 1>, memref<4xi64, 1>)
                outs(%out : memref<5x6xf32, 1>) {mode = "constant"}
  return
}

// -----

// Aspect F (transitional): hip.pad on no-space memrefs verifies — both the
// device and host slots accept an unspecified space under the toggle.
// CHECK-LABEL: func.func @pad_unspecified_ok
// CHECK:         hip.pad
func.func @pad_unspecified_ok(%ctx: !hip.context,
                              %data: memref<3x4xf32>,
                              %pads: memref<4xi64>,
                              %out: memref<5x6xf32>) {
  hip.pad(%ctx) ins(%data, %pads : memref<3x4xf32>, memref<4xi64>)
                outs(%out : memref<5x6xf32>) {mode = "constant"}
  return
}
