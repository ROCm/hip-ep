// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Memory-space operand constraints on hip.pad (the pilot op): data/output are
// device, pads/axes are host, constant_value is a by-value scalar. Covers the
// OK form, the wrong-space diagnostics, the scalar-only cval, and the
// transitional acceptance of memrefs with no hip space (or a legacy int space).

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// -----

// OK: device data/output, host pads/axes, by-value f32 cval.
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

// Reject: host `data` in a device slot.
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

// Reject: device `pads` in a host slot.
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

// Reject: a buffer in the by-value cval slot (even a 0-D device memref); the
// value must be a plain float/integer scalar.
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

// Transitional: legacy integer space (the `, 1` form the current pipeline
// emits) still verifies.
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

// Transitional: no-space memrefs verify (device and host slots both accept an
// unspecified space under the toggle).
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
