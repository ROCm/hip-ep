// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Pins the memory-space operand constraints on the pilot op:
//   * hip.pad     : data / constant_value / output device; pads / axes host.
// Covers the positive (correctly-spaced) form, the negative (wrong-space)
// diagnostics, and the TRANSITIONAL acceptance of memrefs that carry no hip
// memory space (plain, or a legacy integer space) — the form the current
// pipeline still emits.

// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// -----

// Aspect D (positive): hip.pad with device payload (data/cval/output) and host
// shape-control tensors (pads/axes) verifies.
// CHECK-LABEL: func.func @pad_spaces_ok
// CHECK:         hip.pad
func.func @pad_spaces_ok(%ctx: !hip.context,
                         %data: memref<3x4xf32, #hip.mem<device>>,
                         %pads: memref<4xi64, #hip.mem<host>>,
                         %cval: memref<f32, #hip.mem<device>>,
                         %axes: memref<2xi64, #hip.mem<host>>,
                         %out: memref<5x6xf32, #hip.mem<device>>) {
  hip.pad(%ctx) ins(%data, %pads : memref<3x4xf32, #hip.mem<device>>, memref<4xi64, #hip.mem<host>>)
                cval(%cval : memref<f32, #hip.mem<device>>)
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

// Aspect D (negative, scalar slot): a non-scalar (1-D) device constant_value is
// rejected. ONNX Pad.constant_value is a scalar, so hip.pad requires 0-D even
// though the space (device) is correct.
func.func @pad_cval_not_scalar(%ctx: !hip.context,
                               %data: memref<3x4xf32, #hip.mem<device>>,
                               %pads: memref<4xi64, #hip.mem<host>>,
                               %cval: memref<2xf32, #hip.mem<device>>,
                               %out: memref<5x6xf32, #hip.mem<device>>) {
  // expected-error @+1 {{0-D (scalar) ranked tensor or device memref}}
  hip.pad(%ctx) ins(%data, %pads : memref<3x4xf32, #hip.mem<device>>, memref<4xi64, #hip.mem<host>>)
                cval(%cval : memref<2xf32, #hip.mem<device>>)
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
