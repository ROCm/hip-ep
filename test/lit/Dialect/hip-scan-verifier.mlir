// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s -verify-diagnostics

module {
  func.func private @scan_body(%ctx: !hip.context,
                               %out: memref<4xf32>)

  func.func @empty_scan_inputs(%ctx: !hip.context,
                               %out: memref<4xf32>) {
    // expected-error @+1 {{requires at least one scan input}}
    "hip.scan"(%ctx, %out) {
      body_func = @scan_body,
      num_scan_inputs = 0 : i32,
      num_scan_outputs = 1 : i32,
      num_state_variables = 0 : i32,
      operand_segment_sizes = array<i32: 1, 0, 0, 0, 1>
    } : (!hip.context, memref<4xf32>) -> ()
    return
  }
}
