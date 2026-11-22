// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.scaled_masked_softmax indexes row_ids by the second-to-last scores axis
// and col_ids by the last, broadcasting a length-1 vector the way Where does.
// The kernel reads ids[r] (or ids[0]) unchecked, so a statically known length
// that is neither 1 nor the axis extent is rejected here.
//
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @full_and_broadcast_ids
// CHECK:         hip.scaled_masked_softmax
func.func @full_and_broadcast_ids(%ctx: !hip.context, %in: memref<2x4x8xf16, 1>,
                                  %rows: memref<1xi64, 1>, %cols: memref<8xi64, 1>,
                                  %out: memref<2x4x8xf16, 1>) {
  hip.scaled_masked_softmax(%ctx) ins(%in, %rows, %cols : memref<2x4x8xf16, 1>,
      memref<1xi64, 1>, memref<8xi64, 1>) outs(%out : memref<2x4x8xf16, 1>)
      {fill = -6.550400e+04 : f32, scale = 1.250000e-01 : f32}
  return
}

// -----

// A dynamic length or extent is left to the runtime check.
// CHECK-LABEL: func.func @dynamic_ids
// CHECK:         hip.scaled_masked_softmax
func.func @dynamic_ids(%ctx: !hip.context, %in: memref<2x?x8xf16, 1>,
                       %rows: memref<3xi64, 1>, %cols: memref<?xi64, 1>,
                       %out: memref<2x?x8xf16, 1>) {
  hip.scaled_masked_softmax(%ctx) ins(%in, %rows, %cols : memref<2x?x8xf16, 1>,
      memref<3xi64, 1>, memref<?xi64, 1>) outs(%out : memref<2x?x8xf16, 1>)
      {fill = -6.550400e+04 : f32, scale = 1.250000e-01 : f32}
  return
}

// -----

func.func @row_ids_too_short(%ctx: !hip.context, %in: memref<2x4x8xf16, 1>,
                             %rows: memref<3xi64, 1>, %cols: memref<8xi64, 1>,
                             %out: memref<2x4x8xf16, 1>) {
  // expected-error @+1 {{id vector of length 3 neither broadcasts nor matches scores axis 1 of size 4}}
  hip.scaled_masked_softmax(%ctx) ins(%in, %rows, %cols : memref<2x4x8xf16, 1>,
      memref<3xi64, 1>, memref<8xi64, 1>) outs(%out : memref<2x4x8xf16, 1>)
      {fill = -6.550400e+04 : f32, scale = 1.250000e-01 : f32}
  return
}

// -----

func.func @col_ids_too_long(%ctx: !hip.context, %in: memref<2x3x4x8xf32, 1>,
                            %rows: memref<4xi64, 1>, %cols: memref<16xi64, 1>,
                            %out: memref<2x3x4x8xf32, 1>) {
  // expected-error @+1 {{id vector of length 16 neither broadcasts nor matches scores axis 3 of size 8}}
  hip.scaled_masked_softmax(%ctx) ins(%in, %rows, %cols : memref<2x3x4x8xf32, 1>,
      memref<4xi64, 1>, memref<16xi64, 1>) outs(%out : memref<2x3x4x8xf32, 1>)
      {fill = -1.000000e+09 : f32, scale = 5.000000e-01 : f32}
  return
}

// -----

func.func @i32_ids(%ctx: !hip.context, %in: memref<2x4x8xf16, 1>,
                   %rows: memref<4xi32, 1>, %cols: memref<8xi32, 1>,
                   %out: memref<2x4x8xf16, 1>) {
  // expected-error @+1 {{row_ids and col_ids must be rank-1 i64}}
  hip.scaled_masked_softmax(%ctx) ins(%in, %rows, %cols : memref<2x4x8xf16, 1>,
      memref<4xi32, 1>, memref<8xi32, 1>) outs(%out : memref<2x4x8xf16, 1>)
      {fill = -6.550400e+04 : f32, scale = 1.250000e-01 : f32}
  return
}
