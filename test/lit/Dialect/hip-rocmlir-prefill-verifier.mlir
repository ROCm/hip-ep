// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// hip.rocmlir prefill triples are (argument index, bytes per element, bit
// pattern). The lowering fills every element of the argument at that width, so
// the width must be the argument's element width: a wider one writes past the
// buffer and a narrower one leaves part of it unfilled.
//
// RUN: hip-mlir-opt --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @prefill_i32_output
// CHECK:         prefill = array<i64: 2, 4, 0>
func.func @prefill_i32_output(%ctx: !hip.context,
                              %x: memref<1x8x4x4xi8, 1>,
                              %w: memref<4x8x3x3xi8, 1>,
                              %y: memref<1x4x4x4xi32, 1>) {
  hip.rocmlir(%ctx) @rocMlir1 ins(%x, %w : memref<1x8x4x4xi8, 1>,
                                           memref<4x8x3x3xi8, 1>)
      outs(%y : memref<1x4x4x4xi32, 1>)
      {block_size = 128 : i64, grid_size = 4 : i64, kernel_binary = "ELF",
       prefill = array<i64: 2, 4, 0>}
  return
}

// -----

func.func @prefill_wider_than_i8(%ctx: !hip.context,
                                 %x: memref<1x8x4x4xi8, 1>,
                                 %y: memref<1x8x4x4xi8, 1>) {
  // expected-error @+1 {{prefill element size 4 does not match the element type 'i8' of argument 1}}
  hip.rocmlir(%ctx) @rocMlir1 ins(%x : memref<1x8x4x4xi8, 1>)
      outs(%y : memref<1x8x4x4xi8, 1>)
      {block_size = 128 : i64, grid_size = 4 : i64, kernel_binary = "ELF",
       prefill = array<i64: 1, 4, 0>}
  return
}

// -----

func.func @prefill_narrower_than_i32(%ctx: !hip.context,
                                     %x: memref<1x8x4x4xi8, 1>,
                                     %y: memref<1x4x4x4xi32, 1>) {
  // expected-error @+1 {{prefill element size 1 does not match the element type 'i32' of argument 1}}
  hip.rocmlir(%ctx) @rocMlir1 ins(%x : memref<1x8x4x4xi8, 1>)
      outs(%y : memref<1x4x4x4xi32, 1>)
      {block_size = 128 : i64, grid_size = 4 : i64, kernel_binary = "ELF",
       prefill = array<i64: 1, 1, 0>}
  return
}
