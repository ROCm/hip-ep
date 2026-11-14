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

// -----

// A row-major strided layout that starts at the base is the same buffer as the
// identity layout, so it is accepted.
// CHECK-LABEL: func.func @prefill_contiguous_strided
func.func @prefill_contiguous_strided(%ctx: !hip.context,
                                      %x: memref<8xi8, 1>,
                                      %y: memref<2x4xf32, strided<[4, 1]>, 1>) {
  hip.rocmlir(%ctx) @rocMlir1 ins(%x : memref<8xi8, 1>)
      outs(%y : memref<2x4xf32, strided<[4, 1]>, 1>)
      {block_size = 64 : i64, grid_size = 1 : i64, kernel_binary = "ELF",
       prefill = array<i64: 1, 4, 0>}
  return
}

// -----

func.func @prefill_not_triples(%ctx: !hip.context, %x: memref<8xi8, 1>,
                               %y: memref<8xi32, 1>) {
  // expected-error @+1 {{prefill holds (index, bytes, pattern) triples}}
  hip.rocmlir(%ctx) @rocMlir1 ins(%x : memref<8xi8, 1>)
      outs(%y : memref<8xi32, 1>)
      {block_size = 64 : i64, grid_size = 1 : i64, kernel_binary = "ELF",
       prefill = array<i64: 1, 4>}
  return
}

// -----

func.func @prefill_out_of_range(%ctx: !hip.context, %x: memref<8xi8, 1>,
                                %y: memref<8xi32, 1>) {
  // expected-error @+1 {{prefill argument index 2 is out of range}}
  hip.rocmlir(%ctx) @rocMlir1 ins(%x : memref<8xi8, 1>)
      outs(%y : memref<8xi32, 1>)
      {block_size = 64 : i64, grid_size = 1 : i64, kernel_binary = "ELF",
       prefill = array<i64: 2, 4, 0>}
  return
}

// -----

// A fill of an input wipes data that the kernel reads.
func.func @prefill_input(%ctx: !hip.context, %x: memref<8xi32, 1>,
                         %y: memref<8xi32, 1>) {
  // expected-error @+1 {{prefill argument index 0 names an input; only an outs operand is pre-filled}}
  hip.rocmlir(%ctx) @rocMlir1 ins(%x : memref<8xi32, 1>)
      outs(%y : memref<8xi32, 1>)
      {block_size = 64 : i64, grid_size = 1 : i64, kernel_binary = "ELF",
       prefill = array<i64: 0, 4, 0>}
  return
}

// -----

func.func @prefill_output_is_input(%ctx: !hip.context, %y: memref<8xi32, 1>) {
  // expected-error @+1 {{pre-filled argument 1 is also an input}}
  hip.rocmlir(%ctx) @rocMlir1 ins(%y : memref<8xi32, 1>)
      outs(%y : memref<8xi32, 1>)
      {block_size = 64 : i64, grid_size = 1 : i64, kernel_binary = "ELF",
       prefill = array<i64: 1, 4, 0>}
  return
}

// -----

func.func @prefill_twice(%ctx: !hip.context, %x: memref<8xi8, 1>,
                         %y: memref<8xi32, 1>) {
  // expected-error @+1 {{prefill argument index 1 is listed twice}}
  hip.rocmlir(%ctx) @rocMlir1 ins(%x : memref<8xi8, 1>)
      outs(%y : memref<8xi32, 1>)
      {block_size = 64 : i64, grid_size = 1 : i64, kernel_binary = "ELF",
       prefill = array<i64: 1, 4, 0, 1, 4, 7>}
  return
}

// -----

func.func @prefill_dynamic(%ctx: !hip.context, %x: memref<8xi8, 1>,
                           %y: memref<?xi32, 1>) {
  // expected-error @+1 {{pre-filled argument 1 needs a static shape}}
  hip.rocmlir(%ctx) @rocMlir1 ins(%x : memref<8xi8, 1>)
      outs(%y : memref<?xi32, 1>)
      {block_size = 64 : i64, grid_size = 1 : i64, kernel_binary = "ELF",
       prefill = array<i64: 1, 4, 0>}
  return
}

// -----

func.func @prefill_strided(%ctx: !hip.context, %x: memref<8xi8, 1>,
                           %y: memref<2x4xi32, strided<[8, 1]>, 1>) {
  // expected-error @+1 {{pre-filled argument 1 needs a contiguous layout with no offset}}
  hip.rocmlir(%ctx) @rocMlir1 ins(%x : memref<8xi8, 1>)
      outs(%y : memref<2x4xi32, strided<[8, 1]>, 1>)
      {block_size = 64 : i64, grid_size = 1 : i64, kernel_binary = "ELF",
       prefill = array<i64: 1, 4, 0>}
  return
}

// -----

func.func @prefill_offset(%ctx: !hip.context, %x: memref<8xi8, 1>,
                          %y: memref<4xi32, strided<[1], offset: 4>, 1>) {
  // expected-error @+1 {{pre-filled argument 1 needs a contiguous layout with no offset}}
  hip.rocmlir(%ctx) @rocMlir1 ins(%x : memref<8xi8, 1>)
      outs(%y : memref<4xi32, strided<[1], offset: 4>, 1>)
      {block_size = 64 : i64, grid_size = 1 : i64, kernel_binary = "ELF",
       prefill = array<i64: 1, 4, 0>}
  return
}
